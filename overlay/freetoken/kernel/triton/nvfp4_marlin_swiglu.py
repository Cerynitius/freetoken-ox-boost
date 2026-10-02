"""NVFP4 Marlin-style decode down GEMV with GLM-5.3's ``swiglu_clamp`` folded in.

Copy of :func:`freetoken.kernel.triton.nvfp4_fused_moe._decode_nvfp4_marlin_kernel`
(``A_ROW_IS_ROUTE=True`` down GEMV) whose A operand is built on the fly from the
``[routes, 2I]`` gate_up rows, like that kernel's ``FUSE_SILU`` branch -- but with the
clamped SwiGLU of :func:`freetoken.kernel.triton.dsv4.fused_moe._swiglu_kernel`:

    g = min(gate, limit); u = clamp(up, -limit, limit)      (HAS_LIMIT, i.e. limit > 0)
    a = bf16((g * sigmoid(g)) * u)                          (fp32 math, one rounding)

It removes the separate ``fused_swiglu`` launch, its ``[routes, I]`` intermediate and
the ``out.copy_`` launch of ``_run_act`` (twice per layer in the split hit/miss path).

Bit-identical to ``fused_swiglu`` -> ``_decode_nvfp4_marlin_kernel(ic2, ...)``:
  * the activation is the same fp32 op sequence (``tl.minimum``/``tl.maximum`` with the
    same fp32 ``limit``, ``tl.sigmoid``, ``(g * s) * u``) followed by the same
    round-to-nearest-even cast to the gate_up dtype that ``_swiglu_kernel`` stores, then
    the exact upcast the GEMV applies when it loads ``ic2``; elementwise, so the layout
    it is computed in cannot change its bits;
  * the GEMV body (word loads, e2m1 LUT, per-word scale, deferred ``[KW, N]`` partial,
    final ``tl.sum``, global scale, router weight, store) is unchanged, launched with
    the same BLOCK_N / BLOCK_KW / num_warps as the unfused down GEMV.
Masked-out K lanes (``other=0``) give ``a == 0`` exactly as the unfused load does.
"""

from __future__ import annotations

import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32


@triton.jit
def _decode_nvfp4_marlin_swiglu_kernel(
    a_ptr,             # [routes, 2K] gate_up rows ([gate | up] halves, compute dtype)
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k=8*w+j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [M, TOP_K, N] output (compute dtype)
    topk_weights_ptr,  # [M, TOP_K] fp32
    topk_ids_ptr,      # [M, TOP_K] int32 -> cache slot
    lut_ptr,           # [16] fp32
    route_mask_ptr,    # [M * TOP_K] int8, 1 = compute this route (HAS_MASK only)
    act_up_off,        # offset of the up half inside a gate_up row (== K)
    act_limit,         # swiglu clamp limit (fp32), used when HAS_LIMIT
    total_routes,
    N,
    K,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_ck, stride_cn,
    stride_tw_m, stride_tw_k,
    stride_tid_m, stride_tid_k,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,  # int32 words per K-iter (covers 8*KW k-values)
    TOP_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    act_type: tl.constexpr,       # dtype _swiglu_kernel rounds to (== gate_up dtype)
    HAS_MASK: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
):
    route_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    if HAS_MASK:
        # split decode (hit/miss phases): this launch owns only the masked-in routes;
        # the other phase writes the rest of the (disjoint) route rows.
        if tl.load(route_mask_ptr + route_id) == 0:
            return
    token_id = route_id // TOP_K
    route_k = route_id - token_id * TOP_K

    offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N

    slot = tl.load(topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k).to(tl.int64)
    a_base = a_ptr + route_id * stride_am  # A_ROW_IS_ROUTE

    offs_kw = tl.arange(0, BLOCK_SIZE_KW)
    K_WORDS = K // 8
    partial = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)

    packed_slot = packed_ptr + slot * stride_pe
    scale_slot = scale_ptr + slot * stride_se
    for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
        widx = kw_start * BLOCK_SIZE_KW + offs_kw
        w_mask = widx < K_WORDS

        word = tl.load(
            packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
            mask=w_mask[:, None] & n_mask[None, :], other=0,
        )
        # 8 codes/word fall in the same or adjacent 16-wide block -> one scale per word.
        s_ptrs = scale_slot + offs_n[None, :] * stride_sn + (widx[:, None] // 2) * stride_sblk
        s_mask = w_mask[:, None] & n_mask[None, :]
        if e4m3_native_cx():
            scale = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)
        else:
            scale = e4m3_u8_to_f32(tl.load(s_ptrs, mask=s_mask, other=0))

        kbase = 8 * widx
        acc_w = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
        for j in tl.static_range(8):
            code = (word >> (4 * j)) & 0xF
            b = tl.load(lut_ptr + code)
            # a = bf16(swiglu_clamp(gate, up)) straight from the gate_up rows -- the same
            # fp32 math + rounding _swiglu_kernel stores into ic2.
            g_j = tl.load(a_base + (kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
            u_j = tl.load(a_base + (act_up_off + kbase + j) * stride_ak, mask=w_mask, other=0.0).to(tl.float32)
            if HAS_LIMIT:
                g_j = tl.minimum(g_j, act_limit)
                u_j = tl.minimum(tl.maximum(u_j, -act_limit), act_limit)
            a_j = ((g_j * tl.sigmoid(g_j)) * u_j).to(act_type).to(tl.float32)
            acc_w += a_j[:, None] * b
        partial += acc_w * scale

    accumulator = tl.sum(partial, axis=0)
    g = tl.load(global_ptr + slot * stride_ge + offs_n * stride_gn, mask=n_mask, other=0.0).to(tl.float32)
    accumulator = accumulator * g

    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k)
        accumulator = accumulator * weight

    c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(compute_type), mask=(route_id < total_routes) & n_mask)


__all__ = ["_decode_nvfp4_marlin_swiglu_kernel"]
