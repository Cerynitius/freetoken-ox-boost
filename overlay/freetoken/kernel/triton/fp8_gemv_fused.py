"""Single-launch decode FP8 W8A16 split-K GEMVs, bit-identical to the two-launch
``_gemv_splitk_kernel`` (+``_m``) -> ``_splitk_reduce_kernel`` (+``_m``) pairs in
``fp8_pertensor_linear`` (opt-in, default OFF).

FREETOKEN_FP8_GEMV_FUSED_REDUCE=1 -- every M<=4 decode GEMV drops its reduce launch
  (GLM-5.3 bs=1: 247 per step). The split-K partial loop is a VERBATIM copy of the
  original kernels (same BLOCK_N/BLOCK_K/num_warps/split_k/kb_per, same loads, same
  ``tl.sum``), so every partial is the same fp32 value. Each CTA stores its partial,
  then bumps a per-n-tile arrival counter (acq_rel, gpu scope; CUTLASS split-K
  "serial reduction" semaphore pattern); the LAST arrival re-reads all partials in
  the fixed order k = 0..SPLIT_K-1 starting from 0.0 -- exactly the reduce kernel's
  ``acc = 0; acc += part[k]`` chain -- applies the per-row scale and casts. The order
  never depends on which CTA arrives last, so results are deterministic and equal
  to the reduce kernel's bit for bit. SPLIT_K == 1 needs no counter: the epilogue is
  ``(0.0 + acc) * scale`` in registers (== store fp32, reload, reduce). The last CTA
  resets its counter to 0, so the counters are self-cleaning across launches and
  CUDA-graph replays (allocated once, outside capture, per weight).

FREETOKEN_GLM5_MLP_FUSED=1 -- GLM-5.3 clamped-SwiGLU MLP (shared experts + dense
  layers): gate GEMV + up GEMV + both reduces + ``fused_swiglu`` -> ONE launch. Both
  weights keep their own (identical-shape) split-K schedule; the last of the
  2*SPLIT_K arrivals for an n-tile reduces gate then up exactly as above, rounds each
  to the projection dtype (the bf16 the two-launch path stores), and applies the
  ``_swiglu_kernel`` expression verbatim. Bit-identical by construction.

Both are opt-in per process (read once at import) and fall back to the original
launches whenever a precondition is not met (no counter yet during graph capture,
non-fp8 projection, bias, W8A8 input_scale, M > 1 for the MLP path)."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

FUSED_REDUCE_ENV = "FREETOKEN_FP8_GEMV_FUSED_REDUCE"
MLP_FUSED_ENV = "FREETOKEN_GLM5_MLP_FUSED"
FUSED_REDUCE = os.environ.get(FUSED_REDUCE_ENV, "0") == "1"
MLP_FUSED = os.environ.get(MLP_FUSED_ENV, "0") == "1"

_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Same constants/formulas as fp8_pertensor_linear._gemv/_gemv_m (keep in lock-step).
_BLOCK_N = 16
_BLOCK_K = 128
_MAX_CTAS = 1536


def _split(N: int, K: int):
    n_kb = triton.cdiv(K, _BLOCK_K)
    n_tiles = triton.cdiv(N, _BLOCK_N)
    split_k = max(1, min(_MAX_CTAS // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    return n_kb, n_tiles, split_k, kb_per


# ---- per-weight arrival counters (int32 [n_tiles], zero at rest) ----------------------
_COUNTERS: dict = {}


def _counter(weight: torch.Tensor, n_tiles: int, tag: str) -> torch.Tensor | None:
    """Zeroed int32 [n_tiles] owned by (device, weight storage, tag). One per weight, so
    GEMVs on different streams never share a counter. Created on first (eager / warmup)
    use; a capture that would need a NEW counter returns None (caller falls back to the
    two-launch path for that call -- never a captured memset)."""
    key = (weight.device, weight.data_ptr(), tag)
    c = _COUNTERS.get(key)
    if c is not None and c.numel() >= n_tiles:
        return c
    if weight.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    c = torch.zeros(n_tiles, dtype=torch.int32, device=weight.device)
    _COUNTERS[key] = c
    return c


# ======================================================================================
# M == 1: fused split-K GEMV + reduce
# ======================================================================================
@triton.jit
def _gemv_splitk_fused_kernel(
    a_ptr, w_ptr, part_ptr, scale_ptr, out_ptr, cnt_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_pk, stride_pn,
    SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    # ---- verbatim _gemv_splitk_kernel body ----
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            acc += tl.sum(w * a[None, :], axis=1)
    # ---- reduce epilogue (== _splitk_reduce_kernel on this tile's rows) ----
    if SPLIT_K == 1:
        r = tl.zeros((BLOCK_N,), dtype=tl.float32)
        r += acc
        scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs_n, (r * scale).to(OUT), mask=n_mask)
    else:
        tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)
        tl.debug_barrier()  # every thread's partial is issued before the arrival
        arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            r = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k in tl.static_range(SPLIT_K):
                r += tl.load(part_ptr + k * stride_pk + offs_n * stride_pn, mask=n_mask,
                             other=0.0, cache_modifier=".cg")
            scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + offs_n, (r * scale).to(OUT), mask=n_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


def gemv_fused(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
               out_dtype: torch.dtype) -> torch.Tensor | None:
    """Drop-in for ``fp8_pertensor_linear._gemv`` (same args, same result bits).
    None -> caller runs the original two launches."""
    N, K = weight.shape
    n_kb, n_tiles, split_k, kb_per = _split(N, K)
    if split_k > 1:
        cnt = _counter(weight, n_tiles, "gemv")
        if cnt is None:
            return None
        part = torch.empty((split_k, N), dtype=torch.float32, device=a.device)
    else:
        cnt = part = weight_scale  # unused (constexpr-pruned) pointers
    out = torch.empty(N, dtype=out_dtype, device=a.device)
    _gemv_splitk_fused_kernel[(n_tiles, split_k)](
        a, weight, part, weight_scale, out, cnt, N, K, n_kb, kb_per,
        a.stride(0), weight.stride(0), weight.stride(1),
        part.stride(0) if split_k > 1 else 0, part.stride(1) if split_k > 1 else 0,
        SPLIT_K=split_k, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16],
        num_warps=1,
    )
    return out


# ======================================================================================
# 2 <= M <= 4: fused M-tiled split-K GEMV + reduce
# ======================================================================================
@triton.jit
def _gemv_splitk_m_fused_kernel(
    a_ptr, w_ptr, part_ptr, scale_ptr, out_ptr, cnt_ptr, N, K, M, n_kb, kb_per,
    stride_am, stride_ak, stride_wn, stride_wk, stride_pk, stride_pm, stride_pn, stride_om,
    SPLIT_K: tl.constexpr, M_TILE: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, OUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_m = tl.arange(0, M_TILE)
    kb_start = pid_k * kb_per
    # ---- verbatim _gemv_splitk_m_kernel body ----
    acc = tl.zeros((M_TILE, BLOCK_N), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)                               # [N_b, K_b], read ONCE
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            for m in tl.static_range(M_TILE):
                if m < M:  # M_TILE is M padded to pow2 (tl.arange constraint)
                    a_m = tl.load(a_ptr + m * stride_am + offs_k * stride_ak,
                                  mask=k_mask, other=0.0).to(tl.float32)
                    c = tl.sum(w * a_m[None, :], axis=1)       # [N_b]
                    acc = acc + tl.where(offs_m[:, None] == m, c[None, :], 0.0)
    m_mask = offs_m < M
    p_ptrs = part_ptr + pid_k * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    tl.store(p_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
    # ---- reduce epilogue (== _splitk_reduce_m_kernel for every row m < M) ----
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT_K - 1:
        scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        for m in tl.static_range(M_TILE):
            if m < M:
                r = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for k in tl.static_range(SPLIT_K):
                    r += tl.load(part_ptr + k * stride_pk + m * stride_pm + offs_n * stride_pn,
                                 mask=n_mask, other=0.0, cache_modifier=".cg")
                tl.store(out_ptr + m * stride_om + offs_n, (r * scale).to(OUT), mask=n_mask)
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


def gemv_m_fused(a: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                 out_dtype: torch.dtype) -> torch.Tensor | None:
    """Drop-in for ``fp8_pertensor_linear._gemv_m`` (same args, same result bits)."""
    M, K = a.shape
    N = weight.shape[0]
    n_kb, n_tiles, split_k, kb_per = _split(N, K)
    cnt = _counter(weight, n_tiles, "gemv_m")
    if cnt is None:
        return None
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    compute = out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16
    out = torch.empty((M, N), dtype=compute, device=a.device)
    _gemv_splitk_m_fused_kernel[(n_tiles, split_k)](
        a, weight, part, weight_scale, out, cnt, N, K, M, n_kb, kb_per,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
        part.stride(0), part.stride(1), part.stride(2), out.stride(0),
        SPLIT_K=split_k, M_TILE=(1 << (M - 1).bit_length()), BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K, OUT=_TL_DTYPE[compute], num_warps=1,
    )
    return out


# ======================================================================================
# GLM-5.3 clamped-SwiGLU MLP: gate + up GEMV + both reduces + swiglu in one launch (M==1)
# ======================================================================================
@triton.jit
def _gemv2_swiglu_fused_kernel(
    a_ptr, wg_ptr, wu_ptr, part_ptr, sg_ptr, su_ptr, out_ptr, cnt_ptr,
    N, K, n_kb, kb_per, limit,
    stride_ak, stride_gn, stride_gk, stride_un, stride_uk, stride_pz, stride_pk, stride_pn,
    SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PROJ: tl.constexpr, OUT: tl.constexpr, HAS_LIMIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    z = tl.program_id(2)  # 0: gate_proj, 1: up_proj
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Two verbatim copies of the _gemv_splitk_kernel loop, each on its OWN pointer
    # argument (a selected pointer would lose the argument's alignment facts and could
    # change the load layout -> the tl.sum reduction tree).
    if z == 0:
        for i in range(kb_per):
            kb = kb_start + i
            if kb < n_kb:
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K
                a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
                if e4m3_native_cx():
                    w = tl.load(
                        wg_ptr + offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk,
                        mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                    ).to(tl.float32)
                else:
                    w = e4m3_u8_to_f32(tl.load(
                        wg_ptr + offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk,
                        mask=n_mask[:, None] & k_mask[None, :], other=0,
                    ))
                acc += tl.sum(w * a[None, :], axis=1)
    else:
        for i in range(kb_per):
            kb = kb_start + i
            if kb < n_kb:
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K
                a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
                if e4m3_native_cx():
                    w = tl.load(
                        wu_ptr + offs_n[:, None] * stride_un + offs_k[None, :] * stride_uk,
                        mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                    ).to(tl.float32)
                else:
                    w = e4m3_u8_to_f32(tl.load(
                        wu_ptr + offs_n[:, None] * stride_un + offs_k[None, :] * stride_uk,
                        mask=n_mask[:, None] & k_mask[None, :], other=0,
                    ))
                acc += tl.sum(w * a[None, :], axis=1)
    tl.store(part_ptr + z * stride_pz + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if arrived == 2 * SPLIT_K - 1:
        # gate = _splitk_reduce_kernel(part[0]) ; up = _splitk_reduce_kernel(part[1])
        rg = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):
            rg += tl.load(part_ptr + k * stride_pk + offs_n * stride_pn, mask=n_mask,
                          other=0.0, cache_modifier=".cg")
        sg = tl.load(sg_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        gate = (rg * sg).to(PROJ)
        ru = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(SPLIT_K):
            ru += tl.load(part_ptr + stride_pz + k * stride_pk + offs_n * stride_pn,
                          mask=n_mask, other=0.0, cache_modifier=".cg")
        su = tl.load(su_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        up = (ru * su).to(PROJ)
        # ---- verbatim dsv4.swiglu._swiglu_kernel on the projection-dtype values ----
        g = gate.to(tl.float32)
        u = up.to(tl.float32)
        if HAS_LIMIT:
            u = tl.minimum(tl.maximum(u, -limit), limit)
            g = tl.minimum(g, limit)
        g = g * tl.sigmoid(g)
        tl.store(out_ptr + offs_n, (g * u).to(OUT), mask=n_mask)
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


def gate_up_swiglu_fused(x: torch.Tensor, gate_proj, up_proj, limit: float) -> torch.Tensor | None:
    """``fused_swiglu(gate_proj(x), up_proj(x), limit, x.dtype)`` for one decode token, in one
    launch, bit-identical. None when the projections would not take the M==1 W8A16 GEMV path
    (caller then runs the original three ops)."""
    from freetoken.kernel.triton import fp8_pertensor_linear as fpl
    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native

    if not (isinstance(gate_proj, fpl.Fp8PerTensorLinear) and isinstance(up_proj, fpl.Fp8PerTensorLinear)):
        return None
    if fpl._USE_REF or gate_proj.bias is not None or up_proj.bias is not None:
        return None
    if (gate_proj.input_scale is not None or up_proj.input_scale is not None) and e4m3_native():
        return None  # deployment runs W8A8 (_scaled_mm) -- not this kernel's math
    *lead, K = x.shape
    if x.numel() // K != 1:
        return None
    wg, wu = gate_proj.weight, up_proj.weight
    if wg.shape != wu.shape or wg.shape[1] != K:
        return None
    N = wg.shape[0]
    proj_dtype = x.dtype  # fp8_pertensor_linear output dtype == activation dtype
    if proj_dtype not in _TL_DTYPE:
        return None
    n_kb, n_tiles, split_k, kb_per = _split(N, K)
    cnt = _counter(wg, n_tiles, "gate_up")
    if cnt is None:
        return None
    a = x.reshape(K)
    wg, wu = e4m3_kernel_view(wg), e4m3_kernel_view(wu)
    part = torch.empty((2, split_k, N), dtype=torch.float32, device=x.device)
    out = torch.empty((*lead, N), dtype=x.dtype, device=x.device)
    _gemv2_swiglu_fused_kernel[(n_tiles, split_k, 2)](
        a, wg, wu, part, gate_proj.weight_scale, up_proj.weight_scale, out, cnt,
        N, K, n_kb, kb_per, float(limit),
        a.stride(0), wg.stride(0), wg.stride(1), wu.stride(0), wu.stride(1),
        part.stride(0), part.stride(1), part.stride(2),
        SPLIT_K=split_k, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        PROJ=_TL_DTYPE[proj_dtype], OUT=_TL_DTYPE[x.dtype], HAS_LIMIT=limit > 0,
        num_warps=1,
    )
    return out


__all__ = [
    "FUSED_REDUCE", "MLP_FUSED", "gemv_fused", "gemv_m_fused", "gate_up_swiglu_fused",
]
