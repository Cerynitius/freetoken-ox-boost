"""Triton kernels for Naive-N0.5-Flash attention (GQA, K 192 / V 128).

* ``naive_rope_``            in-place partial NeoX rope (first R dims) from fp32 tables.
* ``naive_round_fp8``        per-row fp8-e4m3 rounding == the reference round_indexer_fp8.
* ``naive_decode_attention`` split-K GQA decode; window mode (SWA, page-table slots with a
                             sliding window) or sparse mode (a per-request list of selected
                             slots, -1 = empty). Sinks and the V scale fold into stage 2.
* ``naive_extend_attention`` causal prefill over the pool (all K/V already stored), GQA
                             heads of one KV head share each K/V tile; optional window.
* ``naive_sparse_prefill``   prefill attention over per-token selected slot lists.
* ``naive_index_scores_*``   lightning-indexer scores relu(q.k) weighted over heads.

The 192-wide K head is processed as a 128 + 64 split (no padding to 256).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_D1 = 128  # first chunk of the 192-wide q/k head
_D2 = 64  # second chunk


# ----------------------------------------------------------------------------- rope
@triton.jit
def _rope_kernel(
    x_ptr, pos_ptr, cos_ptr, sin_ptr,
    stride_xt, stride_xh,
    NUM_HEADS: tl.constexpr, HALF: tl.constexpr, BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_r = tl.arange(0, HALF)
    mask = offs_h[:, None] < NUM_HEADS
    base = x_ptr + t * stride_xt + offs_h[:, None] * stride_xh + offs_r[None, :]
    x1 = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + pos * HALF + offs_r)[None, :]
    s = tl.load(sin_ptr + pos * HALF + offs_r)[None, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    tl.store(base, o1.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(base + HALF, o2.to(x_ptr.dtype.element_ty), mask=mask)


def naive_rope_(x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate the first ``2*cos.shape[1]`` dims of every head of ``x`` [T, H, D] in place."""
    T, H, _ = x.shape
    if T == 0:
        return x
    half = cos.shape[1]
    block_h = min(16, triton.next_power_of_2(H))
    _rope_kernel[(T, triton.cdiv(H, block_h))](
        x, positions, cos, sin, x.stride(0), x.stride(1),
        NUM_HEADS=H, HALF=half, BLOCK_H=block_h, num_warps=2,
    )
    return x


def build_rope_tables(max_pos: int, rotary_dim: int, base: float, device) -> tuple:
    """fp32 cos/sin halves [max_pos, rotary_dim/2], computed like HF's default rope
    (inv_freq in fp32, angle = pos * inv_freq rounded in fp32, fp32 cos/sin)."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = t[:, None] * inv_freq[None, :]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


# ----------------------------------------------------------------------------- fp8 rounding
@triton.jit
def _round_fp8_kernel(x_ptr, q_ptr, s_ptr, stride_x, D: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + r * stride_x + offs).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    # torch lowers ``t / 448.`` to ``t * (1/448)`` (fp32 reciprocal); match it bit-exactly
    scale = tl.maximum(amax, 1e-4) * (1.0 / 448.0)
    y = tl.math.div_rn(x, scale)
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    tl.store(q_ptr + r * D + offs, y.to(tl.float8e4nv, fp_downcast_rounding="rtne"))
    tl.store(s_ptr + r, scale)


def naive_round_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``x`` [..., D] -> (fp8-e4m3 [..., D], fp32 scale [...]) with
    value == fp8 * scale exactly as the reference ``round_indexer_fp8``."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    R = x2.shape[0]
    q = torch.empty((R, D), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((R,), dtype=torch.float32, device=x.device)
    if R:
        _round_fp8_kernel[(R,)](x2, q, s, x2.stride(0), D=D, num_warps=1)
    return q.view(*x.shape), s.view(*x.shape[:-1])


# ----------------------------------------------------------------------------- decode
@triton.jit
def _decode_stage1(
    q_ptr, k_ptr, v_ptr, sm_scale,
    indptr_ptr, indices_ptr, qpos_ptr, sel_ptr,
    mid_o_ptr, mid_lse_ptr,
    stride_qb, stride_qh, stride_ks, stride_kh, stride_vs, stride_vh,
    stride_sel,
    stride_mob, stride_moh, stride_mos, stride_lb, stride_lh,
    GROUP: tl.constexpr, DV: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    NSPLIT: tl.constexpr, SPARSE: tl.constexpr, TOPK: tl.constexpr, WINDOW: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    offs_d1 = tl.arange(0, 128)
    offs_d2 = tl.arange(0, 64)
    offs_dv = tl.arange(0, DV)

    if SPARSE:
        n_total = TOPK
        base = 0
        kv_start = 0
    else:
        kv_start = tl.load(indptr_ptr + b)
        kv_len = tl.load(indptr_ptr + b + 1) - kv_start
        pos = tl.load(qpos_ptr + b)
        end = tl.minimum(kv_len, pos + 1)
        base = 0
        if WINDOW > 0:
            base = tl.maximum(0, pos - WINDOW + 1)
        n_total = tl.maximum(end - base, 0)
    per = tl.cdiv(tl.cdiv(n_total, NSPLIT), BLOCK_N) * BLOCK_N
    s0 = sp * per
    s1 = tl.minimum(s0 + per, n_total)

    qb = q_ptr + b * stride_qb + heads[:, None] * stride_qh
    q1 = tl.load(qb + offs_d1[None, :], mask=mask_h[:, None], other=0.0)
    q2 = tl.load(qb + 128 + offs_d2[None, :], mask=mask_h[:, None], other=0.0)

    m_i = tl.zeros((BLOCK_H,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, DV), dtype=tl.float32)
    for r in tl.range(s0, s1, BLOCK_N):
        offs = r + tl.arange(0, BLOCK_N)
        mask_n = offs < s1
        if SPARSE:
            slots = tl.load(sel_ptr + b * stride_sel + offs, mask=mask_n, other=-1)
            mask_n = mask_n & (slots >= 0)
            slots = tl.maximum(slots, 0)
        else:
            slots = tl.load(indices_ptr + kv_start + base + offs, mask=mask_n, other=0)
        slots = slots.to(tl.int64)
        kb = k_ptr + slots[None, :] * stride_ks + kvh * stride_kh
        k1 = tl.load(kb + offs_d1[:, None], mask=mask_n[None, :], other=0.0)
        k2 = tl.load(kb + 128 + offs_d2[:, None], mask=mask_n[None, :], other=0.0)
        s = tl.dot(q1, k1) + tl.dot(q2, k2)
        s = s * sm_scale
        s = tl.where(mask_h[:, None] & mask_n[None, :], s, -float("inf"))
        v = tl.load(
            v_ptr + slots[:, None] * stride_vs + kvh * stride_vh + offs_dv[None, :],
            mask=mask_n[:, None], other=0.0,
        )
        rmax = tl.max(s, axis=1)
        rmax = tl.where(rmax == -float("inf"), -1e20, rmax)
        m_new = tl.maximum(rmax, m_i)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    has = l_i > 0.0
    out = tl.where(has[:, None], acc / tl.where(has, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), -float("inf"))
    tl.store(
        mid_o_ptr + b * stride_mob + heads[:, None] * stride_moh + sp * stride_mos
        + offs_dv[None, :],
        out, mask=mask_h[:, None],
    )
    tl.store(mid_lse_ptr + b * stride_lb + heads * stride_lh + sp, lse, mask=mask_h)


@triton.jit
def _decode_stage2(
    mid_o_ptr, mid_lse_ptr, o_ptr, sinks_ptr,
    stride_mob, stride_moh, stride_mos, stride_lb, stride_lh,
    stride_ob, stride_oh, out_scale,
    DV: tl.constexpr, NSPLIT: tl.constexpr, HAS_SINKS: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_dv = tl.arange(0, DV)
    offs_s = tl.arange(0, NSPLIT)
    lse = tl.load(mid_lse_ptr + b * stride_lb + h * stride_lh + offs_s)
    m = tl.max(lse, axis=0)
    if HAS_SINKS:
        sink = tl.load(sinks_ptr + h).to(tl.float32)
        m = tl.maximum(m, sink)
    m = tl.where(m == -float("inf"), 0.0, m)
    w = tl.exp(lse - m)
    parts = tl.load(
        mid_o_ptr + b * stride_mob + h * stride_moh + offs_s[:, None] * stride_mos
        + offs_dv[None, :]
    )
    num = tl.sum(parts * w[:, None], axis=0)
    den = tl.sum(w, axis=0)
    if HAS_SINKS:
        den += tl.exp(sink - m)
    out = tl.where(den > 0.0, num / tl.where(den > 0.0, den, 1.0), 0.0) * out_scale
    tl.store(o_ptr + b * stride_ob + h * stride_oh + offs_dv, out.to(o_ptr.dtype.element_ty))


def naive_decode_attention(
    q: torch.Tensor,  # [B, H, 192]
    k_cache: torch.Tensor,  # [T, Hk, 192]
    v_cache: torch.Tensor,  # [T, Hk, DV]
    sm_scale: float,
    out_scale: float,
    mid_o: torch.Tensor,  # [>=B, H, NSPLIT, DV] fp32 scratch
    mid_lse: torch.Tensor,  # [>=B, H, NSPLIT] fp32 scratch
    *,
    indptr: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    q_positions: torch.Tensor | None = None,
    window: int = 0,
    sel: torch.Tensor | None = None,  # [B, TOPK] int32 slots, -1 = empty
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, D = q.shape
    Hk = k_cache.shape[1]
    DV = v_cache.shape[-1]
    assert D == _D1 + _D2 and k_cache.shape[-1] == D
    group = H // Hk
    nsplit = mid_lse.shape[-1]
    sparse = sel is not None
    topk = sel.shape[1] if sparse else 0
    o = out if out is not None else torch.empty((B, H, DV), dtype=q.dtype, device=q.device)
    dummy = q
    block_h = max(16, triton.next_power_of_2(group))
    _decode_stage1[(B, Hk, nsplit)](
        q, k_cache, v_cache, sm_scale,
        indptr if indptr is not None else dummy,
        indices if indices is not None else dummy,
        q_positions if q_positions is not None else dummy,
        sel if sparse else dummy,
        mid_o, mid_lse,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1),
        sel.stride(0) if sparse else 0,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        GROUP=group, DV=DV, BLOCK_H=block_h, BLOCK_N=32 if not sparse else 64,
        NSPLIT=nsplit, SPARSE=sparse, TOPK=topk, WINDOW=window,
        num_warps=4, num_stages=2,
    )
    _decode_stage2[(B, H)](
        mid_o, mid_lse, o, sinks if sinks is not None else dummy,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_lse.stride(0), mid_lse.stride(1),
        o.stride(0), o.stride(1), out_scale,
        DV=DV, NSPLIT=nsplit, HAS_SINKS=sinks is not None, num_warps=2,
    )
    return o


# ----------------------------------------------------------------------------- extend
@triton.jit
def _extend_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, sinks_ptr,
    qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr, prefix_ptr,
    sm_scale, out_scale,
    stride_qt, stride_qh, stride_ks, stride_kh, stride_vs, stride_vh, stride_ot, stride_oh,
    GROUP: tl.constexpr, DV: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr, HAS_SINKS: tl.constexpr,
):
    """One program = BLOCK_T query tokens x GROUP heads of one KV head (rows t*GROUP+h)."""
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    mb = tl.program_id(2)
    ROWS: tl.constexpr = BLOCK_T * GROUP
    q_start = tl.load(qo_indptr_ptr + seq)
    q_len = tl.load(qo_indptr_ptr + seq + 1) - q_start
    kv_start = tl.load(kv_indptr_ptr + seq)
    prefix = tl.load(prefix_ptr + seq)
    t0 = mb * BLOCK_T
    if t0 >= q_len:
        return
    rows = tl.arange(0, ROWS)
    tok = t0 + rows // GROUP
    head = kvh * GROUP + rows % GROUP
    mask_r = tok < q_len
    qpos = prefix + tok
    offs_d1 = tl.arange(0, 128)
    offs_d2 = tl.arange(0, 64)
    offs_dv = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    qb = q_ptr + (q_start + tok)[:, None].to(tl.int64) * stride_qt + head[:, None] * stride_qh
    q1 = tl.load(qb + offs_d1[None, :], mask=mask_r[:, None], other=0.0)
    q2 = tl.load(qb + 128 + offs_d2[None, :], mask=mask_r[:, None], other=0.0)

    if HAS_SINKS:
        sink = tl.load(sinks_ptr + head).to(tl.float32)
        m_i = sink
        l_i = tl.zeros((ROWS,), dtype=tl.float32) + 1.0
    else:
        m_i = tl.zeros((ROWS,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((ROWS,), dtype=tl.float32)
    acc = tl.zeros((ROWS, DV), dtype=tl.float32)

    last_pos = prefix + tl.minimum(q_len, t0 + BLOCK_T) - 1
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(0, prefix + t0 - WINDOW + 1)
        lo = (lo // BLOCK_N) * BLOCK_N
    for start in tl.range(lo, last_pos + 1, BLOCK_N):
        kpos = start + offs_n
        mask_n = kpos <= last_pos
        slots = tl.load(kv_indices_ptr + kv_start + kpos, mask=mask_n, other=0).to(tl.int64)
        kb = k_ptr + slots[None, :] * stride_ks + kvh * stride_kh
        k1 = tl.load(kb + offs_d1[:, None], mask=mask_n[None, :], other=0.0)
        k2 = tl.load(kb + 128 + offs_d2[:, None], mask=mask_n[None, :], other=0.0)
        s = (tl.dot(q1, k1) + tl.dot(q2, k2)) * sm_scale
        allowed = mask_r[:, None] & mask_n[None, :] & (kpos[None, :] <= qpos[:, None])
        if WINDOW > 0:
            allowed = allowed & ((kpos[None, :] + WINDOW) > qpos[:, None])
        s = tl.where(allowed, s, -float("inf"))
        v = tl.load(
            v_ptr + slots[:, None] * stride_vs + kvh * stride_vh + offs_dv[None, :],
            mask=mask_n[:, None], other=0.0,
        )
        rmax = tl.max(s, axis=1)
        rmax = tl.where(rmax == -float("inf"), -1e20, rmax)
        m_new = tl.maximum(rmax, m_i)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    out = tl.where(l_i[:, None] > 0.0, acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None], 0.0)
    out = out * out_scale
    ob = o_ptr + (q_start + tok)[:, None].to(tl.int64) * stride_ot + head[:, None] * stride_oh
    tl.store(ob + offs_dv[None, :], out.to(o_ptr.dtype.element_ty), mask=mask_r[:, None])


def naive_extend_attention(
    q: torch.Tensor,  # [N, H, 192]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: float,
    out_scale: float,
    window: int = 0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    N, H, D = q.shape
    Hk = k_cache.shape[1]
    DV = v_cache.shape[-1]
    group = H // Hk
    o = out if out is not None else torch.empty((N, H, DV), dtype=q.dtype, device=q.device)
    block_t = max(1, 128 // group)
    grid = (qo_indptr.numel() - 1, Hk, triton.cdiv(max_q_len, block_t))
    _extend_kernel[grid](
        q, k_cache, v_cache, o, sinks if sinks is not None else q,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        sm_scale, out_scale,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), o.stride(0), o.stride(1),
        GROUP=group, DV=DV, BLOCK_T=block_t, BLOCK_N=64,
        WINDOW=window, HAS_SINKS=sinks is not None,
        num_warps=8, num_stages=2,
    )
    return o


# ----------------------------------------------------------------------------- sparse prefill
@triton.jit
def _sparse_prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, sel_ptr, sinks_ptr,
    sm_scale, out_scale,
    stride_qt, stride_qh, stride_ks, stride_kh, stride_vs, stride_vh, stride_ot, stride_oh,
    stride_sel,
    GROUP: tl.constexpr, DV: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr, HAS_SINKS: tl.constexpr,
):
    t = tl.program_id(0)
    kvh = tl.program_id(1)
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    offs_d1 = tl.arange(0, 128)
    offs_d2 = tl.arange(0, 64)
    offs_dv = tl.arange(0, DV)
    qb = q_ptr + t.to(tl.int64) * stride_qt + heads[:, None] * stride_qh
    q1 = tl.load(qb + offs_d1[None, :], mask=mask_h[:, None], other=0.0)
    q2 = tl.load(qb + 128 + offs_d2[None, :], mask=mask_h[:, None], other=0.0)
    if HAS_SINKS:
        m_i = tl.load(sinks_ptr + heads, mask=mask_h, other=0.0).to(tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32) + 1.0
    else:
        m_i = tl.zeros((BLOCK_H,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, DV), dtype=tl.float32)
    for start in tl.range(0, TOPK, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        slots = tl.load(sel_ptr + t.to(tl.int64) * stride_sel + offs)
        mask_n = slots >= 0
        if tl.max(mask_n.to(tl.int32), axis=0) > 0:
            sl = tl.maximum(slots, 0).to(tl.int64)
            kb = k_ptr + sl[None, :] * stride_ks + kvh * stride_kh
            k1 = tl.load(kb + offs_d1[:, None], mask=mask_n[None, :], other=0.0)
            k2 = tl.load(kb + 128 + offs_d2[:, None], mask=mask_n[None, :], other=0.0)
            s = (tl.dot(q1, k1) + tl.dot(q2, k2)) * sm_scale
            s = tl.where(mask_h[:, None] & mask_n[None, :], s, -float("inf"))
            v = tl.load(
                v_ptr + sl[:, None] * stride_vs + kvh * stride_vh + offs_dv[None, :],
                mask=mask_n[:, None], other=0.0,
            )
            rmax = tl.max(s, axis=1)
            rmax = tl.where(rmax == -float("inf"), -1e20, rmax)
            m_new = tl.maximum(rmax, m_i)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new
    out = tl.where(l_i[:, None] > 0.0, acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None], 0.0)
    out = out * out_scale
    ob = o_ptr + t.to(tl.int64) * stride_ot + heads[:, None] * stride_oh
    tl.store(ob + offs_dv[None, :], out.to(o_ptr.dtype.element_ty), mask=mask_h[:, None])


def naive_sparse_prefill(
    q: torch.Tensor,  # [N, H, 192]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sel: torch.Tensor,  # [N, TOPK] int32 full slots, -1 = empty
    sm_scale: float,
    out_scale: float,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    N, H, D = q.shape
    Hk = k_cache.shape[1]
    DV = v_cache.shape[-1]
    group = H // Hk
    topk = sel.shape[1]
    assert topk % 64 == 0
    o = out if out is not None else torch.empty((N, H, DV), dtype=q.dtype, device=q.device)
    if N == 0:
        return o
    _sparse_prefill_kernel[(N, Hk)](
        q, k_cache, v_cache, o, sel, sinks if sinks is not None else q,
        sm_scale, out_scale,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
        v_cache.stride(0), v_cache.stride(1), o.stride(0), o.stride(1), sel.stride(0),
        GROUP=group, DV=DV, BLOCK_H=max(16, triton.next_power_of_2(group)), BLOCK_N=64,
        TOPK=topk, HAS_SINKS=sinks is not None, num_warps=4, num_stages=2,
    )
    return o


# ----------------------------------------------------------------------------- indexer
@triton.jit
def _index_scores_kernel(
    q8_ptr, qs_ptr, w_ptr, k8_ptr, ks_ptr, slots_ptr, qpos_ptr, out_ptr,
    n_keys_ptr, n_q, n_cols,
    stride_q8t, stride_q8h, stride_qst, stride_wt, stride_out,
    NH: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """scores[t, j] = sum_h w[t,h] * relu(qs[t,h] * ks[j] * <q8[t,h], k8[j]>) for keys
    j < n_keys with pos_j <= qpos[t] (key position == j); -inf elsewhere up to n_cols."""
    tb = tl.program_id(0)
    nb = tl.program_id(1)
    ROWS: tl.constexpr = BLOCK_T * NH
    rows = tl.arange(0, ROWS)
    tok = tb * BLOCK_T + rows // NH
    hh = rows % NH
    mask_r = tok < n_q
    n_keys = tl.load(n_keys_ptr)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    tok_c = tl.minimum(tok, n_q - 1)
    qpos = tl.load(qpos_ptr + tok_c)
    col_ok = offs_n < n_keys
    any_live = tl.max(col_ok.to(tl.int32), axis=0) > 0
    otok = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    if any_live:
        q = tl.load(
            q8_ptr + tok_c[:, None].to(tl.int64) * stride_q8t + hh[:, None] * stride_q8h
            + offs_d[None, :]
        ).to(tl.bfloat16)
        slots = tl.load(slots_ptr + offs_n, mask=col_ok, other=0).to(tl.int64)
        k = tl.load(k8_ptr + slots[None, :] * D + offs_d[:, None], mask=col_ok[None, :], other=0.0)
        k = k.to(tl.bfloat16)
        dot = tl.dot(q, k)
        qs = tl.load(qs_ptr + tok_c * stride_qst + hh)
        ks = tl.load(ks_ptr + slots, mask=col_ok, other=0.0)
        w = tl.load(w_ptr + tok_c * stride_wt + hh)
        sc = tl.maximum(dot * qs[:, None] * ks[None, :], 0.0) * w[:, None]
        sc = tl.reshape(sc, (BLOCK_T, NH, BLOCK_N))
        tot = tl.sum(sc, axis=1)
        opos = tl.load(qpos_ptr + tl.minimum(otok, n_q - 1))
        ok = col_ok[None, :] & (offs_n[None, :] <= opos[:, None])
        tot = tl.where(ok, tot, -float("inf"))
    else:
        tot = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32) - float("inf")
    tl.store(
        out_ptr + otok[:, None].to(tl.int64) * stride_out + offs_n[None, :], tot,
        mask=(otok[:, None] < n_q) & (offs_n[None, :] < n_cols),
    )


def naive_index_scores(
    q8: torch.Tensor,  # [n, NH, D] fp8
    qs: torch.Tensor,  # [n, NH] fp32
    w: torch.Tensor,  # [n, NH] fp32
    k8: torch.Tensor,  # [T, D] fp8 (layer slab)
    ks: torch.Tensor,  # [T] fp32
    slots: torch.Tensor,  # [>= n_cols] int32 key slots in position order
    q_positions: torch.Tensor,  # [n] int (absolute query positions)
    n_keys: torch.Tensor,  # [1] int32 device (live key count)
    n_cols: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    n, NH, D = q8.shape
    o = out if out is not None else torch.empty((n, n_cols), dtype=torch.float32, device=q8.device)
    block_t = max(1, 64 // NH)
    block_n = 128
    grid = (triton.cdiv(n, block_t), triton.cdiv(n_cols, block_n))
    _index_scores_kernel[grid](
        q8, qs, w, k8, ks, slots, q_positions, o, n_keys, n, n_cols,
        q8.stride(0), q8.stride(1), qs.stride(0), w.stride(0), o.stride(0),
        NH=NH, D=D, BLOCK_T=block_t, BLOCK_N=block_n, num_warps=4,
    )
    return o


__all__ = [
    "naive_rope_",
    "build_rope_tables",
    "naive_round_fp8",
    "naive_decode_attention",
    "naive_extend_attention",
    "naive_sparse_prefill",
    "naive_index_scores",
]


# ----------------------------------------------------------------------------- fused prep
@triton.jit
def _rope_head(x_ptr, D: tl.constexpr, BLOCK: tl.constexpr, HALF: tl.constexpr, c, s):
    """Load one head (``D`` values, fp32) with the NeoX rope applied to its first 2*HALF."""
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < D, other=0.0).to(tl.float32)
    offs_h = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + offs_h).to(tl.float32)
    x2 = tl.load(x_ptr + HALF + offs_h).to(tl.float32)
    return x, x1 * c - x2 * s, x2 * c + x1 * s


@triton.jit
def _naive_prep_kernel(
    proj_ptr, stride_pt, pos_ptr, loc_ptr, swa_map_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_cache_ptr, v_cache_ptr, stride_ks, stride_kh, stride_vs, stride_vh,
    iq8_ptr, iqs_ptr, iw_ptr, ik8_ptr, iks_ptr, ln_w_ptr, ln_b_ptr, iw_scale,
    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, HALF: tl.constexpr,
    IS_SWA: tl.constexpr, NIH: tl.constexpr, ID: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.program_id(1)
    row = proj_ptr + t.to(tl.int64) * stride_pt
    pos = tl.load(pos_ptr + t).to(tl.int64)
    offs_h = tl.arange(0, HALF)
    c = tl.load(cos_ptr + pos * HALF + offs_h)
    s = tl.load(sin_ptr + pos * HALF + offs_h)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if IS_SWA:
        slot = tl.load(swa_map_ptr + loc)
    else:
        slot = loc
    OFF_K: tl.constexpr = H * D
    OFF_V: tl.constexpr = OFF_K + HK * D
    OFF_IQ: tl.constexpr = OFF_V + HK * DV
    OFF_IK: tl.constexpr = OFF_IQ + NIH * ID
    OFF_IW: tl.constexpr = OFF_IK + ID
    offs = tl.arange(0, 256)
    dmask = offs < D
    if j < H:  # ---- q head: rope -> q_out
        x, o1, o2 = _rope_head(row + j * D, D, 256, HALF, c, s)
        dst = q_out_ptr + (t.to(tl.int64) * H + j) * D
        tl.store(dst + offs, x.to(q_out_ptr.dtype.element_ty), mask=dmask & (offs >= 2 * HALF))
        tl.store(dst + offs_h, o1.to(q_out_ptr.dtype.element_ty))
        tl.store(dst + HALF + offs_h, o2.to(q_out_ptr.dtype.element_ty))
    elif j < H + HK:  # ---- k head: rope -> k cache slot
        h = j - H
        x, o1, o2 = _rope_head(row + OFF_K + h * D, D, 256, HALF, c, s)
        dst = k_cache_ptr + slot * stride_ks + h * stride_kh
        tl.store(dst + offs, x.to(k_cache_ptr.dtype.element_ty), mask=dmask & (offs >= 2 * HALF))
        tl.store(dst + offs_h, o1.to(k_cache_ptr.dtype.element_ty))
        tl.store(dst + HALF + offs_h, o2.to(k_cache_ptr.dtype.element_ty))
    elif j < H + 2 * HK:  # ---- v head -> v cache slot
        h = j - H - HK
        offs_v = tl.arange(0, DV)
        v = tl.load(row + OFF_V + h * DV + offs_v)
        tl.store(v_cache_ptr + slot * stride_vs + h * stride_vh + offs_v, v)
    elif not IS_SWA:
        jj = j - H - 2 * HK
        offs_i = tl.arange(0, ID)
        if jj < NIH:  # ---- indexer q head: rope (bf16 round) -> fp8 round
            x = tl.load(row + OFF_IQ + jj * ID + offs_i).to(tl.float32)
            x1 = tl.load(row + OFF_IQ + jj * ID + offs_h).to(tl.float32)
            x2 = tl.load(row + OFF_IQ + jj * ID + HALF + offs_h).to(tl.float32)
            r1 = (x1 * c - x2 * s).to(tl.bfloat16).to(tl.float32)
            r2 = (x2 * c + x1 * s).to(tl.bfloat16).to(tl.float32)
            y = tl.where(offs_i < HALF, 0.0, tl.where(offs_i < 2 * HALF, 0.0, x))
            # assemble the roped head in registers: [r1 | r2 | x[2*HALF:]]
            y = y + tl.sum(tl.where(offs_i[:, None] == offs_h[None, :], r1[None, :], 0.0), axis=1)
            y = y + tl.sum(tl.where(offs_i[:, None] == (offs_h + HALF)[None, :], r2[None, :], 0.0), axis=1)
            amax = tl.max(tl.abs(y), axis=0)
            scale = tl.maximum(amax, 1e-4) * (1.0 / 448.0)
            qv = tl.minimum(tl.maximum(tl.math.div_rn(y, scale), -448.0), 448.0)
            tl.store(iq8_ptr + (t.to(tl.int64) * NIH + jj) * ID + offs_i,
                     qv.to(tl.float8e4nv, fp_downcast_rounding="rtne"))
            tl.store(iqs_ptr + t * NIH + jj, scale)
        elif jj == NIH:  # ---- indexer key: LayerNorm (bf16 out) -> rope -> fp8 -> cache
            x = tl.load(row + OFF_IK + offs_i).to(tl.float32)
            mean = tl.sum(x, axis=0) / ID
            xc = x - mean
            var = tl.sum(xc * xc, axis=0) / ID
            lw = tl.load(ln_w_ptr + offs_i).to(tl.float32)
            lb = tl.load(ln_b_ptr + offs_i).to(tl.float32)
            y = (xc * tl.math.rsqrt(var + 1e-5) * lw + lb).to(tl.bfloat16).to(tl.float32)
            y1 = tl.sum(tl.where(offs_i[None, :] == offs_h[:, None], y[None, :], 0.0), axis=1)
            y2 = tl.sum(tl.where(offs_i[None, :] == (offs_h + HALF)[:, None], y[None, :], 0.0), axis=1)
            r1 = (y1 * c - y2 * s).to(tl.bfloat16).to(tl.float32)
            r2 = (y2 * c + y1 * s).to(tl.bfloat16).to(tl.float32)
            z = tl.where(offs_i < 2 * HALF, 0.0, y)
            z = z + tl.sum(tl.where(offs_i[:, None] == offs_h[None, :], r1[None, :], 0.0), axis=1)
            z = z + tl.sum(tl.where(offs_i[:, None] == (offs_h + HALF)[None, :], r2[None, :], 0.0), axis=1)
            amax = tl.max(tl.abs(z), axis=0)
            scale = tl.maximum(amax, 1e-4) * (1.0 / 448.0)
            kv = tl.minimum(tl.maximum(tl.math.div_rn(z, scale), -448.0), 448.0)
            tl.store(ik8_ptr + loc * ID + offs_i, kv.to(tl.float8e4nv, fp_downcast_rounding="rtne"))
            tl.store(iks_ptr + loc, scale)
        else:  # ---- indexer head weights: bf16 proj * H^-0.5 -> fp32
            offs_w = tl.arange(0, NIH)
            wv = tl.load(row + OFF_IW + offs_w).to(tl.float32) * iw_scale
            tl.store(iw_ptr + t * NIH + offs_w, wv)


def naive_prep(
    proj: torch.Tensor,  # [T, NTOT] bf16 (merged projection output)
    positions: torch.Tensor,
    out_loc: torch.Tensor,  # full-pool slots [T]
    swa_map: torch.Tensor | None,  # full->swa mapping (SWA layers) or None
    cos: torch.Tensor, sin: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,  # layer slabs [S, HK, D] / [S, HK, DV]
    H: int, HK: int, D: int, DV: int,
    index: tuple | None = None,  # (ik8_cache [S,ID] fp8, iks_cache [S], ln_w, ln_b, NIH, ID, iw_scale)
):
    """One launch per layer: rope(q) -> q_out, rope(k)/v -> cache, and (DSA) the indexer
    q/k/w prep incl. the index-key cache store. Returns q_out [T,H,D] (+ iq8, iqs, iw)."""
    T = proj.shape[0]
    q_out = torch.empty((T, H, D), dtype=proj.dtype, device=proj.device)
    is_swa = index is None
    if is_swa:
        nih, idim, iw_scale = 1, 128, 0.0
        iq8 = iqs = iw = ik8 = iks = lnw = lnb = q_out
        n_prog = H + 2 * HK
    else:
        ik8, iks, lnw, lnb, nih, idim, iw_scale = index
        iq8 = torch.empty((T, nih, idim), dtype=torch.float8_e4m3fn, device=proj.device)
        iqs = torch.empty((T, nih), dtype=torch.float32, device=proj.device)
        iw = torch.empty((T, nih), dtype=torch.float32, device=proj.device)
        n_prog = H + 2 * HK + nih + 2
    if T:
        _naive_prep_kernel[(T, n_prog)](
            proj, proj.stride(0), positions, out_loc, swa_map if swa_map is not None else out_loc,
            cos, sin, q_out, k_cache, v_cache,
            k_cache.stride(0), k_cache.stride(1), v_cache.stride(0), v_cache.stride(1),
            iq8, iqs, iw, ik8, iks, lnw, lnb, iw_scale,
            H=H, HK=HK, D=D, DV=DV, HALF=cos.shape[1], IS_SWA=is_swa, NIH=nih, ID=idim,
            num_warps=1,
        )
    if is_swa:
        return q_out, None
    return q_out, (iq8, iqs, iw)



# ----------------------------------------------------------------------------- DSA decode top-k
# Exact top-TOPK over the LIVE index scores only, as a fixed-grid (CUDA-graph safe) kernel
# chain whose cost scales with the live key count instead of the graph-static page width
# (torch.topk over a 1M-wide row costs ~86 us/layer even at 100 tokens of context).
# State (int32 [8]): 0 n, 1 fast (n <= TOPK), 2 prefix, 3 pmask, 4 k_rem.
@triton.jit
def _tk_key(x):
    """fp32 -> uint32 whose unsigned order equals the float order (signed-int form: flip
    the magnitude of negatives, then the sign bit)."""
    b = x.to(tl.int32, bitcast=True)
    f = tl.where(b < 0, b ^ 0x7FFFFFFF, b)
    return (f ^ -2147483648).to(tl.uint32, bitcast=True)


@triton.jit
def _tk_init(n_keys_ptr, state_ptr, hist_ptr, TOPK: tl.constexpr, NB: tl.constexpr):
    n = tl.load(n_keys_ptr).to(tl.int32)
    tl.store(state_ptr + 0, n)
    tl.store(state_ptr + 1, (n <= TOPK).to(tl.int32))
    tl.store(state_ptr + 2, 0)
    tl.store(state_ptr + 3, 0)
    tl.store(state_ptr + 4, TOPK)
    tl.store(hist_ptr + tl.arange(0, NB), tl.zeros([NB], dtype=tl.int32))


@triton.jit
def _tk_hist(scores_ptr, state_ptr, hist_ptr, SHIFT: tl.constexpr, NB: tl.constexpr,
             BLOCK: tl.constexpr):
    if tl.load(state_ptr + 1) != 0:
        return
    n = tl.load(state_ptr + 0)
    t0 = tl.program_id(0) * BLOCK
    if t0 >= n:
        return
    prefix = tl.load(state_ptr + 2).to(tl.uint32, bitcast=True)
    pmask = tl.load(state_ptr + 3).to(tl.uint32, bitcast=True)
    i = t0 + tl.arange(0, BLOCK)
    valid = i < n
    key = _tk_key(tl.load(scores_ptr + i, mask=valid, other=0.0))
    match = valid & ((key & pmask) == prefix)
    digit = ((key >> SHIFT) & (NB - 1)).to(tl.int32)
    h = tl.histogram(digit, NB, mask=match)
    b = tl.arange(0, NB)
    tl.atomic_add(hist_ptr + b, h, mask=h > 0)


@triton.jit
def _tk_select(state_ptr, hist_ptr, SHIFT: tl.constexpr, NB: tl.constexpr):
    if tl.load(state_ptr + 1) != 0:
        return
    b = tl.arange(0, NB)
    hist = tl.load(hist_ptr + b)
    k_rem = tl.load(state_ptr + 4)
    total = tl.sum(hist)
    c_ge = total - tl.cumsum(hist, 0) + hist
    d = tl.sum((c_ge >= k_rem).to(tl.int32)) - 1
    c_ge_d = tl.sum(tl.where(b == d, c_ge, 0))
    h_d = tl.sum(tl.where(b == d, hist, 0))
    tl.store(state_ptr + 4, k_rem - (c_ge_d - h_d))
    prefix = tl.load(state_ptr + 2).to(tl.uint32, bitcast=True) | (d.to(tl.uint32) << SHIFT)
    pmask = tl.load(state_ptr + 3).to(tl.uint32, bitcast=True) | ((NB - 1) << SHIFT)
    tl.store(state_ptr + 2, prefix.to(tl.int32, bitcast=True))
    tl.store(state_ptr + 3, pmask.to(tl.int32, bitcast=True))
    tl.store(hist_ptr + b, tl.zeros([NB], dtype=tl.int32))


@triton.jit
def _tk_count(scores_ptr, state_ptr, cnt_ptr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    n = tl.load(state_ptr + 0)
    fast = tl.load(state_ptr + 1)
    t0 = t * BLOCK
    thr = tl.load(state_ptr + 2).to(tl.uint32, bitcast=True)
    i = t0 + tl.arange(0, BLOCK)
    valid = (i < n) & (fast == 0)
    key = _tk_key(tl.load(scores_ptr + i, mask=valid, other=0.0))
    tl.store(cnt_ptr + 2 * t, tl.sum((valid & (key > thr)).to(tl.int32)))
    tl.store(cnt_ptr + 2 * t + 1, tl.sum((valid & (key == thr)).to(tl.int32)))


@triton.jit
def _tk_scan(cnt_ptr, off_ptr, G, GP: tl.constexpr):
    g = tl.arange(0, GP)
    m = g < G
    gt = tl.load(cnt_ptr + 2 * g, mask=m, other=0)
    eq = tl.load(cnt_ptr + 2 * g + 1, mask=m, other=0)
    tot_gt = tl.sum(gt)
    tl.store(off_ptr + 2 * g, tl.cumsum(gt, 0) - gt, mask=m)                 # gt offsets
    tl.store(off_ptr + 2 * g + 1, tot_gt + tl.cumsum(eq, 0) - eq, mask=m)    # eq offsets (after all gt)


@triton.jit
def _tk_write(scores_ptr, row_ptr, state_ptr, off_ptr, sel_ptr, TOPK: tl.constexpr,
              BLOCK: tl.constexpr):
    t = tl.program_id(0)
    n = tl.load(state_ptr + 0)
    t0 = t * BLOCK
    offs = tl.arange(0, BLOCK)
    if tl.load(state_ptr + 1) != 0:
        # dense case: every live key, -1 padding (TOPK <= BLOCK: tile 0 writes it all)
        if t == 0:
            i = offs
            r = tl.load(row_ptr + i, mask=(i < n) & (i < TOPK), other=-1)
            tl.store(sel_ptr + i, tl.where(i < n, r, -1), mask=i < TOPK)
        return
    if t0 >= n:
        return
    thr = tl.load(state_ptr + 2).to(tl.uint32, bitcast=True)
    k_rem = tl.load(state_ptr + 4)                     # how many == thr keys to take
    i = t0 + offs
    valid = i < n
    key = _tk_key(tl.load(scores_ptr + i, mask=valid, other=0.0))
    gt = valid & (key > thr)
    eq = valid & (key == thr)
    g_off = tl.load(off_ptr + 2 * t)
    e_off = tl.load(off_ptr + 2 * t + 1)
    gpos = g_off + tl.cumsum(gt.to(tl.int32), 0) - 1
    epos = e_off + tl.cumsum(eq.to(tl.int32), 0) - 1
    tot_gt = TOPK - k_rem
    take_eq = eq & ((epos - tot_gt) < k_rem)
    r = tl.load(row_ptr + i, mask=gt | take_eq, other=-1)
    tl.store(sel_ptr + gpos, r, mask=gt)
    tl.store(sel_ptr + epos, r, mask=take_eq)


class DsaTopk:
    """Preallocated state for :func:`_tk_*` over a ``width``-wide score row."""

    BLOCK = 4096

    def __init__(self, width: int, topk: int, device):
        assert topk <= self.BLOCK
        self.topk = topk
        self.G = triton.cdiv(width, self.BLOCK)
        self.state = torch.zeros(8, dtype=torch.int32, device=device)
        self.hist = torch.zeros(2048, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(2 * self.G, dtype=torch.int32, device=device)
        self.off = torch.zeros(2 * self.G, dtype=torch.int32, device=device)

    def __call__(self, scores: torch.Tensor, row: torch.Tensor, n_keys: torch.Tensor,
                 out: torch.Tensor) -> torch.Tensor:
        B, G, K = self.BLOCK, self.G, self.topk
        _tk_init[(1,)](n_keys, self.state, self.hist, TOPK=K, NB=2048)
        for shift, nb in ((21, 2048), (10, 2048), (0, 1024)):
            _tk_hist[(G,)](scores, self.state, self.hist, SHIFT=shift, NB=nb, BLOCK=B, num_warps=8)
            _tk_select[(1,)](self.state, self.hist, SHIFT=shift, NB=nb, num_warps=8)
        _tk_count[(G,)](scores, self.state, self.cnt, BLOCK=B, num_warps=8)
        _tk_scan[(1,)](self.cnt, self.off, G, GP=triton.next_power_of_2(G), num_warps=4)
        _tk_write[(G,)](scores, row, self.state, self.off, out, TOPK=K, BLOCK=B, num_warps=8)
        return out
