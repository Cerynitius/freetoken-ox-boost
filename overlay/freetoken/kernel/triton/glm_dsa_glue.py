"""GLM-5.3 DSA decode "glue" fusions (FREETOKEN_GLM5_DSA_GLUE=1, default OFF; see
attention/dsa.py). Every kernel here is integer index arithmetic, gathers, scatters and
exact dtype widenings (bf16 -> fp32) plus ONE fp32 add per element that the original
also performs as a single IEEE add -- no reductions, no multiplies feeding adds (so no
FMA contraction), no transcendental. Outputs are therefore bit-identical to the eager
torch chains they replace, which stay in place for every other configuration:

* ``dsa_store_rows``   -- the decode token's latent row ``[c_kv | k_rope]``, index key and
  k-pool gate scores scattered at ``out_loc`` in one launch (was 3 ``index_put_`` + 4
  int32->int64 index casts per MLA layer).
* ``dsa_pool_prologue`` -- ``_update_pools`` decode prologue: current-pool token ids,
  validity, row gather (invalid lanes redirected to the pool's first token and their key
  zeroed), ``kcache[rg].float()``, ``where(valid, gcache[rg].float() + ape, -inf)`` and the
  pooled-key destination row = the newest token's row (was ~20 tiny int/gather/cast
  kernels). The softmax / product / sum / bf16 cast stay as the SAME torch ops on
  bit-identical contiguous inputs.
* ``dsa_kpool_select`` -- k-pool decode selection epilogue: top-k pool ids -> sentinel
  where -> pool expansion -> always-selected tail pool -> concat -> position->row map ->
  ``-1`` holes, plus the constant ``cnt`` (was ~22 tiny kernels). ``torch.topk`` itself is
  untouched (same call, same tie order).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------------------
# 1) decode KV / index-key / gate row scatter
# ---------------------------------------------------------------------------------------
@triton.jit
def _dsa_store_rows_kernel(
    loc_ptr,
    ckv_ptr, krope_ptr, lat_ptr, kidx_ptr, idx_ptr, gate_ptr, gbuf_ptr,
    R_LAT, R_IDX, R_GATE,
    s_ckv, s_krope, s_lat, s_kidx, s_idx, s_gate, s_gbuf,
    W_CKV: tl.constexpr, W_ROPE: tl.constexpr, W_IDX: tl.constexpr, W_GATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    which = tl.program_id(1)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    if which == 0:  # rows[out_loc, :split] = c_kv
        r = tl.where(loc < 0, loc + R_LAT, loc)  # index_put_ wraps negative indices
        m = offs < W_CKV
        v = tl.load(ckv_ptr + t * s_ckv + offs, mask=m)
        tl.store(lat_ptr + r * s_lat + offs, v, mask=m)
    elif which == 1:  # index_k_cache(slot)[out_loc] = k
        r = tl.where(loc < 0, loc + R_IDX, loc)
        m = offs < W_IDX
        v = tl.load(kidx_ptr + t * s_kidx + offs, mask=m)
        tl.store(idx_ptr + r * s_idx + offs, v, mask=m)
    elif which == 2:  # gate_cache(slot)[out_loc] = gate
        r = tl.where(loc < 0, loc + R_GATE, loc)
        m = offs < W_GATE
        v = tl.load(gate_ptr + t * s_gate + offs, mask=m)
        tl.store(gbuf_ptr + r * s_gbuf + offs, v, mask=m)
    else:  # rows[out_loc, split:] = k_rope   (only launched when W_ROPE > 0)
        r = tl.where(loc < 0, loc + R_LAT, loc)
        m = offs < W_ROPE
        v = tl.load(krope_ptr + t * s_krope + offs, mask=m)
        tl.store(lat_ptr + r * s_lat + W_CKV + offs, v, mask=m)


def dsa_store_rows(c_kv, k_rope, out_loc, latent, k_idx, idx_buf, gate, gate_buf) -> bool:
    """One-launch twin of ``store_kv`` + ``store_index_k`` + ``store_gate``. Returns False
    (nothing written) when the layout/dtypes are not the plain-copy case."""
    T = c_kv.shape[0]
    w_ckv, w_rope = c_kv.shape[-1], k_rope.shape[-1]
    ok = (
        latent.shape[1] == w_ckv + w_rope
        and c_kv.dtype == latent.dtype and k_rope.dtype == latent.dtype
        and k_idx.dtype == idx_buf.dtype and gate.dtype == gate_buf.dtype
        and k_idx.shape == (T, idx_buf.shape[1]) and gate.shape == (T, gate_buf.shape[1])
        and out_loc.dim() == 1 and out_loc.shape[0] == T
        and not out_loc.dtype.is_floating_point
        and all(t.stride(-1) == 1 for t in (c_kv, latent, k_idx, idx_buf, gate, gate_buf))
        and (w_rope == 0 or k_rope.stride(-1) == 1)
    )
    if not ok or T == 0:
        return False
    block = triton.next_power_of_2(max(w_ckv, w_rope, k_idx.shape[1], gate.shape[1]))
    _dsa_store_rows_kernel[(T, 4 if w_rope > 0 else 3)](
        out_loc,
        c_kv, k_rope if w_rope > 0 else c_kv, latent, k_idx, idx_buf, gate, gate_buf,
        latent.shape[0], idx_buf.shape[0], gate_buf.shape[0],
        c_kv.stride(0), k_rope.stride(0) if w_rope > 0 else 0, latent.stride(0),
        k_idx.stride(0), idx_buf.stride(0), gate.stride(0), gate_buf.stride(0),
        W_CKV=w_ckv, W_ROPE=w_rope, W_IDX=k_idx.shape[1], W_GATE=gate.shape[1],
        BLOCK=block, num_warps=4,
    )
    return True


# ---------------------------------------------------------------------------------------
# 2) _update_pools decode prologue
# ---------------------------------------------------------------------------------------
@triton.jit
def _dsa_pool_prologue_kernel(
    rows_ptr, kvv_ptr, kc_ptr, gc_ptr, ape_ptr, kk_ptr, lg_ptr, dst_ptr,
    W, D,
    s_rb, s_rw, s_kc, s_gc, s_ape,
    KP: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    kvl = tl.load(kvv_ptr + b).to(tl.int64)
    # torch: pcur = clamp((kvlen - 1) // kp, min=0); floor-div then clamp == max(x, 0) // kp
    # for every x (negative x floors to <= -1 and clamps to 0), so no signed division.
    pcur = tl.maximum(kvl - 1, 0) // KP
    tok = pcur * KP + j                                   # tok = pcur*kp + arange(kp)
    valid = tok < kvl                                     # tok < kv_valid
    # tok = where(valid, tok, pcur*kp): page-table entries past kvlen are a previous
    # occupant's rows (unmapped after an elastic shrink) -- never dereference them
    tok = tl.where(valid, tok, pcur * KP)
    tc = tl.minimum(tok, W - 1)                           # tok.clamp(max=W-1)
    rg = tl.load(rows_ptr + b * s_rb + tc * s_rw).to(tl.int64)
    rg = tl.maximum(rg, 0)                                # rows.gather(...).clamp(min=0)
    offs = tl.arange(0, BLOCK_D)
    m = offs < D
    # kcache[rg].float().masked_fill(~valid, 0) / gcache[rg].float() (lane masked by -inf below)
    kk = tl.load(kc_ptr + rg * s_kc + offs, mask=m & valid, other=0.0).to(tl.float32)
    gg = tl.load(gc_ptr + rg * s_gc + offs, mask=m & valid, other=0.0).to(tl.float32)
    ap = tl.load(ape_ptr + j * s_ape + offs, mask=m).to(tl.float32)  # ape.float()
    lg = tl.where(valid, gg + ap, float("-inf"))
    o = (b * KP + j) * D + offs
    tl.store(kk_ptr + o, kk, mask=m)
    tl.store(lg_ptr + o, lg, mask=m)
    if j == 0:
        # dst = rows.gather(1, (kvlen-1).clamp(min=0)).clamp(min=0): the newest token's row
        dst = tl.load(rows_ptr + b * s_rb + tl.maximum(kvl - 1, 0) * s_rw).to(tl.int64)
        tl.store(dst_ptr + b, tl.maximum(dst, 0))


def dsa_pool_prologue(rows, kv_valid, kcache, gcache, ape, kp: int):
    """-> (kk fp32 [bs,kp,D], logits fp32 [bs,kp,D], dst int64 [bs]), bit-identical to
    the eager ``_update_pools`` decode chain; all outputs fresh and contiguous. None when
    the slabs are not the plain row-major layout (caller keeps the eager chain)."""
    bs, W = rows.shape
    D = kcache.shape[1]
    if not (
        gcache.shape[1] == D and tuple(ape.shape) == (kp, D) and bs > 0
        and kcache.stride(-1) == 1 and gcache.stride(-1) == 1 and ape.stride(-1) == 1
        and not rows.dtype.is_floating_point and not kv_valid.dtype.is_floating_point
    ):
        return None
    kk = torch.empty(bs, kp, D, dtype=torch.float32, device=rows.device)
    lg = torch.empty(bs, kp, D, dtype=torch.float32, device=rows.device)
    dst = torch.empty(bs, dtype=torch.int64, device=rows.device)
    _dsa_pool_prologue_kernel[(bs, kp)](
        rows, kv_valid, kcache, gcache, ape, kk, lg, dst,
        W, D,
        rows.stride(0), rows.stride(1), kcache.stride(0), gcache.stride(0), ape.stride(0),
        KP=kp, BLOCK_D=triton.next_power_of_2(D), num_warps=1,
    )
    return kk, lg, dst


# ---------------------------------------------------------------------------------------
# 3) k-pool decode selection epilogue
# ---------------------------------------------------------------------------------------
@triton.jit
def _dsa_kpool_select_kernel(
    picks_ptr, kvp_ptr, kvl_ptr, rows_ptr, sel_ptr, cnt_ptr,
    WIDTH,
    s_pb, s_pp, s_rb, s_rw, s_sb,
    KP: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = o < WIDTH
    kvp = tl.load(kvp_ptr + b).to(tl.int64)               # kv_pools = kvlen // kp
    kvl = tl.load(kvl_ptr + b).to(tl.int64)
    # tail = where(kv_pools*kp + ar[:kp-1] < kvlen, ., -1)   (cat position o < kp-1)
    is_tail = o < KP - 1
    t_tail = kvp * KP + o
    t_tail = tl.where(t_tail < kvl, t_tail, -1)
    # tok = where(picks >= 0, picks*kp + ar, -1) with picks = where(raw >= kv_pools, -1, raw)
    q = tl.maximum(o - (KP - 1), 0)
    p = q // KP
    jj = q - p * KP
    raw = tl.load(picks_ptr + b * s_pb + p * s_pp, mask=m & (o >= KP - 1), other=0).to(tl.int64)
    pick = tl.where(raw >= kvp, -1, raw + 0)
    tok = tl.where(pick >= 0, pick * KP + jj, -1)
    pos = tl.where(is_tail, t_tail, tok)
    # dsa_map_rows: where(pos < 0, -1, rows.gather(-1, pos.clamp_min(0)).to(int32))
    r = tl.load(rows_ptr + b * s_rb + tl.maximum(pos, 0) * s_rw, mask=m, other=0)
    sel = tl.where(pos < 0, -1, r.to(tl.int32))
    tl.store(sel_ptr + b * s_sb + o, sel, mask=m)
    tl.store(cnt_ptr + b + o * 0, WIDTH, mask=(o == 0) & (tl.program_id(1) == 0))


def dsa_kpool_select(picks, kv_pools, kvlen, rows, kp: int):
    """``picks`` [bs, P] raw top-k pool ids (int64, any strides), ``kv_pools``/``kvlen``
    [bs] int32, ``rows`` [bs, W] int32 -> (sel int32 [bs, 1, kp-1+P*kp], cnt int32 [bs, 1])."""
    bs, P = picks.shape
    width = (kp - 1) + P * kp
    sel = torch.empty(bs, width, dtype=torch.int32, device=rows.device)
    cnt = torch.empty(bs, 1, dtype=torch.int32, device=rows.device)
    BLOCK = 1024
    _dsa_kpool_select_kernel[(bs, triton.cdiv(width, BLOCK))](
        picks, kv_pools, kvlen, rows, sel, cnt,
        width,
        picks.stride(0), picks.stride(1), rows.stride(0), rows.stride(1), sel.stride(0),
        KP=kp, BLOCK=BLOCK, num_warps=4,
    )
    return sel.view(bs, 1, width), cnt


__all__ = ["dsa_store_rows", "dsa_pool_prologue", "dsa_kpool_select"]
