"""Fused DSV4 compressor decode step (per-row register update + gated pool).

The eager decode_step spends ~8-10 launches per tier-layer on the register roll
(read_carry_blocks -> clone x2 -> scatter x2 -> cat x2 -> gated_pool -> where x2 ->
cat -> write_carry_blocks), and there are 62 tier-instances per decode step (21
ratio-4 layers x 2 tiers + 20 ratio-128 layers x 1). These two kernels collapse the
roll to ONE launch per tier-instance, operating directly on the ring buffer
(``CompressStateRing.buffer``: ``[n_slots+1, 2*item]`` fp32, block base row =
``(window_slot // P) * ring_size``).

Numerics: the pooled softmax accumulates row-by-row in ascending row order, exactly
like the eager ``gated_pool`` kernel (fp32 online max/exp/sum, output cast to the
model dtype by the caller) -- bit-parity with the eager path, not just closeness.
The score contribution adds ``ape[pos % ratio]`` in-kernel (one launch fewer).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _comp_step_ov_kernel(
    ring_ptr,                 # [n_slots+1, 4D] fp32 (rows: kv 2D | score 2D)
    kv_ptr, sc_ptr,           # [B, 2D] fp32 (score WITHOUT ape)
    ape_ptr,                  # [4, 2D] fp32
    prev_base_ptr, cur_base_ptr,  # [B] int64 ring block base rows
    idxmod_ptr, should_ptr,   # [B] int32
    out_ptr,                  # [B, D] fp32 pooled (pre-norm)
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    m2 = mask[None, :]
    item: tl.constexpr = 2 * D
    stride: tl.constexpr = 2 * item
    row4 = tl.arange(0, 4)

    pb = tl.load(prev_base_ptr + b)
    cb = tl.load(cur_base_ptr + b)
    im = tl.load(idxmod_ptr + b)
    promote = tl.load(should_ptr + b) != 0

    a_base = ring_ptr + (pb + row4)[:, None] * stride + offs[None, :]
    b_base = ring_ptr + (pb + 4 + row4)[:, None] * stride + offs[None, :]
    a_k_lo = tl.load(a_base, mask=m2, other=0.0)
    a_k_hi = tl.load(a_base + D, mask=m2, other=0.0)
    a_s_lo = tl.load(a_base + item, mask=m2, other=0.0)
    a_s_hi = tl.load(a_base + item + D, mask=m2, other=0.0)
    b_k_lo = tl.load(b_base, mask=m2, other=0.0)
    b_k_hi = tl.load(b_base + D, mask=m2, other=0.0)
    b_s_lo = tl.load(b_base + item, mask=m2, other=0.0)
    b_s_hi = tl.load(b_base + item + D, mask=m2, other=0.0)

    kv_lo = tl.load(kv_ptr + b * item + offs, mask=mask, other=0.0)
    kv_hi = tl.load(kv_ptr + b * item + D + offs, mask=mask, other=0.0)
    sc_lo = tl.load(sc_ptr + b * item + offs, mask=mask, other=0.0) + tl.load(
        ape_ptr + im * item + offs, mask=mask, other=0.0)
    sc_hi = tl.load(sc_ptr + b * item + D + offs, mask=mask, other=0.0) + tl.load(
        ape_ptr + im * item + D + offs, mask=mask, other=0.0)
    hit = (row4 == im)[:, None]
    b_k_lo = tl.where(hit, kv_lo[None, :], b_k_lo)
    b_k_hi = tl.where(hit, kv_hi[None, :], b_k_hi)
    b_s_lo = tl.where(hit, sc_lo[None, :], b_s_lo)
    b_s_hi = tl.where(hit, sc_hi[None, :], b_s_hi)

    # gated pool over the 8 effective rows (A rows read lo halves, B rows hi),
    # accumulated in ROW ORDER to match the eager gated_pool kernel bit-for-bit.
    # (row extraction via where+sum: other rows contribute exact 0.0, and
    # 0 + (-inf) = -inf keeps empty-slot scores intact)
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    for r in tl.static_range(4):
        m = tl.maximum(m, tl.sum(tl.where((row4 == r)[:, None], a_s_lo, 0.0), axis=0))
    for r in tl.static_range(4):
        m = tl.maximum(m, tl.sum(tl.where((row4 == r)[:, None], b_s_hi, 0.0), axis=0))
    denom = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in tl.static_range(4):
        s = tl.sum(tl.where((row4 == r)[:, None], a_s_lo, 0.0), axis=0)
        k = tl.sum(tl.where((row4 == r)[:, None], a_k_lo, 0.0), axis=0)
        w = tl.exp(s - m)
        denom += w
        acc += w * k
    for r in tl.static_range(4):
        s = tl.sum(tl.where((row4 == r)[:, None], b_s_hi, 0.0), axis=0)
        k = tl.sum(tl.where((row4 == r)[:, None], b_k_hi, 0.0), axis=0)
        w = tl.exp(s - m)
        denom += w
        acc += w * k
    tl.store(out_ptr + b * D + offs, acc / denom, mask=mask)

    a_k_lo = tl.where(promote, b_k_lo, a_k_lo)
    a_k_hi = tl.where(promote, b_k_hi, a_k_hi)
    a_s_lo = tl.where(promote, b_s_lo, a_s_lo)
    a_s_hi = tl.where(promote, b_s_hi, a_s_hi)

    oa = ring_ptr + (cb + row4)[:, None] * stride + offs[None, :]
    ob = ring_ptr + (cb + 4 + row4)[:, None] * stride + offs[None, :]
    tl.store(oa, a_k_lo, mask=m2)
    tl.store(oa + D, a_k_hi, mask=m2)
    tl.store(oa + item, a_s_lo, mask=m2)
    tl.store(oa + item + D, a_s_hi, mask=m2)
    tl.store(ob, b_k_lo, mask=m2)
    tl.store(ob + D, b_k_hi, mask=m2)
    tl.store(ob + item, b_s_lo, mask=m2)
    tl.store(ob + item + D, b_s_hi, mask=m2)


@triton.jit
def _comp_step128_kernel(
    ring_ptr,                 # [n_slots+1, 2D] fp32 (rows: kv D | score D)
    kv_ptr, sc_ptr,           # [B, D] fp32 (score WITHOUT ape)
    ape_ptr,                  # [128, D] fp32
    prev_base_ptr, cur_base_ptr,
    idxmod_ptr,
    out_ptr,                  # [B, D] fp32 pooled (pre-norm)
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    stride: tl.constexpr = 2 * D

    pb = tl.load(prev_base_ptr + b)
    cb = tl.load(cur_base_ptr + b)
    im = tl.load(idxmod_ptr + b)
    kv_new = tl.load(kv_ptr + b * D + offs, mask=mask, other=0.0)
    sc_new = tl.load(sc_ptr + b * D + offs, mask=mask, other=0.0) + tl.load(
        ape_ptr + im * D + offs, mask=mask, other=0.0)

    # pass 1: max over the 128 register rows (row im replaced by the new value)
    m = tl.full((BLOCK,), float("-inf"), tl.float32)
    for r in tl.range(0, 128):
        s = tl.load(ring_ptr + (pb + r) * stride + D + offs, mask=mask,
                    other=float("-inf"))
        s = tl.where(r == im, sc_new, s)
        m = tl.maximum(m, s)
    # pass 2: pooled accumulate in row order + copy-through write to the cur block
    denom = tl.zeros((BLOCK,), tl.float32)
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in tl.range(0, 128):
        k = tl.load(ring_ptr + (pb + r) * stride + offs, mask=mask, other=0.0)
        s = tl.load(ring_ptr + (pb + r) * stride + D + offs, mask=mask,
                    other=float("-inf"))
        k = tl.where(r == im, kv_new, k)
        s = tl.where(r == im, sc_new, s)
        w = tl.exp(s - m)
        denom += w
        acc += w * k
        tl.store(ring_ptr + (cb + r) * stride + offs, k, mask=mask)
        tl.store(ring_ptr + (cb + r) * stride + D + offs, s, mask=mask)
    tl.store(out_ptr + b * D + offs, acc / denom, mask=mask)


def fused_comp_step(ring_buffer: torch.Tensor, kv: torch.Tensor, score: torch.Tensor,
                    ape: torch.Tensor, prev_base: torch.Tensor, cur_base: torch.Tensor,
                    idx_mod: torch.Tensor, should: torch.Tensor, overlap: bool,
                    head_dim: int) -> torch.Tensor:
    """One-launch register roll for a batched decode step. Returns pooled [B, D] fp32.
    ``kv``/``score`` are the raw wkv/wgate projections [B, item] fp32 (NO ape added);
    ``prev_base``/``cur_base`` are the per-row ring block base rows (int64)."""
    B = kv.shape[0]
    D = head_dim
    out = torch.empty(B, D, dtype=torch.float32, device=kv.device)
    BLOCK = 128 if D >= 128 else triton.next_power_of_2(D)
    grid = (B, triton.cdiv(D, BLOCK))
    if overlap:
        _comp_step_ov_kernel[grid](
            ring_buffer, kv.contiguous(), score.contiguous(), ape,
            prev_base.contiguous(), cur_base.contiguous(),
            idx_mod.to(torch.int32).contiguous(), should.to(torch.int32).contiguous(),
            out, D=D, BLOCK=BLOCK, num_warps=4,
        )
    else:
        _comp_step128_kernel[grid](
            ring_buffer, kv.contiguous(), score.contiguous(), ape,
            prev_base.contiguous(), cur_base.contiguous(),
            idx_mod.to(torch.int32).contiguous(),
            out, D=D, BLOCK=BLOCK, num_warps=4,
        )
    return out


__all__ = ["fused_comp_step"]
