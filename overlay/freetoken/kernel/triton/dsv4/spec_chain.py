"""Fused register-chain for the DSV4 compressor under speculative verify (ratio-4).

``Compressor.spec_decode_chain`` advances the rolling carry over B consecutive
positions. Chaining the 8-slot overlap register position-by-position in torch costs
~9 launches per position per tier-layer (21 layers x 2 tiers x 6 positions -> ~2300
kernels/verify). This kernel runs a whole chain in ONE launch: the register lives in
SRAM as [4, BLOCK] tile pairs (A/B half x kv/score x lo/hi columns), positions are an
in-kernel loop, and every per-position register state is written out so the caller
can persist the right block to each ring page.

Column coupling: pooled output column d' reads A-half rows at column d' and B-half
rows at column D+d' (``kv_eff = cat(ks[:, :4, :D], ks[:, 4:, D:], dim=1)``), so each
program loads both column halves of its tile. Promotion (block completion) is a
whole-tile ``where(A <- B)`` on a scalar flag -- no row shuffling.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _chain4_kernel(
    kv_ptr, sc_ptr,          # [B, 2D] fp32 contributions (score pre-ape'd)
    seed_ptr,                # [8, 4D] fp32 ring block (rows 0..3 = A, 4..7 = B; kv | score)
    idxmod_ptr, should_ptr,  # [B] int32 / int32(0|1)
    out_ptr,                 # [B, D] fp32 pooled outputs (junk rows where !should)
    states_ptr,              # [B, 8, 4D] fp32 per-position register states
    B: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    item: tl.constexpr = 2 * D
    stride: tl.constexpr = 2 * item  # per register row: [kv (2D) | score (2D)]
    row4 = tl.arange(0, 4)
    m2 = mask[None, :]

    a_base = seed_ptr + row4[:, None] * stride + offs[None, :]
    b_base = seed_ptr + (row4 + 4)[:, None] * stride + offs[None, :]
    a_k_lo = tl.load(a_base, mask=m2, other=0.0)
    a_k_hi = tl.load(a_base + D, mask=m2, other=0.0)
    a_s_lo = tl.load(a_base + item, mask=m2, other=0.0)
    a_s_hi = tl.load(a_base + item + D, mask=m2, other=0.0)
    b_k_lo = tl.load(b_base, mask=m2, other=0.0)
    b_k_hi = tl.load(b_base + D, mask=m2, other=0.0)
    b_s_lo = tl.load(b_base + item, mask=m2, other=0.0)
    b_s_hi = tl.load(b_base + item + D, mask=m2, other=0.0)

    for i in range(B):
        im = tl.load(idxmod_ptr + i)
        promote = tl.load(should_ptr + i) != 0
        kv_lo = tl.load(kv_ptr + i * item + offs, mask=mask, other=0.0)
        kv_hi = tl.load(kv_ptr + i * item + D + offs, mask=mask, other=0.0)
        sc_lo = tl.load(sc_ptr + i * item + offs, mask=mask, other=0.0)
        sc_hi = tl.load(sc_ptr + i * item + D + offs, mask=mask, other=0.0)
        hit = (row4 == im)[:, None]
        b_k_lo = tl.where(hit, kv_lo[None, :], b_k_lo)
        b_k_hi = tl.where(hit, kv_hi[None, :], b_k_hi)
        b_s_lo = tl.where(hit, sc_lo[None, :], b_s_lo)
        b_s_hi = tl.where(hit, sc_hi[None, :], b_s_hi)

        # pooled over 8 effective rows: A rows read lo halves, B rows read hi halves
        m = tl.maximum(tl.max(a_s_lo, axis=0), tl.max(b_s_hi, axis=0))
        wa = tl.exp(a_s_lo - m[None, :])
        wb = tl.exp(b_s_hi - m[None, :])
        pooled = (tl.sum(wa * a_k_lo, axis=0) + tl.sum(wb * b_k_hi, axis=0)) / (
            tl.sum(wa, axis=0) + tl.sum(wb, axis=0))
        tl.store(out_ptr + i * D + offs, pooled, mask=mask)

        # completion: A <- B (whole tiles, scalar flag)
        a_k_lo = tl.where(promote, b_k_lo, a_k_lo)
        a_k_hi = tl.where(promote, b_k_hi, a_k_hi)
        a_s_lo = tl.where(promote, b_s_lo, a_s_lo)
        a_s_hi = tl.where(promote, b_s_hi, a_s_hi)

        sa = states_ptr + i * 8 * stride + row4[:, None] * stride + offs[None, :]
        sb = states_ptr + i * 8 * stride + (row4 + 4)[:, None] * stride + offs[None, :]
        tl.store(sa, a_k_lo, mask=m2)
        tl.store(sa + D, a_k_hi, mask=m2)
        tl.store(sa + item, a_s_lo, mask=m2)
        tl.store(sa + item + D, a_s_hi, mask=m2)
        tl.store(sb, b_k_lo, mask=m2)
        tl.store(sb + D, b_k_hi, mask=m2)
        tl.store(sb + item, b_s_lo, mask=m2)
        tl.store(sb + item + D, b_s_hi, mask=m2)


def chain4(kv: torch.Tensor, score: torch.Tensor, seed: torch.Tensor,
           idx_mod: torch.Tensor, should: torch.Tensor):
    """kv/score ``[B, 2d]`` fp32 (score pre-ape'd); seed ``[8, 4d]`` fp32 (one ring
    block); idx_mod ``[B]`` int32, should ``[B]`` int32. Returns ``(pooled [B, d]
    fp32, states [B, 8, 4d] fp32)`` -- states[i] is the register AFTER position i."""
    B, item = kv.shape
    D = item // 2
    out = torch.empty(B, D, dtype=torch.float32, device=kv.device)
    states = torch.empty(B, 8, 2 * item, dtype=torch.float32, device=kv.device)
    BLOCK = 128 if D >= 128 else triton.next_power_of_2(D)
    grid = (triton.cdiv(D, BLOCK),)
    _chain4_kernel[grid](
        kv.contiguous(), score.contiguous(), seed.contiguous(),
        idx_mod.to(torch.int32).contiguous(), should.to(torch.int32).contiguous(),
        out, states, B=B, D=D, BLOCK=BLOCK, num_warps=4,
    )
    return out, states


__all__ = ["chain4"]
