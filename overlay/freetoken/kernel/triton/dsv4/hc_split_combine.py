"""mHC pre-mix decode fusion: Sinkhorn split + pre_combine in ONE launch (lossless).

``hc_pre`` is four launches: ``hc_rms_cast`` -> ``hc_mix_gemv`` -> ``hc_split_sinkhorn``
-> ``hc_pre_combine``. :func:`hc_sinkhorn_pre_combine` replaces the last two with one
grid ``(M, NB + 1)``:

  * programs ``blk < NB`` -- ``y[m, blk-slice] = sum_h pre[m,h] * x[m,h,:]``, each
    recomputing the HC ``pre`` gates as scalars straight from ``mixes`` (no ``pre``
    round-trip, no dependency on the Sinkhorn program);
  * program ``blk == NB`` -- ``post`` and the Sinkhorn ``comb`` (code copied verbatim
    from :func:`freetoken.kernel.triton.dsv4.sinkhorn._hc_sinkhorn_kernel`), running in
    parallel with the combine slices.

Bit-identical to ``hc_split_sinkhorn`` + ``hc_pre_combine`` by construction:

  * ``pre[h] = sigmoid(mixes[h] * sc0 + base[h]) + EPS`` is elementwise (no reduction):
    same fp32 ops in the same order as the Sinkhorn kernel, which stored it as fp32 and
    ``hc_pre_combine`` loaded it back unchanged.
  * ``y`` is the same per-element fp32 chain ``acc += p * x_h`` over ``h = 0..HC-1``
    (static order) and one cast to the output dtype; it is elementwise in ``d``, so
    BLOCK / num_warps / layout cannot change its bits.
  * ``post`` / ``comb``: the Sinkhorn code is unchanged and compiled at the SAME
    ``num_warps=1``, so the ``[HC, HC]`` ``tl.max`` / ``tl.sum`` reductions get the same
    layout and therefore the same reduction tree.

CUDA-graph safe: fixed shapes, no host sync, outputs allocated per call like the
kernels it replaces.

``hc_mix_rms`` (only for ``FREETOKEN_GLM5_HC_FUSED=2``) launches DeepSeek-V4's
``_hc_mix_rms_kernel`` (rms folded into the MIX gemv). That one is NOT bit-identical
to ``hc_rms_cast`` + ``hc_mix_gemv``: the sum of squares is accumulated in 2048-lane
tiles with 8 warps instead of 1024-lane tiles with 4 warps, and loading the dot's operand
as bf16 changes its reduction layout (sm_120 TTGIR: sizePerThread 8 instead of 4), so
both fp32 summation trees change -> ``mixes`` can differ in the last bits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}

# Combine slice per program. Elementwise, so any value is bit-identical; at
# num_warps=1, 512 -> 16 elements / thread and NB = 8 slices for D = 4096.
_BLOCK = 512


@triton.jit
def _hc_sinkhorn_pre_combine_kernel(
    mixes_ptr, scale_ptr, base_ptr,   # [M, (2+HC)*HC] fp32, [3] fp32, [(2+HC)*HC] fp32
    x_ptr, y_ptr,                     # [M, HC, D] in, [M, D] out (OUT)
    post_ptr, comb_ptr,               # [M, HC] fp32, [M, HC, HC] fp32
    D,
    stride_mn, stride_xm, stride_xh, stride_ym, stride_pon, stride_cn,
    HC: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    m = mixes_ptr + row * stride_mn
    if blk == tl.num_programs(1) - 1:
        # ---- post + Sinkhorn comb: verbatim _hc_sinkhorn_kernel (pre is not stored) ----
        h = tl.arange(0, HC)
        sc1 = tl.load(scale_ptr + 1)
        sc2 = tl.load(scale_ptr + 2)

        post = 2.0 * tl.sigmoid(tl.load(m + HC + h) * sc1 + tl.load(base_ptr + HC + h))
        tl.store(post_ptr + row * stride_pon + h, post)

        idx = h[:, None] * HC + h[None, :]  # [HC, HC] row-major within the comb block
        c = tl.load(m + 2 * HC + idx) * sc2 + tl.load(base_ptr + 2 * HC + idx)
        # softmax over the last axis (dim=-1)
        c = c - tl.max(c, axis=1)[:, None]
        c = tl.exp(c)
        c = c / tl.sum(c, axis=1)[:, None]
        c = c + EPS
        # initial column normalization (sum over rows = axis 0)
        c = c / (tl.sum(c, axis=0)[None, :] + EPS)
        for _ in range(ITERS - 1):
            c = c / (tl.sum(c, axis=1)[:, None] + EPS)
            c = c / (tl.sum(c, axis=0)[None, :] + EPS)
        tl.store(comb_ptr + row * stride_cn + idx, c)
    else:
        # ---- pre_combine slice: _hc_pre_combine_kernel with pre recomputed inline ----
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D
        sc0 = tl.load(scale_ptr + 0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for hh in tl.static_range(HC):
            xh = tl.load(x_ptr + row * stride_xm + hh * stride_xh + offs, mask=mask, other=0.0).to(tl.float32)
            p = tl.sigmoid(tl.load(m + hh) * sc0 + tl.load(base_ptr + hh)) + EPS
            acc += p * xh
        tl.store(y_ptr + row * stride_ym + offs, acc.to(OUT), mask=mask)


def hc_sinkhorn_pre_combine(
    mixes: torch.Tensor, x: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    hc_mult: int, sinkhorn_iters: int, eps: float, out_dtype: torch.dtype,
):
    """``hc_split_sinkhorn(mixes, ...)`` + ``hc_pre_combine(x, pre, out_dtype)`` in one
    launch. ``mixes`` [M, (2+HC)*HC] fp32, ``x`` [M, HC, D]. Returns
    ``(y [M, D] out_dtype, post [M, HC] fp32, comb [M, HC, HC] fp32)``."""
    assert hc_mult & (hc_mult - 1) == 0, "hc_mult must be a power of two"
    mixes = mixes.contiguous().float()
    x = x.contiguous()
    M, HC, D = x.shape
    assert HC == hc_mult and mixes.shape[0] == M
    dev = x.device
    y = torch.empty((M, D), dtype=out_dtype, device=dev)
    post = torch.empty(M, HC, device=dev, dtype=torch.float32)
    comb = torch.empty(M, HC, HC, device=dev, dtype=torch.float32)
    nb = triton.cdiv(D, _BLOCK)
    _hc_sinkhorn_pre_combine_kernel[(M, nb + 1)](
        mixes, hc_scale.float().contiguous(), hc_base.float().contiguous(),
        x, y, post, comb,
        D,
        mixes.stride(0), x.stride(0), x.stride(1), y.stride(0), post.stride(0), comb.stride(0),
        HC=HC, ITERS=sinkhorn_iters, EPS=eps,
        BLOCK=_BLOCK, OUT=_TL[out_dtype], num_warps=1,
    )
    return y, post, comb


def hc_mix_rms(x2: torch.Tensor, hc_fn: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """``mixes[m, r] = rsqrt(mean(x2[m]^2) + eps) * dot(hc_fn[r], x2[m])`` with the rms
    folded into the gemv (DeepSeek-V4's ``_hc_mix_rms_kernel``, same launch config as
    ``hc_pre_fused``). ``x2`` [M, HC*D] bf16, ``hc_fn`` [MIX, HC*D] fp32.
    NOT bit-identical to ``hc_rms_cast`` + ``hc_mix_gemv`` (see module docstring)."""
    from freetoken.kernel.triton.dsv4.hc_fused import _hc_mix_rms_kernel

    M, KD = x2.shape
    MIX = hc_fn.shape[0]
    mixes = torch.empty(M, MIX, dtype=torch.float32, device=x2.device)
    _hc_mix_rms_kernel[(M * MIX,)](
        x2, hc_fn, mixes, KD, norm_eps,
        x2.stride(0), hc_fn.stride(0), mixes.stride(0),
        MIX=MIX, BLOCK=2048, num_warps=8,
    )
    return mixes


__all__ = ["hc_sinkhorn_pre_combine", "hc_mix_rms"]
