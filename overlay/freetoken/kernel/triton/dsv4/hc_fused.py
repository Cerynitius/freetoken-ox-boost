"""mHC pre-mix decode fusion, take 2 (take 1 -- everything in one CTA -- destroyed
parallelism and measured 39 tok/s; reverted).

This version keeps the parallel grid shape of ``hc_mix_gemv`` ((m, r) programs) and
folds the rms INTO it: each program redundantly accumulates the row's sum of squares
alongside its dot (16K extra FMAs per program, free next to the loads) and reads the
bf16 stream DIRECTLY -- eliminating the ``hc_rms_cast`` launch and the [M, HC*D]
fp32 intermediate (64 KB written+read per call, ~87 calls/step). Sinkhorn and
pre_combine stay as their existing single launches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_mix_rms_kernel(
    x_ptr,                    # [M, KD] bf16
    w_ptr,                    # [MIX, KD] fp32
    out_ptr,                  # [M, MIX] fp32
    KD, norm_eps,
    stride_xm, stride_wr, stride_om,
    MIX: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // MIX
    r = pid - m * MIX
    dot = tl.zeros((BLOCK,), tl.float32)
    ssq = tl.zeros((BLOCK,), tl.float32)
    for t in range(0, tl.cdiv(KD, BLOCK)):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < KD
        xv = tl.load(x_ptr + m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + r * stride_wr + offs, mask=mask, other=0.0)
        dot += xv * wv
        ssq += xv * xv
    rs = 1.0 / tl.sqrt(tl.sum(ssq, axis=0) / KD + norm_eps)
    tl.store(out_ptr + m * stride_om + r, tl.sum(dot, axis=0) * rs)


def hc_pre_fused(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                 hc_base: torch.Tensor, norm_eps: float, hc_eps: float,
                 iters: int, out_dtype: torch.dtype):
    """``x`` [M, HC, D] (bf16); returns (y [M, D] out_dtype, post [M, HC] fp32,
    comb [M, HC, HC] fp32). Three launches (was four + an fp32 intermediate)."""
    from freetoken.kernel.triton.dsv4.hc import hc_pre_combine
    from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn

    M, HC, D = x.shape
    KD = HC * D
    MIX = hc_fn.shape[0]
    xf = x.reshape(M, KD)
    mixes = torch.empty(M, MIX, dtype=torch.float32, device=x.device)
    _hc_mix_rms_kernel[(M * MIX,)](
        xf, hc_fn, mixes, KD, norm_eps,
        xf.stride(0), hc_fn.stride(0), mixes.stride(0),
        MIX=MIX, BLOCK=2048, num_warps=8,
    )
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, HC, iters, hc_eps)
    y = hc_pre_combine(x, pre, out_dtype)
    return y, post, comb


__all__ = ["hc_pre_fused"]
