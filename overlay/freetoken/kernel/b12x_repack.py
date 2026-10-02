"""GPU-batched native-NVFP4 -> flashinfer SM12x W4A16 ("b12x") layout repack.

flashinfer's ``prepare_w4a16_packed_weights`` builds the b12x layout with a Python loop
per expert (~40 s per Naive layer), which only fits a one-time load. This module applies
the SAME transform to a whole expert bank on the GPU at memory speed, so a streamed
prefill layer (native layout, in the offload double buffer) can be repacked in place
right before the b12x tensor-core MoE runs on it (Triton kernels). Byte-identical to
``freetoken.moe.nvfp4_backends.b12x_repack_layer(..., chunk=1)`` (per-expert scale
factor), verified on real Naive layers.

Layout (per expert, bank N x K, native row-major nibbles, nibble j of int32 word w of row
n holds k = 8w + j):
  weights: out[kt, nt*128 + p] (int32, kt < K/16, nt < N/64); nibble s of that word is
      the native nibble (n = 64 nt + col_s(p), k = 16 kt + elem_s(p)) -- flashinfer's
      _repack_4bit_no_perm index math with the transpose/tile permute folded in.
  scales:  out[kb, m] (fp8 byte, kb < K/16, m < N): m -> m2 = 4(m/4) + [0,2,1,3][m%4],
      n = 64(m2/64) + scale_perm[m2%64]; v = fp16(e4m3(s[n,kb] * ratio[n]) * f * 128),
      0 if v < 2, byte = bits 14..7 of v.  f = per-expert power of two
      (flashinfer's _nvfp4_compute_scale_factor), alpha = global * 2^119 / f.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _repack_w_kernel(src_ptr, dst_ptr, N, K, NT_PER: tl.constexpr):
    # grid (E_chunk * K/16, cdiv(N/64, NT_PER)); each program: NT_PER n-tiles x 128 words
    k_tiles = K // 16
    n_tiles = N // 64
    pid = tl.program_id(0)
    e = pid // k_tiles
    kt = pid % k_tiles
    p = tl.arange(0, 128)
    th = p >> 2
    warp = p & 3
    c1 = warp * 16 + (th >> 2)
    tc_row = (th & 3) * 2
    src_e = src_ptr + e.to(tl.int64) * N * (K // 2)
    dst_e = dst_ptr + e.to(tl.int64) * k_tiles * n_tiles * 128 + kt * n_tiles * 128
    for t in tl.static_range(NT_PER):
        nt = tl.program_id(1) * NT_PER + t
        w = tl.zeros([128], dtype=tl.int32)
        for s in tl.static_range(8):
            q = (2 * s) % 8 + s // 4          # pack_idx[s] (constexpr)
            col = c1 + (q >> 2) * 8
            elem = tc_row + (q & 1) + 8 * ((q >> 1) & 1)    # tc_row + [0,1,8,9][q&3]
            n = nt * 64 + col
            k = kt * 16 + elem
            b = tl.load(src_e + n.to(tl.int64) * (K // 2) + (k >> 1), mask=nt < n_tiles, other=0).to(tl.int32)
            nib = tl.where((k & 1) == 1, b >> 4, b & 0xF)
            w = w | (nib << (4 * s))
        tl.store(dst_e + nt * 128 + p, w, mask=(nt < n_tiles) & (p < 128))


@triton.jit
def _repack_s_kernel(s_ptr, ratio_ptr, fac_ptr, dst_ptr, N, KB,
                     HAS_RATIO: tl.constexpr, BLOCK: tl.constexpr):
    # grid (E_chunk, cdiv(KB*N, BLOCK)); dst[e, kb, m] from src[e, n, kb]
    e = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    msk = i < N * KB
    kb = i // N
    m = i % N
    j = m & 3
    m2 = (m - j) + tl.where(j == 1, 2, tl.where(j == 2, 1, j))   # [0,2,1,3]
    ii = m2 & 63
    n = (m2 >> 6) * 64 + (ii >> 3) + 8 * (ii & 7)                 # scale_perm[8a+b] = a+8b
    base = e.to(tl.int64) * N * KB
    raw = tl.load(s_ptr + base + n.to(tl.int64) * KB + kb, mask=msk, other=0)
    x = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    if HAS_RATIO:
        r = tl.load(ratio_ptr + e.to(tl.int64) * N + n, mask=msk, other=1.0)
        x = (x * r).to(tl.float16).to(tl.float32).to(tl.float8e4nv).to(tl.float32)
    f = tl.load(fac_ptr + e)
    h = (x * f * 128.0).to(tl.float16)
    hb = h.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    out = tl.where(h.to(tl.float32) < 2.0, 0, (hb >> 7) & 0xFF)
    tl.store(dst_ptr + base + i, out.to(tl.uint8), mask=msk)


_E4M3_MAX_X128 = 448.0 * 128.0


def _folded_scale_max(s8: torch.Tensor, ratio: torch.Tensor | None, chunk: int = 32) -> torch.Tensor:
    """Per-expert max of the (ratio-folded, e4m3-requantized) block scales."""
    E = s8.shape[0]
    out = torch.empty(E, dtype=torch.float32, device=s8.device)
    for e0 in range(0, E, chunk):
        x = s8[e0:e0 + chunk].view(torch.float8_e4m3fn).float()
        if ratio is not None:
            x = (x * ratio[e0:e0 + chunk, :, None]).to(torch.float16).to(torch.float8_e4m3fn).float()
        out[e0:e0 + chunk] = x.flatten(1).amax(1)
    return out


def _factor(emax: torch.Tensor) -> torch.Tensor:
    """flashinfer _nvfp4_compute_scale_factor, per expert (device-side, no sync)."""
    m = emax * 128.0
    f = torch.exp2(torch.floor(torch.log2(_E4M3_MAX_X128 / m.clamp_min(1e-30))))
    ok = (m > 0) & (m < _E4M3_MAX_X128)
    return torch.where(ok, f, torch.ones_like(f))


@torch.no_grad()
def bank_meta(scale: torch.Tensor, glob: torch.Tensor, *, gated: bool):
    """(ratio | None, per-expert factor, alpha) of one native bank -- static per layer,
    so callers compute it once (it host-syncs on the all-ones ratio check)."""
    g = glob.float()
    if gated:
        gmax = g.amax(dim=1)
        ratio = g / gmax[:, None]
        ratio = None if bool((ratio == 1.0).all()) else ratio.contiguous()
        alpha_g = gmax
    else:
        ratio = None
        alpha_g = g[:, 0].contiguous()
    fac = _factor(_folded_scale_max(scale.view(torch.uint8), ratio)).contiguous()
    return ratio, fac, (alpha_g * (2.0 ** 119) / fac).contiguous()


@torch.no_grad()
def repack_apply_(packed: torch.Tensor, scale: torch.Tensor, ratio, fac: torch.Tensor,
                  tmp_w: torch.Tensor, tmp_s: torch.Tensor, chunk: int = 32) -> None:
    """In-place weight + scale repack of one bank with precomputed ``bank_meta``.
    Stream-ordered on the current stream; no host sync."""
    E, N, K2 = packed.shape
    K = K2 * 2
    KB = K // 16
    s8 = scale.view(torch.uint8)
    wb, sb = N * K2, N * KB
    NT_PER = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        n = e1 - e0
        tw = tmp_w[: n * wb]
        ts = tmp_s[: n * sb]
        _repack_w_kernel[(n * (K // 16), triton.cdiv(N // 64, NT_PER))](
            packed[e0:e1], tw.view(torch.int32), N, K, NT_PER=NT_PER, num_warps=4)
        _repack_s_kernel[(n, triton.cdiv(N * KB, 1024))](
            s8[e0:e1], ratio[e0:e1] if ratio is not None else fac, fac[e0:e1], ts, N, KB,
            HAS_RATIO=ratio is not None, BLOCK=1024, num_warps=4)
        packed[e0:e1].view(-1).copy_(tw)
        s8[e0:e1].reshape(-1).copy_(ts)


@torch.no_grad()
def repack_bank_(packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor,
                 tmp_w: torch.Tensor, tmp_s: torch.Tensor, *, gated: bool, chunk: int = 32):
    """One-shot: ``bank_meta`` + ``repack_apply_``; returns the [E] fp32 alpha."""
    ratio, fac, alpha = bank_meta(scale, glob, gated=gated)
    repack_apply_(packed, scale, ratio, fac, tmp_w, tmp_s, chunk)
    return alpha


def b12x_views(views, H: int, I: int):
    """Native bank views (gu_packed, gu_scale, gu_global, dn_packed, dn_scale, dn_global),
    already repacked in place, reshaped to flashinfer's prepared tensor shapes."""
    gu_p, gu_s, _, dn_p, dn_s, _ = views
    E = gu_p.shape[0]
    return (gu_p.view(torch.int32).view(E, H // 16, (2 * I // 64) * 128),
            gu_s.view(torch.float8_e4m3fn).view(E, H // 16, 2 * I),
            dn_p.view(torch.int32).view(E, I // 16, (H // 64) * 128),
            dn_s.view(torch.float8_e4m3fn).view(E, I // 16, H))


__all__ = ["repack_bank_", "bank_meta", "repack_apply_", "b12x_views"]
