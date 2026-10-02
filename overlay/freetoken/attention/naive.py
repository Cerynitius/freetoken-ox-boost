"""Naive-N0.5-Flash attention backend (SWA + DSA over the Naive pool).

Metadata / CUDA-graph staging is the Triton backend's (page-table slots, full->swa
translated slots, positions, cu_seqlens), reused unchanged; only the compute differs:

* SWA layers: 128-token sliding window with a per-head sink, K 192 / V 128.
* DSA layers: the lightning indexer scores every visible key, the top-2048 are kept,
  and attention runs over those slots only.
  - decode (graph-captured, bs=1): scores for the whole page-table row (fixed width,
    -inf past the live length) -> torch.topk(2048) -> slot list (-1 = empty) -> split-K
    sparse decode. When fewer than 2048 keys exist every live key is selected, which is
    exactly dense attention.
  - prefill (eager): dense causal attention when the request's keys fit the top-k;
    otherwise per query chunk: scores [chunk, L] -> top-k -> sparse prefill kernel.

Rope for both groups also lives here (fp32 cos/sin tables sized to the page table).
The attention V scale (0.707) is applied to the fp32 output before the bf16 store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger
from freetoken.utils.regions import region

from .base import AttentionSpec
from .triton import TritonAttentionBackend, TritonMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)

_DECODE_SPLITS_SWA = 4
_DECODE_SPLITS_DSA = 32
# prefill index-score transient budget (fp32 [chunk, L])
_PREFILL_SCORE_BYTES = 512 << 20


class NaiveAttnBackend(TritonAttentionBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        from freetoken.kernel.triton.naive_attn import build_rope_tables

        self.args = args = config.naive_args
        ctx = get_global_ctx()
        dev = self.device
        self.page_width = int(ctx.page_table.shape[1])
        max_pos = self.page_width + 8
        self.cos_swa, self.sin_swa = build_rope_tables(
            max_pos, args.rotary_dim, args.swa_rope_theta, dev
        )
        self.cos_dsa, self.sin_dsa = build_rope_tables(
            max_pos, args.rotary_dim, args.dsa_rope_theta, dev
        )
        self.sm_scale = args.head_dim ** -0.5
        self.topk = args.index_topk
        H, DV = args.num_heads, args.v_head_dim
        self.mid_o_swa = torch.empty((1, H, _DECODE_SPLITS_SWA, DV), dtype=torch.float32, device=dev)
        self.mid_lse_swa = torch.empty((1, H, _DECODE_SPLITS_SWA), dtype=torch.float32, device=dev)
        self.mid_o_dsa = torch.empty((1, H, _DECODE_SPLITS_DSA, DV), dtype=torch.float32, device=dev)
        self.mid_lse_dsa = torch.empty((1, H, _DECODE_SPLITS_DSA), dtype=torch.float32, device=dev)
        # decode index scores over the full page-table row (graph-static width)
        self.dec_scores = torch.empty((1, self.page_width), dtype=torch.float32, device=dev)
        # exact live-length radix top-k (kernel/triton/naive_attn.DsaTopk): cost scales with
        # the live keys, not the graph-static page width (1M with elastic KV).
        # FREETOKEN_NAIVE_RADIX_TOPK=0 -> torch.topk over the whole row.
        import os as _os_tk

        self._radix_topk = None
        if _os_tk.environ.get("FREETOKEN_NAIVE_RADIX_TOPK", "1") != "0":
            from freetoken.kernel.triton.naive_attn import DsaTopk

            self._radix_topk = DsaTopk(self.page_width, self.topk, dev)
            self.dec_sel = torch.empty((1, self.topk), dtype=torch.int32, device=dev)
        logger.info(
            f"naive backend: page width {self.page_width}, topk {self.topk}, "
            f"rope tables {2 * 2 * self.cos_swa.numel() * 4 / 2**20:.0f} MiB"
        )

    # ------------------------------------------------------------------ rope
    def rope_(self, x: torch.Tensor, positions: torch.Tensor, is_swa: bool) -> None:
        from freetoken.kernel.triton.naive_attn import naive_rope_

        if is_swa:
            naive_rope_(x, positions, self.cos_swa, self.sin_swa)
        else:
            naive_rope_(x, positions, self.cos_dsa, self.sin_dsa)

    # ------------------------------------------------------------------ entry
    def forward(self, q, k, v, layer_id, batch, attn_spec: AttentionSpec | None = None):
        raise RuntimeError("Naive attention is driven through naive_forward()")

    def naive_forward(
        self,
        q: torch.Tensor,  # [T, H, 192] (pre-rope)
        k: torch.Tensor,  # [T, Hk, 192] (pre-rope)
        v: torch.Tensor,  # [T, Hk, 128]
        layer_id: int,
        batch: Batch,
        *,
        sinks: torch.Tensor | None,
        index: tuple | None,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.naive_attn import naive_round_fp8

        md = batch.attn_metadata
        assert isinstance(md, TritonMetadata)
        is_swa = index is None
        pos = batch.positions
        q = q.contiguous()
        k = k.contiguous()
        self.rope_(q, pos, is_swa)
        self.rope_(k, pos, is_swa)
        kv = self.kvcache
        kv.store_kv(k, v, batch.out_loc, layer_id)
        k_cache = kv.k_cache(layer_id)
        v_cache = kv.v_cache(layer_id)
        if is_swa:
            return self._swa(q, k_cache, v_cache, md, sinks)
        iq, ik, iw = index
        iq = iq.contiguous()
        ik = ik.to(q.dtype).unsqueeze(1).contiguous()  # [T, 1, 128]
        self.rope_(iq, pos, False)
        self.rope_(ik, pos, False)
        iq8, iqs = naive_round_fp8(iq)
        ik8, iks = naive_round_fp8(ik.squeeze(1))
        kv.store_index(ik8, iks, batch.out_loc, layer_id)
        if md.is_decode:
            return self._dsa_decode(q, k_cache, v_cache, md, sinks, layer_id, iq8, iqs, iw)
        return self._dsa_prefill(q, k_cache, v_cache, md, batch, sinks, layer_id, iq8, iqs, iw)

    def naive_forward_proj(
        self,
        proj: torch.Tensor,  # [T, NTOT] merged projection output
        layer_id: int,
        batch: Batch,
        *,
        attn,  # NaiveAttention module (geometry + indexer norm weights)
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fused front-end: one prep launch does rope(q)->q, rope(k)/v->cache and (DSA)
        the indexer q/k/w prep + index-key store; then the attention kernels."""
        from freetoken.kernel.triton.naive_attn import naive_prep

        md = batch.attn_metadata
        kv = self.kvcache
        k_cache = kv.k_cache(layer_id)
        v_cache = kv.v_cache(layer_id)
        is_swa = attn.is_swa
        if is_swa:
            with region("attn.swa.prep"):
                q, _ = naive_prep(
                    proj, batch.positions, batch.out_loc, kv.full_to_swa_index_mapping,
                    self.cos_swa, self.sin_swa, k_cache, v_cache,
                    attn.num_heads, attn.kv_heads, attn.head_dim, attn.v_dim,
                )
            with region("attn.swa.core"):
                return self._swa(q, k_cache, v_cache, md, sinks)
        k8, ks = kv.index_cache(layer_id)
        with region("attn.dsa.prep"):
          q, (iq8, iqs, iw) = naive_prep(
            proj, batch.positions, batch.out_loc, None,
            self.cos_dsa, self.sin_dsa, k_cache, v_cache,
            attn.num_heads, attn.kv_heads, attn.head_dim, attn.v_dim,
            index=(k8, ks, attn.index_k_norm_weight, attn.index_k_norm_bias,
                   attn.index_heads, attn.index_dim, attn.index_w_scale),
        )
        if md.is_decode:
            return self._dsa_decode(q, k_cache, v_cache, md, sinks, layer_id, iq8, iqs, iw)
        return self._dsa_prefill(q, k_cache, v_cache, md, batch, sinks, layer_id, iq8, iqs, iw)

    # ------------------------------------------------------------------ SWA
    def _swa(self, q, k_cache, v_cache, md: TritonMetadata, sinks):
        from freetoken.kernel.triton.naive_attn import (
            naive_decode_attention,
            naive_extend_attention,
        )

        W = self.args.sliding_window
        scale = self.args.value_scale
        if md.is_decode:
            return naive_decode_attention(
                q, k_cache, v_cache, self.sm_scale, scale,
                self.mid_o_swa, self.mid_lse_swa,
                indptr=md.indptr, indices=md.swa_indices, q_positions=md.q_positions,
                window=W, sinks=sinks,
            )
        return naive_extend_attention(
            q, k_cache, v_cache, md.cu_seqlens_q_gpu, md.indptr, md.swa_indices,
            md.prefix_lens, md.max_q_len, self.sm_scale, scale, window=W, sinks=sinks,
        )

    # ------------------------------------------------------------------ DSA
    def _dsa_decode(self, q, k_cache, v_cache, md, sinks, layer_id, iq8, iqs, iw):
        from freetoken.kernel.triton.naive_attn import (
            naive_decode_attention,
            naive_index_scores,
        )

        assert q.shape[0] == 1, "Naive decode is single-stream"
        k8, ks = self.kvcache.index_cache(layer_id)
        row = md.indices  # bs=1: the request's page-table row (graph: full capture row)
        if row.numel() < self.page_width:  # eager decode: live row only -> pad to width
            row = torch.nn.functional.pad(row, (0, self.page_width - row.numel()))
        n_keys = md.indptr[1:2]
        with region("attn.dsa.index_scores"):
            scores = naive_index_scores(
                iq8, iqs, iw, k8, ks, row, md.q_positions, n_keys, self.page_width,
                out=self.dec_scores,
            )
        with region("attn.dsa.topk"):
            if self._radix_topk is not None:
                sel = self._radix_topk(scores, row, n_keys, self.dec_sel)
            else:
                vals, idx = torch.topk(scores, self.topk, dim=-1, sorted=False)
                slots = torch.gather(row[: self.page_width].unsqueeze(0), 1, idx)
                sel = torch.where(vals > float("-inf"), slots, torch.full_like(slots, -1))
        with region("attn.dsa.sparse_attn"):
            return naive_decode_attention(
                q, k_cache, v_cache, self.sm_scale, self.args.value_scale,
                self.mid_o_dsa, self.mid_lse_dsa, sel=sel, sinks=sinks,
            )

    def _dsa_prefill(self, q, k_cache, v_cache, md, batch, sinks, layer_id, iq8, iqs, iw):
        from freetoken.kernel.triton.naive_attn import (
            naive_extend_attention,
            naive_index_scores,
            naive_sparse_prefill,
        )

        reqs = batch.padded_reqs
        seqlens_q = [r.extend_len for r in reqs]
        seqlens_k = [r.device_len for r in reqs]
        scale = self.args.value_scale
        if max(seqlens_k) <= self.topk:
            return naive_extend_attention(
                q, k_cache, v_cache, md.cu_seqlens_q_gpu, md.indptr, md.indices,
                md.prefix_lens, md.max_q_len, self.sm_scale, scale, sinks=sinks,
            )
        k8, ks = self.kvcache.index_cache(layer_id)
        out = torch.empty(
            (q.shape[0], q.shape[1], v_cache.shape[-1]), dtype=q.dtype, device=q.device
        )
        q_off = 0
        k_off = 0
        for n_q, n_k in zip(seqlens_q, seqlens_k):
            if n_q == 0:
                k_off += n_k
                continue
            row = md.indices[k_off : k_off + n_k]
            n_keys = torch.tensor([n_k], dtype=torch.int32, device=q.device)
            kk = min(self.topk, n_k)
            chunk = max(1, min(n_q, _PREFILL_SCORE_BYTES // (4 * n_k)))
            for c0 in range(0, n_q, chunk):
                c1 = min(n_q, c0 + chunk)
                a, b = q_off + c0, q_off + c1
                scores = naive_index_scores(
                    iq8[a:b], iqs[a:b], iw[a:b], k8, ks, row, md.q_positions[a:b],
                    n_keys, n_k,
                )
                vals, idx = torch.topk(scores, kk, dim=-1, sorted=False)
                del scores
                slots = row[idx.reshape(-1)].view(b - a, kk)
                sel = torch.full((b - a, self.topk), -1, dtype=torch.int32, device=q.device)
                sel[:, :kk] = torch.where(vals > float("-inf"), slots, -1)
                naive_sparse_prefill(
                    q[a:b], k_cache, v_cache, sel, self.sm_scale, scale, sinks=sinks,
                    out=out[a:b],
                )
            q_off += n_q
            k_off += n_k
        return out


__all__ = ["NaiveAttnBackend"]
