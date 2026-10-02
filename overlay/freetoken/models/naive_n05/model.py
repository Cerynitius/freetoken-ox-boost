"""Naive-N0.5-Flash model (see config.py for the architecture summary).

One merged projection per layer feeds everything the attention needs:
SWA ``[q 64x192 | k 8x192 | v 8x128]``; DSA additionally
``[index_q 16x128 | index_k 128 | index_w 16]`` -- one GEMV reads all of a layer's
input-side attention weights at decode. RoPE, the indexer LayerNorm / fp8 rounding and
the attention itself live in the ``naive`` backend (attention/naive.py), which owns the
rope tables sized to the engine's page table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate
from freetoken.utils.regions import region

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


import os as _os0

# fused attention front-end (one prep kernel per layer); =0 -> the unfused reference path
_FUSED_PREP = _os0.environ.get("FREETOKEN_NAIVE_FUSED_PREP", "1") != "0"


class NaiveAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.naive_args
        self.layer_id = layer_id
        self.is_swa = args.is_swa(layer_id)
        self.num_heads = args.num_heads
        self.kv_heads = args.kv_heads(layer_id)
        self.head_dim = args.head_dim
        self.v_dim = args.v_head_dim
        q, k, v = (
            self.num_heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.kv_heads * self.v_dim,
        )
        splits = [q, k, v]
        if not self.is_swa:
            self.index_heads = args.index_heads
            self.index_dim = args.index_dim
            splits += [args.index_heads * args.index_dim, args.index_dim, args.index_heads]
            # indexer k LayerNorm (with bias, eps 1e-5 as nn.LayerNorm default)
            self.index_k_norm_weight = torch.empty(args.index_dim)
            self.index_k_norm_bias = torch.empty(args.index_dim)
            self.index_w_scale = args.index_heads ** -0.5
        self._splits = splits
        if args.attn_fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear as _Lin
        else:
            _Lin = LinearReplicated
        self.qkv_proj = _Lin(args.hidden_size, sum(splits), has_bias=False)
        self.o_proj = _Lin(self.num_heads * self.v_dim, args.hidden_size, has_bias=False)
        sink = args.swa_sink if self.is_swa else args.dsa_sink
        self.attention_sink_bias = torch.empty(self.num_heads) if sink else None

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        backend = ctx.attn_backend
        kind = "swa" if self.is_swa else "dsa"
        with region(f"attn.{kind}.qkv_gemv"):
            proj = self.qkv_proj.forward(x)
        del x
        if _FUSED_PREP:
            o = backend.naive_forward_proj(
                proj, self.layer_id, ctx.batch, attn=self, sinks=self.attention_sink_bias
            )
            with region(f"attn.{kind}.o_gemv"):
                return self.o_proj.forward(o.view(proj.shape[0], self.num_heads * self.v_dim))
        parts = proj.split(self._splits, dim=-1)
        T = proj.shape[0]
        q = parts[0].reshape(T, self.num_heads, self.head_dim)
        k = parts[1].reshape(T, self.kv_heads, self.head_dim)
        v = parts[2].reshape(T, self.kv_heads, self.v_dim)
        index = None
        if not self.is_swa:
            iq = parts[3].reshape(T, self.index_heads, self.index_dim)
            ik = F.layer_norm(
                parts[4], (self.index_dim,), self.index_k_norm_weight,
                self.index_k_norm_bias, 1e-5,
            )
            # reference: weights_proj(x) (bf16) * H**-0.5 (power of two -> exact), then fp32
            iw = parts[5].float() * self.index_w_scale
            index = (iq, ik, iw)
        o = backend.naive_forward(
            q, k, v, self.layer_id, ctx.batch, sinks=self.attention_sink_bias, index=index,
        )
        return self.o_proj.forward(o.view(T, self.num_heads * self.v_dim))


class NaiveDenseMLP(BaseOP):
    def __init__(self, hidden_size: int, intermediate_size: int, fp8: bool = False):
        if fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import (
                Fp8PerTensorColMerged,
                Fp8PerTensorLinear,
            )

            self.gate_up_proj = Fp8PerTensorColMerged(
                hidden_size, [intermediate_size, intermediate_size], has_bias=False
            )
            self.down_proj = Fp8PerTensorLinear(intermediate_size, hidden_size, has_bias=False)
            return
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(intermediate_size, hidden_size, has_bias=False)

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


def offload_moe_layers(config: ModelConfig) -> list[int]:
    """MoE layers served from the offload cache, in order (bank id == list index)."""
    res = frozenset(config.naive_args.resident_layer_ids)
    return [l for l in range(config.first_k_dense_replace, config.num_layers) if l not in res]


class NaiveMoE(BaseOP):
    """DeepSeek-V3 router (fp32 gate, sigmoid + selection bias, top-8, renorm, x1.0),
    routed NVFP4 experts from the offload cache, no shared expert."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        # HF keeps the router weight and the selection bias fp32 (_keep_in_fp32_modules_strict)
        self.gate.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self.resident = layer_id in config.naive_args.resident_layer_ids
        if self.resident:
            from freetoken.models.glm5_next.experts_resident import ResidentNvfp4Experts

            self.experts = ResidentNvfp4Experts(config, "silu", layer_id=layer_id)
        else:
            # offload bank ids are dense over the NON-resident MoE layers (same list the
            # host-bank builder in weight.py uses)
            self.experts = make_moe_layer(
                config,
                layer_id=offload_moe_layers(config).index(layer_id),
                renormalize=config.norm_topk_prob,
                activation="silu",
            )
            if hasattr(self.experts, "spec_set_block"):
                self.experts.spec_set_block(self)

    def _route(self, x: torch.Tensor):
        from freetoken.kernel.triton.fused_route import fused_route, fused_route_ranked

        logits = F.linear(x.float(), self.gate.weight)
        cache = get_global_ctx().moe_offload_cache
        # resident layers never consume _soft_rank (it would leak into the next
        # offload layer's ensure), so they take the plain route
        if (cache is not None and cache.cache_policy == "soft" and x.shape[0] == 1
                and not self.resident):
            # same routing result + the ranked near-miss list the soft cache policy uses
            w, ids, rid, rsc = fused_route_ranked(
                logits, self.e_score_correction_bias, self.top_k,
                self.norm_topk_prob, self.routed_scaling_factor, n_rank=16,
            )
            cache._soft_rank = (rid, rsc)
            return w, ids
        return fused_route(
            logits, self.e_score_correction_bias, self.top_k,
            self.norm_topk_prob, self.routed_scaling_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with region("moe.router"):
            w, ids = self._route(x)
        return self.experts.routed_forward(x, w, ids)


class NaiveDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = NaiveAttention(config, layer_id)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = NaiveMoE(config, layer_id)
        else:
            self.mlp = NaiveDenseMLP(
                config.hidden_size, config.intermediate_size, fp8=config.naive_args.mlp_fp8
            )
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with region("norm.input"):
            x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        with region("norm.post_attn"):
            x, residual = self.post_attention_layernorm.forward(x, residual)
        with region("mlp.dense" if not self.mlp.__class__.__name__ == "NaiveMoE" else "moe"):
            x = self.mlp.forward(x)
        return x, residual


class NaiveHostEmbedding(BaseOP):
    """Token embedding kept in pinned host memory (1.25 GiB of VRAM returned to the
    expert cache). Each forward gathers only the rows it needs over PCIe with the
    UVA index-copy kernel (8 KiB per token; CUDA-graph safe). Exact: a plain row copy.
    FREETOKEN_NAIVE_HOST_EMBED=0 keeps the regular GPU table."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)
        self._dim = embedding_dim
        self._host: torch.Tensor | None = None

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False):
        from freetoken.kernel.pinned import copy_to_pinned_tensor

        w = state_dict.pop(f"{prefix}.weight")
        self._host = copy_to_pinned_tensor(w.to("cpu"))
        self.weight = None  # not a tensor -> skipped by state_dict walks
        del w
        torch.cuda.empty_cache()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fast_index_copy import fast_index_copy_jit

        T = input_ids.shape[0]
        out = torch.empty((T, self._dim), dtype=self._host.dtype, device=input_ids.device)
        dst_idx = torch.arange(T, dtype=torch.int32, device=input_ids.device)
        fast_index_copy_jit(out, dst_idx, self._host, input_ids.to(torch.int32))
        return out


class NaiveModel(BaseOP):
    def __init__(self, config: ModelConfig):
        import os

        if os.environ.get("FREETOKEN_NAIVE_HOST_EMBED", "1") != "0" and not config.tie_word_embeddings:
            self.embed_tokens = NaiveHostEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        self.layers = OPList(
            [NaiveDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with region("embed"):
            x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        dump = _dump_active()
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
            if dump is not None:
                dump.setdefault("layers", []).append((x + residual)[-_DUMP_KEEP:].float().cpu())
        with region("norm.final"):
            return self.norm.forward(x, residual)[0]


# Validation hook (off by default): FREETOKEN_NAIVE_DUMP=<dir> saves, per EAGER forward,
# the residual stream after every layer (last _DUMP_KEEP rows), the positions and the
# logits. Never active while a CUDA graph is being captured.
import os as _os

_DUMP_DIR = _os.environ.get("FREETOKEN_NAIVE_DUMP", "")
_DUMP_KEEP = int(_os.environ.get("FREETOKEN_NAIVE_DUMP_KEEP", "64"))
_DUMP_STATE: dict = {"n": 0, "cur": None}


def _dump_active():
    if not _DUMP_DIR or torch.cuda.is_current_stream_capturing():
        return None
    return _DUMP_STATE["cur"]


class NaiveN05ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = NaiveModel(config)
        if config.naive_args.head_fp8 and not config.tie_word_embeddings:
            from freetoken.models.glm_moe_dsa.model import GlmFp8LMHead

            self.lm_head = GlmFp8LMHead(config.vocab_size, config.hidden_size)
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        if _DUMP_DIR and not torch.cuda.is_current_stream_capturing():
            _DUMP_STATE["cur"] = {"positions": batch.positions.cpu(), "ids": batch.input_ids.cpu()}
        output = self.model.forward(batch.input_ids)
        with region("lm_head"):
            logits = self.lm_head.forward(output)
        cur = _DUMP_STATE["cur"]
        if cur is not None:
            cur["logits"] = logits.float().cpu()
            _os.makedirs(_DUMP_DIR, exist_ok=True)
            torch.save(cur, f"{_DUMP_DIR}/fwd{_DUMP_STATE['n']:04d}.pt")
            _DUMP_STATE["n"] += 1
            _DUMP_STATE["cur"] = None
        return logits


__all__ = ["NaiveN05ForCausalLM"]
