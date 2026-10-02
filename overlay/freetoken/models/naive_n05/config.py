"""Engine-facing config for Naive-N0.5-Flash (``naive_n05_flash``).

Architecture (HF reference ``modeling_naive_n05_flash.py``):

* 48 layers; layer 0 has a dense SwiGLU MLP (16384), layers 1..47 route 8 of 256
  sigmoid/noaux_tc experts (2048 wide, no shared expert, renormalized, scale 1.0).
* Attention is plain split-projection GQA, 64 query heads, K head 192 / V head 128,
  partial NeoX rope over the first 64 dims, softmax scale 192**-0.5 and V scaled by
  ``attention_value_scale`` (0.707) before PV.
  - 40 SWA layers (``hybrid_layer_pattern == 1``): 8 KV heads, 128-token window,
    rope theta 1e4, a per-head attention sink logit.
  - 8 DSA layers (0, 5, 11, ..., 47): 4 KV heads, rope theta 1e7, no sink, and a
    lightning indexer (16 heads x 128, one shared key head with LayerNorm, fp8-e4m3
    rounded q/k, relu-weighted head sum) that keeps the top-2048 causal keys per query.

The engine sees one SWA group and one FULL group (the DSA layers); the Naive pool
(``kvcache/naive_pool.py``) stores K and V at their own widths plus the DSA index-key
slab, and the ``naive`` backend (``attention/naive.py``) serves both groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)


@dataclass(frozen=True)
class NaiveArgs:
    hidden_size: int
    num_heads: int
    head_dim: int  # K / Q head width (192)
    v_head_dim: int  # V head width (128)
    rotary_dim: int  # 64
    swa_layer_ids: tuple[int, ...]
    dsa_layer_ids: tuple[int, ...]
    swa_kv_heads: int
    dsa_kv_heads: int
    swa_rope_theta: float
    dsa_rope_theta: float
    sliding_window: int
    swa_sink: bool
    dsa_sink: bool
    value_scale: float
    index_heads: int
    index_dim: int
    index_topk: int
    index_fp8: bool
    dense_intermediate_size: int
    max_position: int
    norm_eps: float
    # MoE layers whose full expert banks live on the GPU as model weights (no host
    # bank, no cache slots, no PCIe): FREETOKEN_NAIVE_RESIDENT_LAYERS="1-5,9,14,17".
    # This is what keeps the host pin inside the RAM safety margin.
    resident_layer_ids: tuple[int, ...] = ()
    # Non-expert W8A16: attention qkv/o (incl. the DSA indexer rows) and the dense layer-0
    # MLP requantized at load to fp8-e4m3 with per-output-row scales (GLM recipe). Halves
    # the attention GEMV bytes and returns ~4.7 GiB of VRAM to the expert cache.
    # FREETOKEN_NAIVE_ATTN_FP8 / FREETOKEN_NAIVE_MLP_FP8 (=0 keeps bf16). lm_head stays bf16.
    attn_fp8: bool = False
    mlp_fp8: bool = False
    # FREETOKEN_NAIVE_HEAD_FP8=1: W8A16 lm_head too (GLM's GlmFp8LMHead); off by default.
    head_fp8: bool = False

    def is_swa(self, layer_id: int) -> bool:
        return layer_id in self.swa_layer_ids

    def kv_heads(self, layer_id: int) -> int:
        return self.swa_kv_heads if self.is_swa(layer_id) else self.dsa_kv_heads


def load_args(hf: Any, num_layers: int) -> NaiveArgs:
    pattern = list(hf.hybrid_layer_pattern)[:num_layers]
    head_dim = int(hf.head_dim)
    assert int(hf.swa_head_dim) == head_dim and int(hf.swa_v_head_dim) == int(hf.v_head_dim)
    assert int(hf.swa_num_attention_heads) == int(hf.num_attention_heads)
    assert getattr(hf, "attention_projection_layout", "split") == "split"
    assert int(getattr(hf, "index_n_kv_heads", 1)) == 1
    rotary_dim = int(head_dim * float(hf.partial_rotary_factor))
    return NaiveArgs(
        hidden_size=int(hf.hidden_size),
        num_heads=int(hf.num_attention_heads),
        head_dim=head_dim,
        v_head_dim=int(hf.v_head_dim),
        rotary_dim=rotary_dim,
        swa_layer_ids=tuple(i for i, p in enumerate(pattern) if p),
        dsa_layer_ids=tuple(i for i, p in enumerate(pattern) if not p),
        swa_kv_heads=int(hf.swa_num_key_value_heads),
        dsa_kv_heads=int(hf.num_key_value_heads),
        swa_rope_theta=float(hf.swa_rope_theta),
        dsa_rope_theta=float(hf.rope_theta),
        sliding_window=int(hf.sliding_window),
        swa_sink=bool(hf.add_swa_attention_sink_bias),
        dsa_sink=bool(hf.add_full_attention_sink_bias),
        value_scale=float(hf.attention_value_scale or 1.0),
        index_heads=int(hf.index_n_heads),
        index_dim=int(hf.index_head_dim),
        index_topk=int(hf.index_top_k),
        index_fp8=str(hf.indexer_activation_dtype) == "fp8_e4m3",
        dense_intermediate_size=int(hf.intermediate_size),
        max_position=int(hf.max_position_embeddings),
        norm_eps=float(hf.layernorm_epsilon),
        resident_layer_ids=_parse_resident_env(num_layers),
        attn_fp8=_env_flag("FREETOKEN_NAIVE_ATTN_FP8", "1"),
        mlp_fp8=_env_flag("FREETOKEN_NAIVE_MLP_FP8", "1"),
        head_fp8=_env_flag("FREETOKEN_NAIVE_HEAD_FP8", "0"),
    )


def _env_flag(name: str, default: str) -> bool:
    import os

    return os.environ.get(name, default) != "0"


def _parse_resident_env(num_layers: int) -> tuple[int, ...]:
    import os

    raw = os.environ.get("FREETOKEN_NAIVE_RESIDENT_LAYERS", "").strip()
    out = []
    for part in filter(None, (p.strip() for p in raw.split(","))):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return tuple(sorted(l for l in set(out) if l < num_layers))


def parse_config(hf_config: Any) -> ModelConfig:
    import os

    num_layers = int(hf_config.num_hidden_layers)
    # Dev only: truncate the stack (bring-up without pinning all 159 GiB of experts).
    cap = os.environ.get("FREETOKEN_NAIVE_MAX_LAYERS")
    if cap and int(cap) < num_layers:
        num_layers = int(cap)
    args = load_args(hf_config, num_layers)
    moe_freq = list(hf_config.moe_layer_freq)[:num_layers]
    first_k_dense = moe_freq.index(1) if 1 in moe_freq else num_layers
    assert all(moe_freq[first_k_dense:]), "Naive expects a contiguous dense prefix"

    swa_rope = RotaryConfig(
        head_dim=args.head_dim, rotary_dim=args.rotary_dim, max_position=args.max_position,
        base=args.swa_rope_theta, scaling=None,
    )
    dsa_rope = RotaryConfig(
        head_dim=args.head_dim, rotary_dim=args.rotary_dim, max_position=args.max_position,
        base=args.dsa_rope_theta, scaling=None,
    )
    quant = getattr(hf_config, "quantization_config", None)
    expert_quant = "nvfp4" if quant is not None else "none"
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=args.num_heads,
        num_kv_heads=args.dsa_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        vocab_size=int(hf_config.vocab_size),
        intermediate_size=args.dense_intermediate_size,
        rms_norm_eps=args.norm_eps,
        rotary_config=dsa_rope,
        hidden_act="silu",
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        num_experts=int(hf_config.n_routed_experts),
        num_experts_per_tok=int(hf_config.num_experts_per_tok),
        moe_intermediate_size=int(hf_config.moe_intermediate_size),
        norm_topk_prob=bool(hf_config.norm_topk_prob),
        model_type="naive_n05_flash",
        architectures=["NaiveN05FlashForCausalLM"],
        moe_enabled=True,
        expert_quant=expert_quant,
        first_k_dense_replace=first_k_dense,
        n_shared_experts=0,
        routed_scaling_factor=float(hf_config.routed_scaling_factor or 1.0),
        n_group=int(hf_config.n_group),
        topk_group=int(hf_config.topk_group),
        has_router_bias=True,
        attn_sm_scale=args.head_dim ** -0.5,
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=args.swa_layer_ids,
                num_kv_heads=args.swa_kv_heads,
                head_dim=args.head_dim,
                rotary_config=swa_rope,
                sliding_window=args.sliding_window,
            ),
            FullAttentionGroupConfig(
                name="full",
                layer_ids=args.dsa_layer_ids,
                num_kv_heads=args.dsa_kv_heads,
                head_dim=args.head_dim,
                rotary_config=dsa_rope,
            ),
        ),
        single_stream_only=True,
        naive_args=args,
    )


__all__ = ["NaiveArgs", "parse_config", "load_args"]
