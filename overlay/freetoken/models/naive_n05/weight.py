"""Weight loading for Naive-N0.5-Flash (NVFP4 community checkpoint or the bf16 original).

Resident weights stream bf16 (checkpoint-faithful) into the merged layouts model.py
builds: per layer ``self_attn.qkv_proj`` = cat(q, k, v[, indexer wq, wk, weights_proj]),
``mlp.gate_up_proj`` = cat(gate, up) for the dense layer 0. The router weight and
selection bias stay fp32. Routed experts (ModelOpt NVFP4, same key layout as GLM) go to
the offload cache through the shared glm4_moe bank loader.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm4_moe.weight import _ROUTED_EXPERT_KEY_RE, _ShardReader
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config


def _layer_to_bank(layer: int, config) -> int | None:
    """Host-bank id: dense over the non-resident MoE layers (resident layers load
    their full banks as GPU model weights in iter_weights)."""
    from .model import offload_moe_layers

    if layer < config.first_k_dense_replace or layer >= config.num_layers:
        return None
    if layer in config.naive_args.resident_layer_ids:
        return None
    return offload_moe_layers(config).index(layer)


_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_ROUTED_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="Naive NVFP4 experts",
)


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
):
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path, config, _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers, chunk=chunk, layer_sink=layer_sink,
    )


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "Naive-N0.5-Flash routed experts are NVFP4 and load into the offload cache "
        "(load_nvfp4_expert_sources); use --moe-backend offload/hybrid."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    args = config.naive_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    try:
        for layer in tqdm(
            range(config.num_layers), desc="Loading Naive dense weights", disable=not primary
        ):
            p = f"model.layers.{layer}"
            a = f"{p}.self_attn"
            parts = [reader.get(f"{a}.{n}.weight") for n in ("q_proj", "k_proj", "v_proj")]
            if not args.is_swa(layer):
                parts += [
                    reader.get(f"{a}.indexer.{n}.weight") for n in ("wq", "wk", "weights_proj")
                ]
                yield f"{a}.index_k_norm_weight", reader.get(f"{a}.indexer.k_norm.weight")
                yield f"{a}.index_k_norm_bias", reader.get(f"{a}.indexer.k_norm.bias")
            yield from _dense(f"{a}.qkv_proj", torch.cat([t.to(torch.bfloat16) for t in parts], 0),
                              args.attn_fp8)
            del parts
            yield from _dense(f"{a}.o_proj", reader.get(f"{a}.o_proj.weight"), args.attn_fp8)
            if reader.has(f"{a}.attention_sink_bias"):
                yield f"{a}.attention_sink_bias", reader.get(f"{a}.attention_sink_bias")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{p}.{norm}.weight", reader.get(f"{p}.{norm}.weight")
            m = f"{p}.mlp"
            if layer < config.first_k_dense_replace:
                yield from _dense(f"{m}.gate_up_proj", torch.cat(
                    [reader.get(f"{m}.gate_proj.weight"), reader.get(f"{m}.up_proj.weight")], 0
                ), args.mlp_fp8)
                yield from _dense(f"{m}.down_proj", reader.get(f"{m}.down_proj.weight"), args.mlp_fp8)
            else:
                if layer in args.resident_layer_ids:
                    yield from _resident_banks(reader, f"{m}", config)
                yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight").float()
                yield (
                    f"{m}.e_score_correction_bias",
                    reader.get(f"{m}.gate.e_score_correction_bias").float(),
                )
        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight")
        yield "model.norm.weight", reader.get("model.norm.weight")
        yield from _dense("lm_head", reader.get("lm_head.weight"),
                          args.head_fp8 and not config.tie_word_embeddings)
    finally:
        reader.close()


def _dense(prefix: str, w: torch.Tensor, fp8: bool):
    """A resident dense projection: bf16 as stored, or per-output-row fp8-e4m3 W8A16
    (``weight`` + ``weight_scale``, the GLM quantizer) when ``fp8``."""
    if not fp8:
        yield f"{prefix}.weight", w
        return
    from freetoken.models.glm_moe_dsa.weight import _quant_fp8_per_row

    q, scale = _quant_fp8_per_row(w)
    yield f"{prefix}.weight", q
    yield f"{prefix}.weight_scale", scale


def _resident_banks(reader, m: str, config):
    """A resident layer's six native ModelOpt banks (position == expert id), stacked
    the way the offload bank builder lays them out (per-row fp16 global scale)."""
    E = config.num_experts

    def stack(proj, kind):
        return torch.stack([reader.get(f"{m}.experts.{e}.{proj}.{kind}") for e in range(E)])

    def glob(proj, rows):
        return torch.stack([
            reader.get(f"{m}.experts.{e}.{proj}.weight_scale_2").reshape(1).to(torch.float16).expand(rows)
            for e in range(E)
        ]).contiguous()

    i_sz = config.moe_intermediate_size
    yield f"{m}.experts.gate_up_packed", torch.cat([stack("gate_proj", "weight"), stack("up_proj", "weight")], 1)
    yield f"{m}.experts.gate_up_scale", torch.cat(
        [stack("gate_proj", "weight_scale"), stack("up_proj", "weight_scale")], 1)
    yield f"{m}.experts.gate_up_global", torch.cat([glob("gate_proj", i_sz), glob("up_proj", i_sz)], 1)
    yield f"{m}.experts.down_packed", stack("down_proj", "weight")
    yield f"{m}.experts.down_scale", stack("down_proj", "weight_scale")
    yield f"{m}.experts.down_global", glob("down_proj", config.hidden_size)


__all__ = ["iter_weights", "load_nvfp4_expert_sources", "load_nvfp4_expert_sources_parallel"]
