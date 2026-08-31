"""DSpark (mtp.*) weight loading for DeepSeek-V4-Flash.

Called from ``DeepseekV4ForCausalLM.load_state_dict`` when ``FREETOKEN_DSV4_SPEC=1``.
Non-expert tensors are dequantized to BF16 on device (a few hundred MB); the routed
experts stay packed ds_fp4 in VRAM slabs shaped exactly like the offload banks
(``[E, 2I, H//2]`` gate_up, ``[E, H, I//2]`` down), so the shared grouped decode kernel
runs on them with ``slots == expert ids``. Runs BEFORE the MoE offload cache sizes
itself, so ``--moe-cache-auto`` sees the reduced free VRAM and adapts.

Also stamps the transformer with the spec-side hooks' state: ``_spec_targets`` (the
hc-mean staging layer ids), ``spec_hidden_decode`` (the decode-graph staging buffer),
``freqs_dspark`` (the plain-rope table the DSpark stages use), and the ``_dspark``
weight dicts consumed by ``dspark.DsparkDraft``.
"""

from __future__ import annotations

import json
import os

import torch


def _deq_block128(w: torch.Tensor, s: torch.Tensor, device) -> torch.Tensor:
    """FP8(e4m3) weight + e8m0 128x128 block scales -> BF16."""
    w = w.to(device)
    s = s.to(device)
    sc = torch.exp2(s.view(torch.uint8).float() - 127.0)
    O, I = w.shape
    v = w.float().view(O // 128, 128, I // 128, 128) * sc[:, None, :, None]
    return v.view(O, I).to(torch.bfloat16)


def load_dspark(transformer, ckpt_dir: str) -> None:
    from safetensors import safe_open

    device = transformer.embed.weight.device
    args = transformer.args

    # dspark hyperparams ship in the reference inference config, not the HF config
    icfg = json.load(open(os.path.join(ckpt_dir, "inference", "config.json")))
    for k in ("dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank",
              "n_mtp_layers"):
        object.__setattr__(args, k, icfg[k])
    object.__setattr__(args, "dspark_target_layer_ids", tuple(icfg["dspark_target_layer_ids"]))
    n_stages = args.n_mtp_layers
    last = n_stages - 1

    idx = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
    names = [k for k in idx if k.startswith("mtp.")]
    byshard: dict[str, list[str]] = {}
    for k in names:
        byshard.setdefault(idx[k], []).append(k)
    raw: dict[str, torch.Tensor] = {}
    for shard, keys in byshard.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
            for k in keys:
                raw[k] = f.get_tensor(k)

    def deq(prefix: str) -> torch.Tensor:
        return _deq_block128(raw[prefix + ".weight"], raw[prefix + ".scale"], device)

    def bf(name: str) -> torch.Tensor:
        return raw[name].to(device=device, dtype=torch.bfloat16)

    def f32(name: str) -> torch.Tensor:
        return raw[name].to(device=device, dtype=torch.float32)

    E = args.n_routed_experts
    I = args.moe_inter_dim
    H = args.dim
    e8m0 = torch.float8_e8m0fnu

    stages = []
    for s in range(n_stages):
        p = f"mtp.{s}"
        w: dict[str, torch.Tensor] = {
            "attn.wq_a": deq(f"{p}.attn.wq_a"),
            "attn.wq_b": deq(f"{p}.attn.wq_b"),
            "attn.wkv": deq(f"{p}.attn.wkv"),
            "attn.wo_a": deq(f"{p}.attn.wo_a"),
            "attn.wo_b": deq(f"{p}.attn.wo_b"),
            "attn.q_norm": bf(f"{p}.attn.q_norm.weight"),
            "attn.kv_norm": bf(f"{p}.attn.kv_norm.weight"),
            "attn.attn_sink": f32(f"{p}.attn.attn_sink"),
            "attn_norm": bf(f"{p}.attn_norm.weight"),
            "ffn_norm": bf(f"{p}.ffn_norm.weight"),
            "gate.weight": bf(f"{p}.ffn.gate.weight"),
            "gate.bias": f32(f"{p}.ffn.gate.bias"),
            "shared.w1": deq(f"{p}.ffn.shared_experts.w1"),
            "shared.w2": deq(f"{p}.ffn.shared_experts.w2"),
            "shared.w3": deq(f"{p}.ffn.shared_experts.w3"),
        }
        for hc in ("hc_attn", "hc_ffn"):
            w[f"{hc}_fn"] = f32(f"{p}.{hc}_fn")
            w[f"{hc}_base"] = f32(f"{p}.{hc}_base")
            w[f"{hc}_scale"] = f32(f"{p}.{hc}_scale")

        gu_packed = torch.empty(E, 2 * I, H // 2, dtype=torch.uint8, device=device)
        gu_scale = torch.empty(E, 2 * I, H // 32, dtype=e8m0, device=device)
        down_packed = torch.empty(E, H, I // 2, dtype=torch.uint8, device=device)
        down_scale = torch.empty(E, H, I // 32, dtype=e8m0, device=device)
        for e in range(E):
            ep = f"{p}.ffn.experts.{e}"
            gu_packed[e, :I] = raw[f"{ep}.w1.weight"].view(torch.uint8).to(device)
            gu_packed[e, I:] = raw[f"{ep}.w3.weight"].view(torch.uint8).to(device)
            gu_scale[e, :I] = raw[f"{ep}.w1.scale"].view(e8m0).to(device)
            gu_scale[e, I:] = raw[f"{ep}.w3.scale"].view(e8m0).to(device)
            down_packed[e] = raw[f"{ep}.w2.weight"].view(torch.uint8).to(device)
            down_scale[e] = raw[f"{ep}.w2.scale"].view(e8m0).to(device)
        w["exp.gu_packed"] = gu_packed
        w["exp.gu_scale"] = gu_scale
        w["exp.down_packed"] = down_packed
        w["exp.down_scale"] = down_scale
        stages.append(w)

    top: dict[str, torch.Tensor] = {
        "main_proj": _deq_block128(raw["mtp.0.main_proj.weight"],
                                   raw["mtp.0.main_proj.scale"], device),
        "main_norm": bf("mtp.0.main_norm.weight"),
        "head_norm": bf(f"mtp.{last}.norm.weight"),
        "hc_head_fn": f32(f"mtp.{last}.hc_head_fn"),
        "hc_head_base": f32(f"mtp.{last}.hc_head_base"),
        "hc_head_scale": f32(f"mtp.{last}.hc_head_scale"),
        "markov_w1": bf(f"mtp.{last}.markov_head.markov_w1.weight"),
        "markov_w2": bf(f"mtp.{last}.markov_head.markov_w2.weight"),
        "confidence": f32(f"mtp.{last}.confidence_head.proj.weight"),
    }

    from .ops import get_freqs_cis

    transformer._dspark = {"stages": stages, "top": top}
    transformer._spec_targets = tuple(args.dspark_target_layer_ids)
    transformer.spec_hidden_decode = torch.zeros(
        64, len(transformer._spec_targets) * args.dim, dtype=torch.bfloat16, device=device)
    transformer.freqs_dspark = get_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0, args.rope_theta, args.rope_factor,
        args.beta_fast, args.beta_slow, device)
    gb = sum(t.numel() * t.element_size() for w in stages for t in w.values()) / (1 << 30)
    print(f"[dspark] loaded {n_stages} stages ({gb:.2f} GiB VRAM), "
          f"block={args.dspark_block_size} targets={transformer._spec_targets}", flush=True)


__all__ = ["load_dspark"]
