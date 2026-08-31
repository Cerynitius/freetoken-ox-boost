"""DSpark speculative decoding for DeepSeek-V4-Flash (block-parallel MTP draft).

The checkpoint ships three DSpark stages under ``mtp.*``: full DSV4 blocks (HC mixing +
MQA-style windowed sink attention + routed MoE) that draft a 5-token block in ONE parallel
forward -- position 0 is the real next token, positions 1..4 are noise tokens the stages
denoise into drafts. A low-rank markov head biases the per-position logits during the
(cheap, sequential) draft sampling, and a confidence head scores each draft position.

Draft-side state per request is tiny and self-contained: each stage keeps a 128-slot ring
of ``main_kv`` latents -- wkv projections of ``main_x = main_norm(main_proj(concat of the
hc-mean hidden at layers dspark_target_layer_ids)))`` -- one per CONSUMED position. The
reference (checkpoint ``inference/model.py``) writes one ring entry per decode step; with
speculation we emit several positions per round, so ``ring_write`` takes a VECTOR of
(position, main_hidden) pairs collected from the verify pass. Nothing here touches the
main model's paged pools.

Verify runs on the main model's prefill/extend path (exact sequential compressor/indexer
semantics; the on-demand MoE prefill path fetches through the slot cache, so a 5-token
extend costs decode-like PCIe). See ``DsparkSpec`` for the round layout and the carry
rollback contract.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.hc_norm import hc_mix_gemv, hc_rms_cast
from freetoken.kernel.triton.dsv4.norm import rms_norm
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.moe.fused_ds_fp4 import routed_experts_fp4


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return rms_norm(x, w, eps)


def _fp8_block64_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Reference ``act_quant(kv[..., :-rd], 64, ..., inplace)``: fp8(e4m3) round-trip with
    per-64 ue8m0 scales. The rings and the block's own kv both store round-tripped values,
    matching what the stages saw in training."""
    shape = x.shape
    v = x.reshape(-1, 64).float()
    amax = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    sc = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (v / sc).to(torch.float8_e4m3fn).float() * sc
    return q.reshape(shape).to(x.dtype)


class _HcMix:
    """Duck-typed carrier for the Block.hc_pre/hc_post math (sinkhorn HC mixing) -- the
    DSpark stages use the exact same kernels/params as main layers, just with mtp weights."""

    def __init__(self, hc_mult: int, dim: int, norm_eps: float, hc_eps: float,
                 sinkhorn_iters: int):
        self.hc_mult = hc_mult
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = sinkhorn_iters

    def pre(self, x, hc_fn, hc_scale, hc_base):
        shape = x.shape  # [1, T, hc_mult, dim]
        dtype = x.dtype
        x2 = x
        M = shape[0] * shape[1]
        xf, rs = hc_rms_cast(x2.reshape(M, -1), self.norm_eps)
        mixes = hc_mix_gemv(xf, hc_fn, rs)
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = hc_pre_combine(x2.reshape(M, self.hc_mult, self.dim), pre, dtype).view(
            *shape[:2], self.dim)
        return y, post.view(M, self.hc_mult), comb.view(M, self.hc_mult, self.hc_mult)

    def post(self, x, residual, post, comb):
        M = x.shape[0] * x.shape[1]
        y = hc_post_combine(
            x.reshape(M, self.dim), residual.reshape(M, self.hc_mult, self.dim), post, comb
        )
        return y.view(residual.shape)

    def head(self, x, fn, scale, base):
        # Transformer.hc_head: sigmoid mix over streams -> merged [M, dim]
        shape = x.shape  # [1, T, hc_mult, dim]
        M = shape[0] * shape[1]
        xf, rs = hc_rms_cast(x.reshape(M, -1), self.norm_eps)
        mixes = hc_mix_gemv(xf, fn, rs)
        pre = torch.sigmoid(mixes * scale + base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.reshape(M, self.hc_mult, self.dim), dim=1)
        return y.view(shape[0], shape[1], self.dim).to(x.dtype)


class DsparkStage:
    """One mtp.* stage: windowed-sink MQA attention + shared/routed MoE, all VRAM-resident.

    Weights are dequantized to BF16 at load (a few hundred MB across the three stages);
    the routed experts stay in ds_fp4 slabs and run through the shared grouped decode
    kernel with ``slots == expert ids`` (identity mapping over the resident slab)."""

    def __init__(self, args, stage_id: int):
        self.stage_id = stage_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window = args.window_size
        self.eps = args.norm_eps
        self.softmax_scale = args.head_dim ** -0.5
        self.topk = args.n_activated_experts
        self.route_scale = args.route_scale
        self.swiglu_limit = args.swiglu_limit
        self.hc = _HcMix(args.hc_mult, args.dim, args.norm_eps, args.hc_eps,
                         args.hc_sinkhorn_iters)
        self.w: dict[str, torch.Tensor] = {}  # loaded by weight.load_dspark

    # ---- attention (reference DSparkAttention, eager BF16) ----------------------------
    def main_kv(self, main_x: torch.Tensor, positions: torch.Tensor,
                freqs_cis: torch.Tensor) -> torch.Tensor:
        """[N, 512] ring entries for consumed positions: wkv -> kv_norm -> rope -> fp8 rt."""
        rd = self.rope_head_dim
        kv = F.linear(main_x, self.w["attn.wkv"])
        kv = _rms(kv, self.w["attn.kv_norm"], self.eps)
        f = freqs_cis.index_select(0, positions)  # [N, rd//2] complex
        kv = kv.clone()
        kv[..., -rd:] = _rope_apply(kv[..., -rd:], f)
        kv[..., :-rd] = _fp8_block64_roundtrip(kv[..., :-rd])
        return kv

    def attend(self, x: torch.Tensor, ring: torch.Tensor, ring_valid: int,
               start_pos: int, freqs_cis: torch.Tensor) -> torch.Tensor:
        """x [1, B, dim] (the draft block, B = block_size), ring [win, 512] with
        ``ring_valid`` filled entries ending at absolute position ``start_pos``. The block
        occupies absolute positions start_pos+1 .. start_pos+B and attends
        [valid ring window | whole block] (non-causal within the block) plus the sink."""
        rd, hd, nh = self.rope_head_dim, self.head_dim, self.n_heads
        B = x.shape[1]
        f = freqs_cis[start_pos + 1: start_pos + 1 + B]

        q = _rms(F.linear(x, self.w["attn.wq_a"]), self.w["attn.q_norm"], self.eps)
        q = F.linear(q, self.w["attn.wq_b"]).unflatten(-1, (nh, hd))
        q = rms_norm(q, None, self.eps)
        q = q.clone()
        q[..., -rd:] = _rope_apply(q[..., -rd:], f.unsqueeze(1))

        kv = _rms(F.linear(x, self.w["attn.wkv"]), self.w["attn.kv_norm"], self.eps)
        kv = kv.clone()
        kv[..., -rd:] = _rope_apply(kv[..., -rd:], f)
        kv[..., :-rd] = _fp8_block64_roundtrip(kv[..., :-rd])

        win_part = ring[:0] if ring_valid <= 0 else _ring_ordered(ring, ring_valid,
                                                                  start_pos, self.window)
        keys = torch.cat([win_part, kv[0]], dim=0)  # [S, 512]; values == keys (MQA latent)
        # logits [B, nh, S] + per-head sink column
        lg = torch.einsum("bhd,sd->bhs", q[0].float(), keys.float()) * self.softmax_scale
        sink = self.w["attn.attn_sink"].float().view(1, nh, 1).expand(B, nh, 1)
        p = torch.softmax(torch.cat([lg, sink], dim=-1), dim=-1)[..., :-1]
        o = torch.einsum("bhs,sd->bhd", p, keys.float()).to(x.dtype)  # [B, nh, 512]
        o = o.clone()
        o[..., -rd:] = _rope_apply(o[..., -rd:], f.unsqueeze(1), inverse=True)

        o = o.view(1, B, self.n_groups, -1)
        wo_a = self.w["attn.wo_a"].view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return F.linear(o, self.w["attn.wo_b"])

    # ---- MoE --------------------------------------------------------------------------
    def moe(self, x: torch.Tensor) -> torch.Tensor:
        """x [1, B, dim]. sqrtsoftplus router + shared expert (BF16) + routed fp4 slab."""
        B = x.shape[1]
        xf = x.reshape(B, self.dim)
        scores = F.linear(xf.float(), self.w["gate.weight"].float())
        scores = F.softplus(scores).sqrt()
        picked = scores + self.w["gate.bias"]
        idx = picked.topk(self.topk, dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / wts.sum(dim=-1, keepdim=True) * self.route_scale

        shared = F.linear(
            fused_swiglu(F.linear(xf, self.w["shared.w1"]), F.linear(xf, self.w["shared.w3"]),
                         self.swiglu_limit, xf.dtype),
            self.w["shared.w2"])
        routed = routed_experts_fp4(
            xf, idx.to(torch.int32), wts.float(),
            self.w["exp.gu_packed"], self.w["exp.gu_scale"],
            self.w["exp.down_packed"], self.w["exp.down_scale"],
            self.swiglu_limit,
        )
        return (shared + routed).view(1, B, self.dim)

    # ---- full stage -------------------------------------------------------------------
    def forward(self, h: torch.Tensor, ring: torch.Tensor, ring_valid: int,
                start_pos: int, freqs_cis: torch.Tensor) -> torch.Tensor:
        """h [1, B, hc_mult, dim] -> same shape."""
        residual = h
        x, post, comb = self.hc.pre(h, self.w["hc_attn_fn"], self.w["hc_attn_scale"],
                                    self.w["hc_attn_base"])
        x = _rms(x, self.w["attn_norm"], self.eps)
        x = self.attend(x, ring, ring_valid, start_pos, freqs_cis)
        h = self.hc.post(x, residual, post, comb)

        residual = h
        x, post, comb = self.hc.pre(h, self.w["hc_ffn_fn"], self.w["hc_ffn_scale"],
                                    self.w["hc_ffn_base"])
        x = _rms(x, self.w["ffn_norm"], self.eps)
        x = self.moe(x)
        return self.hc.post(x, residual, post, comb)


def _rope_apply(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Reference ``apply_rotary_emb`` on the trailing rope dims. ``freqs`` complex,
    broadcastable to x's leading dims."""
    xs = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    f = freqs.conj() if inverse else freqs
    y = torch.view_as_real(xs * f).flatten(-2)
    return y.to(x.dtype)


def _ring_ordered(ring: torch.Tensor, valid: int, last_pos: int, win: int) -> torch.Tensor:
    """The reference block layout keeps ring slot = position % win, and the window
    candidate ORDER in get_dspark_topk_idxs is plain slot order 0..win-1 (positions are
    interchangeable under attention -- order only matters for reproducibility). Restrict
    to slots whose position is within the valid trailing range."""
    if valid >= win:
        return ring
    # slots holding positions (last_pos-valid, last_pos]: slot = p % win
    p = torch.arange(last_pos - valid + 1, last_pos + 1, device=ring.device)
    return ring.index_select(0, p % win)


class DsparkDraft:
    """The three-stage draft head + per-request ring state (single slot, keyed by uid)."""

    def __init__(self, args, model, device: torch.device):
        self.args = args
        self.model = model  # DeepseekV4 Transformer (embed / head / freqs access)
        self.device = device
        self.block = args.dspark_block_size            # 5
        self.noise_id = args.dspark_noise_token_id
        self.n_targets = len(args.dspark_target_layer_ids)
        self.stages = [DsparkStage(args, i) for i in range(args.n_mtp_layers)]
        packed = getattr(model, "_dspark", None)       # set by weight.load_dspark
        assert packed is not None, "DSpark weights not loaded (FREETOKEN_DSV4_SPEC=1?)"
        self.w: dict[str, torch.Tensor] = packed["top"]
        for st, sw in zip(self.stages, packed["stages"]):
            st.w = sw
        self.hc = self.stages[0].hc
        win = args.window_size
        # per-stage main_kv rings (single request slot; multi-slot is a v2 concern)
        self.rings = torch.zeros(args.n_mtp_layers, win, args.head_dim,
                                 dtype=torch.bfloat16, device=device)
        self.ring_uid: int | None = None
        self.ring_last = -1     # absolute position of the newest ring entry
        self.ring_valid = 0

    # ---- ring maintenance -------------------------------------------------------------
    def ring_reset(self, uid: int) -> None:
        self.ring_uid = uid
        self.ring_last = -1
        self.ring_valid = 0

    def ring_write(self, positions: torch.Tensor, main_hidden: torch.Tensor) -> None:
        """positions [N] int64 ascending, main_hidden [N, n_targets*dim] (hc means at the
        target layers). Projects to main_x once, then per-stage wkv into the rings."""
        n = positions.numel()
        if n == 0:
            return
        main_x = _rms(F.linear(main_hidden, self.w["main_proj"]), self.w["main_norm"],
                      self.args.norm_eps)
        freqs = self._freqs()
        slots = (positions % self.args.window_size).long()
        for si, st in enumerate(self.stages):
            kv = st.main_kv(main_x, positions, freqs)
            self.rings[si].index_copy_(0, slots, kv)
        last = int(positions[-1].item())
        if self.ring_last >= 0 and int(positions[0].item()) == self.ring_last + 1:
            self.ring_valid = min(self.ring_valid + n, self.args.window_size)
        else:  # gap (fresh prompt / radix continuation): only the new run is trustworthy
            self.ring_valid = min(n, self.args.window_size)
        self.ring_last = last

    def _freqs(self) -> torch.Tensor:
        # non-compressed rope table (DSpark attention runs the plain rope_theta variant,
        # matching reference DSparkAttention.freqs_cis == Attention's at compress_ratio 0)
        return self.model.freqs_dspark

    # ---- draft ------------------------------------------------------------------------
    @torch.inference_mode()
    def draft(self, next_token: torch.Tensor):
        """next_token [1] (the just-emitted token, consumed at position ring_last+1).
        Returns (draft_tokens [block] int64, confidence [block] fp32).

        Note the ring holds positions <= ring_last; the block occupies
        ring_last+1 .. ring_last+block. The reference writes main_kv@start_pos before
        attending; here the consumed position's main_kv was already ring_written by the
        caller (verify collects hiddens for every emitted position, next_token included
        via its consumed predecessor)."""
        B = self.block
        start = self.ring_last  # reference start_pos: newest ring position
        ids = torch.full((1, B), self.noise_id, dtype=torch.long, device=self.device)
        ids[0, 0] = next_token
        emb_w = self.model.embed.weight
        h = emb_w[ids].unsqueeze(2).repeat(1, 1, self.hc.hc_mult, 1)

        freqs = self._freqs()
        for si, st in enumerate(self.stages):
            h = st.forward(h, self.rings[si], self.ring_valid, start, freqs)

        x = self.hc.head(h, self.w["hc_head_fn"], self.w["hc_head_scale"],
                         self.w["hc_head_base"])          # [1, B, dim] (pre-norm: confidence input)
        xn = _rms(x, self.w["head_norm"], self.args.norm_eps)
        logits = self._lm_head(xn[0])                      # [B, V] fp32/bf16

        out = torch.empty(B + 1, dtype=torch.long, device=self.device)
        out[0] = next_token
        m_embeds = []
        logits = logits.float()
        for i in range(B):
            me = self.w["markov_w1"][out[i]]               # [rank]
            bias = F.linear(me, self.w["markov_w2"])       # [V]
            logits[i] += bias.float()
            m_embeds.append(me)
            out[i + 1] = logits[i].argmax()
        m_emb = torch.stack(m_embeds, dim=0)               # [B, rank]
        conf_in = torch.cat([x[0].float(), m_emb.float()], dim=-1)
        confidence = F.linear(conf_in, self.w["confidence"]).squeeze(-1)  # [B]
        return out[1:], confidence

    def _lm_head(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        if getattr(m, "_head_fp8", False):
            from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear
            return fp8_pertensor_linear(x.contiguous(), m.head, m.head_scale)
        return F.linear(x, m.head)


__all__ = ["DsparkDraft", "DsparkStage"]
