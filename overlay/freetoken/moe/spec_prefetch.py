"""Speculative expert prefetch (Mixtral-offloading style, measured for GLM-5.3).

Layer L predicts layer L+HOP's routing by applying that layer's REAL router (gate +
noaux_tc bias + group limit, referenced from its sparse block) to L's router-input
hidden, ensures the top-``count`` predicted experts (maps on the MAIN stream,
dedicated double-buffered plan), and runs the H2D copy on a SIDE stream that waits
for L's expert GEMM (a prefetch eviction may target a slot that GEMM still reads).
Layer L+HOP's GEMM waits the copy event before touching the slots.

HOP=2 default: measured hop-2 top-16 coverage 79% (hop-1 87%), but the copy gets
TWO layers of main-stream work to hide behind, taking it off the critical path
(hop-1's copy sat between consecutive GEMMs and net -4%).

Double-buffer proof: side copies run in source order; the plan for source S lives
in buffer[S%2]. The next writer of buffer[S%2] is source S+2, whose ensure is
ordered (main stream) after GEMM(S+2), which waits ev_copy[S] -- so the side stream
has consumed S's plan first. CUDA-graph safe: events/streams created here (outside
capture), re-recorded inside; all per-step work is fixed-shape device ops.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# <0 = follow the bs==1 count. Sweep 2026-08-29 (conc2/conc4 agg): P_multi 0 -> 38.1/44.5,
# 4 -> 39.8/49.2, 8 -> 37.3/41.5 -- uniform P=4 wins at every batch size.
_COUNT_MULTI = int(os.environ.get("FREETOKEN_MOE_SPEC_PREFETCH_MULTI", "-1") or -1)
HOP = int(os.environ.get("FREETOKEN_MOE_SPEC_HOP", "1") or 1)
_HOP_MULTI = int(os.environ.get("FREETOKEN_MOE_SPEC_HOP_MULTI", "1") or 1)  # hop-2 at bs>1: conc2 39.7 vs 39.8, conc4 46 vs 49 -- no win (2026-08-29)


class SpecPrefetch:
    def __init__(self, cache, count: int):
        self.cache = cache
        self.count = count
        dev = cache.device
        nb = cache.num_layers
        self.blocks: dict = {}
        self.stream = torch.cuda.Stream(device=dev)
        self.ev_gemm = [torch.cuda.Event() for _ in range(nb)]
        self.ev_copy = [torch.cuda.Event() for _ in range(nb)]
        # Buffers must fit the LARGEST per-launch id set: max_bs (8) tokens x the
        # larger of the bs==1 count and the multi-stream count (_COUNT_MULTI).
        cmax = max(count, _COUNT_MULTI, 0) or count
        cmax = max(count, cmax)
        plan = max(cache.num_experts, 8 * cmax)
        self.pf_rows = [torch.empty(plan, dtype=torch.int32, device=dev) for _ in range(2)]
        self.pf_slots = [torch.empty(plan, dtype=torch.int32, device=dev) for _ in range(2)]
        self.pf_num = [torch.zeros(1, dtype=torch.int64, device=dev) for _ in range(2)]
        self.ids_buf = [torch.empty(8 * cmax, dtype=torch.int32, device=dev) for _ in range(2)]
        self.pending: dict = {}  # target bank -> copy event

    def before_gemm(self, bank_id: int) -> None:
        ev = self.pending.pop(bank_id, None)
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)

    def launch(self, bank_id: int, hidden: torch.Tensor) -> None:
        # bs>1: prefetch bytes scale with batch on an already-saturated link, so use
        # the (smaller) multi-stream budget; 0 disables. Static per captured graph --
        # each cuda-graph batch size bakes its own branch (bs==1 keeps the tuned P).
        count = self.count if hidden.shape[0] == 1 or _COUNT_MULTI < 0 else _COUNT_MULTI
        if count <= 0:
            return
        # bs>1: fetch bytes double but the single-layer hiding window does not
        # (conc2 profile 2026-08-29: PCIe 42.5ms/step, compute lane 56% busy).
        # hop-2 trades ~8pp coverage for a two-layer window -- the binding
        # constraint flips at bs>=2. Static per captured graph.
        hop = HOP if hidden.shape[0] == 1 else _HOP_MULTI
        target = bank_id + hop
        blk = self.blocks.get(target)
        if blk is None:
            return
        from flashlib.kernels.slot_cache import lru_ensure

        c = self.cache
        b = bank_id % 2
        logits = F.linear(hidden.float(), blk.gate.weight.float())
        scores = torch.sigmoid(logits) + blk.e_score_correction_bias.float()
        if blk.n_group > 1:
            scores = blk._group_limited(scores)
        ids = torch.topk(scores, count, dim=-1)[1].to(torch.int32).reshape(-1)
        buf = self.ids_buf[b][: ids.numel()]
        buf.copy_(ids)
        lru_ensure(
            buf, c.slot_for_id.view(-1), c.id_of_slot, c.usage, c.step,
            buf, self.pf_rows[b], self.pf_slots[b], self.pf_num[b],
            id_base=target * c.num_experts,
        )
        # No wait on this layer's GEMM: eviction picks argmin-usage (the OLDEST
        # slots), while every slot an in-flight GEMM reads was touched by an ensure
        # within the current step -- with ~2k slots and 2-4 evictions/call, a
        # just-touched slot cannot become argmin inside one step. Starting the copy
        # immediately gives it the whole rest of this layer + the next layer's
        # attention to hide in; the target GEMM's ev_copy wait covers readiness.
        ev_g = self.ev_gemm[bank_id]
        ev_g.record()  # order the side copy after the ensure's map/plan writes
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ev_g)
            self._copy(target, b)
        ev_c = self.ev_copy[bank_id]
        ev_c.record(self.stream)
        self.pending[target] = ev_c

    def _copy(self, bank_id: int, b: int) -> None:
        c = self.cache
        if c._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            fast_index_copy_multi_jit(
                c._copy_dst_ptrs, c._copy_src_ptrs[bank_id], c._copy_feat_bytes,
                self.pf_slots[b], self.pf_rows[b], self.pf_num[b],
            )
        else:
            from freetoken.kernel import fast_index_copy_jit

            for per_layer, slab in c.banks:
                fast_index_copy_jit(slab, self.pf_slots[b], per_layer[bank_id],
                                    self.pf_rows[b], self.pf_num[b])


__all__ = ["SpecPrefetch"]
