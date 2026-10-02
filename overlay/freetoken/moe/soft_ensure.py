"""Score-aware slot-cache admission (soft LRFU) for flat sigmoid routing (Naive-N0.5-Flash).

Replacement policy: every (layer, expert) id carries a combined recency/frequency value
``CRF(t) = sum_i w_i * 2^(-lam * (t - t_i))`` over its touches, and eviction takes the
resident id with the smallest CRF. A touch is not only a routed use: the router's
near-misses (ranks TOPK..NRANK-1 by selection score) are touched with their raw sigmoid
score as weight, selected experts with ``1 + score``. Experts the router almost picked
are the likeliest next picks, so they are kept over cold ones even before their first
use. Measured on a 3.2k-token Naive decode trace (5454 slots): 44.2 misses/token vs 48.8
for LRU (-9.4%); Belady's offline bound is ~19.

Ordering trick: ``log2(CRF(now)) = P - lam*now`` with ``P = log2(CRF(t_last)) + lam*t_last``
fixed between touches, so ranking by the static P equals ranking by the live CRF. P is
kept in int64 fixed point (``SCALE`` units per log2 unit) with ``lam = 2^-10`` per call,
i.e. exactly ``LAMQ = 64`` units per step -- no float drift over long uptimes.

Same contract/buffers as flashlib ``lru_ensure``: rewrites ``query`` to slot ids in
place, fills the copy plan (``src``/``dst``/``num_copy``), single CTA, fixed shapes,
CUDA-graph capturable. Changes only WHICH expert is evicted, never any math.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

SCALE_BITS = 16
LAMQ = 64  # lam = 2^-10 log2-units per call, in 2^-16 units
P_NONE = -(1 << 62)  # "never touched"

# Tuning knobs (2026-10-01; the constants above were tuned on Naive, 47 offload lookups per
# token -- GLM-5.3 base3 does 34). Each default is the constant above, and with all of them
# unset the kernel takes its original expressions (constexpr branches), i.e. unchanged.
# They change only WHICH expert is evicted, never any math.
#   FREETOKEN_SOFT_DECAY_LAMQ   decay per ensure call in 2^-16 log2 units (64 = 2^-10: the
#                               CRF half-life is 1024 calls = 1024/L tokens at L lookups/token)
#   FREETOKEN_SOFT_DECAY_TICKS  clock quantum in ensure calls: the decay clock reads
#                               ((step-1)//T)*T, so with T = lookups per token every layer of
#                               one token touches at the same clock (no intra-token recency
#                               skew); same average decay. 1 = per-call clock (default);
#                               0 = auto (cache.num_layers, the offload layer count)
#   FREETOKEN_SOFT_WSEL         touch bonus of a selected route (default 1.0)
#   FREETOKEN_SOFT_WSCORE       multiplier of the raw router score in every touch (1.0)
_LAMQ = int(os.environ.get("FREETOKEN_SOFT_DECAY_LAMQ", "") or LAMQ)
_TICKS = int(os.environ.get("FREETOKEN_SOFT_DECAY_TICKS", "") or 1)
_WSEL = float(os.environ.get("FREETOKEN_SOFT_WSEL", "") or 1.0)
_WSCORE = float(os.environ.get("FREETOKEN_SOFT_WSCORE", "") or 1.0)
assert _LAMQ >= 0 and _TICKS >= 0, "FREETOKEN_SOFT_DECAY_LAMQ/_TICKS must be >= 0"
assert _WSCORE > 0 and _WSEL >= 0, "FREETOKEN_SOFT_WSCORE must be > 0 (a zero-weight first touch is log2(0))"


@triton.jit(do_not_specialize=["K", "R", "num_cached", "id_base"])
def _soft_ensure_kernel(
    query_ptr, rank_id_ptr, rank_sc_ptr,
    prio_ptr, slot_of_id_ptr, id_of_slot_ptr, usage_ptr, guard_ptr, step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr, hit_mask_ptr, miss_mask_ptr,
    K, R, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    COLLECT_STATS: tl.constexpr, WRITE_MASKS: tl.constexpr,
    LAMQ_C: tl.constexpr, TICKS: tl.constexpr,
    TUNED_W: tl.constexpr, W_SEL: tl.constexpr, W_SCORE: tl.constexpr,
    MERGE_NR: tl.constexpr, KSEL: tl.constexpr,
):
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    if TICKS == 1:
        nowq = step * LAMQ_C  # LAMQ (default 64: the original clock)
    else:
        nowq = ((step - 1) // TICKS) * (TICKS * LAMQ_C)  # quantized clock, same mean decay
    SCALE: tl.constexpr = 65536.0

    # ---- 1. CRF touches for the ranked list (distinct ids by construction)
    r = tl.arange(0, BLOCK_R)
    rmask = r < R
    g = tl.load(rank_id_ptr + r, mask=rmask, other=0) + id_base
    sc = tl.load(rank_sc_ptr + r, mask=rmask, other=0.0)
    if MERGE_NR > 0:
        sel = (r % MERGE_NR) < KSEL  # [M, MERGE_NR] rows: ranks < KSEL of each row
    else:
        sel = r < K
    if TUNED_W:
        w = sc * W_SCORE + tl.where(sel, W_SEL, 0.0)
    else:
        w = sc + tl.where(sel, 1.0, 0.0)
    if MERGE_NR > 0:
        # per-token rows share ids: touch each DISTINCT id once, at its first occurrence,
        # with the summed weight of all its occurrences (fixed shape, deterministic)
        same = (g[:, None] == g[None, :]) & rmask[:, None] & rmask[None, :]
        dup = same & (r[None, :] < r[:, None])
        w = tl.sum(tl.where(same, w[None, :], 0.0), axis=1)
        rmask = rmask & (tl.sum(dup.to(tl.int32), axis=1) == 0)
    p_old = tl.load(prio_ptr + g, mask=rmask, other=-(1 << 62))
    never = p_old <= -(1 << 61)
    d = (p_old - nowq).to(tl.float32) / SCALE
    crf = tl.where(never, 0.0, tl.exp2(tl.minimum(d, 60.0))) + w
    p_new = nowq + (tl.log2(crf) * SCALE).to(tl.int64)
    tl.store(prio_ptr + g, p_new, mask=rmask)
    s_r = tl.load(slot_of_id_ptr + g, mask=rmask, other=-1)
    tl.store(usage_ptr + s_r, p_new, mask=rmask & (s_r >= 0))

    # ---- 2. phase 1 over the query (mirrors lru_ensure): dedup, hit/miss, miss ranks
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & kmask[:, None] & kmask[None, :]
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    tl.store(guard_ptr + s, step, mask=hit)  # this call's hits are never victims
    out = tl.where(hit, s, -1)
    if WRITE_MASKS:  # per-route residency for the hit/miss split decode (int8 0/1)
        tl.store(hit_mask_ptr + k, hit.to(tl.int8), mask=kmask)
        tl.store(miss_mask_ptr + k, miss.to(tl.int8), mask=kmask)

    if num_missing > 0:
        tl.debug_barrier()  # usage/guard scatters above must be visible to the reload
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(usage_ptr + c, mask=cmask, other=0)
        gd = tl.load(guard_ptr + c, mask=cmask, other=0)
        kmax = tl.full([BLOCK_C], 0x7FFFFFFFFFFFFFFF, tl.int64)
        key = tl.where((gd == step) | (~cmask), kmax, u)
        for i in tl.range(num_missing):
            victim = tl.argmin(key, axis=0).to(tl.int32)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(usage_ptr + victim, tl.load(prio_ptr + e))
            tl.store(guard_ptr + victim, step)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            out = tl.where((rank == i) & miss, victim, out)
            key = tl.where(c == victim, kmax, key)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        si = tl.arange(0, 4)
        v = tl.where(si == 0, tl.sum(first.to(tl.int32)), tl.where(si == 1, num_missing, 1))
        tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=si < 3)


def soft_ensure(cache, layer_id: int, expert_ids: torch.Tensor, rank_ids=None, rank_sc=None,
                stats=None, masks=None) -> None:
    """Drop-in for the ``lru_ensure`` call in ``offload_kernels.ensure_experts``."""
    if getattr(cache, "soft_prio", None) is None:
        n_ids = cache.num_layers * cache.num_experts
        cache.soft_prio = torch.full((n_ids,), P_NONE, dtype=torch.int64, device=cache.device)
        cache.soft_guard = torch.zeros((cache.cache_size,), dtype=torch.int64, device=cache.device)
    q = expert_ids.reshape(-1)
    merge_nr = ksel = 0
    if rank_ids is None:
        rank_ids = q
        rank_sc = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    elif rank_ids.dim() == 2 and rank_ids.shape[0] > 1:
        # [M, NR] per-token ranked rows (GLM FREETOKEN_GLM5_SOFT_RANK_BS2; nothing else
        # emits M > 1): merged to distinct ids in the kernel, ranks < top_k of each row
        # are that token's selected experts
        merge_nr = rank_ids.shape[1]
        ksel = q.numel() // rank_ids.shape[0]
        assert ksel * rank_ids.shape[0] == q.numel() and ksel <= merge_nr
    rank_ids = rank_ids.reshape(-1)
    rank_sc = rank_sc.reshape(-1)
    K = q.numel()
    R = rank_ids.numel()
    assert R >= K
    num_cached = cache.id_of_slot.numel()
    block_c = triton.next_power_of_2(num_cached)
    ticks = _TICKS if _TICKS > 0 else max(1, int(cache.num_layers))
    _soft_ensure_kernel[(1,)](
        q, rank_ids, rank_sc,
        cache.soft_prio, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage,
        cache.soft_guard, cache.step,
        q, cache.src_indices, cache.evict_slots, cache.num_indices,
        stats if stats is not None else cache.num_indices,
        masks[0] if masks is not None else cache.num_indices,
        masks[1] if masks is not None else cache.num_indices,
        K, R, num_cached, layer_id * cache.num_experts,
        BLOCK_K=triton.next_power_of_2(K), BLOCK_R=triton.next_power_of_2(R), BLOCK_C=block_c,
        COLLECT_STATS=stats is not None,
        WRITE_MASKS=masks is not None,
        LAMQ_C=_LAMQ, TICKS=ticks,
        TUNED_W=(_WSEL != 1.0 or _WSCORE != 1.0), W_SEL=_WSEL, W_SCORE=_WSCORE,
        MERGE_NR=merge_nr, KSEL=ksel,
        num_warps=8 if block_c >= 2048 else 4,
    )


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _soft_prefetch_kernel(
    query_ptr, prio_ptr, slot_of_id_ptr, id_of_slot_ptr, usage_ptr, guard_ptr, step_ptr,
    src_ptr, dst_ptr, num_copy_ptr,
    K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Map the MISSING ids of a predicted (distinct) query into victim slots and emit the
    copy plan. Unlike an ensure it touches nothing: resident predictions keep their
    priority, the clock does not advance, and slots guarded at the current step (the
    routes the in-flight layer is computing) are never victims. A new slot takes the
    expert's own CRF priority, so a wrong guess (never or rarely used) is the next victim."""
    step = tl.load(step_ptr)
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=0) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=0)
    miss = kmask & (s == -1)
    rank = tl.cumsum(miss.to(tl.int32), axis=0) - 1
    num_missing = tl.sum(miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    if num_missing > 0:
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(usage_ptr + c, mask=cmask, other=0)
        gd = tl.load(guard_ptr + c, mask=cmask, other=0)
        kmax = tl.full([BLOCK_C], 0x7FFFFFFFFFFFFFFF, tl.int64)
        key = tl.where((gd == step) | (~cmask), kmax, u)
        for i in tl.range(num_missing):
            victim = tl.argmin(key, axis=0).to(tl.int32)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(usage_ptr + victim, tl.load(prio_ptr + e))
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            key = tl.where(c == victim, kmax, key)


def soft_prefetch(cache, layer_id: int, ids: torch.Tensor, src: torch.Tensor,
                  dst: torch.Tensor, num: torch.Tensor) -> None:
    """Prefetch counterpart of :func:`soft_ensure` (see the kernel). Needs the soft state,
    which the target's preceding soft_ensure calls have already created."""
    q = ids.reshape(-1)
    num_cached = cache.id_of_slot.numel()
    _soft_prefetch_kernel[(1,)](
        q, cache.soft_prio, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage,
        cache.soft_guard, cache.step, src, dst, num,
        q.numel(), num_cached, layer_id * cache.num_experts,
        BLOCK_K=triton.next_power_of_2(q.numel()), BLOCK_C=triton.next_power_of_2(num_cached),
        num_warps=8,
    )


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _soft_prefetch_guard_kernel(
    query_ptr, prio_ptr, slot_of_id_ptr, id_of_slot_ptr, usage_ptr, guard_ptr, step_ptr,
    src_ptr, dst_ptr, num_copy_ptr,
    K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """``_soft_prefetch_kernel`` for the post-GEMM prefetch (``SpecPrefetch.launch``), whose
    copy runs on its OWN stream and is only waited by the target GEMM -- i.e. AFTER the
    target layer's ensure + demand copy. Two additions:
      * every slot it fills gets ``guard = step + 1``: the next soft_ensure (the target's,
        hop 1) runs at exactly that step and excludes it from its victims, so it cannot
        re-map a slot whose prefetch copy may still be in flight. One call later the main
        stream has waited the copy event, so the guard can lapse.
      * duplicate ids (bs>1: per-token top-P unions) are mapped once (first occurrence)."""
    step = tl.load(step_ptr)
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=0) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=0)
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & kmask[:, None] & kmask[None, :]
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    miss = first & (s == -1)
    rank = tl.cumsum(miss.to(tl.int32), axis=0) - 1
    num_missing = tl.sum(miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    if num_missing > 0:
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(usage_ptr + c, mask=cmask, other=0)
        gd = tl.load(guard_ptr + c, mask=cmask, other=0)
        kmax = tl.full([BLOCK_C], 0x7FFFFFFFFFFFFFFF, tl.int64)
        # exact matches only (cache.reset() rewinds step but keeps soft_guard): the current
        # layer's routes (== step) and slots already guarded for the next ensure (== step+1)
        key = tl.where((gd == step) | (gd == step + 1) | (~cmask), kmax, u)
        for i in tl.range(num_missing):
            victim = tl.argmin(key, axis=0).to(tl.int32)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(usage_ptr + victim, tl.load(prio_ptr + e))
            tl.store(guard_ptr + victim, step + 1)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            key = tl.where(c == victim, kmax, key)


def soft_prefetch_guarded(cache, layer_id: int, ids: torch.Tensor, src: torch.Tensor,
                          dst: torch.Tensor, num: torch.Tensor) -> None:
    """:func:`soft_prefetch` for a copy on a separate stream (see the kernel). Needs the soft
    state (created by the first soft_ensure). Fixed shapes, no host sync: graph-capturable."""
    q = ids.reshape(-1)
    num_cached = cache.id_of_slot.numel()
    _soft_prefetch_guard_kernel[(1,)](
        q, cache.soft_prio, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage,
        cache.soft_guard, cache.step, src, dst, num,
        q.numel(), num_cached, layer_id * cache.num_experts,
        BLOCK_K=triton.next_power_of_2(q.numel()), BLOCK_C=triton.next_power_of_2(num_cached),
        num_warps=8,
    )


__all__ = ["soft_ensure", "soft_prefetch", "soft_prefetch_guarded"]
