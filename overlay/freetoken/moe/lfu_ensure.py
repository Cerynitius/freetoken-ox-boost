"""Frequency-aware slot-cache admission for FLAT expert routing (GLM-5.3, Hn=0.87).

Same contract as flashlib's ``lru_ensure`` (sequential strategy) plus one per-slot
``freq`` table: hits bump a saturating counter, eviction ranks by ``(freq, usage,
slot)`` -- one-touch experts leave before repeat visitors (TinyLFU-style doorkeeper),
which is exactly the failure mode of pure LRU under a flat routing tail. A periodic
halving sweep (every DECAY_PERIOD calls) ages hotness so the cache cannot ossify.

Bounds honesty: measured LRU 60% vs Belady-oracle 69% at similar capacity -- every
online policy lives inside that 9pp. Device-side, fixed shapes, CUDA-graph capturable,
single CTA like the original (num_cached ~3k -> register-resident scan).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FREQ_CAP = 7          # 3-bit saturating counter
DECAY_PERIOD = 4096   # kernel calls (~105 decode steps at 39 offload layers)
_USAGE_BITS = 48      # composite key: freq << 48 | (usage & (2^48-1))


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _lfu_ensure_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    freq_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr, K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
    FCAP: tl.constexpr, DECAY: tl.constexpr, UBITS: tl.constexpr,
):
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    c = tl.arange(0, BLOCK_C)
    cmask = c < num_cached
    if step % DECAY == 0:  # aging sweep: halve every slot's hotness
        fdec = tl.load(freq_ptr + c, mask=cmask, other=0)
        tl.store(freq_ptr + c, fdec >> 1, mask=cmask)
        tl.debug_barrier()

    # ---- phase 1 (mirrors flashlib _phase1, plus the hit-side freq bump) ----
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (
        (q[:, None] == q[None, :])
        & (k[:, None] > k[None, :])
        & kmask[:, None]
        & kmask[None, :]
    )
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    tl.store(lru_usage_ptr + s, step, mask=hit)
    fh = tl.load(freq_ptr + s, mask=hit & first, other=0)
    tl.store(freq_ptr + s, tl.minimum(fh + 1, FCAP), mask=hit & first)
    out = tl.where(hit, s, -1)

    if num_missing > 0:
        tl.debug_barrier()  # same fence the original needs: hit bumps before reload
        u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
        f = tl.load(freq_ptr + c, mask=cmask, other=FCAP)
        umask48 = (tl.full([BLOCK_C], 1, tl.int64) << UBITS) - 1
        key = (f.to(tl.int64) << UBITS) | (u.to(tl.int64) & umask48)
        kmax = tl.full([BLOCK_C], 0x7FFFFFFFFFFFFFFF, tl.int64)
        key = tl.where((u == step) | (~cmask), kmax, key)
        for i in tl.range(num_missing):
            victim = tl.argmin(key, axis=0).to(tl.int32)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(lru_usage_ptr + victim, step)
            tl.store(freq_ptr + victim, 0)  # fresh install: doorkeeper starts cold
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            out = tl.where((rank == i) & miss, victim, out)
            key = tl.where(c == victim, kmax, key)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        si = tl.arange(0, 4)
        v = tl.where(si == 0, tl.sum(first.to(tl.int32)),
                     tl.where(si == 1, num_missing, 1))
        tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=si < 3)


def lfu_ensure(
    query, slot_of_id, id_of_slot, lru_usage, lru_step, freq,
    out_indices, src_indices, dst_indices, num_copy,
    stats=None, id_base: int = 0,
) -> None:
    """Drop-in for flashlib ``lru_ensure`` (seq strategy) with the freq table added."""
    k = query.numel()
    num_cached = id_of_slot.numel()
    assert freq.numel() == num_cached and freq.dtype == torch.int32
    block_c = triton.next_power_of_2(num_cached)
    _lfu_ensure_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step, freq,
        out_indices, src_indices, dst_indices, num_copy, stats, k, num_cached, id_base,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        FCAP=FREQ_CAP, DECAY=DECAY_PERIOD, UBITS=_USAGE_BITS,
        num_warps=8 if block_c >= 2048 else 4,
    )


__all__ = ["lfu_ensure", "FREQ_CAP", "DECAY_PERIOD"]
