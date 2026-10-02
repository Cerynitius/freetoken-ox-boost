"""Elastic KV <-> expert-cache budget (Naive single-stream offload serving).

One physical VRAM budget is shared by the offload expert slot cache and the DSA KV pool.
Both live in reserved virtual ranges (kernel/vmm.py) whose tensors never move, so CUDA
graphs and kernels are untouched; only the physical backing changes:

* KV is organised in STEPS of ``step`` tokens (+ a permanently mapped ``tail`` that also
  holds the engine's dummy token). Step 0 and the tail are mapped at startup; the page
  budget solver reserves only that much, the expert cache gets everything else.
* grow(): retire expert slots from the TOP of the slot range (evict their experts, pin
  their usage at INT64_MAX so no ensure ever picks them, unmap their bank bytes) until one
  KV step fits, then map that step in every DSA layer. Called by the scheduler's page
  allocator only when LIVE requests need more tokens than free + evictable pages.
* shrink(): when the server is idle, compact the prefix-cache KV that lives in the top
  step down into free low pages (radix node values + full->swa mapping are rewritten,
  the DSA K/V/index rows copied), unmap the step, and hand the bytes back to the expert
  cache. The conversation's prefix cache survives the shrink.

FREETOKEN_ELASTIC_KV=1 enables it; FREETOKEN_ELASTIC_MAX (default 1048576) caps the
logical context, FREETOKEN_ELASTIC_STEP (163840) / _TAIL (16384) set the geometry,
FREETOKEN_ELASTIC_IDLE_S (30) the idle time before a shrink.
"""

from __future__ import annotations

import os
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

EXPERT_CHUNK = 16  # expert slots per VMM mapping (every big bank row x16 is 2 MiB aligned)


def enabled() -> bool:
    return os.environ.get("FREETOKEN_ELASTIC_KV", "0") == "1"


def geometry() -> tuple[int, int, int]:
    """(step, tail, steps_max) -- tokens per step, permanent tail tokens, max steps."""
    step = int(os.environ.get("FREETOKEN_ELASTIC_STEP", "163840"))
    tail = int(os.environ.get("FREETOKEN_ELASTIC_TAIL", "16384"))
    mx = int(os.environ.get("FREETOKEN_ELASTIC_MAX", "1048576"))
    assert step % 16384 == 0 and tail % 16384 == 0, "step/tail must be multiples of 16384"
    return step, tail, -(-mx // step)


def logical_tokens() -> int:
    step, tail, steps_max = geometry()
    return steps_max * step + tail


def initial_tokens() -> int:
    step, tail, _ = geometry()
    return step + tail


class LayerSlabs:
    """Layer-major ``[L, T, ...]`` KV slabs in reserved VA (kernel/vmm.py), mapped per
    elastic step: step 0 + the tail at allocation, further steps on grow(). Shared by the
    KV pools that support elastic mode (the MLA/DSA pool; Naive has its own copy)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.items: list = []  # (region, L, T, row_bytes, tensor, zero_on_map)

    def alloc(self, shape, dtype, zero: bool = False) -> torch.Tensor:
        from freetoken.kernel.vmm import VmmRegion

        step, tail, steps_max = geometry()
        L, T = shape[0], shape[1]
        rb = torch.empty((), dtype=dtype).element_size()
        for d in shape[2:]:
            rb *= d
        reg = VmmRegion(L * T * rb, self.device)
        assert T == steps_max * step + tail, (T, steps_max, step, tail)
        assert (T * rb) % reg.gran == 0 and (step * rb) % reg.gran == 0 and (tail * rb) % reg.gran == 0
        for l in range(L):
            reg.map(l * T * rb, step * rb)
            reg.map(l * T * rb + steps_max * step * rb, tail * rb)
        t = reg.bytes_tensor()[: L * T * rb].view(dtype).view(shape)
        if zero:
            t[:, :step].view(torch.uint8).zero_()
            t[:, steps_max * step:].view(torch.uint8).zero_()
        self.items.append((reg, L, T, rb, t, zero))
        return t

    def step_bytes(self) -> int:
        step = geometry()[0]
        return sum(L * step * rb for _, L, _, rb, _, _ in self.items)

    def map_step(self, s: int) -> None:
        step = geometry()[0]
        for reg, L, T, rb, t, zero in self.items:
            for l in range(L):
                reg.map(l * T * rb + s * step * rb, step * rb)
            if zero:
                t[:, s * step:(s + 1) * step].view(torch.uint8).zero_()

    def unmap_step(self, s: int) -> None:
        step = geometry()[0]
        for reg, L, T, rb, _, _ in self.items:
            for l in range(L):
                reg.unmap(l * T * rb + s * step * rb)

    def move(self, old: torch.Tensor, new: torch.Tensor, chunk: int = 32768) -> None:
        for _, _, _, _, t, _ in self.items:
            u = t.view(t.shape[0], t.shape[1], -1).view(torch.uint8) if t.element_size() == 1 else t
            for a in range(0, old.numel(), chunk):
                o, n = old[a:a + chunk], new[a:a + chunk]
                u[:, n] = u[:, o]


class ElasticKV:
    def __init__(self, engine) -> None:
        self.cache = engine.moe_offload_cache
        self.kv = engine.kv_cache
        self.step, self.tail, self.steps_max = geometry()
        self.steps = 1
        self.retired: list[int] = []  # expert chunks retired per grown step (stack)
        self.min_slots = 2 * self.cache.num_experts + 512
        self.idle_s = float(os.environ.get("FREETOKEN_ELASTIC_IDLE_S", "30"))
        # grown capacity no LIVE request needed for this long may evict prefix cache to shrink
        self.evict_after_s = float(os.environ.get("FREETOKEN_ELASTIC_EVICT_AFTER_S", "600"))
        self.top_needed_ts = time.monotonic()
        self.last_busy = time.monotonic()
        logger.info_rank0(
            f"elastic KV: step {self.step} tok, tail {self.tail}, max {self.steps_max} steps "
            f"({self.steps_max * self.step + self.tail} tok logical); expert slots "
            f"{self.cache.elastic_active_slots()} active"
        )

    # ------------------------------------------------------------------ geometry
    def mapped_tokens(self) -> int:
        return self.steps * self.step + self.tail

    def growable_tokens(self) -> int:
        return (self.steps_max - self.steps) * self.step

    def step_range(self, s: int) -> torch.Tensor:
        return torch.arange(s * self.step, (s + 1) * self.step, dtype=torch.int32,
                            device=self.cache.device)

    def initial_free(self, num_pages: int) -> torch.Tensor:
        """Allocatable token slots at startup: step 0 + the tail minus the dummy slot."""
        dev = self.cache.device
        lo = torch.arange(0, self.step, dtype=torch.int32, device=dev)
        hi = torch.arange(self.steps_max * self.step, num_pages, dtype=torch.int32, device=dev)
        return torch.cat([lo, hi])

    # ------------------------------------------------------------------ grow / shrink
    def grow(self) -> torch.Tensor | None:
        """Map one more KV step (retiring expert slots); returns its token slots or None."""
        if self.steps >= self.steps_max:
            return None
        need = self.kv.elastic_step_bytes()
        torch.cuda.synchronize(self.cache.device)
        freed, k = self.cache.elastic_retire_until(need, self.min_slots)
        if freed < need:
            self.cache.elastic_restore(k)
            logger.warning(f"elastic KV: cannot grow (only {freed >> 20} MiB retirable)")
            return None
        s = self.steps
        self.kv.elastic_map_step(s)
        self.steps += 1
        self.retired.append(k)
        self.top_needed_ts = time.monotonic()
        logger.info_rank0(
            f"elastic KV grow -> {self.mapped_tokens()} tok mapped; expert slots "
            f"{self.cache.elastic_active_slots()} active (-{k * EXPERT_CHUNK})"
        )
        return self.step_range(s)

    def shrink(self, cache_manager, allow_evict: bool = False) -> bool:
        """Idle-only: compact the top step's live prefix KV below it, unmap, return slots.
        ``allow_evict``: if the cached KV does not fit below the top step, evict prefix
        cache (LRU) until it does."""
        if self.steps <= 1:
            return False
        s = self.steps - 1
        if not self._compact(cache_manager, s):
            if not allow_evict:
                return False
            short = self._top_live(cache_manager, s) - self._low_free(cache_manager, s)
            cache_manager.evict_tokens(short)
            if not self._compact(cache_manager, s):
                return False
        torch.cuda.synchronize(self.cache.device)
        self.kv.elastic_unmap_step(s)
        self.steps -= 1
        k = self.retired.pop()
        self.cache.elastic_restore(k)
        logger.info_rank0(
            f"elastic KV shrink -> {self.mapped_tokens()} tok mapped; expert slots "
            f"{self.cache.elastic_active_slots()} active (+{k * EXPERT_CHUNK})"
        )
        return True

    def _low_free(self, cm, s: int) -> int:
        lo, hi = s * self.step, (s + 1) * self.step
        f = cm.free_slots
        return int(((f < lo) | (f >= hi)).sum())

    def _top_live(self, cm, s: int) -> int:
        """Tree-held tokens inside step s (idle: every allocated slot is tree-held)."""
        lo, hi = s * self.step, (s + 1) * self.step
        f = cm.free_slots
        return self.step - int(((f >= lo) & (f < hi)).sum())

    def _compact(self, cm, s: int) -> bool:
        """Move every tree-held token slot of step ``s`` into a free slot below it."""
        lo, hi = s * self.step, (s + 1) * self.step
        free = cm.free_slots
        in_top = (free >= lo) & (free < hi)
        low_free = free[~in_top]
        nodes, olds = [], []
        stack = [cm.prefix_cache.root]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            v = getattr(n, "_value", None)
            if v is None or v.numel() == 0:
                continue
            m = (v >= lo) & (v < hi)
            if bool(m.any()):
                nodes.append((n, m))
                olds.append(v[m])
        moved = int(sum(o.numel() for o in olds))
        if moved > low_free.numel():
            return False
        if moved:
            old = torch.cat(olds).to(torch.int64)
            new = torch.sort(low_free).values[:moved].to(torch.int64)
            self.kv.elastic_move_tokens(old, new)
            off = 0
            for (n, m), o in zip(nodes, olds):
                v = n._value.clone()
                v[m] = new[off: off + o.numel()].to(v.dtype)
                n._value = v
                off += o.numel()
            taken = torch.zeros(self.kv.elastic_total_tokens(), dtype=torch.bool, device=free.device)
            taken[new] = True
            low_free = low_free[~taken[low_free.to(torch.int64)]]
        cm.free_slots = low_free
        logger.info_rank0(f"elastic KV compaction: moved {moved} tokens out of step {s}")
        return True

    def note_busy(self) -> None:
        self.last_busy = time.monotonic()

    def top_step_unneeded(self) -> bool:
        return time.monotonic() - self.top_needed_ts >= self.evict_after_s

    def maybe_shrink_idle(self, cache_manager, running: bool) -> None:
        if running:
            self.last_busy = time.monotonic()
            return
        if self.steps > 1 and time.monotonic() - self.last_busy >= self.idle_s:
            while self.steps > 1 and self.shrink(cache_manager):
                pass
            self.last_busy = time.monotonic()


__all__ = ["ElasticKV", "LayerSlabs", "enabled", "geometry", "logical_tokens", "initial_tokens", "EXPERT_CHUNK"]
