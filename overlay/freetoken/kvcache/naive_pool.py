"""KV pool for Naive-N0.5-Flash: the hybrid full/SWA pool with K and V at their own
widths (K 192, V 128) plus the DSA index-key slab.

Same global-paged SWA machinery as ``HybridSWAKVCache`` (full->swa slot mapping, swa
free-list, translate/alloc/free), so the SWA radix/naive cache managers drive it
unchanged. What differs is storage:

* each group keeps separate K ``[L, T, H, 192]`` and V ``[L, T, H, 128]`` slabs -- the
  generic pool would force V to 192 and burn 50% more V bytes and bandwidth;
* the DSA (full-group) layers also own a per-token index key: the indexer's LayerNorm'd,
  roped key rounded to fp8-e4m3 with a per-token fp32 scale, exactly the reference's
  ``round_indexer_fp8`` (value == fp8 * scale), stored as ``[Ld, T, 128]`` fp8 plus
  ``[Ld, T]`` fp32 scales, addressed by the FULL slot like the K/V it belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from freetoken.models.config import KVCacheGroupSpec

from .hybrid_swa_pool import (
    HybridSWAKVCache,
    _LayerRef,
    _naive_swa_num_tokens,
    _swa_paged_num_tokens,
    _swa_pool_floor,
)


@dataclass
class _NaiveGroupStorage:
    k_buffer: torch.Tensor  # [L, T, H, Dk]
    v_buffer: torch.Tensor  # [L, T, H, Dv]
    num_tokens: int


def _index_bytes_per_token(index_dim: int, num_index_layers: int) -> int:
    return num_index_layers * (index_dim + 4)  # fp8 key + fp32 scale


def _group_bytes_per_token(spec: KVCacheGroupSpec, v_dim: int, itemsize: int) -> int:
    return spec.num_layers * spec.num_kv_heads * (spec.head_dim + v_dim) * itemsize


class NaiveKVCache(HybridSWAKVCache):
    def __init__(
        self,
        groups: Sequence[KVCacheGroupSpec],
        num_layers: int,
        num_full_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_swa_tokens: int | None,
        v_head_dim: int,
        index_dim: int,
    ) -> None:
        assert page_size == 1, "Naive pool addresses tokens (page_size 1)"
        specs = {g.name: g for g in groups if g.num_layers > 0}
        assert set(specs) == {"full", "swa"}, sorted(specs)
        self._specs = specs
        self._v_dim = v_head_dim
        self._index_dim = index_dim
        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        self._page_size = page_size
        self._full_num_tokens = num_full_pages * page_size
        self._swa_num_tokens = (
            num_swa_tokens if num_swa_tokens is not None else self._full_num_tokens
        )
        self._swa_paged = True
        self._alloc_all()
        self.layers_mapping = self._build_layers_mapping(num_layers, specs)
        self._init_swa_paged_state()

    # ---------------------------------------------------------------- storage
    def _alloc(self, spec: KVCacheGroupSpec, num_tokens: int) -> _NaiveGroupStorage:
        k = torch.empty(
            (spec.num_layers, num_tokens, spec.num_kv_heads, spec.head_dim),
            device=self._device, dtype=self._dtype,
        )
        v = torch.empty(
            (spec.num_layers, num_tokens, spec.num_kv_heads, self._v_dim),
            device=self._device, dtype=self._dtype,
        )
        return _NaiveGroupStorage(k, v, num_tokens)

    def _alloc_all(self) -> None:
        from freetoken.engine import elastic

        self._elastic = []
        el = elastic.enabled() and self._full_num_tokens == elastic.logical_tokens()
        if el:
            spec = self._specs["full"]
            k = self._vmm((spec.num_layers, self._full_num_tokens, spec.num_kv_heads, spec.head_dim), self._dtype)
            v = self._vmm((spec.num_layers, self._full_num_tokens, spec.num_kv_heads, self._v_dim), self._dtype)
            self.full_kv_pool = _NaiveGroupStorage(k, v, self._full_num_tokens)
        else:
            self.full_kv_pool = self._alloc(self._specs["full"], self._full_num_tokens)
        self.swa_kv_pool = self._alloc(self._specs["swa"], self._swa_num_tokens)
        self._storages = {"full": self.full_kv_pool, "swa": self.swa_kv_pool}
        n_idx = self._specs["full"].num_layers
        # Zero-filled: never-written slots (dummy page, padding) must score as finite
        # zeros, not NaN garbage, when a fixed-shape decode kernel touches them.
        if el:
            self.index_k = self._vmm((n_idx, self._full_num_tokens, self._index_dim),
                                     torch.float8_e4m3fn, zero=True)
        else:
            self.index_k = torch.zeros(
                (n_idx, self._full_num_tokens, self._index_dim),
                device=self._device, dtype=torch.float8_e4m3fn,
            )
        self.index_scale = torch.zeros(
            (n_idx, self._full_num_tokens), device=self._device, dtype=torch.float32
        )

    # ---------------------------------------------------------------- elastic (engine/elastic.py)
    def _vmm(self, shape, dtype, zero: bool = False) -> torch.Tensor:
        """Layer-major [L, T, ...] tensor in a reserved VA range; step 0 + the tail mapped."""
        from freetoken.engine import elastic
        from freetoken.kernel.vmm import VmmRegion

        step, tail, steps_max = elastic.geometry()
        L, T = shape[0], shape[1]
        rb = torch.empty((), dtype=dtype).element_size()
        for d in shape[2:]:
            rb *= d
        reg = VmmRegion(L * T * rb, self._device)
        assert (T * rb) % reg.gran == 0 and (step * rb) % reg.gran == 0 and (tail * rb) % reg.gran == 0
        for l in range(L):
            reg.map(l * T * rb, step * rb)
            reg.map(l * T * rb + steps_max * step * rb, tail * rb)
        t = reg.bytes_tensor()[: L * T * rb].view(dtype).view(shape)
        if zero:
            t[:, :step].view(torch.uint8).zero_()
            t[:, steps_max * step:].view(torch.uint8).zero_()
        self._elastic.append((reg, L, T, rb, t, zero))
        return t

    def elastic_total_tokens(self) -> int:
        return self._full_num_tokens

    def elastic_step_bytes(self) -> int:
        from freetoken.engine import elastic

        step = elastic.geometry()[0]
        return sum(L * step * rb for _, L, _, rb, _, _ in self._elastic)

    def elastic_map_step(self, s: int) -> None:
        from freetoken.engine import elastic

        step = elastic.geometry()[0]
        for reg, L, T, rb, t, zero in self._elastic:
            for l in range(L):
                reg.map(l * T * rb + s * step * rb, step * rb)
            if zero:
                t[:, s * step:(s + 1) * step].view(torch.uint8).zero_()

    def elastic_unmap_step(self, s: int) -> None:
        from freetoken.engine import elastic

        step = elastic.geometry()[0]
        for reg, L, T, rb, _, _ in self._elastic:
            for l in range(L):
                reg.unmap(l * T * rb + s * step * rb)

    def elastic_move_tokens(self, old: torch.Tensor, new: torch.Tensor, chunk: int = 32768) -> None:
        """Copy the DSA K/V/index rows (all layers) of full slots ``old`` -> ``new`` and move
        their full->swa mapping entries (idle-only compaction)."""
        fk, fv = self.full_kv_pool.k_buffer, self.full_kv_pool.v_buffer
        for a in range(0, old.numel(), chunk):
            o, n = old[a:a + chunk], new[a:a + chunk]
            fk[:, n] = fk[:, o]
            fv[:, n] = fv[:, o]
            self.index_k.view(torch.uint8)[:, n] = self.index_k.view(torch.uint8)[:, o]
            self.index_scale[:, n] = self.index_scale[:, o]
            m = self.full_to_swa_index_mapping
            m[n] = m[o]
            m[o] = 0

    def rebuild(self, num_full_pages: int, num_swa_tokens: int | None = None) -> None:
        assert not self._elastic, "elastic KV pool does not support runtime rebuild"
        self._full_num_tokens = num_full_pages * self._page_size
        self._swa_num_tokens = (
            num_swa_tokens if num_swa_tokens is not None else self._full_num_tokens
        )
        self.full_kv_pool = self.swa_kv_pool = None
        self._storages = {}
        self.index_k = self.index_scale = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._alloc_all()
        self._init_swa_paged_state()

    # ---------------------------------------------------------------- access
    def group_index(self, layer_id: int) -> _LayerRef:
        return self.layers_mapping[layer_id]

    def k_cache(self, index: int) -> torch.Tensor:
        ref = self.layers_mapping[index]
        return self._storages[ref.group].k_buffer[ref.index]

    def v_cache(self, index: int) -> torch.Tensor:
        ref = self.layers_mapping[index]
        return self._storages[ref.group].v_buffer[ref.index]

    def index_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self.layers_mapping[layer_id]
        assert ref.group == "full", f"layer {layer_id} has no index keys"
        return self.index_k[ref.index], self.index_scale[ref.index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        ref = self.layers_mapping[layer_id]
        storage = self._storages[ref.group]
        loc = out_loc
        if ref.group == "swa":
            loc = self.translate_loc_from_full_to_swa(out_loc)
        loc = loc.to(torch.int64)
        kb = storage.k_buffer[ref.index]
        vb = storage.v_buffer[ref.index]
        kb.index_copy_(0, loc, k.view(-1, kb.shape[1], kb.shape[2]))
        vb.index_copy_(0, loc, v.view(-1, vb.shape[1], vb.shape[2]))

    def store_index(
        self, k8: torch.Tensor, scale: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        ik, isc = self.index_cache(layer_id)
        loc = out_loc.to(torch.int64)
        ik.view(torch.uint8).index_copy_(0, loc, k8.view(torch.uint8))  # no fp8 index_copy
        isc.index_copy_(0, loc, scale)

    # ---------------------------------------------------------------- sizing
    @classmethod
    def kv_cost(cls, config, *, num_swa_pages: int | None = None) -> tuple[int, int, int, int]:
        mc = config.model_config
        args = mc.naive_args
        itemsize = config.dtype.itemsize
        swa_pin = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
        cache_per_page = 0
        fixed = 0
        for spec in mc.kv_cache_group_specs():
            per_token = _group_bytes_per_token(spec, args.v_head_dim, itemsize)
            if not spec.is_swa:
                per_token += _index_bytes_per_token(args.index_dim, spec.num_layers)
                cache_per_page += per_token * config.page_size
                continue
            cache_per_page += 8 * config.page_size  # full_to_swa mapping
            if config.cache_type != "swa_radix":
                fixed += per_token * _naive_swa_num_tokens(config)
            elif swa_pin is not None:
                fixed += per_token * (max(_swa_pool_floor(config), int(swa_pin)) + 1)
            else:
                cache_per_page += int(per_token * config.page_size * config.swa_full_tokens_ratio)
                fixed += per_token * _swa_pool_floor(config)
        return cache_per_page, fixed, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        num_swa_tokens = (
            _swa_paged_num_tokens(config, num_pages + 1, num_swa_pages=num_swa_pages)
            if config.cache_type == "swa_radix"
            else _naive_swa_num_tokens(config)
        )
        self.rebuild(num_full_pages=num_pages + 1, num_swa_tokens=num_swa_tokens)

    def unit_bytes(self) -> tuple[int, int]:
        f, s = self.full_kv_pool, self.swa_kv_pool
        full = (f.k_buffer.numel() + f.v_buffer.numel()) * f.k_buffer.element_size()
        full += self.index_k.numel() + self.index_scale.numel() * 4
        swa = (s.k_buffer.numel() + s.v_buffer.numel()) * s.k_buffer.element_size()
        return full // self._full_num_tokens, swa // self._swa_num_tokens


__all__ = ["NaiveKVCache"]
