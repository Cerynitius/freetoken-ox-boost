"""CUDA virtual-memory regions with growable/shrinkable physical backing.

A region reserves a fixed virtual range once; physical memory is mapped/unmapped in
granularity-aligned sub-ranges at runtime. Tensors viewing the region keep their
address, shape and strides forever, so kernels and captured CUDA graphs stay valid
while the physical footprint changes (elastic KV <-> expert cache, engine/elastic.py).
Touching an unmapped sub-range is an illegal address -- callers guarantee that only
mapped offsets are ever referenced.
"""

from __future__ import annotations

import torch

_GRAN: dict[int, int] = {}


def _check(res):
    err = res[0] if isinstance(res, tuple) else res
    from cuda.bindings import driver as d

    if err != d.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return res[1] if isinstance(res, tuple) and len(res) == 2 else res


def _prop(dev: int):
    from cuda.bindings import driver as d

    p = d.CUmemAllocationProp()
    p.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


def granularity(dev: int) -> int:
    if dev not in _GRAN:
        from cuda.bindings import driver as d

        torch.cuda.init()
        _GRAN[dev] = int(_check(d.cuMemGetAllocationGranularity(
            _prop(dev), d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)))
    return _GRAN[dev]


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class _CAI:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3,
            "strides": None, "stream": None,
        }


class VmmRegion:
    """Reserve ``nbytes`` of VA on ``device``; map/unmap aligned sub-ranges on demand."""

    def __init__(self, nbytes: int, device: torch.device):
        from cuda.bindings import driver as d

        self.dev = device.index if device.index is not None else torch.cuda.current_device()
        self.gran = granularity(self.dev)
        self.size = align_up(nbytes, self.gran)
        self.base = int(_check(d.cuMemAddressReserve(self.size, self.gran, 0, 0)))
        self.maps: dict[int, tuple[int, object]] = {}  # offset -> (size, handle)
        self.device = torch.device("cuda", self.dev)

    @property
    def mapped_bytes(self) -> int:
        return sum(s for s, _ in self.maps.values())

    def map(self, off: int, size: int) -> None:
        from cuda.bindings import driver as d

        size = align_up(size, self.gran)
        assert off % self.gran == 0 and off + size <= self.size, (off, size, self.size)
        assert off not in self.maps
        h = _check(d.cuMemCreate(size, _prop(self.dev), 0))
        try:
            _check(d.cuMemMap(self.base + off, size, 0, h, 0))
        except Exception:
            d.cuMemRelease(h)
            raise
        acc = d.CUmemAccessDesc()
        acc.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = self.dev
        acc.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(d.cuMemSetAccess(self.base + off, size, [acc], 1))
        self.maps[off] = (size, h)

    def unmap(self, off: int) -> None:
        """Unmap the sub-range previously mapped at ``off`` (caller has synchronized)."""
        from cuda.bindings import driver as d

        size, h = self.maps.pop(off)
        _check(d.cuMemUnmap(self.base + off, size))
        _check(d.cuMemRelease(h))

    def bytes_tensor(self) -> torch.Tensor:
        """uint8 view over the WHOLE reserved range (only mapped parts may be touched)."""
        return torch.as_tensor(_CAI(self.base, self.size), device=self.device)

    def free(self) -> None:
        from cuda.bindings import driver as d

        torch.cuda.synchronize(self.device)
        for off in list(self.maps):
            self.unmap(off)
        d.cuMemAddressFree(self.base, self.size)
        self.base = 0


__all__ = ["VmmRegion", "granularity", "align_up"]
