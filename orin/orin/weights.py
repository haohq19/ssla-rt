"""Load OpenEVA's `export.py` output (weights.npz + meta.json) into
CUDA managed memory so a persistent kernel can read it without a copy.

The exporter writes every parameter and required buffer as a key in
`weights.npz` (key separator `/`, all float32 C-order). For each tensor
this loader:
  - allocates a managed-memory block of the right size,
  - exposes a host-visible `np.ndarray` view (for inspection / numpy
    consumers),
  - returns a `CUdeviceptr` integer for kernel launches.

The loader is method-agnostic — `Weights` will load every npz key
regardless of method. SSLA-specific helpers live in
`weights_ssla.py` (later) so the kernel side can pull tensors by their
semantic role (stage, layer, projection) rather than by raw key.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


class Weights:
    def __init__(self, export_dir: str | Path, *, allocator: str = "managed"):
        self.export_dir = Path(export_dir)
        meta_path = self.export_dir / "meta.json"
        npz_path = self.export_dir / "weights.npz"
        if not meta_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(
                f"export dir missing weights.npz or meta.json: {self.export_dir}")
        with open(meta_path) as f:
            self.meta: Dict = json.load(f)
        self._allocator = allocator
        self._views:    Dict[str, np.ndarray] = {}
        self._devptrs:  Dict[str, int]        = {}
        self._shapes:   Dict[str, Tuple[int, ...]] = {}
        self._keepalive: list = []
        self._total_bytes = 0

        npz = np.load(npz_path)
        try:
            for name in npz.files:
                arr = np.ascontiguousarray(npz[name].astype(np.float32, copy=False))
                self._add(name, arr)
        finally:
            npz.close()

    def _add(self, name: str, arr: np.ndarray) -> None:
        nbytes = arr.nbytes
        if self._allocator == "managed":
            from . import cuda_util
            ptr, ka = cuda_util.alloc_managed(nbytes)
            view = np.frombuffer(ka, dtype=np.float32).reshape(arr.shape)
            view[...] = arr
            self._views[name]   = view
            self._devptrs[name] = ptr
            self._keepalive.append(ka)
        elif self._allocator == "numpy":
            view = arr.copy()
            self._views[name]   = view
            self._devptrs[name] = view.ctypes.data
        else:
            raise ValueError(f"unknown allocator {self._allocator!r}")
        self._shapes[name] = arr.shape
        self._total_bytes += nbytes

    def keys(self) -> Iterable[str]:
        return self._views.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._views

    def view(self, name: str) -> np.ndarray:
        return self._views[name]

    def devptr(self, name: str) -> int:
        return self._devptrs[name]

    def shape(self, name: str) -> Tuple[int, ...]:
        return self._shapes[name]

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def allocator(self) -> str:
        return self._allocator

    def close(self):
        if self._allocator == "managed":
            from . import cuda_util
            for p in self._devptrs.values():
                cuda_util.free_managed(p)
        self._devptrs.clear()
        self._views.clear()
        self._keepalive.clear()
        self._shapes.clear()
