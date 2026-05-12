"""SPSC ring buffer + DVXplorer camera producer thread.

The ring stores events as a 4×float32 record (`t, x, y, p` — matches
`openeva::Event` from the parent deploy). Capacity is power-of-two so the
mask `idx & (cap-1)` replaces the modulo. Head and tail are `np.uint64`
single-element arrays — under CPython's GIL their reads/writes are
visible across threads, which is enough for SPSC at the Python level.

When the ring is later swapped to `cuMemAllocManaged` for kernel
consumption, the only thing that changes is how `_buf`, `_head`, `_tail`
are allocated — the API is identical.

The camera producer runs in a Python thread, reads
`dv_processing.io.camera.getNextEventBatch()` (releases the GIL inside
its USB read), converts the structured int batch into the float32
event-record dtype, and pushes to the ring. It subtracts a per-session
timestamp epoch before casting to keep µs resolution within fp32.

Drop policy: if the ring is full, the producer spins (blocks) by default
to preserve event order — drops would break bit-equivalence with
deploy/S1. Set `block_on_full=False` to drop instead and count them in
`stats.drops`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


EVENT_DTYPE = np.dtype([
    ("t", "<f4"),
    ("x", "<f4"),
    ("y", "<f4"),
    ("p", "<f4"),
])


def _is_pow2(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


class Ring:
    """Single-producer, single-consumer ring of `EVENT_DTYPE` records.

    `try_push_batch(arr)` and `try_pop_batch(out)` return the count
    actually transferred. They handle wrap-around with at most two
    contiguous slice copies. Producer thread owns `head`; consumer
    thread owns `tail`.
    """

    def __init__(self, capacity_pow2: int, *, allocator: str = "numpy"):
        if not _is_pow2(capacity_pow2):
            raise ValueError(f"capacity must be power of two >= 2, got {capacity_pow2}")
        self._cap = capacity_pow2
        self._mask = capacity_pow2 - 1
        self._allocator = allocator
        self._devptrs = None
        self._keepalive = None
        if allocator == "numpy":
            self._buf = np.zeros(capacity_pow2, dtype=EVENT_DTYPE)
            self._head = np.zeros(1, dtype=np.uint64)
            self._tail = np.zeros(1, dtype=np.uint64)
        elif allocator in ("managed", "pinned"):
            from . import cuda_util
            alloc = (cuda_util.alloc_pinned if allocator == "pinned"
                     else cuda_util.alloc_managed)
            buf_bytes = capacity_pow2 * EVENT_DTYPE.itemsize
            buf_ptr,  buf_ka  = alloc(buf_bytes)
            head_ptr, head_ka = alloc(8)
            tail_ptr, tail_ka = alloc(8)
            self._buf = np.frombuffer(buf_ka, dtype=EVENT_DTYPE)
            self._head = np.frombuffer(head_ka, dtype=np.uint64)
            self._tail = np.frombuffer(tail_ka, dtype=np.uint64)
            self._devptrs = (buf_ptr, head_ptr, tail_ptr)
            self._keepalive = (buf_ka, head_ka, tail_ka)
        else:
            raise ValueError(f"unknown allocator {allocator!r}")

    def device_ptrs(self):
        if self._devptrs is None:
            raise RuntimeError("device_ptrs() requires allocator='managed'")
        return self._devptrs

    def close(self):
        if self._allocator in ("managed", "pinned") and self._devptrs is not None:
            from . import cuda_util
            free = (cuda_util.free_pinned if self._allocator == "pinned"
                    else cuda_util.free_managed)
            for p in self._devptrs:
                free(p)
            self._devptrs = None
            self._keepalive = None

    @property
    def capacity(self) -> int:
        return self._cap

    def occupancy(self) -> int:
        return int(self._head[0] - self._tail[0])

    def free_space(self) -> int:
        return self._cap - self.occupancy()

    def try_push_batch(self, arr: np.ndarray) -> int:
        if arr.dtype != EVENT_DTYPE:
            raise TypeError(f"arr must be EVENT_DTYPE, got {arr.dtype}")
        n = arr.shape[0]
        if n == 0:
            return 0
        h = int(self._head[0])
        t = int(self._tail[0])
        free = self._cap - (h - t)
        if free <= 0:
            return 0
        m = min(n, free)
        start = h & self._mask
        end = start + m
        if end <= self._cap:
            self._buf[start:end] = arr[:m]
        else:
            first = self._cap - start
            self._buf[start:] = arr[:first]
            self._buf[: m - first] = arr[first:m]
        self._head[0] = h + m
        return m

    def try_pop_batch(self, out: np.ndarray) -> int:
        if out.dtype != EVENT_DTYPE:
            raise TypeError(f"out must be EVENT_DTYPE, got {out.dtype}")
        n = out.shape[0]
        if n == 0:
            return 0
        h = int(self._head[0])
        t = int(self._tail[0])
        avail = h - t
        if avail <= 0:
            return 0
        m = min(n, avail)
        start = t & self._mask
        end = start + m
        if end <= self._cap:
            out[:m] = self._buf[start:end]
        else:
            first = self._cap - start
            out[:first] = self._buf[start:]
            out[first:m] = self._buf[: m - first]
        self._tail[0] = t + m
        return m


@dataclass
class ProducerStats:
    events_pushed: int = 0
    batches: int = 0
    drops: int = 0
    max_occupancy: int = 0
    spin_us_total: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def rate_mev_s(self) -> float:
        elapsed = time.monotonic() - self.started_at
        return self.events_pushed / elapsed / 1e6 if elapsed > 0 else 0.0


class CameraProducer:
    """Thread that drains a DVXplorer camera into a `Ring`.

    Knobs (`contrast_on`, `contrast_off`, `subsample`, `readout_fps`)
    mirror the visualizer's tuning panel; pass enum values from
    `dv.io.camera.DVXplorer.SubSample` / `.ReadoutFPS`.

    Usage:
        ring = Ring(1 << 20)
        prod = CameraProducer(ring)
        prod.start()
        ...
        prod.stop()
    """

    def __init__(
        self,
        ring: Ring,
        *,
        contrast_on: int = 9,
        contrast_off: int = 9,
        subsample=None,
        readout_fps=None,
        block_on_full: bool = True,
        spin_quantum_us: int = 50,
    ):
        self.ring = ring
        self._contrast_on = contrast_on
        self._contrast_off = contrast_off
        self._subsample = subsample
        self._readout_fps = readout_fps
        self._block = block_on_full
        self._spin_us = spin_quantum_us
        self.stats = ProducerStats()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cam = None
        self._t0_us: Optional[int] = None

    def _open(self):
        import dv_processing as dv
        cam = dv.io.camera.open()
        try:
            cam.setContrastThresholdOn(max(1, self._contrast_on))
            cam.setContrastThresholdOff(max(1, self._contrast_off))
            if self._subsample is not None:
                cam.setSubSampleHorizontal(self._subsample)
                cam.setSubSampleVertical(self._subsample)
            if self._readout_fps is not None:
                cam.setReadoutFPS(self._readout_fps)
        except Exception as e:
            print(f"[producer] camera knob setup warning: {e}", flush=True)
        self._cam = cam
        return cam

    def _convert(self, batch_np: np.ndarray) -> np.ndarray:
        if self._t0_us is None and batch_np.shape[0] > 0:
            self._t0_us = int(batch_np["timestamp"][0])
        out = np.empty(batch_np.shape[0], dtype=EVENT_DTYPE)
        out["t"] = (batch_np["timestamp"] - self._t0_us).astype(np.float32)
        out["x"] = batch_np["x"].astype(np.float32)
        out["y"] = batch_np["y"].astype(np.float32)
        out["p"] = batch_np["polarity"].astype(np.float32)
        return out

    def _run(self):
        cam = self._open()
        spin_quantum_s = self._spin_us / 1e6
        while not self._stop.is_set() and cam.isRunning():
            batch = cam.getNextEventBatch()
            if batch is None or batch.size() == 0:
                time.sleep(0.0005)
                continue
            ev = self._convert(batch.numpy())
            n = ev.shape[0]
            self.stats.batches += 1

            if self._block:
                pushed_total = 0
                while pushed_total < n and not self._stop.is_set():
                    pushed = self.ring.try_push_batch(ev[pushed_total:])
                    pushed_total += pushed
                    if pushed == 0:
                        self.stats.spin_us_total += self._spin_us
                        time.sleep(spin_quantum_s)
                self.stats.events_pushed += pushed_total
            else:
                pushed = self.ring.try_push_batch(ev)
                self.stats.events_pushed += pushed
                self.stats.drops += n - pushed

            occ = self.ring.occupancy()
            if occ > self.stats.max_occupancy:
                self.stats.max_occupancy = occ

    def start(self):
        if self._thread is not None:
            raise RuntimeError("already started")
        self.stats = ProducerStats()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dvx-producer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._cam = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
