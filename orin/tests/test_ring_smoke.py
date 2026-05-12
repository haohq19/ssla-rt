"""Live-camera smoke test: producer + dummy consumer, no drops over N seconds.

The producer thread reads the live DVXplorer Micro and pushes events into
a managed-memory-style ring; the dummy consumer (this main thread) drains
as fast as it can and counts total bytes. At the end, assert:
    events_pushed == events_consumed
    drops == 0
    producer keeps up with the camera (no wedged thread)

Print per-second throughput so we know the producer can hit the
6.6–7.5 Mev/s the camera saw in the visualizer.

Run:
    python3 deploy/orin/tests/test_ring_smoke.py [--seconds 5]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin.ring import EVENT_DTYPE, CameraProducer, Ring  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--ring-pow2", type=int, default=20,
                    help="ring capacity = 2**this (default 20 = 1 048 576 events ≈ 16 MB)")
    ap.add_argument("--pop-batch", type=int, default=8192)
    ap.add_argument("--no-camera", action="store_true",
                    help="Run a synthetic in-process producer instead of the camera")
    ap.add_argument("--allocator", choices=["numpy", "managed"], default="numpy",
                    help="Ring backing storage. 'managed' = CUDA Unified Memory (Tegra)")
    args = ap.parse_args()

    cap = 1 << args.ring_pow2
    ring = Ring(cap, allocator=args.allocator)
    print(f"allocator = {args.allocator}", flush=True)
    print(f"ring capacity = {cap} events ({cap * EVENT_DTYPE.itemsize / 1024 / 1024:.1f} MB)",
          flush=True)

    if args.no_camera:
        prod = _SyntheticProducer(ring, target_meps=8.0)
    else:
        prod = CameraProducer(ring, block_on_full=True)
    prod.start()
    print("producer started", flush=True)

    out = np.empty(args.pop_batch, dtype=EVENT_DTYPE)
    consumed = 0
    last_t = time.monotonic()
    last_consumed = 0
    deadline = time.monotonic() + args.seconds

    while time.monotonic() < deadline:
        n = ring.try_pop_batch(out)
        if n == 0:
            time.sleep(0.0005)
            continue
        consumed += n

        now = time.monotonic()
        if now - last_t >= 1.0:
            consumer_meps = (consumed - last_consumed) / (now - last_t) / 1e6
            stats = prod.stats
            print(f"  produced={stats.events_pushed:>10d} ({stats.rate_mev_s():5.2f} Mev/s)   "
                  f"consumed={consumed:>10d} ({consumer_meps:5.2f} Mev/s)   "
                  f"occ={ring.occupancy():>6d}  max_occ={stats.max_occupancy:>6d}  "
                  f"drops={stats.drops}", flush=True)
            last_t = now
            last_consumed = consumed

    # Drain any leftover before stopping the producer so the final count
    # equals what the producer pushed.
    prod.stop()
    while True:
        n = ring.try_pop_batch(out)
        if n == 0:
            break
        consumed += n

    pushed = prod.stats.events_pushed
    drops = prod.stats.drops
    print(f"\nfinal: pushed={pushed}  consumed={consumed}  drops={drops}  "
          f"max_occ={prod.stats.max_occupancy}/{cap}  spin={prod.stats.spin_us_total/1000:.1f}ms",
          flush=True)

    if drops != 0:
        print("FAIL: drops > 0 (consumer too slow or ring too small)", flush=True)
        return 1
    if pushed != consumed:
        print("FAIL: pushed != consumed", flush=True)
        return 2
    if pushed == 0:
        print("FAIL: producer pushed zero events (camera not delivering?)", flush=True)
        return 3
    print("OK", flush=True)
    return 0


class _SyntheticProducer:
    """Cheap stand-in for CameraProducer when --no-camera is passed.

    Pushes a fixed-rate stream of generated events using the same Ring
    and the same `stats` shape, so the smoke test can exercise the
    plumbing without the DVXplorer attached.
    """
    def __init__(self, ring, target_meps=8.0, batch=4096):
        from orin.ring import ProducerStats
        import threading
        self.ring = ring
        self.stats = ProducerStats()
        self._stop = threading.Event()
        self._thread = None
        self._target = target_meps * 1e6
        self._batch = batch
        self._ev = np.empty(batch, dtype=EVENT_DTYPE)
        self._ev["t"] = np.arange(batch, dtype=np.float32)
        self._ev["x"] = (np.arange(batch) % 640).astype(np.float32)
        self._ev["y"] = (np.arange(batch) % 480).astype(np.float32)
        self._ev["p"] = (np.arange(batch) % 2).astype(np.float32)

    def start(self):
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        period = self._batch / self._target
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.001, next_t - now))
                continue
            pushed_total = 0
            while pushed_total < self._batch and not self._stop.is_set():
                pushed = self.ring.try_push_batch(self._ev[pushed_total:])
                pushed_total += pushed
                if pushed == 0:
                    time.sleep(0.00005)
            self.stats.events_pushed += pushed_total
            self.stats.batches += 1
            occ = self.ring.occupancy()
            if occ > self.stats.max_occupancy:
                self.stats.max_occupancy = occ
            next_t += period


if __name__ == "__main__":
    sys.exit(main())
