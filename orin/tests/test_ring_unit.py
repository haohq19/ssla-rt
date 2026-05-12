"""Unit test: Ring is FIFO-correct under wrap-around with random batch sizes.

No camera, no threads — single-threaded determinism check that the
ring's batch slicing math survives wrap-around.

Run:
    python3 deploy/orin/tests/test_ring_unit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin.ring import EVENT_DTYPE, Ring  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    cap = 1 << 12   # 4096 — small so we wrap many times
    ring = Ring(cap)

    # Generate a long stream of unique-t events, then push/pop in random
    # batch sizes. Verify the popped sequence equals the input sequence.
    n_total = cap * 50
    src = np.empty(n_total, dtype=EVENT_DTYPE)
    src["t"] = np.arange(n_total, dtype=np.float32)
    src["x"] = rng.integers(0, 640, n_total).astype(np.float32)
    src["y"] = rng.integers(0, 480, n_total).astype(np.float32)
    src["p"] = rng.integers(0, 2, n_total).astype(np.float32)

    out = np.empty(n_total, dtype=EVENT_DTYPE)
    pushed_idx = 0
    popped_idx = 0
    while popped_idx < n_total:
        if pushed_idx < n_total:
            push_n = int(rng.integers(1, cap // 2))
            push_n = min(push_n, n_total - pushed_idx)
            actually = ring.try_push_batch(src[pushed_idx : pushed_idx + push_n])
            pushed_idx += actually

        pop_n = int(rng.integers(1, cap // 2))
        pop_n = min(pop_n, n_total - popped_idx)
        scratch = np.empty(pop_n, dtype=EVENT_DTYPE)
        actually_popped = ring.try_pop_batch(scratch)
        out[popped_idx : popped_idx + actually_popped] = scratch[:actually_popped]
        popped_idx += actually_popped

    # FIFO assertion — the popped sequence must equal the source byte-for-byte.
    assert np.array_equal(src, out), "FIFO violated"

    # Empty-pop / full-push sanity.
    full_buf = np.zeros(cap, dtype=EVENT_DTYPE)
    full_buf["t"] = np.arange(cap, dtype=np.float32)
    assert ring.try_push_batch(full_buf) == cap
    assert ring.try_push_batch(full_buf[:1]) == 0  # full
    drained = np.empty(cap, dtype=EVENT_DTYPE)
    assert ring.try_pop_batch(drained) == cap
    assert ring.try_pop_batch(drained[:1]) == 0    # empty
    assert np.array_equal(full_buf, drained)

    print(f"OK  {n_total} events round-tripped through cap={cap} ring (wrapped ~{n_total // cap}×)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
