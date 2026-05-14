"""SSLA-S throughput micro-benchmark on Orin GPU.

Pre-pushes N synthetic events into a pinned ring, runs `k_ssla_drain_n`
(synchronous, no polling overhead), measures wall-clock time. Reports
ev/s plus analytical GMAC/s (from the network topology + tdrop ratios).

Use as the canonical baseline / regression measurement for P1+
optimizations. Validates a sample of outputs against the numpy
reference each run.

Run:
    python3 deploy/orin/bench_throughput.py             # default: 5k events
    python3 deploy/orin/bench_throughput.py --n 20000   # longer sample
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import (
    CudaModule, arg_ptr, arg_i32, arg_u64,  # noqa: E402
)
from orin.ring import EVENT_DTYPE, Ring  # noqa: E402
from orin.ssla_ref import StepRefState, step_ref  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_runner_drain import (  # noqa: E402
    CLayerWeights, CStepConfig, COutputSlot, _build_layers, _stage_grid,
    SSLA_S_IN_OUT, SSLA_S_KERNELS,
)

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def _layer_macs(in_d: int, out_d: int, K: int) -> int:
    """Per-event MAC cost of a single SSLA layer in the kernel.
    Counts: input_proj (if in!=out), 9-position qvgIn matvec, 9-position
    goW matvec. LRU element-wise ops not counted (they're FLOPs, not
    MACs — the analytical GMAC/s number is what we compare to peak
    fp32 GMAC/s of the GPU).
    """
    A = K * K
    macs = A * (3 * out_d * in_d) + A * (out_d * out_d)
    if in_d != out_d:
        macs += out_d * in_d
    return macs


def _stage_macs(stage: int) -> int:
    """Total MAC cost for a single event running the two layers of stage s."""
    total = 0
    for li in (2 * stage, 2 * stage + 1):
        in_d, out_d = SSLA_S_IN_OUT[li]
        K = SSLA_S_KERNELS[li]
        total += _layer_macs(in_d, out_d, K)
    return total


def _count_passes(tdrop_dev_ptr: int, n_cells: int, window: int) -> int:
    """Count events that passed a tdrop gate by reading its byte
    counter array. At each cell the number of passes is
    ceil(touches / window). Bytes wrap at 256 — fine for our N."""
    arr = (ctypes.c_ubyte * n_cells).from_address(tdrop_dev_ptr)
    K = np.frombuffer(arr, dtype=np.uint8)
    return int(np.sum((K.astype(np.int32) + window - 1) // window))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="events to drain")
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--validate", type=int, default=64,
                    help="how many sample outputs to compare to numpy ref")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H0, W0 = 480, 640
    n = args.n

    src   = open(os.path.join(KERNELS_DIR, "ssla_step.cuh")).read()
    prim  = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
    layer = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()
    print(f"compile  ({len(src)+len(prim)+len(layer)} bytes)...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_step.cu",
                     headers={"ssla_primitives.cuh": prim,
                              "ssla_layer.cuh":      layer})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    print(f"build weights for {H0}x{W0}...", flush=True)
    kernel_layers, ref_layers, keepalive = _build_layers(rng, H0, W0)

    p_tdrop = []
    ref_tdrop = []
    for s in range(3):
        Hl, Wl = _stage_grid(H0, W0, s + 1)
        n_cells = Hl * Wl
        p, ka = cuda_util.alloc_managed(n_cells)
        ctypes.memset(p, 0, n_cells)
        keepalive.append(ka)
        p_tdrop.append(p)
        ref_tdrop.append(np.zeros(n_cells, dtype=np.uint8))

    cfg = CStepConfig()
    cfg.H0, cfg.W0, cfg.tdrop_window = H0, W0, args.tdrop
    for li, kl in enumerate(kernel_layers):
        cfg.layers[li].qvgIn      = kl["qvgIn"]
        cfg.layers[li].goW        = kl["goW"]
        cfg.layers[li].input_proj = kl["input_proj"]
        cfg.layers[li].ln_gamma   = kl["ln_gamma"]
        cfg.layers[li].ln_beta    = kl["ln_beta"]
        cfg.hidden[li]            = kl["hidden_dev"]
    for s in range(3):
        cfg.tdrop[s] = p_tdrop[s]
    p_cfg, cfg_ka = cuda_util.alloc_managed(ctypes.sizeof(CStepConfig))
    ctypes.memmove(p_cfg, ctypes.addressof(cfg), ctypes.sizeof(CStepConfig))
    keepalive.append(cfg_ka)

    # Synthetic events — uniform in sensor space, dt_norm in [0,1).
    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = rng.random(n).astype(np.float32)
    events["x"] = rng.integers(0, W0, n).astype(np.float32)
    events["y"] = rng.integers(0, H0, n).astype(np.float32)
    events["p"] = rng.integers(0, 2, n).astype(np.float32)

    # Pinned ring sized to fit all events (drain_n reads from start=0).
    cap = 1
    while cap < n: cap <<= 1
    ring = Ring(cap, allocator="pinned")
    pushed = ring.try_push_batch(events)
    assert pushed == n, pushed
    ring_buf_dev, _, _ = ring.device_ptrs()

    p_out, out_ka = cuda_util.alloc_pinned(ctypes.sizeof(COutputSlot) * n)
    keepalive.append(out_ka)

    func = mod.get_function("k_ssla_drain_n")
    SMEM = (2 + 96 + 5 * 96) * 4   # SSLA_STEP_SCRATCH_FLOATS(96) * sizeof(float)

    # Warmup: 1 event so SASS / page faults are paid before timing.
    func.launch((1,1,1), (256,1,1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_buf_dev), arg_u64(cap - 1),
        arg_u64(0), arg_i32(1),
        arg_ptr(p_out),
    ], smem=SMEM)
    func.sync()
    # Reset hidden state + tdrop after warmup so reference matches.
    for kl in kernel_layers:
        kl["hidden_view"].fill(0.0)
    for li, ref_l in enumerate(ref_layers):
        ref_l.H.fill(0.0)
    for p in p_tdrop:
        Hl, Wl = _stage_grid(H0, W0, p_tdrop.index(p) + 1)
        ctypes.memset(p, 0, Hl * Wl)
    for arr in ref_tdrop:
        arr.fill(0)
    # Re-init hidden from rng to match _build_layers (so reference still matches).
    rng2 = np.random.default_rng(args.seed)
    for li in range(8):
        in_d, out_d = SSLA_S_IN_OUT[li]
        K = SSLA_S_KERNELS[li]
        A = K * K
        # consume same RNG sequence as _build_layers up to H_init
        rng2.normal(scale=0.3, size=(in_d, (1 if K == 1 else A) * out_d))
        rng2.normal(scale=0.3, size=((1 if K == 1 else A) * out_d, out_d))
        for _ in range(4):
            rng2.normal(scale=0.3, size=(out_d, out_d))
        if in_d != out_d:
            rng2.normal(scale=0.3, size=(out_d, in_d))
        rng2.normal(scale=0.1, size=out_d)
        rng2.normal(scale=0.1, size=out_d)
        Hl, Wl = _stage_grid(H0, W0, li // 2)
        H_init = rng2.normal(scale=0.05, size=(Hl * Wl, out_d)).astype(np.float32)
        kernel_layers[li]["hidden_view"][...] = H_init
        ref_layers[li].H[...] = H_init

    # Time the drain.
    t0 = time.monotonic()
    func.launch((1,1,1), (256,1,1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_buf_dev), arg_u64(cap - 1),
        arg_u64(0), arg_i32(n),
        arg_ptr(p_out),
    ], smem=SMEM)
    func.sync()
    dt = time.monotonic() - t0
    ev_s = n / dt

    # Validate a sample of outputs.
    SlotArr = COutputSlot * n
    slots = SlotArr.from_address(p_out)
    state = StepRefState(H0=H0, W0=W0, tdrop_window=args.tdrop,
                         layers=ref_layers, tdrop=ref_tdrop)
    val_n = min(args.validate, n)
    fails = 0
    max_err = 0.0
    pass_count_kernel = 0
    pass_count_ref = 0
    for i in range(val_n):
        ev = events[i]
        passed_r, s0r, s1r, s2r, s3r, txr, tyr = step_ref(
            state, int(ev["x"]), int(ev["y"]),
            float(ev["t"]), float(ev["p"]))
        slot = slots[i]
        if bool(slot.passed) != passed_r:
            fails += 1
            continue
        if not passed_r:
            continue
        pass_count_kernel += 1
        pass_count_ref += 1
        for arr, ref, dim in [
            (slot.s0, s0r, 12), (slot.s1, s1r, 24),
            (slot.s2, s2r, 48), (slot.s3, s3r, 96),
        ]:
            kbuf = np.frombuffer(arr, dtype=np.float32, count=dim)
            err = float(np.max(np.abs(kbuf - ref)))
            max_err = max(max_err, err)
    # Continue ref past the validation window for kernel-side passed counting.
    for i in range(val_n, n):
        ev = events[i]
        step_ref(state, int(ev["x"]), int(ev["y"]),
                 float(ev["t"]), float(ev["p"]))

    total_passed = sum(1 for i in range(n) if slots[i].passed)
    P = [n, 0, 0, total_passed]   # P[s] = events that ran stage s (s=0..3)
    for s in range(3):
        Hl, Wl = _stage_grid(H0, W0, s + 1)
        P[s + 1] = _count_passes(p_tdrop[s], Hl * Wl, args.tdrop)
    # Sanity: P should be non-increasing; tdrop gate at idx s feeds stage s+1.
    macs_total = sum(P[s] * _stage_macs(s) for s in range(4))
    avg_macs   = macs_total / max(1, n)
    gmacs      = macs_total / dt / 1e9

    print(f"\nbench result")
    print(f"  H={H0}  W={W0}  N={n}  tdrop={args.tdrop}")
    print(f"  drain         {dt*1000:8.1f} ms")
    print(f"  throughput    {ev_s:8.0f} ev/s   ({ev_s/1e6:.4f} Mev/s)")
    print(f"  stage reach   s0={P[0]}  s1={P[1]}  s2={P[2]}  s3={P[3]}")
    print(f"  avg MACs/event   {avg_macs:8.0f}   (steady-state @1/{args.tdrop**3} ≈ 30912)")
    print(f"  achieved GMAC/s   {gmacs:8.3f}")
    print(f"  fp32 peak  ~2400 GMAC/s   util  {100*gmacs/2400:.3f}%")
    print(f"  validate  {val_n} samples  fails={fails}  max|Δ|={max_err:.2e}")

    cuda_util.free_managed(p_cfg)
    cuda_util.free_pinned(p_out)
    for kl in kernel_layers:
        for k in ("qvgIn", "goW", "ln_gamma", "ln_beta", "hidden_dev"):
            cuda_util.free_managed(kl[k])
        if kl["input_proj"] != 0:
            cuda_util.free_managed(kl["input_proj"])
    for p in p_tdrop:
        cuda_util.free_managed(p)
    ring.close()

    ok = fails == 0 and max_err < 1e-2
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
