"""Persistent-kernel smoke test.

The kernel is launched once and runs forever (until *stop_flag != 0).
The host:
  1. Pushes N events into the SPSC input ring.
  2. Polls `events_done` until it equals N.
  3. Sets stop_flag, syncs, and reads outputs.

Verifies the same algorithmic correctness as test_runner_drain.py but
through the polling path the live-camera runner will use. Importantly
this proves:
  • host writes to ring + ring_head are visible to the GPU
  • GPU writes to ring_tail / out_head / events_done / out_buf are
    visible to the host (under managed-memory + __threadfence_system)
  • stop_flag termination cleanly halts the loop

Run:
    python3 deploy/orin/tests/test_runner_persistent.py
"""
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import (
    CudaModule, arg_ptr, arg_i32, arg_u32, arg_u64,  # noqa: E402
)
from orin.ring import EVENT_DTYPE, Ring  # noqa: E402
from orin.ssla_ref import LayerRef, StepRefState, step_ref  # noqa: E402
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402

# Reuse the helper module structures + builders.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_runner_drain import (
    CLayerWeights, CStepConfig, COutputSlot,
    SSLA_S_CHANNELS, SSLA_S_KERNELS, SSLA_S_IN_OUT,
    _alloc_and_fill, _stage_grid, _build_layers,
)  # noqa: E402


KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "kernels")


def main() -> int:
    src   = open(os.path.join(KERNELS_DIR, "ssla_step.cuh")).read()
    prim  = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
    layer = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()
    t0 = time.monotonic()
    print(f"loading ssla_step.cuh  ({len(src)}+{len(prim)}+{len(layer)} bytes)", flush=True)
    mod = CudaModule(src, name="ssla_step.cu",
                     headers={"ssla_primitives.cuh": prim,
                              "ssla_layer.cuh":      layer})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s", flush=True)

    H0, W0 = 24, 32
    n_events = 80
    tdrop_window = 1
    rng = np.random.default_rng(11)

    kernel_layers, ref_layers, keepalive = _build_layers(rng, H0, W0)

    # tdrop counters

    # tdrop counters
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

    # StepConfig
    cfg = CStepConfig()
    cfg.H0, cfg.W0, cfg.tdrop_window = H0, W0, tdrop_window
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

    # Input ring + output buffer + control scalars in PINNED host
    # memory (Orin NX has CONCURRENT_MANAGED_ACCESS=0 so managed memory
    # is unsafe for concurrent CPU+GPU access).
    in_cap = 1 << 8
    ring = Ring(in_cap, allocator="pinned")
    ring_buf_dev, ring_head_dev, ring_tail_dev = ring.device_ptrs()

    out_cap = 1 << 8
    out_bytes = ctypes.sizeof(COutputSlot) * out_cap
    p_out, out_ka = cuda_util.alloc_pinned(out_bytes)
    keepalive.append(out_ka)

    p_stop, stop_ka = cuda_util.alloc_pinned(4)
    keepalive.append(stop_ka)
    stop_view = np.frombuffer(stop_ka, dtype=np.int32).reshape(1)

    p_done, done_ka = cuda_util.alloc_pinned(8)
    keepalive.append(done_ka)
    done_view = np.frombuffer(done_ka, dtype=np.uint64).reshape(1)

    p_outhead, outhead_ka = cuda_util.alloc_pinned(8)
    keepalive.append(outhead_ka)
    outhead_view = np.frombuffer(outhead_ka, dtype=np.uint64).reshape(1)

    func = mod.get_function("k_ssla_persistent_loop")

    # Pre-push all events into the pinned ring before launch, so the
    # kernel drains as fast as it can. (Real-time interleaved push +
    # drain is exercised by future live-camera runner tests.)
    events = np.empty(n_events, dtype=EVENT_DTYPE)
    events["t"] = rng.random(n_events).astype(np.float32)
    events["x"] = rng.integers(0, W0, n_events).astype(np.float32)
    events["y"] = rng.integers(0, H0, n_events).astype(np.float32)
    events["p"] = rng.integers(0, 2, n_events).astype(np.float32)
    pushed = ring.try_push_batch(events)
    assert pushed == n_events, pushed

    # Launch persistent kernel ASYNC (don't sync after).
    SMEM = (2 + 96 + 5 * 96) * 4   # SSLA_STEP_SCRATCH_FLOATS(96) * sizeof(float)
    func.launch((1,1,1), (256,1,1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_buf_dev),
        arg_u64(in_cap - 1),
        arg_ptr(ring_head_dev), arg_ptr(ring_tail_dev),
        arg_ptr(p_out), arg_u64(out_cap - 1),
        arg_ptr(p_outhead),
        arg_ptr(p_stop), arg_ptr(p_done),
        arg_u32(2_000),  # 2 µs nanosleep on empty
    ], smem=SMEM)
    # Poll done_view from the main thread while the kernel runs async.
    poll_t0 = time.monotonic()
    while int(done_view[0]) < n_events:
        time.sleep(0.01)
        if time.monotonic() - poll_t0 > 5.0:
            print(f"TIMEOUT done={int(done_view[0])}/{n_events}", flush=True)
            break
    poll_dt = time.monotonic() - poll_t0

    stop_view[0] = 1
    from cuda import cuda
    err, = cuda.cuCtxSynchronize()
    if err != cuda.CUresult.CUDA_SUCCESS:
        print(f"cuCtxSynchronize err: {err}", flush=True)
        return 1

    final_done = int(done_view[0])
    final_outhead = int(outhead_view[0])
    print(f"  drain {poll_dt*1000:.0f}ms  events_done={final_done}  "
          f"out_head={final_outhead}  ({final_done/poll_dt:.0f} ev/s)",
          flush=True)

    # Verify outputs vs ref.
    SlotArr = COutputSlot * out_cap
    slots   = SlotArr.from_address(p_out)

    ref_state = StepRefState(H0=H0, W0=W0, tdrop_window=tdrop_window,
                             layers=ref_layers, tdrop=ref_tdrop)
    fails = 0
    max_err = 0.0
    pass_count = 0
    for i in range(n_events):
        ev = events[i]
        passed_r, s0r, s1r, s2r, s3r, txr, tyr = step_ref(
            ref_state, int(ev["x"]), int(ev["y"]),
            float(ev["t"]), float(ev["p"]))

        slot = slots[i]
        passed_k = bool(slot.passed)
        if passed_k != passed_r:
            print(f"  ev[{i}] FAIL: passed mismatch", flush=True)
            fails += 1
            continue
        if not passed_r:
            continue
        pass_count += 1
        for k_ in range(4):
            if slot.touched_x[k_] != txr[k_] or slot.touched_y[k_] != tyr[k_]:
                print(f"  ev[{i}] FAIL: touched mismatch at {k_}", flush=True)
                fails += 1
        for slot_arr, s_ref, dim in [
            (slot.s0, s0r, 12), (slot.s1, s1r, 24),
            (slot.s2, s2r, 48), (slot.s3, s3r, 96),
        ]:
            kbuf = np.frombuffer(slot_arr, dtype=np.float32, count=dim)
            err = float(np.max(np.abs(kbuf - s_ref)))
            max_err = max(max_err, err)

    # Free.
    cuda_util.free_managed(p_cfg)
    cuda_util.free_pinned(p_out)
    cuda_util.free_pinned(p_stop)
    cuda_util.free_pinned(p_done)
    cuda_util.free_pinned(p_outhead)
    for kl in kernel_layers:
        for k in ("qvgIn", "goW", "ln_gamma", "ln_beta", "hidden_dev"):
            cuda_util.free_managed(kl[k])
        if kl["input_proj"] != 0:
            cuda_util.free_managed(kl["input_proj"])
    for p in p_tdrop:
        cuda_util.free_managed(p)
    ring.close()

    atol = 1e-2
    ok = (final_done == n_events) and (fails == 0) and (max_err < atol)
    print(f"events={n_events} passed={pass_count} failed={fails}  "
          f"max|Δ|={max_err:.2e}  atol={atol:.0e}", flush=True)
    print("OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
