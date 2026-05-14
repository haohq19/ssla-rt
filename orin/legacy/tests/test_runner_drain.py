"""End-to-end smoke test: SPSC ring → drain_n CUDA kernel → output slots.

Validates the full data path the persistent runner will use:
  • Ring (managed memory) populated by host with N events.
  • Drain kernel reads events from ring, runs the SSLA-S step, writes
    each event's per-stage features into a managed output slot array.
  • Host reads output slots and compares against the pure-numpy
    reference (`step_ref`).

This is the same algorithm as test_step.py but plumbed through the
ring — proving the wire format the persistent kernel + Python
producer will share.

Run:
    python3 deploy/orin/tests/test_runner_drain.py
"""
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import CudaModule, arg_ptr, arg_i32, arg_u64  # noqa: E402
from orin.ring import EVENT_DTYPE, Ring  # noqa: E402
from orin.ssla_ref import LayerRef, StepRefState, step_ref  # noqa: E402
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402


KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "kernels")


# ctypes mirrors of ssla_step.cuh structs.
class CLayerWeights(ctypes.Structure):
    _fields_ = [
        ("qvgIn",      ctypes.c_uint64),
        ("goW",        ctypes.c_uint64),
        ("input_proj", ctypes.c_uint64),
        ("ln_gamma",   ctypes.c_uint64),
        ("ln_beta",    ctypes.c_uint64),
    ]


class CStepConfig(ctypes.Structure):
    _fields_ = [
        ("H0",           ctypes.c_int32),
        ("W0",           ctypes.c_int32),
        ("tdrop_window", ctypes.c_int32),
        ("layers",       CLayerWeights * 8),
        ("hidden",       ctypes.c_uint64 * 8),
        ("tdrop",        ctypes.c_uint64 * 3),
    ]


class COutputSlot(ctypes.Structure):
    _fields_ = [
        ("passed",     ctypes.c_int32),
        ("touched_x",  ctypes.c_int32 * 4),
        ("touched_y",  ctypes.c_int32 * 4),
        ("s0",         ctypes.c_float * 12),
        ("s1",         ctypes.c_float * 24),
        ("s2",         ctypes.c_float * 48),
        ("s3",         ctypes.c_float * 96),
    ]


SSLA_S_CHANNELS = (12, 24, 48, 96)
SSLA_S_KERNELS  = (1, 3, 3, 3, 3, 3, 3, 3)
SSLA_S_IN_OUT   = (
    (2, 12),  (12, 12),
    (12, 24), (24, 24),
    (24, 48), (48, 48),
    (48, 96), (96, 96),
)


def _alloc_and_fill(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    ptr, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=arr.dtype).reshape(arr.shape)
    view[...] = arr
    return ptr, view, ka


def _stage_grid(H0, W0, stage):
    return H0 >> stage, W0 >> stage


def _build_layers(rng, H0, W0):
    kernel_layers = []
    ref_layers    = []
    keepalive     = []
    for li in range(8):
        in_d, out_d = SSLA_S_IN_OUT[li]
        K = SSLA_S_KERNELS[li]
        A = K * K
        n_pos_in = n_pos_out = (1 if K == 1 else A)
        Win  = rng.normal(scale=0.3, size=(in_d, n_pos_in * out_d)).astype(np.float32)
        Wout = rng.normal(scale=0.3, size=(n_pos_out * out_d, out_d)).astype(np.float32)
        qW = rng.normal(scale=0.3, size=(out_d, out_d)).astype(np.float32)
        vW = rng.normal(scale=0.3, size=(out_d, out_d)).astype(np.float32)
        gW = rng.normal(scale=0.3, size=(out_d, out_d)).astype(np.float32)
        oW = rng.normal(scale=0.3, size=(out_d, out_d)).astype(np.float32)
        qvgIn, goW = reshape_ssla_layer(Win, Wout, qW, vW, gW, oW, in_d, out_d)
        if in_d != out_d:
            input_proj = rng.normal(scale=0.3, size=(out_d, in_d)).astype(np.float32)
        else:
            input_proj = None
        ln_gamma = (rng.normal(scale=0.1, size=out_d) + 1.0).astype(np.float32)
        ln_beta  = rng.normal(scale=0.1, size=out_d).astype(np.float32)
        Hl, Wl = _stage_grid(H0, W0, li // 2)
        H_init = rng.normal(scale=0.05, size=(Hl * Wl, out_d)).astype(np.float32)

        # Transpose to (IN, OUT) layout for coalesced GPU matvec —
        # adjacent warp lanes access contiguous addresses.
        qvgIn_t = np.ascontiguousarray(qvgIn.transpose(0, 2, 1))   # (A, in_d, 3*out_d)
        goW_t   = np.ascontiguousarray(goW.transpose(0, 2, 1))     # (A, out_d, out_d) transposed
        p_qvg, _,  ka1 = _alloc_and_fill(qvgIn_t);  keepalive.append(ka1)
        p_go,  _,  ka2 = _alloc_and_fill(goW_t);    keepalive.append(ka2)
        if input_proj is not None:
            input_proj_t = np.ascontiguousarray(input_proj.T)      # (in_d, out_d)
            p_ip, _, ka3 = _alloc_and_fill(input_proj_t); keepalive.append(ka3)
        else:
            p_ip = 0
        p_lng, _,  ka4 = _alloc_and_fill(ln_gamma); keepalive.append(ka4)
        p_lnb, _,  ka5 = _alloc_and_fill(ln_beta);  keepalive.append(ka5)
        p_h, H_view, ka6 = _alloc_and_fill(H_init); keepalive.append(ka6)

        kernel_layers.append({
            "qvgIn": p_qvg, "goW": p_go, "input_proj": p_ip,
            "ln_gamma": p_lng, "ln_beta": p_lnb,
            "hidden_dev": p_h, "hidden_view": H_view,
        })
        ref_layers.append(LayerRef(
            qvgIn=qvgIn.copy(), goW=goW.copy(),
            input_proj=None if input_proj is None else input_proj.copy(),
            ln_gamma=ln_gamma.copy(), ln_beta=ln_beta.copy(),
            H=H_init.copy(),
            K=K,
        ))
    return kernel_layers, ref_layers, keepalive


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
    tdrop_window = 1   # all events pass — exercises every stage
    rng = np.random.default_rng(7)

    kernel_layers, ref_layers, keepalive = _build_layers(rng, H0, W0)

    # Tdrop counters in managed memory (3 stages).
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

    # StepConfig in managed memory.
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

    # Build the input ring with managed memory + push events.
    cap = 1 << 8   # 256 — power of 2, > n_events
    ring = Ring(cap, allocator="managed")
    events = np.empty(n_events, dtype=EVENT_DTYPE)
    events["t"] = rng.random(n_events).astype(np.float32)
    events["x"] = rng.integers(0, W0, n_events).astype(np.float32)
    events["y"] = rng.integers(0, H0, n_events).astype(np.float32)
    events["p"] = rng.integers(0, 2, n_events).astype(np.float32)
    pushed = ring.try_push_batch(events)
    assert pushed == n_events, f"only pushed {pushed}/{n_events}"

    # Output slot array in managed memory.
    out_bytes = ctypes.sizeof(COutputSlot) * n_events
    p_out, out_ka = cuda_util.alloc_managed(out_bytes)
    ctypes.memset(p_out, 0, out_bytes)
    keepalive.append(out_ka)

    ring_buf_dev, ring_head_dev, ring_tail_dev = ring.device_ptrs()

    func = mod.get_function("k_ssla_drain_n")
    SMEM = (2 + 96 + 5 * 96) * 4   # SSLA_STEP_SCRATCH_FLOATS(96) * sizeof(float)
    t0 = time.monotonic()
    func.launch((1,1,1), (256,1,1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_buf_dev),
        arg_u64(cap - 1),                   # ring mask
        arg_u64(0),                          # start index
        arg_i32(n_events),
        arg_ptr(p_out),
    ], smem=SMEM)
    func.sync()
    kernel_dt = time.monotonic() - t0

    # Read kernel outputs as a numpy structured array view.
    SlotArr = COutputSlot * n_events
    slots = SlotArr.from_address(p_out)

    # Compare each event to numpy reference.
    fails = 0
    max_err = 0.0
    pass_count = 0
    for i in range(n_events):
        ev = events[i]
        passed_r, s0r, s1r, s2r, s3r, txr, tyr = step_ref(
            StepRefState(H0=H0, W0=W0, tdrop_window=tdrop_window,
                         layers=ref_layers, tdrop=ref_tdrop),
            int(ev["x"]), int(ev["y"]), float(ev["t"]), float(ev["p"]))

        slot = slots[i]
        passed_k = bool(slot.passed)
        if passed_k != passed_r:
            print(f"  ev[{i}] FAIL: passed mismatch", flush=True)
            fails += 1
            continue
        if not passed_r:
            continue
        pass_count += 1

        # Compare touched cells.
        for k_ in range(4):
            if slot.touched_x[k_] != txr[k_] or slot.touched_y[k_] != tyr[k_]:
                print(f"  ev[{i}] FAIL: touched mismatch at stage {k_}", flush=True)
                fails += 1

        # Compare per-stage features.
        for k_, (slot_arr, s_ref, dim) in enumerate([
            (slot.s0, s0r, 12),
            (slot.s1, s1r, 24),
            (slot.s2, s2r, 48),
            (slot.s3, s3r, 96),
        ]):
            kbuf = np.frombuffer(slot_arr, dtype=np.float32, count=dim)
            err = float(np.max(np.abs(kbuf - s_ref)))
            max_err = max(max_err, err)

    # Final hidden state comparison.
    for li, (kl, ref_l) in enumerate(zip(kernel_layers, ref_layers)):
        diff = float(np.max(np.abs(kl["hidden_view"] - ref_l.H)))
        max_err = max(max_err, diff)

    # Free.
    cuda_util.free_managed(p_cfg)
    cuda_util.free_managed(p_out)
    for kl in kernel_layers:
        for k in ("qvgIn", "goW", "ln_gamma", "ln_beta", "hidden_dev"):
            cuda_util.free_managed(kl[k])
        if kl["input_proj"] != 0:
            cuda_util.free_managed(kl["input_proj"])
    for p in p_tdrop:
        cuda_util.free_managed(p)
    ring.close()

    atol = 1e-2
    ok = (fails == 0) and (max_err < atol)
    print(f"\nevents={n_events} passed={pass_count} failed={fails}  "
          f"max|Δ|={max_err:.2e}  atol={atol:.0e}  kernel={kernel_dt*1000:.1f}ms",
          flush=True)
    print("OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
