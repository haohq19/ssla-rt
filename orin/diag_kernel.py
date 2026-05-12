"""Kernel diagnostics: register usage, shared/local memory footprint, and
per-event GPU-duration distribution from k_ssla_drain_n.

Reports:
  - cuFuncGetAttribute for each kernel of interest:
      NUM_REGS, MAX_THREADS_PER_BLOCK, SHARED_SIZE_BYTES,
      LOCAL_SIZE_BYTES (= per-thread spill bucket — non-zero => register
      spills to local memory), CACHE_MODE_CA
  - Per-event GPU-clock duration (clock64() at start + end of each
    event's processing in a modified drain kernel), converted to µs
    using GPU SM frequency. Reports p50, p90, p99, max.

This script does NOT modify the existing kernels; it compiles a separate
timing variant inline (k_ssla_drain_n_timed) that wraps ssla_step_ct
with clock64() bracketing.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_runner_drain import (  # noqa: E402
    CLayerWeights, CStepConfig, COutputSlot, _build_layers, _stage_grid,
)

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# Inline timing kernel: same as k_ssla_drain_n but with clock64() bracketing.
TIMING_SHIM = r"""
extern "C" __global__ void k_ssla_drain_n_timed(
    const StepConfig*    cfg,
    const EventRec*      ring,
    unsigned long long   ring_mask,
    unsigned long long   start,
    int                  n,
    OutputSlot*          out,
    unsigned long long*  timings)         // (n,) — GPU clock cycles per event
{
    if (blockIdx.x != 0) return;
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        const EventRec& ev = ring[(start + (unsigned long long)i) & ring_mask];
        OutputSlot* slot = &out[i];
        unsigned long long t0 = 0;
        if (tid == 0) t0 = clock64();
        __syncthreads();
        bool passed = ssla_step_ct<12, 24, 48, 96>(
            *cfg, (int)ev.x, (int)ev.y, ev.t, ev.p,
            slot->s0, slot->s1, slot->s2, slot->s3,
            slot->touched_x, slot->touched_y, smem);
        __syncthreads();
        if (tid == 0) {
            slot->passed = passed ? 1 : 0;
            timings[i] = clock64() - t0;
        }
        __syncthreads();
    }
}
"""


def _query_func_attrs(func, name):
    from cuda import cuda
    attrs_to_query = {
        "NUM_REGS": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
        "MAX_THREADS_PER_BLOCK": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        "SHARED_SIZE_BYTES": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        "LOCAL_SIZE_BYTES": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        "CONST_SIZE_BYTES": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        "PTX_VERSION": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION,
        "BINARY_VERSION": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION,
    }
    out = {}
    for k, attr in attrs_to_query.items():
        err, val = cuda.cuFuncGetAttribute(attr, func)
        if err == cuda.CUresult.CUDA_SUCCESS:
            out[k] = int(val)
        else:
            out[k] = f"err({err})"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gpu-mhz", type=float, default=918.0,
                    help="SM clock for cycle→time conversion")
    args = ap.parse_args()

    H0, W0 = 480, 640
    n = args.n

    src   = open(os.path.join(KERNELS_DIR, "ssla_step.cuh")).read()
    prim  = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
    layer = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()
    src_with_timed = src + "\n" + TIMING_SHIM

    print(f"=== compile ({len(src_with_timed)+len(prim)+len(layer)} bytes) ===", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src_with_timed, name="ssla_step.cu",
                     headers={"ssla_primitives.cuh": prim,
                              "ssla_layer.cuh":      layer})
    print(f"  compile: {time.monotonic()-t0:.1f}s ({mod.cache_status})", flush=True)

    # Query function attrs for all kernels we care about.
    func_drain  = mod.get_function("k_ssla_drain_n")
    func_timed  = mod.get_function("k_ssla_drain_n_timed")
    func_persistent = mod.get_function("k_ssla_persistent_loop")

    print("\n=== kernel attributes (cuFuncGetAttribute) ===")
    for name, f in [("k_ssla_drain_n", func_drain._func),
                    ("k_ssla_drain_n_timed", func_timed._func),
                    ("k_ssla_persistent_loop", func_persistent._func)]:
        a = _query_func_attrs(f, name)
        print(f"  {name}:")
        for k, v in a.items():
            print(f"    {k:24s} = {v}")
        # Spill check
        if isinstance(a.get("LOCAL_SIZE_BYTES"), int) and a["LOCAL_SIZE_BYTES"] > 0:
            print(f"    >>> SPILLS to local memory: {a['LOCAL_SIZE_BYTES']} bytes/thread")
        else:
            print(f"    >>> no register spills")

    # Build weights + run timed bench.
    rng = np.random.default_rng(0)
    print(f"\n=== build weights at {H0}x{W0} ===", flush=True)
    kernel_layers, _ref_layers, keepalive = _build_layers(rng, H0, W0)

    p_tdrop = []
    for s in range(3):
        Hl, Wl = _stage_grid(H0, W0, s + 1)
        n_cells = Hl * Wl
        p, ka = cuda_util.alloc_managed(n_cells)
        ctypes.memset(p, 0, n_cells)
        keepalive.append(ka)
        p_tdrop.append(p)

    cfg = CStepConfig()
    cfg.H0, cfg.W0, cfg.tdrop_window = H0, W0, 4
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

    events = np.empty(n, dtype=EVENT_DTYPE)
    events["t"] = rng.random(n).astype(np.float32)
    events["x"] = rng.integers(0, W0, n).astype(np.float32)
    events["y"] = rng.integers(0, H0, n).astype(np.float32)
    events["p"] = rng.integers(0, 2, n).astype(np.float32)

    cap = 1
    while cap < n: cap <<= 1
    ring = Ring(cap, allocator="pinned")
    ring.try_push_batch(events)
    ring_buf_dev, _, _ = ring.device_ptrs()

    p_out, out_ka = cuda_util.alloc_pinned(ctypes.sizeof(COutputSlot) * n)
    keepalive.append(out_ka)

    p_t, t_ka = cuda_util.alloc_pinned(8 * n)
    keepalive.append(t_ka)
    timings_view = np.frombuffer(t_ka, dtype=np.uint64).reshape(n)

    SMEM = (2 + 96 + 5 * 96) * 4

    # Pre-warmup so the GPU governor has spun the SM clock up to max.
    for _ in range(3):
        func_timed.launch((1,1,1), (256,1,1), [
            arg_ptr(p_cfg), arg_ptr(ring_buf_dev), arg_u64(cap-1),
            arg_u64(0), arg_i32(min(50, n)), arg_ptr(p_out), arg_ptr(p_t),
        ], smem=SMEM)
        func_timed.sync()

    # Reset hidden state + tdrop after warmup.
    for kl in kernel_layers:
        kl["hidden_view"].fill(0.0)
    for s, p in enumerate(p_tdrop):
        Hl, Wl = _stage_grid(H0, W0, s + 1)
        ctypes.memset(p, 0, Hl * Wl)

    print(f"\n=== bench: drain n={n} events ===", flush=True)
    t0 = time.monotonic()
    func_timed.launch((1,1,1), (256,1,1), [
        arg_ptr(p_cfg), arg_ptr(ring_buf_dev), arg_u64(cap-1),
        arg_u64(0), arg_i32(n), arg_ptr(p_out), arg_ptr(p_t),
    ], smem=SMEM)
    func_timed.sync()
    wall_dt = time.monotonic() - t0
    print(f"  wall time     {wall_dt*1000:.1f} ms")
    print(f"  throughput    {n/wall_dt:.0f} ev/s = {n/wall_dt/1e6:.4f} Mev/s")

    # Per-event latency distribution.
    cycles = timings_view.copy().astype(np.int64)
    us = cycles.astype(np.float64) / (args.gpu_mhz * 1.0)
    print(f"\n=== per-event GPU cycle counts (assuming SM clock = {args.gpu_mhz:.0f} MHz) ===")
    pct = lambda p: float(np.percentile(us, p))
    print(f"  count            {len(us)}")
    print(f"  mean             {us.mean():9.2f} µs")
    print(f"  p50              {pct(50):9.2f} µs")
    print(f"  p90              {pct(90):9.2f} µs")
    print(f"  p99              {pct(99):9.2f} µs")
    print(f"  max              {us.max():9.2f} µs")
    print(f"  min              {us.min():9.2f} µs")

    SlotArr = COutputSlot * n
    slots = SlotArr.from_address(p_out)
    passed_count = sum(1 for i in range(n) if slots[i].passed)
    print(f"  passed           {passed_count}/{n} = {100*passed_count/n:.2f}%")

    # Cleanup
    cuda_util.free_managed(p_cfg)
    cuda_util.free_pinned(p_out)
    cuda_util.free_pinned(p_t)
    for kl in kernel_layers:
        for k in ("qvgIn", "goW", "ln_gamma", "ln_beta", "hidden_dev"):
            cuda_util.free_managed(kl[k])
        if kl["input_proj"] != 0:
            cuda_util.free_managed(kl["input_proj"])
    for p in p_tdrop:
        cuda_util.free_managed(p)
    ring.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
