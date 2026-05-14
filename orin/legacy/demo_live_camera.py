"""Small live-camera demo for the S9-base persistent kernel.

  DVXplorer Micro  →  CameraProducer thread  →  pinned SPSC ring
                                                       │
                                                       ▼
                          single-block persistent SSLA-S kernel
                                                       │
                                                       ▼
                                            pinned output ring + counters
                                                       │
                                                       ▼
                              main thread polls counters, prints stats

Random SSLA-S weights — outputs are nonsense, the goal is to show the
plumbing runs end-to-end at whatever throughput the unoptimized
single-thread baseline supports.

Stop with Ctrl-C; the demo cleanly halts the kernel via stop_flag.
"""
import ctypes
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import (
    CudaModule, arg_ptr, arg_u32, arg_u64,  # noqa: E402
)
from orin.ring import EVENT_DTYPE, Ring, CameraProducer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_runner_drain import (  # noqa: E402
    CLayerWeights, CStepConfig, COutputSlot, _build_layers, _stage_grid,
)

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# DVXplorer Micro sensor resolution (matches deploy/orin README §1).
H0, W0 = 480, 640


class DemoCameraProducer(CameraProducer):
    """Override `_convert` to fold the timestamp into a safe `dt_norm`
    in [0, 1). Random weights would NaN on raw µs timestamps; a real
    runner will compute per-pixel dt against last-fired-t in a kernel
    preprocess pass.
    """
    def _convert(self, batch_np: np.ndarray) -> np.ndarray:
        out = np.empty(batch_np.shape[0], dtype=EVENT_DTYPE)
        ts = batch_np["timestamp"].astype(np.int64)
        out["t"] = ((ts % 100_000).astype(np.float32) / 100_000.0)
        out["x"] = batch_np["x"].astype(np.float32)
        out["y"] = batch_np["y"].astype(np.float32)
        out["p"] = batch_np["polarity"].astype(np.float32)
        return out


def main() -> int:
    src   = open(os.path.join(KERNELS_DIR, "ssla_step.cuh")).read()
    prim  = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
    layer = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()

    print(f"compiling kernel ({len(src)+len(prim)+len(layer)} bytes)...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_step.cu",
                     headers={"ssla_primitives.cuh": prim,
                              "ssla_layer.cuh":      layer})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s", flush=True)

    # ---- weights, hidden state, tdrop counters (managed) ----
    rng = np.random.default_rng(0)
    print(f"allocating SSLA-S random weights for {H0}x{W0}...", flush=True)
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
    cfg.H0, cfg.W0, cfg.tdrop_window = H0, W0, 4   # 1/64 events reach output
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

    # ---- pinned ring + outputs + control scalars (CPU+GPU concurrent) ----
    in_cap  = 1 << 14   # 16384 slots
    out_cap = 1 << 14
    ring = Ring(in_cap, allocator="pinned")
    ring_buf_dev, ring_head_dev, ring_tail_dev = ring.device_ptrs()

    p_out, out_ka = cuda_util.alloc_pinned(ctypes.sizeof(COutputSlot) * out_cap)
    p_stop, stop_ka = cuda_util.alloc_pinned(4)
    p_done, done_ka = cuda_util.alloc_pinned(8)
    p_outhead, outhead_ka = cuda_util.alloc_pinned(8)
    stop_view    = np.frombuffer(stop_ka,    dtype=np.int32).reshape(1)
    done_view    = np.frombuffer(done_ka,    dtype=np.uint64).reshape(1)
    outhead_view = np.frombuffer(outhead_ka, dtype=np.uint64).reshape(1)

    # ---- launch persistent kernel async ----
    func = mod.get_function("k_ssla_persistent_loop")
    SMEM = (2 + 96 + 5 * 96) * 4   # SSLA_STEP_SCRATCH_FLOATS(96) * sizeof(float)
    print("launching persistent kernel (async)...", flush=True)
    func.launch((1,1,1), (256,1,1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_buf_dev),
        arg_u64(in_cap - 1),
        arg_ptr(ring_head_dev), arg_ptr(ring_tail_dev),
        arg_ptr(p_out), arg_u64(out_cap - 1),
        arg_ptr(p_outhead),
        arg_ptr(p_stop), arg_ptr(p_done),
        arg_u32(2_000),   # 2 µs nanosleep when ring empty
    ], smem=SMEM)

    # ---- start camera producer ----
    print("opening DVXplorer + starting producer...", flush=True)
    producer = DemoCameraProducer(ring, block_on_full=False)
    producer.start()

    # ---- live stats loop ----
    interrupt = {"flag": False}
    def _sigint(_a, _b): interrupt["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    print(f"\nrunning — Ctrl-C to stop")
    print(f"  H={H0} W={W0}  tdrop_window=4 (1/64 reach stage-3 output)")
    print(f"  ring={in_cap} pinned slots, single-block persistent kernel")
    print()
    print(f"{'time':>5s}  {'in Mev/s':>9s}  {'drop %':>7s}  "
          f"{'kernel ev/s':>11s}  {'occ':>5s}  {'pushed':>9s}  {'done':>9s}")

    t_session = time.monotonic()
    last_t = t_session
    last_pushed = 0
    last_drops = 0
    last_done = 0
    try:
        while not interrupt["flag"]:
            time.sleep(1.0)
            now = time.monotonic()
            dt  = now - last_t
            pushed_total = producer.stats.events_pushed
            drops_total  = producer.stats.drops
            done_total   = int(done_view[0])
            d_pushed = pushed_total - last_pushed
            d_drops  = drops_total  - last_drops
            d_done   = done_total   - last_done
            d_in     = d_pushed + d_drops
            in_mev   = d_in / dt / 1e6
            drop_pct = 100.0 * d_drops / max(1, d_in)
            print(f"{now-t_session:5.1f}  {in_mev:9.2f}  {drop_pct:7.1f}  "
                  f"{d_done/dt:11.0f}  {ring.occupancy():5d}  "
                  f"{pushed_total:9d}  {done_total:9d}", flush=True)
            last_t = now
            last_pushed = pushed_total
            last_drops = drops_total
            last_done = done_total
    except KeyboardInterrupt:
        pass

    print("\nstopping...", flush=True)
    producer.stop()

    stop_view[0] = 1
    from cuda import cuda
    err, = cuda.cuCtxSynchronize()
    if err != cuda.CUresult.CUDA_SUCCESS:
        print(f"cuCtxSynchronize err: {err}", flush=True)

    print(f"final  in={producer.stats.events_pushed}  drops={producer.stats.drops}  "
          f"kernel_done={int(done_view[0])}  out_head={int(outhead_view[0])}",
          flush=True)

    # ---- free ----
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
