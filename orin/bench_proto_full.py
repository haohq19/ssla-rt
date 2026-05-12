"""Bench for the full-pipeline multi-warp persistent kernel with strict
spatial sharding.

Measures the five user-requested metrics:
  1. multi-warp speedup over single-warp fused prototype
  2. persistent-kernel overhead per event
  3. halo / sharding overhead
  4. full-pipeline worst-case latency
  5. full-pipeline steady-state throughput

Usage:
    python3 bench_proto_full.py --strips 1 --warps 1   # single-warp baseline
    python3 bench_proto_full.py --strips 1 --warps 4   # multi-warp single-block
    python3 bench_proto_full.py --strips 8 --warps 4   # sharded multi-warp
"""
import argparse
import ctypes
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import (
    CudaModule, arg_ptr, arg_u32,  # noqa: E402
)
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# SSLA-S widths
DIMS = (12, 24, 48, 96)
SSLA_S_IN_OUT = (
    (2, 12),  (12, 12),
    (12, 24), (24, 24),
    (24, 48), (48, 48),
    (48, 96), (96, 96),
)
SSLA_S_K = (1, 3, 3, 3, 3, 3, 3, 3)


# ===== ctypes mirrors of the kernel structs =====
class CLayerWeights(ctypes.Structure):
    _fields_ = [
        ("qvgIn",      ctypes.c_uint64),
        ("goW",        ctypes.c_uint64),
        ("input_proj", ctypes.c_uint64),
        ("ln_gamma",   ctypes.c_uint64),
        ("ln_beta",    ctypes.c_uint64),
    ]

class CStripConfig(ctypes.Structure):
    _fields_ = [
        ("Hs0",          ctypes.c_int32),
        ("Ws0",          ctypes.c_int32),
        ("strip_x_base", ctypes.c_int32),
        ("tdrop_window", ctypes.c_int32),
        ("layers",       CLayerWeights * 8),
        ("hidden",       ctypes.c_uint64 * 8),
        ("tdrop",        ctypes.c_uint64 * 3),
    ]

# EventRecW: 4 floats
class CEventRecW(ctypes.Structure):
    _fields_ = [("t", ctypes.c_float),
                ("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("p", ctypes.c_float)]

class COutputSlotW(ctypes.Structure):
    _fields_ = [("passed",     ctypes.c_int32),
                ("touched_x",  ctypes.c_int32 * 4),
                ("touched_y",  ctypes.c_int32 * 4),
                ("s0",         ctypes.c_float * 12),
                ("s1",         ctypes.c_float * 24),
                ("s2",         ctypes.c_float * 48),
                ("s3",         ctypes.c_float * 96)]


def _alloc_managed_array(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    p, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=arr.dtype).reshape(arr.shape)
    view[...] = arr
    return p, view, ka


def _alloc_pinned_zeroed(nbytes: int):
    p, ka = cuda_util.alloc_pinned(nbytes)
    ctypes.memset(p, 0, nbytes)
    return p, ka


def _build_shared_weights(rng, seed):
    """Build the 8 layers' weights (shared across all strips since SSLA-S
    weights are spatially-invariant within a layer)."""
    weights = []
    for li in range(8):
        in_d, out_d = SSLA_S_IN_OUT[li]
        K = SSLA_S_K[li]
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

        # Transpose to kernel layout (IN, OUT).
        qvgIn_t = np.ascontiguousarray(qvgIn.transpose(0, 2, 1))
        goW_t   = np.ascontiguousarray(goW.transpose(0, 2, 1))
        ip_t    = np.ascontiguousarray(input_proj.T) if input_proj is not None else None

        p_qvg, _, k1 = _alloc_managed_array(qvgIn_t)
        p_go,  _, k2 = _alloc_managed_array(goW_t)
        p_ip = 0; k3 = None
        if ip_t is not None:
            p_ip, _, k3 = _alloc_managed_array(ip_t)
        p_lng, _, k4 = _alloc_managed_array(ln_gamma)
        p_lnb, _, k5 = _alloc_managed_array(ln_beta)
        weights.append({"qvgIn": p_qvg, "goW": p_go, "input_proj": p_ip,
                        "ln_gamma": p_lng, "ln_beta": p_lnb,
                        "_keepalive": [k1, k2, k4, k5] + ([k3] if k3 else [])})
    return weights


def _build_strip(rng_init, weights, Hs0, Ws0, strip_x_base, tdrop_window):
    """Allocate per-strip hidden state + tdrop counters + pinned ring buffers."""
    keepalive = []
    cfg = CStripConfig()
    cfg.Hs0 = Hs0
    cfg.Ws0 = Ws0
    cfg.strip_x_base = strip_x_base
    cfg.tdrop_window = tdrop_window

    # Hidden state per layer.
    for li in range(8):
        _, out_d = SSLA_S_IN_OUT[li]
        stage = li // 2
        Hl = Hs0 >> stage
        Wl = Ws0 >> stage
        H_init = rng_init.normal(scale=0.05, size=(Hl * Wl, out_d)).astype(np.float32)
        p_h, _, k_h = _alloc_managed_array(H_init)
        keepalive.append(k_h)

        cfg.layers[li].qvgIn      = weights[li]["qvgIn"]
        cfg.layers[li].goW        = weights[li]["goW"]
        cfg.layers[li].input_proj = weights[li]["input_proj"]
        cfg.layers[li].ln_gamma   = weights[li]["ln_gamma"]
        cfg.layers[li].ln_beta    = weights[li]["ln_beta"]
        cfg.hidden[li]            = p_h

    # tdrop counters (3 stages, post-pool dims).
    for s in range(3):
        Hl = Hs0 >> (s + 1)
        Wl = Ws0 >> (s + 1)
        n_cells = Hl * Wl
        p_t, k_t = cuda_util.alloc_managed(n_cells)
        ctypes.memset(p_t, 0, n_cells)
        keepalive.append(k_t)
        cfg.tdrop[s] = p_t

    return cfg, keepalive


def _push_events(ring_buf_view, ring_head_view, n_to_push, events):
    """Single-threaded host push: write events into pinned ring + bump ring_head.
    Caller ensures ring is large enough."""
    cap = ring_buf_view.shape[0]
    head = int(ring_head_view[0])
    n = min(n_to_push, cap)   # never overflow
    for i in range(n):
        idx = (head + i) % cap
        ring_buf_view[idx] = events[i]
    ring_head_view[0] = head + n
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips",  type=int, default=1, help="1 = full image, 8 = sharded")
    ap.add_argument("--warps",   type=int, default=4, help="warps per block")
    ap.add_argument("--n",       type=int, default=4000)
    ap.add_argument("--tdrop",   type=int, default=4)
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--gpu-mhz", type=float, default=918.0)
    ap.add_argument("--mode",    type=str, default="uniform",
                    choices=["uniform", "hot", "replay"])
    args = ap.parse_args()

    H_full, W_full = 480, 640
    n_strips = args.strips
    Wstrip = W_full // n_strips
    n_warps = args.warps
    threads_per_block = n_warps * 32
    n_events = args.n

    cuda_util.ensure_context()

    src1 = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
    src2 = open(os.path.join(KERNELS_DIR, "proto_full.cuh")).read()
    print(f"=== compile proto_full.cuh ({len(src1)+len(src2)} bytes) ===", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src2, name="proto_full.cu",
                     headers={"proto_layer_pair.cuh": src1})
    print(f"  compile: {time.monotonic()-t0:.1f}s ({mod.cache_status})", flush=True)

    func = mod.get_function("k_proto_persistent_full")
    from cuda import cuda
    err, num_regs = cuda.cuFuncGetAttribute(
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, func._func)
    err, local_size = cuda.cuFuncGetAttribute(
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func._func)
    err, max_threads = cuda.cuFuncGetAttribute(
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func._func)
    print(f"  NUM_REGS={int(num_regs)}  LOCAL={int(local_size)}B  MAX_THREADS_PER_BLOCK={int(max_threads)}")

    # Shared weights across all strips.
    print(f"=== build shared weights ===", flush=True)
    rng = np.random.default_rng(args.seed)
    weights = _build_shared_weights(rng, args.seed)

    # Per-strip hidden state + tdrop.
    print(f"=== build {n_strips} strip(s) at {H_full}x{Wstrip} per strip ===", flush=True)
    rng_h = np.random.default_rng(args.seed + 100)
    strip_cfgs = []
    keepalive_strips = []
    for s in range(n_strips):
        cfg, ka = _build_strip(rng_h, weights, H_full, Wstrip, s * Wstrip, args.tdrop)
        strip_cfgs.append(cfg)
        keepalive_strips.append(ka)

    # Pinned per-strip rings + counters.
    ring_cap = 1
    while ring_cap < n_events // n_strips + 100:
        ring_cap <<= 1
    ring_cap = max(ring_cap, 1024)
    out_cap = ring_cap

    ring_buf_ptrs = []
    ring_buf_views = []
    ring_head_ptrs = []
    ring_head_views = []
    ring_tail_ptrs = []
    out_buf_ptrs = []
    out_head_ptrs = []
    events_done_ptrs = []
    events_done_views = []
    stripped_keepalive = []

    for s in range(n_strips):
        rb_size = ring_cap * ctypes.sizeof(CEventRecW)
        p_rb, ka_rb = cuda_util.alloc_pinned(rb_size)
        ctypes.memset(p_rb, 0, rb_size)
        rb_view = np.frombuffer(ka_rb, dtype=[("t","<f4"),("x","<f4"),("y","<f4"),("p","<f4")])
        ring_buf_ptrs.append(p_rb)
        ring_buf_views.append(rb_view)

        p_rh, ka_rh = _alloc_pinned_zeroed(8)
        rh_view = np.frombuffer(ka_rh, dtype=np.uint64)
        ring_head_ptrs.append(p_rh)
        ring_head_views.append(rh_view)

        p_rt, ka_rt = _alloc_pinned_zeroed(8)
        ring_tail_ptrs.append(p_rt)

        ob_size = out_cap * ctypes.sizeof(COutputSlotW)
        p_ob, ka_ob = _alloc_pinned_zeroed(ob_size)
        out_buf_ptrs.append(p_ob)

        p_oh, ka_oh = _alloc_pinned_zeroed(8)
        out_head_ptrs.append(p_oh)

        p_ed, ka_ed = _alloc_pinned_zeroed(8)
        ed_view = np.frombuffer(ka_ed, dtype=np.uint64)
        events_done_ptrs.append(p_ed)
        events_done_views.append(ed_view)

        stripped_keepalive.extend([ka_rb, ka_rh, ka_rt, ka_ob, ka_oh, ka_ed])

    p_stop, ka_stop = _alloc_pinned_zeroed(4)
    stop_view = np.frombuffer(ka_stop, dtype=np.int32)
    stripped_keepalive.append(ka_stop)

    # Per-block timing arrays.
    timing_ptrs = []
    timing_views = []
    for s in range(n_strips):
        p_tm, ka_tm = _alloc_pinned_zeroed(8 * out_cap)
        timing_ptrs.append(p_tm)
        timing_views.append(np.frombuffer(ka_tm, dtype=np.uint64).reshape(out_cap))
        stripped_keepalive.append(ka_tm)

    # ---- Pointer arrays (pinned, dereferenced by kernel) ----
    def _alloc_pinned_u64_array(values):
        n = len(values)
        p, ka = _alloc_pinned_zeroed(8 * n)
        view = np.frombuffer(ka, dtype=np.uint64).reshape(n)
        view[:] = np.array(values, dtype=np.uint64)
        return p, ka

    cfg_p, cfg_ka = _alloc_pinned_zeroed(ctypes.sizeof(CStripConfig) * n_strips)
    cfg_arr = (CStripConfig * n_strips).from_address(cfg_p)
    for s, c in enumerate(strip_cfgs):
        cfg_arr[s] = c
    stripped_keepalive.append(cfg_ka)

    p_ring_bufs, _    = _alloc_pinned_u64_array(ring_buf_ptrs)
    p_ring_masks, _   = _alloc_pinned_u64_array([ring_cap - 1] * n_strips)
    p_ring_heads, _   = _alloc_pinned_u64_array(ring_head_ptrs)
    p_ring_tails, _   = _alloc_pinned_u64_array(ring_tail_ptrs)
    p_out_bufs, _     = _alloc_pinned_u64_array(out_buf_ptrs)
    p_out_masks, _    = _alloc_pinned_u64_array([out_cap - 1] * n_strips)
    p_out_heads, _    = _alloc_pinned_u64_array(out_head_ptrs)
    p_events_done, _  = _alloc_pinned_u64_array(events_done_ptrs)
    p_timings, _      = _alloc_pinned_u64_array(timing_ptrs)

    # ---- Generate events ----
    if args.mode == "replay" and os.path.exists("/tmp/dvxplorer_replay.npy"):
        ev = np.load("/tmp/dvxplorer_replay.npy")
        # tile/clip to n_events
        if len(ev) < n_events:
            reps = (n_events + len(ev) - 1) // len(ev)
            ev = np.tile(ev, reps)
        ev = ev[:n_events]
        ts = (ev["t"] - ev["t"][0]).astype(np.int64)
        evt = ((ts % 100_000) / 100_000.0).astype(np.float32)
        evx = ev["x"].astype(np.float32)
        evy = ev["y"].astype(np.float32)
        evp = ev["p"].astype(np.float32)
    elif args.mode == "hot":
        rng_e = np.random.default_rng(args.seed + 7)
        evx = rng_e.integers(W_full // 4, W_full // 2, n_events).astype(np.float32)
        evy = rng_e.integers(H_full // 4, H_full // 2, n_events).astype(np.float32)
        evt = rng_e.random(n_events).astype(np.float32)
        evp = rng_e.integers(0, 2, n_events).astype(np.float32)
    else:
        rng_e = np.random.default_rng(args.seed + 7)
        evx = rng_e.integers(0, W_full, n_events).astype(np.float32)
        evy = rng_e.integers(0, H_full, n_events).astype(np.float32)
        evt = rng_e.random(n_events).astype(np.float32)
        evp = rng_e.integers(0, 2, n_events).astype(np.float32)

    # Route to strips by global x.
    strip_ids = (evx.astype(np.int32) // Wstrip).clip(0, n_strips - 1)
    per_strip_counts = np.bincount(strip_ids, minlength=n_strips)
    print(f"  strip event counts: {per_strip_counts.tolist()}  (max/mean = "
          f"{per_strip_counts.max()/max(1,per_strip_counts.mean()):.2f}x)")

    # Pre-build per-strip event arrays (strip-local x).
    # Pad each strip to a multiple of n_warps so the kernel's batched
    # consumption never deadlocks at the tail (out-of-bounds events
    # short-circuit in ssla_event_w).
    per_strip_events = []
    per_strip_targets = []
    for s in range(n_strips):
        mask = strip_ids == s
        n_real = int(mask.sum())
        pad = (-n_real) % n_warps
        n_s = n_real + pad
        arr = np.empty(n_s, dtype=[("t","<f4"),("x","<f4"),("y","<f4"),("p","<f4")])
        arr["t"][:n_real] = evt[mask]
        arr["x"][:n_real] = (evx[mask] - s * Wstrip)
        arr["y"][:n_real] = evy[mask]
        arr["p"][:n_real] = evp[mask]
        # Padding events: out-of-bounds → ssla_event_w returns false immediately.
        if pad:
            arr["t"][n_real:] = 0.0
            arr["x"][n_real:] = -1.0
            arr["y"][n_real:] = -1.0
            arr["p"][n_real:] = 0.0
        per_strip_events.append(arr)
        per_strip_targets.append(n_s)

    # ---- Pre-fill rings (push all events) ----
    for s in range(n_strips):
        ev_s = per_strip_events[s]
        n_pushed = _push_events(ring_buf_views[s], ring_head_views[s], len(ev_s), ev_s)
        assert n_pushed == len(ev_s), (n_pushed, len(ev_s), ring_cap)

    # ---- Launch persistent kernel async ----
    SMEM = n_warps * 192 * 4    # SCRATCH_PER_WARP_FLOATS × n_warps × 4 bytes
    print(f"\n=== launch persistent kernel (strips={n_strips}, warps/block={n_warps}, "
          f"threads/block={threads_per_block}, smem={SMEM}B) ===", flush=True)

    t_launch = time.monotonic()
    func.launch((n_strips, 1, 1), (threads_per_block, 1, 1), [
        arg_ptr(cfg_p),
        arg_ptr(p_ring_bufs),
        arg_ptr(p_ring_masks),
        arg_ptr(p_ring_heads),
        arg_ptr(p_ring_tails),
        arg_ptr(p_out_bufs),
        arg_ptr(p_out_masks),
        arg_ptr(p_out_heads),
        arg_ptr(p_stop),
        arg_ptr(p_events_done),
        arg_u32(2_000),
        arg_ptr(p_timings),
    ], smem=SMEM)

    # ---- Poll until all strips done ----
    target_per_strip = per_strip_targets
    poll_start = time.monotonic()
    deadline = poll_start + 60.0   # 60s timeout
    while True:
        all_done = True
        for s in range(n_strips):
            done = int(events_done_views[s][0])
            if done < target_per_strip[s]:
                all_done = False
                break
        if all_done:
            break
        if time.monotonic() > deadline:
            print(f"  TIMEOUT — events_done = {[int(v[0]) for v in events_done_views]}, target = {target_per_strip}", flush=True)
            break
        time.sleep(0.001)
    drain_time = time.monotonic() - poll_start

    stop_view[0] = 1
    err, = cuda.cuCtxSynchronize()
    sync_err = (err != cuda.CUresult.CUDA_SUCCESS)

    # Throughput numbers — per-strip + aggregate.
    e_done = [int(v[0]) for v in events_done_views]
    total_done = sum(e_done)
    overall_evps = total_done / drain_time if drain_time > 0 else 0
    per_event_us = (drain_time * 1e6 / total_done) if total_done else 0
    print(f"  events_done per strip: {e_done}")
    print(f"  drain time: {drain_time*1000:.1f} ms  total events: {total_done}")
    print(f"  aggregate throughput: {overall_evps:.0f} ev/s = {overall_evps/1e6:.4f} Mev/s")
    print(f"  per-event aggregate: {per_event_us:.2f} µs (= {drain_time*1e6/total_done:.2f})")

    # Per-event GPU cycles (lane 0 of each warp records its own event).
    all_cycles = []
    for s in range(n_strips):
        view = timing_views[s][:e_done[s]]
        view = view[view > 0]   # drop unwritten slots
        all_cycles.append(view)
    if all_cycles:
        cycles = np.concatenate(all_cycles)
        us = cycles.astype(np.float64) / args.gpu_mhz
        print(f"  per-event kernel-only latency (clock64 inside warp, {args.gpu_mhz:.0f} MHz):")
        print(f"    p50 = {np.percentile(us, 50):.1f} µs")
        print(f"    p90 = {np.percentile(us, 90):.1f} µs")
        print(f"    p99 = {np.percentile(us, 99):.1f} µs")
        print(f"    max = {us.max():.1f} µs")
        print(f"    min = {us.min():.1f} µs")
        print(f"    mean= {us.mean():.1f} µs")

    if sync_err:
        print(f"  ERROR: cuCtxSynchronize returned err")
        return 2

    # Cleanup
    for kalist in keepalive_strips:
        for ka in kalist: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
