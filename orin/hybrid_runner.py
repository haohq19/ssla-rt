"""SSLA Hybrid live driver — DVXplorer → CPU stage 0+1 → GPU stage 2+3.

End-to-end P2 driver. Runs:
  1. dv_processing camera reader thread → libstage01_to_gpu.so submit
  2. libstage01_to_gpu.so: 4 CPU shards, halo=2, owner-pass-through to
     pinned-host MPSC rings (one per GPU block)
  3. GPU persistent kernel `k_ssla_s2s3_persistent`: 2 blocks, coop-per-event,
     poll their per-block ring forever until *stop_flag != 0
  4. Hidden state lives in managed memory; tdrop counters in managed memory;
     ring buffers + atomic counters + stop_flag in pinned host memory
     (CONCURRENT_MANAGED_ACCESS = 0 on Tegra → managed not safe for
     live concurrent CPU/GPU writes; pinned is).

Usage (run for 30 s on live camera):
    python3 deploy/orin/hybrid_runner.py --weights /tmp/ssla_s_64x80/

The weights dir must contain weights.npz + meta.json with height / width
matching --h-full / --w-full (default 64×80, post-EVERY_EIGHTH camera res).
GPU-side weights are independently random-initialised here for P2's
non-zero-prediction acceptance gate; matching the CPU's weights is a
P3 concern.

Acceptance gate (HYBRID_DESIGN.md §6 P2): runs 30 s without deadlock,
both blocks write non-zero predictions / advance their tail.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import datetime as _dt
import os
import signal
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import (  # noqa: E402
    CudaModule, arg_ptr, arg_i32, arg_u32, arg_u64,
)
from orin.hybrid_common import (  # noqa: E402
    INPUT_DTYPE, C1, C2, C3, N_BLOCKS, HEAD_OUT_DEFAULT, TIMING_DTYPE,
    CHybridS2S3Config,
    build_random_layers, build_config, alloc_tdrop_counters,
    build_head_weights, alloc_predictions, alloc_timing, alloc_kernel_start_clk,
    block_proc_range, S01gAPI,
)


# Orin NX SM clock at MAXN power mode. STATUS.md confirms 918 MHz from
# t = 1.2 s onward (warmup ramps from 306 → 714 → 918). Used to convert
# clock64() ticks to nanoseconds. ~5 % error if the governor scales
# back during the run.
SM_CLK_HZ = 918_000_000


KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
LIB_PATH    = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "build", "libstage01_to_gpu.so"))


# ---------------------------------------------------------------------------
# Pinned-host buffer wrapper. On Tegra unified memory the host pointer
# returned by cuMemHostAlloc IS the device pointer — same value works for
# kernel args and for ctypes / numpy host writes.
# ---------------------------------------------------------------------------

def _alloc_u64() -> int:
    """Pinned host alloc of a single uint64, zeroed."""
    p, _ = cuda_util.alloc_pinned(8)
    ctypes.memset(p, 0, 8)
    return p


def _read_u64(devptr: int) -> int:
    return int.from_bytes(
        bytes((ctypes.c_ubyte * 8).from_address(devptr)), "little")


def _write_i32(devptr: int, v: int) -> None:
    ctypes.memmove(devptr, ctypes.byref(ctypes.c_int32(v)), 4)


# ---------------------------------------------------------------------------
# Camera reader thread
# ---------------------------------------------------------------------------

class SyntheticReader(threading.Thread):
    """Drop-in replacement for CameraReader that pushes random events at a
    target rate via s01g_submit_batch — for saturation testing without a
    real camera. Coordinates: full-res H_full × W_full (the lib's grid)."""

    def __init__(self, api: S01gAPI, *, target_mev: float,
                 H_full: int, W_full: int, seed: int,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.api = api
        self.target_mev = target_mev
        self.H_full = H_full
        self.W_full = W_full
        self.rng = np.random.default_rng(seed)
        self.stop_event = stop_event
        self.events_in = 0
        self.batches_in = 0

    def run(self):
        # Generate ~1 ms worth of events per batch; pace via wall clock.
        target_eps = self.target_mev * 1e6
        batch_n = max(1, int(target_eps * 1e-3))
        period_s = batch_n / target_eps if target_eps > 0 else 0.001
        t_next = time.monotonic()
        t_us = 0.0
        dt_us_per_batch = period_s * 1e6
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now < t_next:
                time.sleep(min(t_next - now, 0.001))
                continue
            packed = np.empty((batch_n, 4), dtype=np.float32, order="C")
            packed[:, 0] = (t_us + np.arange(batch_n, dtype=np.float32)
                            * (dt_us_per_batch / batch_n))
            packed[:, 1] = self.rng.integers(0, self.W_full, batch_n
                                              ).astype(np.float32)
            packed[:, 2] = self.rng.integers(0, self.H_full, batch_n
                                              ).astype(np.float32)
            packed[:, 3] = self.rng.integers(0, 2, batch_n).astype(np.float32)
            self.events_in += self.api.submit(packed)
            self.batches_in += 1
            t_us  += dt_us_per_batch
            t_next += period_s


class CameraReader(threading.Thread):
    """Pulls batches from dv_processing, packs into the (n, 4) float32 layout
    libstage01_to_gpu expects, calls s01g_submit_batch."""

    def __init__(self, api: S01gAPI, *, cam_interval_us: int,
                 contrast: int, subsample: str, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.api = api
        self.cam_interval_us = cam_interval_us
        self.contrast = contrast
        self.subsample = subsample
        self.stop_event = stop_event
        self.events_in = 0
        self.batches_in = 0
        self.cam = None

    def run(self):
        import dv_processing as dv
        self.cam = dv.io.camera.open()
        W_cam, H_cam = self.cam.getEventResolution()
        print(f"[cam] opened {self.cam.getCameraName()} "
              f"(serial {self.cam.getSerialNumber()})  resolution={W_cam}×{H_cam}",
              flush=True)
        SS = dv.io.camera.DVXplorer.SubSample
        ss_val = getattr(SS, self.subsample)
        try:
            self.cam.setSubSampleHorizontal(ss_val)
            self.cam.setSubSampleVertical(ss_val)
            self.cam.setContrastThresholdOn(self.contrast)
            self.cam.setContrastThresholdOff(self.contrast)
            self.cam.setTimeInterval(_dt.timedelta(microseconds=self.cam_interval_us))
            self.cam.setMIPITimeoutValue(_dt.timedelta(microseconds=self.cam_interval_us))
        except Exception as e:
            print(f"[cam] knob set failed: {e}", flush=True)
        # Subsample x scale: events come back in original sensor coords; we
        # downsample by the subsample factor and feed coords compacted to
        # the lib's H_full × W_full grid.
        ladder = {"EVERY_PIXEL": 1, "EVERY_SECOND": 2, "EVERY_FOURTH": 4,
                  "EVERY_EIGHTH": 8}
        ds = ladder.get(self.subsample, 8)
        # Camera sends t as int64 µs; keep an epoch so float32 has resolution.
        t0 = None
        while not self.stop_event.is_set():
            if not self.cam.isRunning():
                time.sleep(0.001)
                continue
            batch = self.cam.getNextEventBatch()
            if batch is None or batch.size() == 0:
                time.sleep(0.0005)
                continue
            ev = batch.numpy()
            if t0 is None:
                t0 = int(ev["timestamp"][0])
            n = ev.size
            packed = np.empty((n, 4), dtype=np.float32, order="C")
            packed[:, 0] = (ev["timestamp"] - t0).astype(np.float32)
            packed[:, 1] = (ev["x"].astype(np.int32) // ds).astype(np.float32)
            packed[:, 2] = (ev["y"].astype(np.int32) // ds).astype(np.float32)
            packed[:, 3] = ev["polarity"].astype(np.float32)
            accepted = self.api.submit(packed)
            self.events_in += accepted
            self.batches_in += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="weights export dir for the CPU-side SSLA pipeline "
                         "(weights.npz + meta.json with H, W matching --h-full, --w-full)")
    ap.add_argument("--h-full", type=int, default=64,
                    help="full-res H. After 2 pools (s0+s1) gives s2 H. "
                         "Must be even-divisible by 4 to keep s2/s3 even.")
    ap.add_argument("--w-full", type=int, default=80)
    ap.add_argument("--shards", type=int, default=4,
                    help="CPU shard count for libstage01_to_gpu")
    ap.add_argument("--halo", type=int, default=2)
    ap.add_argument("--base-core", type=int, default=0)
    ap.add_argument("--pin-python-main", action="store_true",
                    help="Pin Python main + cam_thread to core 0 so C++ "
                         "workers (which self-pin to base_core..base_core+N) "
                         "don't share cores with the dispatcher")
    ap.add_argument("--python-pin-core", type=int, default=0,
                    help="If --pin-python-main, pin Python main thread to "
                         "this core (default 0).")
    ap.add_argument("--cpp-synth", action="store_true",
                    help="Use C++ synthetic dispatcher (tsc-paced) instead of "
                         "Python SyntheticReader. Eliminates time.sleep ms "
                         "jitter and GIL interference. Caller specifies a pin "
                         "core via --synth-pin-core.")
    ap.add_argument("--synth-pin-core", type=int, default=-1,
                    help="Core to pin C++ synth dispatcher thread to. -1 = "
                         "no pin (lets OS schedule). Recommended: pick a core "
                         "outside base_core..base_core+shards range.")
    ap.add_argument("--shard-ring-cap", type=int, default=1 << 16,
                    help="Per-shard SPSC ring capacity (power of 2, ≥ 4). "
                         "Smaller = lower max queue latency, but more "
                         "back-pressure on the dispatcher. At 1.8 Mev/s admit "
                         "& 3 µs/msg, 65536 → ~200 ms worst-case queue wait; "
                         "64 → ~200 µs; 16 → ~50 µs.")
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--ring-cap", type=int, default=1 << 16,
                    help="per-block ring capacity (power of 2)")
    ap.add_argument("--spin-ns", type=int, default=200,
                    help="GPU back-off __nanosleep when ring empty")
    ap.add_argument("--cam-interval-us", type=int, default=2000)
    ap.add_argument("--contrast", type=int, default=40)
    ap.add_argument("--subsample", type=str, default="EVERY_EIGHTH",
                    choices=["EVERY_PIXEL", "EVERY_SECOND", "EVERY_FOURTH", "EVERY_EIGHTH"])
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stats-interval-s", type=float, default=1.0)
    ap.add_argument("--timing-cap", type=int, default=1 << 14,
                    help="GPU timing-slot ring capacity (power of 2). "
                         "Last N events' timings retained per block.")
    ap.add_argument("--synthetic-mev", type=float, default=0.0,
                    help="If > 0, skip camera and feed synthetic events at "
                         "this rate (Mev/s) for saturation testing.")
    ap.add_argument("--kernel-variant", type=str, default="coop",
                    choices=["coop", "celled"],
                    help="coop = block-cooperative (ssla_s2_s3_head.cuh); "
                         "celled = cell-owner warps (ssla_s2_s3_head_celled.cuh)")
    args = ap.parse_args()

    # Pin the Python main thread (+ inherited by cam_thread and stats poller)
    # to core 0. C++ shard workers self-pin to base_core..base_core+N via
    # pthread_setaffinity_np inside shard_worker, escaping this mask. This
    # keeps the dispatcher (Python) off the worker cores when --base-core ≥ 1.
    if args.pin_python_main:
        import os
        os.sched_setaffinity(0, {args.python_pin_core})

    if args.ring_cap & (args.ring_cap - 1):
        sys.exit(f"--ring-cap must be power of 2, got {args.ring_cap}")
    if args.h_full % 4 or args.w_full % 4:
        sys.exit("--h-full and --w-full must be divisible by 4 (two pools).")

    H_full, W_full = args.h_full, args.w_full
    H2, W2 = H_full >> 2, W_full >> 2
    H3, W3 = H2 >> 1,    W2 >> 1
    print(f"Geometry: full {W_full}×{H_full}  s2 {W2}×{H2}  s3 {W3}×{H3}", flush=True)
    proc_lo, proc_hi = block_proc_range(W2, halo_s2=args.halo)
    print(f"Strip layout: block 0 owned [0,{W2//N_BLOCKS}) proc [{proc_lo[0]},{proc_hi[0]}); "
          f"block 1 owned [{W2//N_BLOCKS},{W2}) proc [{proc_lo[1]},{proc_hi[1]})",
          flush=True)

    # ---- 1. Build GPU weights + hidden state + tdrop counters ----------
    rng = np.random.default_rng(args.seed)
    cpu_layers, gpu_layers, ka_layers = build_random_layers(rng, H2, W2, H3, W3)
    tdrop_s2_dev, tdrop_s3_dev, ka_tdrop = alloc_tdrop_counters(H2, W2, H3, W3)
    head_out_dim = HEAD_OUT_DEFAULT
    head_W_dev, head_b_dev, ka_head = build_head_weights(rng, head_out_dim)
    preds_dev, preds_view, version_dev, version_view, ka_preds = alloc_predictions(
        H3, W2, head_out_dim)
    timing_dev, timing_view, timing_mask, ka_timing = alloc_timing(
        args.timing_cap)
    kstart_dev, kstart_view, ka_kstart = alloc_kernel_start_clk()
    kend_dev,   kend_view,   ka_kend   = alloc_kernel_start_clk()
    p_cfg, ka_cfg = build_config(
        H2, W2, H3, W3, args.tdrop,
        gpu_layers, tdrop_s2_dev, tdrop_s3_dev,
        head_W=head_W_dev, head_b=head_b_dev, head_out_dim=head_out_dim,
        preds=(preds_dev[0], preds_dev[1]),
        version=(version_dev[0], version_dev[1]),
        timing=(timing_dev[0], timing_dev[1]),
        timing_mask=timing_mask,
        kernel_start_clk=(kstart_dev[0], kstart_dev[1]),
        kernel_end_clk=(kend_dev[0], kend_dev[1]),
    )

    # ---- 2. Pinned host memory for rings + control --------------------
    rec_bytes = INPUT_DTYPE.itemsize          # 112
    ring_nbytes = args.ring_cap * rec_bytes
    print(f"Pinned alloc: 2 × ring ({ring_nbytes/1e6:.1f} MB each), "
          f"6 × u64, 1 × i32", flush=True)
    ring_dev   = []
    ring_view  = []
    ring_head  = []   # producer side, multi-producer atomic
    ring_tail  = []   # consumer side (GPU writes)
    events_done = []  # GPU mirror of tail, host-readable
    for b in range(N_BLOCKS):
        p, ka = cuda_util.alloc_pinned(ring_nbytes)
        ctypes.memset(p, 0, ring_nbytes)
        ring_dev.append(p)
        ring_view.append(np.frombuffer(ka, dtype=INPUT_DTYPE).reshape(-1))
        ring_head.append(_alloc_u64())
        ring_tail.append(_alloc_u64())
        events_done.append(_alloc_u64())
    stop_flag, _ka_stop = cuda_util.alloc_pinned(4)
    ctypes.memset(stop_flag, 0, 4)

    # ---- 3. Compile + launch persistent kernel ------------------------
    if args.kernel_variant == "coop":
        src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head.cuh")).read()
        prim  = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
        layer = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()
        headers = {"ssla_primitives.cuh": prim, "ssla_layer.cuh": layer}
        mod_name   = "ssla_s2_s3_head.cu"
        func_name  = "k_ssla_s2s3_persistent"
        threads_pb = 256
        SMEM       = (C1 + 2 * C3 + 5 * C3) * 4    # 2784 B
    else:  # celled
        src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head_celled.cuh")).read()
        proto = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
        headers = {"proto_layer_pair.cuh": proto}
        mod_name   = "ssla_s2_s3_head_celled.cu"
        func_name  = "k_ssla_s2s3_celled_persistent"
        threads_pb = 9 * 32    # = 288 (9 cell-owner warps)
        # SMEM layout must match bench_s2_s3_head_celled.py::_smem_size_bytes
        BATCH_K = 8
        N_WARPS_K = 9
        OUT_MAX_K = C3
        C1_K, C2_K = 24, 48
        event_slot = OUT_MAX_K * 4 + OUT_MAX_K * 4 + 5 * 4 + 4 + 8 + 8  # 5 ints + pad + 2 u64 (t_push_ns, t_emit_ns)
        SMEM = (event_slot * BATCH_K
                + BATCH_K * N_WARPS_K * OUT_MAX_K * 4
                + N_WARPS_K * OUT_MAX_K * 4
                + N_WARPS_K * OUT_MAX_K * 4
                + N_WARPS_K * BATCH_K * 4
                + N_WARPS_K * BATCH_K * 4
                + N_WARPS_K * 4
                + BATCH_K * 4
                + BATCH_K * 4
                + 9 * 3 * C2_K * C1_K * 4)   # L4 qvgIn smem cache (121 KB)
        SMEM = ((SMEM + 15) // 16) * 16
    print(f"Compiling {args.kernel_variant} kernel ...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name=mod_name, headers=headers)
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s", flush=True)
    func = mod.get_function(func_name)
    # Opt-in to dynamic smem >48 KB (sm_87 supports up to ~167 KB / block).
    if SMEM > 48 * 1024:
        from cuda import cuda as _cuda
        _err, = _cuda.cuFuncSetAttribute(
            func._func,
            _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM)
        if int(_err) != 0:
            raise RuntimeError(f"cuFuncSetAttribute(MAX_DYNAMIC_SHARED): err={int(_err)}, smem={SMEM}")
    print(f"Launching persistent kernel: gridDim=2, blockDim={threads_pb}, "
          f"smem={SMEM} B", flush=True)
    # CPU/GPU clock-anchor for end-to-end latency. We record CPU monotonic
    # right before cuLaunch, then spin-poll kernel_start_clk after launch
    # until the GPU writes its clock64() at kernel entry. The kernel-entry
    # event happens between t_launch_cpu_ns and t_anchor_cpu_ns; we use
    # t_anchor_cpu_ns (the observation time) as the anchor — its bias from
    # the true entry time is bounded by the spin-poll latency (~µs).
    t_launch_cpu_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    func.launch((2, 1, 1), (threads_pb, 1, 1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_dev[0]),
        arg_ptr(ring_dev[1]),
        arg_u64(args.ring_cap - 1),
        arg_ptr(ring_tail[0]),
        arg_ptr(ring_tail[1]),
        arg_ptr(stop_flag),
        arg_ptr(events_done[0]),
        arg_ptr(events_done[1]),
        arg_u32(args.spin_ns),
    ], smem=SMEM)
    # No sync — kernel runs in background until *stop_flag != 0.

    # Busy-spin (NOT time.sleep) until both blocks have written their
    # kernel-entry clock64. time.sleep on a non-RT process has ms-scale
    # granularity, which corrupts the anchor by milliseconds and biases the
    # two-point slope calibration. Busy-spin observes within ~µs.
    a_view = kstart_view[0]; b_view = kstart_view[1]
    t_deadline = time.monotonic() + 2.0
    while a_view[0] == 0 or b_view[0] == 0:
        if time.monotonic() > t_deadline:
            print("[warn] kernel_start_clk did not arrive within 2 s; "
                  "latency anchor will be imprecise", flush=True)
            break
    t_anchor_cpu_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    kernel_start_clk_blk = [int(kstart_view[0][0]), int(kstart_view[1][0])]
    print(f"[anchor] CPU ns = {t_anchor_cpu_ns}, "
          f"GPU clk[0] = {kernel_start_clk_blk[0]}, "
          f"clk[1] = {kernel_start_clk_blk[1]}, "
          f"launch→anchor = {(t_anchor_cpu_ns - t_launch_cpu_ns)/1000:.1f} µs",
          flush=True)

    # ---- 4. Init libstage01_to_gpu and attach rings -------------------
    if not os.path.exists(LIB_PATH):
        sys.exit(f"missing {LIB_PATH} — build with `cmake --build deploy/build`")
    print(f"Loading {LIB_PATH}", flush=True)
    api = S01gAPI(LIB_PATH)
    api.init(args.weights, n_shards=args.shards, halo=args.halo,
             base_core=args.base_core, sample_cap=65536,
             shard_ring_cap=args.shard_ring_cap)
    api.attach(ring_dev[0], ring_head[0], ring_dev[1], ring_head[1],
                args.ring_cap - 1, W2, H2,
                proc_lo[0], proc_hi[0], proc_lo[1], proc_hi[1])
    api.reset()
    print("CPU lib + GPU rings attached, opening camera ...", flush=True)

    # ---- 5. Event source: live camera or synthetic ---------------------
    stop_event = threading.Event()
    cam_thread = None
    cpp_synth = args.synthetic_mev > 0 and args.cpp_synth
    if cpp_synth:
        print(f"C++ synth dispatcher @ {args.synthetic_mev:.2f} Mev/s "
              f"(pin core {args.synth_pin_core})", flush=True)
        api.start_synthetic(args.synthetic_mev, args.synth_pin_core, args.seed)
    elif args.synthetic_mev > 0:
        cam_thread = SyntheticReader(
            api,
            target_mev=args.synthetic_mev,
            H_full=H_full, W_full=W_full,
            seed=args.seed,
            stop_event=stop_event,
        )
        print(f"Python synthetic source @ {args.synthetic_mev:.2f} Mev/s "
              f"(no camera)", flush=True)
    else:
        cam_thread = CameraReader(
            api,
            cam_interval_us=args.cam_interval_us,
            contrast=args.contrast,
            subsample=args.subsample,
            stop_event=stop_event,
        )

    def _sigint(_sig, _frm):
        print("\n[main] SIGINT, requesting stop", flush=True)
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    if cam_thread is not None:
        cam_thread.start()
    t_start = time.monotonic()

    # ---- 6. Stats loop ------------------------------------------------
    headers = ("t   |  cam_in    |  raw   kept |"
               "  push0   push1 |  done0   done1 |"
               " ring0  ring1 |  CPU p50/p99/MAX µs")
    print(headers, flush=True)
    last_done = [0, 0]
    last_pushed = [0, 0]
    last_t = t_start
    while not stop_event.is_set():
        time.sleep(args.stats_interval_s)
        t_now = time.monotonic()
        elapsed = t_now - t_start
        if elapsed >= args.duration_s:
            print(f"[main] duration {args.duration_s}s reached, requesting stop",
                  flush=True)
            stop_event.set()
            break
        st = api.stats()
        head0, head1 = _read_u64(ring_head[0]), _read_u64(ring_head[1])
        done0, done1 = _read_u64(events_done[0]), _read_u64(events_done[1])
        ring_used0, ring_used1 = head0 - done0, head1 - done1
        rate_in    = (cam_thread.events_in if cam_thread is not None
                       else api.synthetic_n_pushed())
        push_b0    = int(st[14])
        push_b1    = int(st[15])
        d_done0 = done0 - last_done[0]
        d_done1 = done1 - last_done[1]
        d_push0 = push_b0 - last_pushed[0]
        d_push1 = push_b1 - last_pushed[1]
        last_done = [done0, done1]
        last_pushed = [push_b0, push_b1]
        dt = max(t_now - last_t, 1e-6)
        last_t = t_now
        print(f"{elapsed:5.1f}s | {rate_in:9d} | "
              f"raw {int(st[0])+int(st[1]):7d}  k{int(st[13]):6d} | "
              f"+{d_push0:5d} +{d_push1:5d} | +{d_done0:5d} +{d_done1:5d} | "
              f"{ring_used0:5d} {ring_used1:5d} | "
              f"{st[6]:5.1f}/{st[8]:6.1f}/{st[10]:7.1f}",
              flush=True)

    # ---- 7. Shutdown ---------------------------------------------------
    synth_total = 0
    if cpp_synth:
        synth_total = api.synthetic_n_pushed()
        print("[main] stopping C++ synth dispatcher ...", flush=True)
        api.stop_synthetic()
    if cam_thread is not None:
        print("[main] stopping camera thread ...", flush=True)
        cam_thread.join(timeout=2.0)
    # Snapshot per-segment timing BEFORE shutdown — api.stats() requires
    # the lib handle which shutdown destroys.
    final_stats = api.stats()
    print("[main] setting GPU stop_flag ...", flush=True)
    _write_i32(stop_flag, 1)
    # Wait for kernel to exit.
    from cuda import cuda
    print("[main] cuCtxSynchronize (waiting for GPU kernel exit) ...", flush=True)
    err, = cuda.cuCtxSynchronize()
    t_exit_cpu_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if err != cuda.CUresult.CUDA_SUCCESS:
        print(f"[main] cuCtxSynchronize: {err}", flush=True)
    kernel_end_clk_blk = [int(kend_view[0][0]), int(kend_view[1][0])]
    print(f"[anchor exit] CPU ns = {t_exit_cpu_ns}, "
          f"GPU end clk[0] = {kernel_end_clk_blk[0]}, "
          f"clk[1] = {kernel_end_clk_blk[1]}", flush=True)
    print("[main] shutdown libstage01_to_gpu ...", flush=True)
    api.shutdown()

    # Final stats.
    head0, head1 = _read_u64(ring_head[0]), _read_u64(ring_head[1])
    done0, done1 = _read_u64(events_done[0]), _read_u64(events_done[1])
    print(f"\n[final] cam events submitted: {(cam_thread.events_in if cam_thread is not None else synth_total)}")
    print(f"[final] ring head:    block 0 = {head0:9d}, block 1 = {head1:9d}")
    print(f"[final] events_done:  block 0 = {done0:9d}, block 1 = {done1:9d}")
    print(f"[final] ring lag:     block 0 = {head0-done0}, block 1 = {head1-done1}")

    # Predictions written by both blocks?
    pred_nonzero = [int((preds_view[b] != 0).any()) for b in range(N_BLOCKS)]
    pred_nz_cells = [int((version_view[b] > 0).sum()) for b in range(N_BLOCKS)]
    pred_total_writes = [int(version_view[b].sum()) for b in range(N_BLOCKS)]
    print(f"[final] predictions:  block 0 nonzero={pred_nonzero[0]}  "
          f"cells_written={pred_nz_cells[0]}/{version_view[0].size}  "
          f"total_writes={pred_total_writes[0]}")
    print(f"[final] predictions:  block 1 nonzero={pred_nonzero[1]}  "
          f"cells_written={pred_nz_cells[1]}/{version_view[1].size}  "
          f"total_writes={pred_total_writes[1]}")

    # Per-segment CPU timing breakdown (Task #1). Slots 16..21 are sum_ns
    # summed across shards; 22..27 are call counts. mean_us = sum_ns /
    # count / 1000. See src/lib_stage01_to_gpu.cpp::s01g_snapshot_stats.
    seg_names = ["preprocess", "stage_forward(0)", "tdrop_and_pool(0)",
                 "stage_forward(1)", "tdrop_and_pool(1)", "ring push"]
    print(f"\n[cpu seg] per-segment mean (sum across shards):")
    print(f"  {'segment':>20} | {'mean_us':>10}  {'count':>12}  "
          f"{'share_%':>8}")
    seg_sum_ns = [final_stats[16 + s] for s in range(6)]
    seg_cnt    = [int(final_stats[22 + s]) for s in range(6)]
    total_ns = sum(seg_sum_ns)
    for name, sn, cn in zip(seg_names, seg_sum_ns, seg_cnt):
        mean_us = (sn / cn / 1000.0) if cn > 0 else 0.0
        share = (sn / total_ns * 100.0) if total_ns > 0 else 0.0
        print(f"  {name:>20} | {mean_us:10.3f}  {cn:12d}  {share:7.1f}%")

    # GPU end-to-end latency: CPU push → GPU done.
    #   t_done_ns_cpu = t_anchor_cpu_ns + (t_done_clk - kernel_start_clk[blk])
    #                                      * 1e9 / SM_CLK_HZ
    #   latency_ns    = t_done_ns_cpu - t_push_ns
    # The previous (t_done_clk - t_pop_clk) metric is also reported as
    # "kernel µs" to show how much of the total is pure GPU compute vs
    # ring-wait time.
    # NOTE on calibration accuracy:
    #   `krn p50/p99` (t_done_clk − t_pop_clk, converted at SM rate) is the
    #   pure GPU compute time per batch — exact.
    #   `e2e p50/p99/max` adds (push→pop) ring-wait time, but the CPU/GPU
    #   clock cross-correlation is approximate: anchored by host CPU
    #   CLOCK_MONOTONIC_RAW recorded at cuLaunch / cuCtxSynchronize, with
    #   per-block (kernel_start_clk, kernel_end_clk) bounding the SM cycles.
    #   Without root + jetson_clocks the SM clock is DVFS-controlled and
    #   the linear fit between the two anchors carries a ms-scale residual
    #   error. Treat e2e numbers as upper-bounded by ~5 ms calibration bias.
    # `owner == 1` means: spatial owner AND passed both tdrop_s2 AND
    # tdrop_s3 — i.e. event produces a final prediction. Latency stats in
    # the "n_pred" column cover only these events (downstream-visible).
    # The "n_all" column is every timed slot (output-producing + dropped)
    # for sanity comparison — at saturation they share the same per-batch
    # t_pop/t_done so values should look similar.
    print(f"\n[gpu lat] CPU-push → GPU-done (calibration ±~5 ms, "
          f"ring cap = {args.timing_cap})")
    print(f"{'block':>5} | {'n_pred':>7} {'e2e p50':>8} {'e2e p99':>8} "
          f"{'e2e max':>8} | {'krn p50':>8} {'krn p99':>8} | "
          f"{'n_all':>7} {'e2e p50':>8} {'e2e p99':>8} µs")
    for b in range(N_BLOCKS):
        tv = timing_view[b]
        valid = (tv["t_done_clk"] != 0) & (tv["t_push_ns"] != 0)
        if not valid.any():
            print(f"{b:>5} | (no samples)")
            continue
        idx = np.nonzero(valid)[0]
        head_skip = max(1, len(idx) // 20)
        tail_skip = max(1, len(idx) // 20)
        idx = idx[head_skip:-tail_skip] if len(idx) > head_skip + tail_skip else idx

        # End-to-end latency in µs.
        # Two-point calibration: (t_anchor_cpu_ns, kernel_start_clk[b]) at
        # kernel entry and (t_exit_cpu_ns, kernel_end_clk[b]) at exit.
        # Effective ns/cycle = slope between the two anchors.
        cyc_span = kernel_end_clk_blk[b] - kernel_start_clk_blk[b]
        ns_span  = t_exit_cpu_ns - t_anchor_cpu_ns
        ns_per_cycle = (ns_span / cyc_span) if cyc_span > 0 else (1e9 / SM_CLK_HZ)
        eff_sm_clk_mhz = 1000.0 / ns_per_cycle
        t_done_cyc = tv["t_done_clk"][idx].astype(np.int64)
        t_push_ns_arr = tv["t_push_ns"][idx].astype(np.int64).astype(np.float64)
        dt_cyc = (t_done_cyc - kernel_start_clk_blk[b]).astype(np.float64)
        t_done_ns_cpu = t_anchor_cpu_ns + dt_cyc * ns_per_cycle
        e2e_us = (t_done_ns_cpu - t_push_ns_arr) / 1000.0

        # Pure GPU compute time (t_done_clk - t_pop_clk).
        krn_us = (t_done_cyc - tv["t_pop_clk"][idx].astype(np.int64)) \
                 * (1e6 / SM_CLK_HZ)

        pred_mask = tv["owner"][idx] == 1
        own_e2e = e2e_us[pred_mask]
        own_krn = krn_us[pred_mask]
        def _pct(a, p):
            return float(np.percentile(a, p)) if a.size else 0.0
        def _max(a):
            return float(a.max()) if a.size else 0.0
        print(f"{b:>5} | {own_e2e.size:>7d} "
              f"{_pct(own_e2e,50):>8.1f} {_pct(own_e2e,99):>8.1f} {_max(own_e2e):>8.1f} | "
              f"{_pct(own_krn,50):>8.1f} {_pct(own_krn,99):>8.1f} | "
              f"{e2e_us.size:>7d} {_pct(e2e_us,50):>8.1f} {_pct(e2e_us,99):>8.1f}  "
              f"(eff SM clk = {eff_sm_clk_mhz:.1f} MHz)")

    p2_pass = ((cam_thread.events_in if cam_thread is not None else synth_total) > 0
               and done0 > 0 and done1 > 0
               and abs(head0 - done0) < args.ring_cap
               and abs(head1 - done1) < args.ring_cap
               and pred_nonzero[0] and pred_nonzero[1])
    if p2_pass:
        print(f"\nP2 PASS — both blocks advanced and wrote non-zero predictions; "
              f"no ring deadlock.")
        return 0
    print(f"\nP2 FAIL — events_in={(cam_thread.events_in if cam_thread is not None else synth_total)} "
          f"done={done0,done1} preds_nonzero={pred_nonzero}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
