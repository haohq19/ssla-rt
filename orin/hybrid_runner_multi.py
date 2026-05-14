"""Multi-block live runner: launches k_ssla_s2s3_celled_persistent_multi
with N independent rings + N CPU shards routed by 2D (x, y) cell ownership.

Mirrors hybrid_runner.py's structure but parametrised on --gpu-blocks
∈ {2, 4, 8}. For 2 it's equivalent to the legacy 2-block path with
H_owned = [0, H2) on both strips.
"""
import argparse
import ctypes
import datetime as _dt
import os
import sys
import signal
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import CudaModule, arg_ptr, arg_u32, arg_u64  # noqa: E402
from orin.hybrid_common import (  # noqa: E402
    INPUT_DTYPE, OUTPUT_DTYPE, TIMING_DTYPE,
    C1, C2, C3, MAX_BLOCKS, HEAD_OUT_DEFAULT,
    build_head_weights, S01gAPI,
)
from orin.multi_block import (  # noqa: E402
    BlockTopo, grid_topology, build_n_block_random_layers,
    alloc_tdrop_n_block, alloc_predictions_n_block,
    alloc_timing_n_block, alloc_kernel_clk_n_block,
    build_n_block_config,
)


SM_CLK_HZ = 918_000_000
LIB_PATH = "/home/nanod/ssla-rt/build/libstage01_to_gpu.so"
KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def _alloc_u64():
    p, _ka = cuda_util.alloc_pinned(8)
    ctypes.memset(p, 0, 8)
    return p


def _write_i32(devptr, val):
    arr = (ctypes.c_int32 * 1).from_address(devptr)
    arr[0] = val


def _read_u64(devptr):
    arr = (ctypes.c_uint64 * 1).from_address(devptr)
    return int(arr[0])


class CameraReader(threading.Thread):
    """Read DVXplorer events, apply camera-side ROI/threshold/subsample,
    pack into the (n, 4) float32 layout libstage01_to_gpu expects, push
    via S01gAPI.submit.

    Coordinate flow:
      sensor event at (x_sensor, y_sensor)
        → divide by subsample factor `ds` (chip-side subsample, so events
          actually arrive already at /ds coords on DVXplorer Micro — but
          we divide defensively in case future cameras don't subsample
          server-side)
        → if outside H_full × W_full grid: dropped
        → submitted to CPU pipeline at (x // ds, y // ds)

    ROI is set via cam.setCropArea(x, y, w, h); events outside the crop
    are not delivered by the camera, so they don't reach this code.
    """
    SS_LADDER = {"EVERY_PIXEL": 1, "EVERY_SECOND": 2,
                 "EVERY_FOURTH": 4, "EVERY_EIGHTH": 8}

    def __init__(self, api, args, stop_event):
        super().__init__(daemon=True)
        self.api = api
        self.args = args
        self.stop_event = stop_event
        self.events_in = 0
        self.batches_in = 0
        self.cam = None

    def run(self):
        import dv_processing as dv
        self.cam = dv.io.camera.open()
        W_cam, H_cam = self.cam.getEventResolution()
        print(f"[cam] opened {self.cam.getCameraName()} "
              f"(serial {self.cam.getSerialNumber()})  sensor={W_cam}×{H_cam}",
              flush=True)

        SS = dv.io.camera.DVXplorer.SubSample
        ss_name = self.args.cam_subsample
        ss_val = getattr(SS, ss_name)
        ds = self.SS_LADDER[ss_name]

        # Centered ROI from --roi-pct (linear, both dims). pct=100 → full.
        roi_pct = max(1.0, min(100.0, self.args.roi_pct))
        roi_w = int(W_cam * roi_pct / 100.0)
        roi_h = int(H_cam * roi_pct / 100.0)
        roi_x = (W_cam - roi_w) // 2
        roi_y = (H_cam - roi_h) // 2

        # Clamp contrast to chip-level valid 0-17 (chip silently clamps
        # higher values — see docs/DEV_GUIDE.md §6.2).
        contrast = max(0, min(17, self.args.contrast))
        try:
            self.cam.setSubSampleHorizontal(ss_val)
            self.cam.setSubSampleVertical(ss_val)
            self.cam.setContrastThresholdOn(contrast)
            self.cam.setContrastThresholdOff(contrast)
            self.cam.setTimeInterval(_dt.timedelta(microseconds=self.args.cam_interval_us))
            self.cam.setMIPITimeoutValue(_dt.timedelta(microseconds=self.args.cam_interval_us))
            if roi_pct < 100.0:
                self.cam.setCropArea((roi_x, roi_y, roi_w, roi_h))
        except Exception as e:
            print(f"[cam] knob set failed: {e}", flush=True)

        print(f"[cam] contrast={contrast}  subsample={ss_name}(/{ds})  "
              f"roi_pct={roi_pct} → crop ({roi_x},{roi_y}) {roi_w}×{roi_h}",
              flush=True)

        t0 = None
        H_full, W_full = self.args.h_full, self.args.w_full
        # Translation: events inside the ROI crop are mapped to grid (0,0)
        # origin. After translation + (optional) subsample, valid coord
        # range is [0, W_full) × [0, H_full). Sanity check the user's
        # configuration:
        expected_w = roi_w // ds
        expected_h = roi_h // ds
        if expected_w != W_full or expected_h != H_full:
            print(f"[cam] WARNING: ROI/subsample gives {expected_w}×{expected_h} "
                  f"events but SSLA grid is {W_full}×{H_full}. "
                  f"Set --h-full={expected_h} --w-full={expected_w} for a tight fit.",
                  flush=True)
        while not self.stop_event.is_set():
            if not self.cam.isRunning():
                time.sleep(0.001); continue
            batch = self.cam.getNextEventBatch()
            if batch is None or batch.size() == 0:
                time.sleep(0.0005); continue
            ev = batch.numpy()
            if t0 is None:
                t0 = int(ev["timestamp"][0])
            n = ev.size
            packed = np.empty((n, 4), dtype=np.float32, order="C")
            packed[:, 0] = (ev["timestamp"] - t0).astype(np.float32)
            # Translate to ROI origin then (optionally) subsample.
            packed[:, 1] = ((ev["x"].astype(np.int32) - roi_x) // ds).astype(np.float32)
            packed[:, 2] = ((ev["y"].astype(np.int32) - roi_y) // ds).astype(np.float32)
            packed[:, 3] = ev["polarity"].astype(np.float32)
            # Defensive clip: drop anything that landed outside grid (the
            # camera's setCropArea should make this empty when configured
            # correctly, but stragglers can appear during the brief window
            # before the chip applies the crop).
            valid = ((packed[:, 1] >= 0) & (packed[:, 1] < W_full) &
                     (packed[:, 2] >= 0) & (packed[:, 2] < H_full))
            if not valid.all():
                packed = packed[valid]
                if packed.shape[0] == 0:
                    continue
            self.events_in += self.api.submit(packed)
            self.batches_in += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--h-full", type=int, default=64)
    ap.add_argument("--w-full", type=int, default=80)
    ap.add_argument("--gpu-blocks", type=int, default=8, choices=[2, 4, 8])
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--halo", type=int, default=2)
    ap.add_argument("--base-core", type=int, default=1)
    ap.add_argument("--pin-python-main", action="store_true")
    ap.add_argument("--python-pin-core", type=int, default=0)
    ap.add_argument("--cpp-synth", action="store_true",
                    help="Use synthetic event generator (mutually exclusive "
                         "with camera input; default if no camera flags set).")
    ap.add_argument("--synth-pin-core", type=int, default=7)
    # ---- Camera mode (active if --cpp-synth NOT set) ----
    ap.add_argument("--contrast", type=int, default=17,
                    help="DVXplorer contrast threshold ON/OFF (clamped to 0-17; "
                         "chip silently clamps above 17 — see DEV_GUIDE §6.2)")
    ap.add_argument("--cam-subsample", type=str, default="EVERY_PIXEL",
                    choices=["EVERY_PIXEL", "EVERY_SECOND",
                             "EVERY_FOURTH", "EVERY_EIGHTH"],
                    help="Chip-side subsample factor (1/2/4/8 x). Default is "
                         "EVERY_PIXEL — events at the camera ROI's native "
                         "resolution. If you want the chip to subsample, pass "
                         "EVERY_SECOND/FOURTH/EIGHTH and pick --h-full / "
                         "--w-full accordingly.")
    ap.add_argument("--roi-pct", type=float, default=100.0,
                    help="Centered crop to N%% × N%% of sensor (LINEAR, both "
                         "dims). 100 = full sensor. 25 = 25%%×25%% linear = "
                         "6.25%% of sensor area. Events inside the crop are "
                         "translated so the ROI origin maps to grid (0,0); "
                         "this means --h-full and --w-full must match the "
                         "ROI-and-subsample-derived grid size, NOT the full "
                         "sensor.")
    ap.add_argument("--cam-interval-us", type=int, default=2000,
                    help="DVXplorer batch interval (µs). Lower = lower latency, "
                         "more frequent small batches.")
    ap.add_argument("--shard-ring-cap", type=int, default=16)
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--ring-cap", type=int, default=1 << 16)
    ap.add_argument("--spin-ns", type=int, default=200)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timing-cap", type=int, default=1 << 14)
    ap.add_argument("--synthetic-mev", type=float, default=2.0)
    args = ap.parse_args()

    if args.pin_python_main:
        os.sched_setaffinity(0, {args.python_pin_core})

    H_full, W_full = args.h_full, args.w_full
    H2, W2 = H_full >> 2, W_full >> 2
    H3, W3 = H2 >> 1, W2 >> 1
    print(f"Geometry: full {W_full}×{H_full}  s2 {W2}×{H2}  s3 {W3}×{H3}",
          flush=True)

    topo = grid_topology(args.gpu_blocks, W2, H2)
    print(f"Topology: {args.gpu_blocks} blocks", flush=True)
    for i, t in enumerate(topo):
        print(f"  blk{i}: owned X[{t.owned_x_lo},{t.owned_x_hi}) "
              f"Y[{t.owned_y_lo},{t.owned_y_hi}) "
              f"proc X[{t.proc_x_lo},{t.proc_x_hi}) Y[{t.proc_y_lo},{t.proc_y_hi})")

    # ---- 1. GPU resources --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    cpu_layers, gpu_layers, ka_layers = build_n_block_random_layers(
        rng, args.gpu_blocks, H2, W2, H3, W3)
    tdrop_s2_dev, tdrop_s3_dev, ka_tdrop = alloc_tdrop_n_block(
        H2, W2, H3, W3, args.gpu_blocks)
    head_W_dev, head_b_dev, ka_head = build_head_weights(rng, HEAD_OUT_DEFAULT)
    preds_dev, preds_view, version_dev, version_view, ka_preds = \
        alloc_predictions_n_block(topo, HEAD_OUT_DEFAULT)
    timing_dev, timing_view, timing_mask, ka_timing = alloc_timing_n_block(
        args.timing_cap, args.gpu_blocks)
    kstart_dev, kstart_view, kend_dev, kend_view, ka_kclk = \
        alloc_kernel_clk_n_block(args.gpu_blocks)
    keepalive = list(ka_layers) + list(ka_tdrop) + list(ka_head) \
              + list(ka_preds) + list(ka_timing) + list(ka_kclk)

    p_cfg, ka_cfg = build_n_block_config(
        H2, W2, H3, W3, args.tdrop,
        gpu_layers, tdrop_s2_dev, tdrop_s3_dev, topo,
        head_W=head_W_dev, head_b=head_b_dev,
        preds_dev=preds_dev, version_dev=version_dev,
        timing_dev=timing_dev, timing_mask=timing_mask,
        kernel_start_clk=kstart_dev, kernel_end_clk=kend_dev,
    )
    keepalive.append(ka_cfg)

    # ---- 2. Pinned rings + tails + events_done ----------------------------
    rec_bytes = INPUT_DTYPE.itemsize
    ring_nbytes = args.ring_cap * rec_bytes
    print(f"\nPinned alloc: {args.gpu_blocks} × ring "
          f"({ring_nbytes/1e6:.1f} MB each)", flush=True)
    ring_dev, ring_view, ring_head, ring_tail, events_done = [], [], [], [], []
    for b in range(args.gpu_blocks):
        p, ka = cuda_util.alloc_pinned(ring_nbytes)
        ctypes.memset(p, 0, ring_nbytes)
        keepalive.append(ka)
        ring_dev.append(p)
        ring_view.append(np.frombuffer(ka, dtype=INPUT_DTYPE).reshape(-1))
        ring_head.append(_alloc_u64())
        ring_tail.append(_alloc_u64())
        events_done.append(_alloc_u64())
    stop_flag, _ka_stop = cuda_util.alloc_pinned(4)
    ctypes.memset(stop_flag, 0, 4)
    keepalive.append(_ka_stop)

    # ---- 3. Compile + launch persistent_multi kernel ----------------------
    src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head_celled.cuh")).read()
    proto = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
    print(f"Compiling celled kernel ...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_s2_s3_head_celled.cu",
                     headers={"proto_layer_pair.cuh": proto})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s",
          flush=True)
    func = mod.get_function("k_ssla_s2s3_celled_persistent_multi")
    threads_pb = 9 * 32
    BATCH_K, N_WARPS_K, OUT_MAX_K = 8, 9, C3
    C1_K, C2_K = 24, 48
    event_slot = OUT_MAX_K * 4 + OUT_MAX_K * 4 + 5 * 4 + 4 + 8 + 8
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

    # Opt-in to dynamic smem >48 KB (sm_87 supports up to ~167 KB / block).
    from cuda import cuda as _cuda
    _err, = _cuda.cuFuncSetAttribute(
        func._func,
        _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SMEM)
    if int(_err) != 0:
        raise RuntimeError(f"cuFuncSetAttribute(MAX_DYNAMIC_SHARED): err={int(_err)}, smem={SMEM}")

    # Pointer arrays for kernel: rings[], tails[], events_dones[].
    def _alloc_ptr_array(values):
        nbytes = args.gpu_blocks * 8
        p, ka = cuda_util.alloc_managed(nbytes)
        view = np.frombuffer(ka, dtype=np.uint64).reshape(-1)
        view[:] = values
        keepalive.append(ka)
        return p

    rings_arr_dev = _alloc_ptr_array(ring_dev)
    tails_arr_dev = _alloc_ptr_array(ring_tail)
    edone_arr_dev = _alloc_ptr_array(events_done)

    print(f"Launching: gridDim={args.gpu_blocks}, blockDim={threads_pb}, "
          f"smem={SMEM} B", flush=True)
    t_launch_cpu_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    func.launch((args.gpu_blocks, 1, 1), (threads_pb, 1, 1), [
        arg_ptr(p_cfg),
        arg_ptr(rings_arr_dev),
        arg_u64(args.ring_cap - 1),
        arg_ptr(tails_arr_dev),
        arg_ptr(stop_flag),
        arg_ptr(edone_arr_dev),
        arg_u32(args.spin_ns),
    ], smem=SMEM)

    # Busy-spin for kernel_start_clk anchor.
    t_dead = time.monotonic() + 2.0
    while True:
        if all(int(kstart_view[b][0]) != 0 for b in range(args.gpu_blocks)):
            break
        if time.monotonic() > t_dead:
            print("[warn] kernel_start_clk timeout"); break
    t_anchor_cpu_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    print(f"[anchor] launch→anchor = "
          f"{(t_anchor_cpu_ns-t_launch_cpu_ns)/1000:.1f} µs", flush=True)

    # ---- 4. Init libstage01_to_gpu and attach multi rings -----------------
    api = S01gAPI(LIB_PATH)
    api.init(args.weights, n_shards=args.shards, halo=args.halo,
             base_core=args.base_core, sample_cap=65536,
             shard_ring_cap=args.shard_ring_cap)
    proc_x_los = [t.proc_x_lo for t in topo]
    proc_x_his = [t.proc_x_hi for t in topo]
    proc_y_los = [t.proc_y_lo for t in topo]
    proc_y_his = [t.proc_y_hi for t in topo]
    api.attach_multi(ring_dev, ring_head, args.ring_cap - 1, W2, H2,
                      proc_x_los, proc_x_his, proc_y_los, proc_y_his)
    api.reset()
    print("CPU + GPU rings attached", flush=True)

    # ---- 5. Synth source --------------------------------------------------
    if args.cpp_synth:
        api.lib.s01g_start_synthetic(api.handle,
            ctypes.c_double(args.synthetic_mev),
            ctypes.c_int(args.synth_pin_core),
            ctypes.c_uint64(args.seed))
        print(f"C++ synth dispatcher @ {args.synthetic_mev:.2f} Mev/s "
              f"(pin core {args.synth_pin_core})", flush=True)

    # If not synth, start camera reader thread.
    cam_stop_event = threading.Event()
    cam_reader = None
    if not args.cpp_synth:
        cam_reader = CameraReader(api, args, cam_stop_event)
        cam_reader.start()
        time.sleep(0.3)   # let camera open before main loop polls events_done

    stop_flag_view = (ctypes.c_int32 * 1).from_address(stop_flag)

    def on_sigint(sig, frame):
        print("\n[SIGINT] stopping", flush=True)
        stop_flag_view[0] = 1
    signal.signal(signal.SIGINT, on_sigint)

    # ---- 6. Run for duration_s, periodic stats ----------------------------
    t_run_start = time.monotonic()
    t_next_stats = t_run_start + 1.0
    print(f"\n{'t':>5} | {'kev/s GPU':>10} | "
          + " ".join(f"d{b}" for b in range(args.gpu_blocks))
          + " | CPU per-block n_pushed", flush=True)
    last_done = [0] * args.gpu_blocks
    while time.monotonic() - t_run_start < args.duration_s:
        if time.monotonic() >= t_next_stats:
            done = [_read_u64(events_done[b]) for b in range(args.gpu_blocks)]
            dones = [done[b] - last_done[b] for b in range(args.gpu_blocks)]
            total = sum(dones)
            print(f"{time.monotonic()-t_run_start:5.1f}s | {total/1e3:10.1f} | "
                  + " ".join(f"{d:6d}" for d in dones), flush=True)
            last_done = done
            t_next_stats += 1.0
        time.sleep(0.05)

    print("[main] duration reached, stopping", flush=True)
    if args.cpp_synth:
        api.lib.s01g_stop_synthetic(api.handle)
    if cam_reader is not None:
        cam_stop_event.set()
        cam_reader.join(timeout=1.0)
        print(f"[cam] {cam_reader.events_in} events submitted from "
              f"{cam_reader.batches_in} batches", flush=True)
    stop_flag_view[0] = 1
    from cuda import cuda
    err, = cuda.cuCtxSynchronize()
    t_exit_cpu_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    print(f"[main] cuCtxSynchronize: {err}", flush=True)
    api.shutdown()

    # ---- 7. Final stats ---------------------------------------------------
    head_done = sum(_read_u64(events_done[b]) for b in range(args.gpu_blocks))
    print(f"\n[final] total events_done across {args.gpu_blocks} blocks = "
          f"{head_done}  → {head_done/args.duration_s/1e3:.1f} kev/s avg")
    for b in range(args.gpu_blocks):
        head = _read_u64(ring_head[b])
        done = _read_u64(events_done[b])
        print(f"  blk{b}: head={head:8d} done={done:8d} lag={head-done}")

    # ---- 8. End-to-end latency from timing slots --------------------------
    # Latency stats are computed ONLY over events that produce a final
    # prediction (`is_owner AND pass_tdrop_s2 AND pass_tdrop_s3` — flagged
    # as `owner == 1` in the timing slot). Events dropped by either tdrop
    # are excluded because their predictions are never read; their kernel-
    # internal timing is the same as output-producing events in the same
    # batch anyway (per-batch t_pop_clk and t_done_clk are shared), so the
    # filter doesn't change the underlying numbers — it just keeps the
    # report focused on the events that matter for downstream consumers.
    #
    # Three components per timing slot:
    #   emit→push  = CPU pipeline latency (synth/camera → CPU stage 0+1 → GPU ring publish)
    #   kernel     = GPU push→done (ring wait + 4 GPU layers + head)
    #   emit→done  = whole pipeline ≈ emit→push + kernel (no cross-clock bias)
    # GPU-side t_done_clk is in SM cycles — translated via the per-block
    # two-point calibration (kernel_start_clk + kernel_end_clk anchored
    # against CPU CLOCK_MONOTONIC_RAW recorded at launch / sync).
    print(f"\n[latency] WHOLE pipeline (emit → GPU done), output-producing events only:")
    print(f"{'blk':>3} | {'n_pred':>7} | "
          f"{'emit→push':>10} {'kernel':>10} (µs p50/p99/max)")
    cyc0 = [int(kstart_view[b][0]) for b in range(args.gpu_blocks)]
    cyc1 = [int(kend_view[b][0])   for b in range(args.gpu_blocks)]
    for b in range(args.gpu_blocks):
        tv = timing_view[b]
        valid = (tv["t_done_clk"] != 0) & (tv["t_emit_ns"] != 0) & (tv["t_push_ns"] != 0)
        if not valid.any():
            print(f"{b:>3} | (no samples)")
            continue
        idx = np.nonzero(valid)[0]
        head_skip = max(1, len(idx) // 20); tail_skip = max(1, len(idx) // 20)
        idx = idx[head_skip:-tail_skip] if len(idx) > head_skip + tail_skip else idx

        cyc_span = max(1, cyc1[b] - cyc0[b])
        ns_span  = max(1, t_exit_cpu_ns - t_anchor_cpu_ns)
        ns_per_cycle = ns_span / cyc_span
        dt_cyc = (tv["t_done_clk"][idx].astype(np.int64) - cyc0[b]).astype(np.float64)
        t_done_ns_cpu = t_anchor_cpu_ns + dt_cyc * ns_per_cycle

        t_push = tv["t_push_ns"][idx].astype(np.int64).astype(np.float64)
        t_emit = tv["t_emit_ns"][idx].astype(np.int64).astype(np.float64)

        emit_to_push = (t_push - t_emit) / 1000.0
        push_to_done = (t_done_ns_cpu - t_push) / 1000.0
        emit_to_done = (t_done_ns_cpu - t_emit) / 1000.0
        # GPU-internal kernel time t_pop_clk → t_done_clk (same SM clock,
        # no cross-clock bias). Use the per-block ns_per_cycle which is
        # what we just derived from the two-anchor calibration.
        kernel_us = ((tv["t_done_clk"][idx].astype(np.int64) -
                      tv["t_pop_clk"][idx].astype(np.int64))
                     * ns_per_cycle / 1000.0)

        # `owner == 1` in the timing slot means: spatial owner AND passed
        # both tdrop_s2 AND tdrop_s3 — i.e. the event produces a final
        # prediction. Latency stats below cover only these events.
        pred_mask = tv["owner"][idx] == 1
        def _pct3(arr, mask):
            sub = arr[mask] if mask.any() else arr
            if sub.size == 0:
                return "n/a", "n/a", "n/a"
            return (f"{float(np.percentile(sub,50)):>5.0f}",
                    f"{float(np.percentile(sub,99)):>5.0f}",
                    f"{float(sub.max()):>5.0f}")
        a50, a99, amx = _pct3(emit_to_push, pred_mask)
        k50, k99, kmx = _pct3(kernel_us, pred_mask)
        n_pred = int(pred_mask.sum())
        # CLEAN latency components (no cross-clock bias):
        #   emit→push : same CPU clock domain (CLOCK_MONOTONIC_RAW)
        #   kernel    : same GPU SM clock domain (t_done_clk − t_pop_clk)
        # Sum ≈ whole-pipeline lower bound (omits push→pop ring-wait,
        # which is ~few µs at low load).
        print(f"{b:>3} | {n_pred:>7d} | "
              f"emit→push {a50}/{a99}/{amx}  "
              f"kernel    {k50}/{k99}/{kmx}")


if __name__ == "__main__":
    sys.exit(main() or 0)
