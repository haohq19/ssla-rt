"""Live DVXplorer + stage-0+1 multi-core benchmark.

Architecture:
  Python main thread:
    - opens dv_processing camera, applies camera-side rate caps (subsample,
      contrast threshold, time interval).
    - tight non-blocking loop reading event batches from camera.
    - token-bucket rate cap as a final safety net before the C++ pipeline.
    - ctypes-calls libstage01_capi.so::s0_submit_batch with packed events.

  C++ background threads (libstage01_capi.so):
    - N shard threads, halo=2 routing.
    - Owner runs stage 0; if tdrop passes, runs stage 1 (which works at
      half-resolution); if stage-1 tdrop passes, the event would
      propagate downstream (counted but not processed further here).
    - Halo shards run stage 0 only (state sync).

Usage:
    python3 deploy/orin/live_stage01_bench.py --weights /tmp/ssla_s_random
        --shards 6 --halo 2 --base-core 1 --rate-mev 1.0 --bucket-events 1000
        --subsample EVERY_FOURTH --contrast-threshold 40 --temporal-ds 2
        --cam-interval-us 100 --duration 30
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import sys
import time

import numpy as np

try:
    import dv_processing as dv
except ImportError as e:
    print(f"[fatal] dv_processing not available: {e}", file=sys.stderr)
    sys.exit(1)


def _read_model_dims(weights_dir: str) -> tuple[int, int]:
    """Read (H, W) from meta.json. Falls back to (240, 304) if missing."""
    p = os.path.join(weights_dir, "meta.json")
    if os.path.exists(p):
        with open(p) as f:
            m = json.load(f)
        return int(m.get("height", 240)), int(m.get("width", 304))
    return 240, 304


# ---------------------------------------------------------------------------
# Shared library loader
# ---------------------------------------------------------------------------

LIB_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "build", "libstage01_capi.so"))


def load_lib():
    if not os.path.exists(LIB_PATH):
        sys.exit(f"[fatal] {LIB_PATH} not found — build first:\n"
                 f"    cmake --build deploy/build --target stage01_capi -j6")
    lib = ctypes.CDLL(LIB_PATH)
    lib.s0_init.restype = ctypes.c_void_p
    lib.s0_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int]
    lib.s0_submit_batch.restype = ctypes.c_int
    lib.s0_submit_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.s0_snapshot_stats.restype = None
    lib.s0_snapshot_stats.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_double)]
    lib.s0_reset_stats.restype = None
    lib.s0_reset_stats.argtypes = [ctypes.c_void_p]
    lib.s0_shutdown.restype = None
    lib.s0_shutdown.argtypes = [ctypes.c_void_p]
    return lib


# ---------------------------------------------------------------------------
# Token-bucket rate cap
# ---------------------------------------------------------------------------

class TokenBucket:
    """Steady rate (Hz) with bucket capacity (events). Refill is real-time."""
    __slots__ = ("rate_hz", "capacity", "tokens", "last_t")

    def __init__(self, rate_hz: float, capacity: float):
        self.rate_hz = rate_hz
        self.capacity = capacity
        self.tokens = capacity
        self.last_t = time.monotonic()

    def admit(self, n_requested: int) -> int:
        now = time.monotonic()
        dt = now - self.last_t
        self.last_t = now
        self.tokens = min(self.capacity, self.tokens + dt * self.rate_hz)
        if n_requested <= self.tokens:
            self.tokens -= n_requested
            return n_requested
        accepted = int(self.tokens)
        self.tokens -= accepted
        return accepted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/tmp/ssla_s_random")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--halo", type=int, default=2)
    ap.add_argument("--base-core", type=int, default=2,
                    help="shard k pinned to core base_core+k; -1 = no pin")
    ap.add_argument("--rate-mev", type=float, default=1.0,
                    help="rate cap in Mev/s (token bucket steady rate)")
    ap.add_argument("--bucket-events", type=int, default=100_000,
                    help="bucket capacity (events) — burst tolerance")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--sample-cap", type=int, default=65536,
                    help="per-shard rolling latency sample buffer size")
    ap.add_argument("--subsample", choices=["EVERY_PIXEL", "EVERY_SECOND",
                                              "EVERY_FOURTH", "EVERY_EIGHTH"],
                    default="EVERY_EIGHTH",
                    help="hardware spatial subsample for rate cap")
    ap.add_argument("--contrast-threshold", type=int, default=40,
                    help="setContrastThresholdOn/Off — higher = fewer events "
                         "(default cam value is 9; 40 strongly suppresses noise)")
    ap.add_argument("--temporal-ds", type=int, default=1,
                    help="software temporal subsample: keep 1 of every K events "
                         "(1 = no subsample)")
    ap.add_argument("--cam-interval-us", type=int, default=500,
                    help="setTimeInterval (camera-side batch flush deadline)")
    args = ap.parse_args()

    lib = load_lib()
    handle = lib.s0_init(args.weights.encode("utf-8"),
                          args.shards, args.halo,
                          args.base_core, args.sample_cap)
    if not handle:
        sys.exit("[fatal] s0_init failed")
    print(f"[lib] handle={handle:#x}  shards={args.shards} halo={args.halo} "
          f"base_core={args.base_core}")

    model_h, model_w = _read_model_dims(args.weights)
    cam = dv.io.camera.open()
    # Apply camera-side subsample BEFORE reading getEventResolution so the
    # effective sensor extent matches what we'll receive.
    SS = type(cam.getSubSampleHorizontal())
    cam.setSubSampleHorizontal(getattr(SS, args.subsample))
    cam.setSubSampleVertical(getattr(SS, args.subsample))
    # Contrast threshold gate: only ON/OFF transitions exceeding this
    # delta-log-intensity emit events. Default cam value 9 produces lots
    # of noise events; 40 cleans most of them out.
    cam.setContrastThresholdOn(args.contrast_threshold)
    cam.setContrastThresholdOff(args.contrast_threshold)
    # Flush deadline: events are committed at most this often. Smaller =
    # lower per-batch wait latency at the cost of slightly less efficient
    # USB transfers.
    cam.setTimeInterval(datetime.timedelta(microseconds=args.cam_interval_us))
    # MIPI timeout also gates batch flushing on the DVXplorer M; setTimeInterval
    # alone leaves the default ~2ms MIPI timeout in place. Set both.
    cam.setMIPITimeoutValue(datetime.timedelta(microseconds=args.cam_interval_us))

    cam_w_full, cam_h_full = cam.getEventResolution()
    # Note: hardware subsampling reduces event count but coords STAY in the
    # full-sensor range (0..639 / 0..479). Scale to model grid using the
    # full extent.
    div = {"EVERY_PIXEL": 1, "EVERY_SECOND": 2,
           "EVERY_FOURTH": 4, "EVERY_EIGHTH": 8}[args.subsample]
    print(f"[cam] {cam.getCameraName()}  full={cam_w_full}×{cam_h_full}  "
          f"subsample={args.subsample} (count÷{div*div}; coords stay full-range)  "
          f"interval={args.cam_interval_us}µs  contrast_thresh={args.contrast_threshold}  "
          f"temporal_ds={args.temporal_ds}")
    print(f"[model] grid={model_w}×{model_h}  → coord scale "
          f"x×={model_w/cam_w_full:.4f} y×={model_h/cam_h_full:.4f}")
    sx = float(model_w) / float(cam_w_full)
    sy = float(model_h) / float(cam_h_full)

    bucket = TokenBucket(rate_hz=args.rate_mev * 1e6,
                          capacity=float(args.bucket_events))

    BUF_SLOTS = 8192
    buf = np.empty((BUF_SLOTS, 4), dtype=np.float32)

    stats_arr = (ctypes.c_double * 16)()

    cam_total = 0    # raw events from camera
    pass_total = 0   # events that survived the rate cap
    drop_total = 0   # events the bucket rejected
    t0 = None        # first DVXplorer timestamp (rebase to keep fp32 precision)

    print()
    print("=" * 70)
    print(f"  START — wave the camera now! Test runs for {args.duration:.0f} s.")
    print(f"  Rate cap: {args.rate_mev:.2f} Mev/s, bucket {args.bucket_events} ev")
    print("=" * 70)
    print()
    print(f"  {'time':>5} {'cam_kev/s':>10} {'pass_kev/s':>10} {'drop':>8}  "
          f"{'p50':>7} {'p99':>8}  ring/cap")
    sys.stdout.flush()

    lib.s0_reset_stats(handle)
    t_start = time.monotonic()
    t_last_print = t_start
    cam_last = 0
    pass_last = 0

    while True:
        now = time.monotonic()
        if now - t_start >= args.duration:
            break

        b = cam.getNextEventBatch()
        if b is None or b.size() == 0:
            # No yield — keep polling for minimum latency.
            if now - t_last_print >= 1.0:
                # still emit a status row even if no events
                _print_row(now - t_start, cam_total - cam_last,
                            pass_total - pass_last, drop_total,
                            lib, handle, stats_arr,
                            args.shards * 65536)
                cam_last = cam_total
                pass_last = pass_total
                t_last_print = now
            continue

        arr = b.numpy()              # struct dtype: x, y, polarity, timestamp
        n = arr.shape[0]
        cam_total += n

        # Software temporal subsample: keep one of every K events.
        if args.temporal_ds > 1:
            arr = arr[::args.temporal_ds]
            n_after_tds = arr.shape[0]
        else:
            n_after_tds = n

        # Token-bucket cap: only the first `n_pass` events pass.
        n_pass = bucket.admit(n_after_tds)
        drop_total += n_after_tds - n_pass

        if n_pass > 0:
            if t0 is None:
                t0 = int(arr["timestamp"][0])
            # Resize buf if needed (rare — batches are typically <2k).
            if n_pass > buf.shape[0]:
                buf = np.empty((n_pass, 4), dtype=np.float32)
            # Pack (t_us - t0, x_model, y_model, polarity) as float32.
            # Rescale camera coords to model grid: cam (640×480) → model (304×240).
            buf[:n_pass, 0] = (arr["timestamp"][:n_pass].astype(np.int64) - t0
                                ).astype(np.float32)
            buf[:n_pass, 1] = (arr["x"][:n_pass].astype(np.float32) * sx)
            buf[:n_pass, 2] = (arr["y"][:n_pass].astype(np.float32) * sy)
            buf[:n_pass, 3] = arr["polarity"][:n_pass].astype(np.float32)
            lib.s0_submit_batch(handle, buf.ctypes.data, n_pass)
            pass_total += n_pass

        if now - t_last_print >= 1.0:
            _print_row(now - t_start, cam_total - cam_last,
                        pass_total - pass_last, drop_total,
                        lib, handle, stats_arr,
                        args.shards * 65536)
            cam_last = cam_total
            pass_last = pass_total
            t_last_print = now

    print()
    print("=" * 70)
    print("  STOP — draining shard rings...")
    print("=" * 70)

    # Final summary — keep stats unreset so we get the cumulative window.
    lib.s0_snapshot_stats(handle, stats_arr)
    print(f"\n  Cumulative over {args.duration:.0f} s:")
    print(f"    camera events  : {cam_total:>10}  ({cam_total/args.duration/1e3:.1f} kev/s avg)")
    print(f"    after rate cap : {pass_total:>10}  ({pass_total/args.duration/1e3:.1f} kev/s avg)")
    print(f"    bucket-dropped : {drop_total:>10}  ({100.0*drop_total/max(1,cam_total):.1f}%)")
    print(f"    owner-processed: {int(stats_arr[0]):>10}")
    print(f"    halo-processed : {int(stats_arr[1]):>10}")
    print(f"    stage-0 passed : {int(stats_arr[2]):>10}  "
          f"({100.0*stats_arr[2]/max(1,stats_arr[0]):.2f}% of owner)")
    print(f"    stage-1 passed : {int(stats_arr[13]):>10}  "
          f"({100.0*stats_arr[13]/max(1,stats_arr[0]):.2f}% of owner; "
          f"{100.0*stats_arr[13]/max(1,stats_arr[2]):.2f}% of stage-0 survivors)")
    print(f"    out-of-bounds  : {int(stats_arr[3]):>10}")
    print(f"\n  Per-event latency (rolling window, {int(stats_arr[12])} samples):")
    print(f"    mean  : {stats_arr[11]:>8.2f} µs")
    print(f"    p50   : {stats_arr[6]:>8.2f} µs")
    print(f"    p90   : {stats_arr[7]:>8.2f} µs")
    print(f"    p99   : {stats_arr[8]:>8.2f} µs")
    print(f"    p99.9 : {stats_arr[9]:>8.2f} µs")
    print(f"    max   : {stats_arr[10]:>8.2f} µs")

    lib.s0_shutdown(handle)


def _print_row(t_rel, cam_d, pass_d, drop_total, lib, handle, stats_arr,
                ring_cap):
    lib.s0_snapshot_stats(handle, stats_arr)
    print(f"  {t_rel:>5.1f} {cam_d/1e3:>10.1f} {pass_d/1e3:>10.1f} "
          f"{drop_total:>8}  {stats_arr[6]:>5.1f}us {stats_arr[8]:>6.1f}us  "
          f"{int(stats_arr[5]):>5}/{ring_cap}", flush=True)


if __name__ == "__main__":
    main()
