#!/usr/bin/env python3
"""Live event viewer + on-screen tuner for the iniVation DVXplorer Micro.

Realtime architecture:
  - Background drainer thread pulls batches non-stop, applies (ROI mask,
    hot-pixel filter), parks accepted events under a lock.
  - Main thread renders at fixed FPS (default 30): takes pending events,
    concatenates once, draws frame + rate oscilloscope, imshow + waitKey.
  - All trackbars live in a separate "Controls" window, placed to the
    right of the display window — the display itself shows nothing but
    the event frame and the rate strip.

Knobs (all on the Controls window):
    Contrast ON / OFF   : raise to require larger intensity changes
                          (default 9; bigger = fewer events). Range 1..200.
                          Use --probe-thresholds to find the chip's
                          actual saturation point empirically.
    Subsample           : 0=full, 1=1/2, 2=1/4, 3=1/8 in each axis.
    ROI cx / cy / W / H : center-anchored region-of-interest, % of full
                          sensor. Tries hardware ROI; otherwise software.
    HotPixelFilter      : 0/1 toggle (sticky mask, 50× active median).
    Window (ms)         : viz accumulation window per displayed frame.
    ReadoutFPS          : caps chip-output rate (LOSSY = drop on overflow).

Modes:
    --probe-thresholds  : sweep contrast values, print rate per value, exit.
    --list-methods      : print dir(cam) at startup.

ESC or Ctrl-C to quit.
"""
import argparse
import collections
import datetime as _dt
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import dv_processing as dv

SS = dv.io.camera.DVXplorer.SubSample
RF = dv.io.camera.DVXplorer.ReadoutFPS

SUBSAMPLE_LADDER = [
    ("full",  SS.EVERY_PIXEL),
    ("1/2",   SS.EVERY_SECOND),
    ("1/4",   SS.EVERY_FOURTH),
    ("1/8",   SS.EVERY_EIGHTH),
]

READOUT_FPS_LADDER = [
    ("var-2k",   RF.VARIABLE_2000),
    ("var-5k",   RF.VARIABLE_5000),
    ("var-10k",  RF.VARIABLE_10000),
    ("var-15k",  RF.VARIABLE_15000),
    ("c-100",    RF.CONSTANT_100),
    ("c-200",    RF.CONSTANT_200),
    ("c-500",    RF.CONSTANT_500),
    ("c-1k",     RF.CONSTANT_1000),
    ("clos-2k",  RF.CONSTANT_LOSSY_2000),
    ("clos-5k",  RF.CONSTANT_LOSSY_5000),
    ("clos-10k", RF.CONSTANT_LOSSY_10000),
]

CONTRAST_MAX     = 200
RATE_BIN_MS      = 50
RATE_HISTORY_S   = 10.0
STRIP_H          = 150
DEFAULT_FPS      = 30
RECENT_KEEP_MULT = 2
CTRL_W           = 420
CTRL_H           = 60   # placeholder canvas (trackbars stack above)

HW_ROI_CANDIDATES = (
    "setRegionOfInterest",
    "setROI",
    "setEventROI",
    "setEventRegionOfInterest",
)


def _safe(setter, value, label):
    try:
        setter(value)
    except Exception as e:
        print(f"  [warn] {label} <- {value!r} failed: {e}", flush=True)


def _try_set_hw_roi(cam, x, y, w, h):
    for name in HW_ROI_CANDIDATES:
        if hasattr(cam, name):
            try:
                getattr(cam, name)(x, y, w, h)
                return name
            except Exception as e:
                print(f"  [warn] {name}({x},{y},{w},{h}) failed: {e}",
                      flush=True)
                return None
    return None


def _nice_y_top(rmax):
    candidates = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for c in candidates:
        if c >= rmax:
            return c
    return rmax * 1.1


def _draw_rate_strip(strip, samples, win_s, header_extras=()):
    h, w, _ = strip.shape
    strip[:] = (12, 12, 12)
    if not samples:
        cv2.putText(strip, "no samples yet", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return

    now = time.monotonic()
    pts = [(now - t, r) for (t, r) in samples if now - t <= win_s]
    if not pts:
        return
    ages  = np.fromiter((a for (a, _) in pts), dtype=np.float32, count=len(pts))
    rates = np.fromiter((r for (_, r) in pts), dtype=np.float32, count=len(pts))

    rmax = float(rates.max())
    y_top = _nice_y_top(max(rmax, 0.02))

    margin_top, margin_bot = 18, 28
    plot_h = h - margin_top - margin_bot
    for frac, color in [(0.25, (40, 40, 40)),
                        (0.50, (70, 70, 70)),
                        (0.75, (40, 40, 40))]:
        py = margin_top + int(plot_h * (1.0 - frac))
        cv2.line(strip, (0, py), (w - 1, py), color, 1)
        cv2.putText(strip, f"{y_top * frac:.2f}", (4, py - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)

    xs = (w - 1) - (ages / win_s * (w - 1)).astype(np.int32)
    ys = margin_top + (plot_h * (1.0 - rates / y_top)).astype(np.int32)
    ys = np.clip(ys, margin_top, margin_top + plot_h)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    polyline = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(strip, [polyline], False, (0, 230, 0), 1, cv2.LINE_AA)
    cv2.circle(strip, (int(xs[-1]), int(ys[-1])), 3, (0, 255, 255), -1, cv2.LINE_AA)

    cur = float(rates[np.argmin(ages)])
    p50 = float(np.percentile(rates, 50))
    p99 = float(np.percentile(rates, 99))
    rmean = float(rates.mean())
    cv2.putText(strip,
                f"now {cur:5.3f}  mean {rmean:5.3f}  "
                f"p50 {p50:5.3f}  p99 {p99:5.3f}  max {rmax:5.3f}  Mev/s",
                (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    if header_extras:
        cv2.putText(strip, "  ".join(header_extras), (8, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1, cv2.LINE_AA)


class HotPixelFilter:
    """Sticky hot-pixel mask. Cheap O(events) check per batch; the
    expensive bincount-based histogram update runs at ~20 Hz (every
    `update_every_s`), so we don't allocate a 2.4 MB int64 array every
    millisecond."""

    def __init__(self, H, W, refresh_s=4.0, hot_factor=50.0,
                 min_active=200, update_every_s=0.05):
        self.H = H
        self.W = W
        self.size = H * W
        self.count = np.zeros(self.size, dtype=np.int64)
        self.hot_mask = np.zeros(self.size, dtype=bool)
        self.refresh_s = refresh_s
        self.hot_factor = hot_factor
        self.min_active = min_active
        self.update_every_s = update_every_s
        self.next_refresh_s = None
        self.next_update_s = None
        self.pending_flat = []
        self._n_hot_cached = 0

    def reset(self):
        self.count[:] = 0
        self.hot_mask[:] = False
        self.next_refresh_s = None
        self.next_update_s = None
        self.pending_flat = []
        self._n_hot_cached = 0

    def filter(self, ev, t_now_s):
        if ev.size == 0:
            return ev
        if self.next_refresh_s is None:
            self.next_refresh_s = t_now_s + self.refresh_s
            self.next_update_s  = t_now_s + self.update_every_s
        flat = (ev['y'].astype(np.int64) * self.W +
                ev['x'].astype(np.int64))
        keep = ~self.hot_mask[flat]
        kept = ev[keep]
        if kept.size > 0:
            self.pending_flat.append(flat[keep])
        if t_now_s >= self.next_update_s:
            self._batched_update(t_now_s)
            self.next_update_s = t_now_s + self.update_every_s
        return kept

    def _batched_update(self, t_now_s):
        if self.pending_flat:
            big = np.concatenate(self.pending_flat)
            self.pending_flat = []
            self.count += np.bincount(big, minlength=self.size).astype(np.int64)
        if t_now_s >= self.next_refresh_s:
            self._recompute()
            self.next_refresh_s = t_now_s + self.refresh_s

    def _recompute(self):
        active = self.count[self.count > 0]
        if active.size < self.min_active:
            self.count[:] = 0
            return
        median = max(int(np.median(active)), 1)
        thr = max(median * self.hot_factor, 50)
        self.hot_mask |= (self.count > thr)
        self.count[:] = 0
        self._n_hot_cached = int(self.hot_mask.sum())

    def n_hot(self):
        return self._n_hot_cached


class EventDrainer(threading.Thread):
    """Pulls batches as fast as the camera can deliver. Applies ROI
    mask + HPF. Parks accepted events under a lock for the render
    thread to take()."""

    def __init__(self, cam, hpf, state):
        super().__init__(daemon=True)
        self.cam = cam
        self.hpf = hpf
        self.state = state
        self.lock = threading.Lock()
        self._chunks = []
        self._n_kept = 0
        self._n_raw = 0
        self._stop = False

    def stop(self):
        self._stop = True

    def take(self):
        with self.lock:
            chunks  = self._chunks
            n_kept  = self._n_kept
            n_raw   = self._n_raw
            self._chunks = []
            self._n_kept = 0
            self._n_raw = 0
        return chunks, n_raw, n_kept

    def run(self):
        while not self._stop:
            try:
                if not self.cam.isRunning():
                    break
                batch = self.cam.getNextEventBatch()
            except Exception as e:
                print(f"  [drainer] cam read failed: {e}", flush=True)
                time.sleep(0.001)
                continue
            if batch is None or batch.size() == 0:
                # No backoff sleep needed if the camera blocks; if it returns
                # immediately on empty, briefly yield.
                time.sleep(0.0001)
                continue

            ev = batch.numpy()
            n_raw = ev.size

            roi_x, roi_y, roi_w, roi_h = self.state["roi_xywh"]
            if (roi_x != 0 or roi_y != 0
                    or roi_w != self.state["W_full"]
                    or roi_h != self.state["H_full"]):
                m_roi = ((ev['x'] >= roi_x) & (ev['x'] < roi_x + roi_w) &
                         (ev['y'] >= roi_y) & (ev['y'] < roi_y + roi_h))
                ev = ev[m_roi]

            if self.state["use_hpf"]:
                ev = self.hpf.filter(ev, time.monotonic())

            n_kept = ev.size
            with self.lock:
                self._n_raw  += n_raw
                self._n_kept += n_kept
                if n_kept > 0:
                    self._chunks.append(ev)


def probe_thresholds(cam, dwell_s=1.5):
    candidates = [1, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    print(f"Probing contrast thresholds, dwell={dwell_s}s each.")
    print(f"{'threshold':>10} | {'rate Mev/s':>12} | result")
    for v in candidates:
        try:
            cam.setContrastThresholdOn(v)
            cam.setContrastThresholdOff(v)
        except Exception as e:
            print(f"{v:10d} | {'---':>12} | rejected: {e}")
            continue
        t_prep = time.monotonic() + 0.4
        while time.monotonic() < t_prep:
            cam.getNextEventBatch()
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < dwell_s:
            b = cam.getNextEventBatch()
            if b is not None:
                n += b.size()
        rate_mev = n / dwell_s / 1e6
        print(f"{v:10d} | {rate_mev:12.4f} | accepted (n={n})")
    print("Done. Useful max ≈ value where rate hits your noise floor.")


def motion_probe(cam, thresholds, dwell_s):
    """Sweep contrast thresholds while the user keeps producing a consistent
    motion in front of the camera. Same motion → comparable rates across
    thresholds. Each threshold held for `dwell_s` seconds.

    Why: when you wave by hand, the motion pattern differs between trials,
    so eyeballing rate change across threshold settings in the GUI is noisy.
    This routine cycles thresholds automatically so you only have to keep
    moving — the script handles the rest.
    """
    print(f"\n=== motion probe ===")
    print(f"Wave / move in front of camera CONTINUOUSLY.")
    print(f"Will cycle through {len(thresholds)} thresholds, "
          f"{dwell_s}s each.")
    print(f"Total time: ~{len(thresholds) * (dwell_s + 0.5):.0f}s.")
    print(f"Starting in 3s — get ready...")
    for i in range(3, 0, -1):
        print(f"  {i}..."); time.sleep(1.0)
    print(f"GO. Keep moving.\n")
    print(f"{'threshold':>10} | {'rate kev/s':>12} | {'n_events':>10} | bar")
    print("-" * 70)
    results = []
    max_rate = 0
    for v in thresholds:
        try:
            cam.setContrastThresholdOn(v)
            cam.setContrastThresholdOff(v)
        except Exception as e:
            print(f"{v:10d} | rejected: {e}")
            continue
        # 0.3s warmup: drain stale batches, let chip settle to new threshold.
        t_warm = time.monotonic() + 0.3
        while time.monotonic() < t_warm:
            cam.getNextEventBatch()
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < dwell_s:
            b = cam.getNextEventBatch()
            if b is not None:
                n += b.size()
        rate_kev = n / dwell_s / 1e3
        max_rate = max(max_rate, rate_kev)
        results.append((v, rate_kev, n))
        bar_len = int(40 * rate_kev / max(max_rate, 1.0))
        bar = "█" * bar_len
        print(f"{v:10d} | {rate_kev:12.1f} | {n:10d} | {bar}", flush=True)
    print()
    print("Done. Stop moving.")
    print()
    if len(results) >= 2:
        rates = [r for _, r, _ in results]
        if max(rates) > 1.5 * min(rates):
            print(f"✓ Threshold IS having an effect "
                  f"(min={min(rates):.0f} max={max(rates):.0f} kev/s).")
            print(f"  Pick the threshold giving you the event rate you want.")
        else:
            print(f"⚠  Rates barely change ({min(rates):.0f}–{max(rates):.0f}"
                  f" kev/s) — either chip floor dominates or motion was too "
                  f"intermittent. Try more vigorous / consistent motion, or "
                  f"narrow the threshold range to where you expect transitions.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-ms", type=int, default=33)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--contrast-on",  type=int, default=9)
    ap.add_argument("--contrast-off", type=int, default=9)
    ap.add_argument("--snapshot-dir", type=str, default=None)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--rate-window-s", type=float, default=RATE_HISTORY_S)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--hot-pixel-filter", action="store_true")
    ap.add_argument("--roi", type=str, default=None,
                    help="Initial ROI 'cx_pct,cy_pct,w_pct,h_pct'")
    ap.add_argument("--cam-interval-us", type=int, default=2000,
                    help="DVXplorer setTimeInterval / setMIPITimeoutValue [us]")
    ap.add_argument("--probe-thresholds", action="store_true")
    ap.add_argument("--motion-probe", action="store_true",
                    help="Cycle through thresholds while you keep moving in "
                         "front of the camera. Compares rates under one "
                         "consistent motion. Pairs with --motion-probe-* below.")
    ap.add_argument("--motion-probe-thresholds", type=str,
                    default="5,8,10,12,15,20,30,50",
                    help="Comma-separated list (default: 5,8,10,12,15,20,30,50)")
    ap.add_argument("--motion-probe-dwell-s", type=float, default=3.0,
                    help="Seconds to hold each threshold (default: 3.0)")
    ap.add_argument("--list-methods", action="store_true")
    args = ap.parse_args()

    cam = dv.io.camera.open()
    W, H = cam.getEventResolution()
    print(f"opened {cam.getCameraName()} (serial {cam.getSerialNumber()})  "
          f"resolution={W}x{H}", flush=True)

    if args.list_methods:
        print("Camera methods:")
        for name in sorted(dir(cam)):
            if not name.startswith("_"):
                print(f"  {name}")

    # Pull the camera-side batch interval down for low-latency drain.
    if hasattr(cam, "setTimeInterval"):
        _safe(cam.setTimeInterval,
              _dt.timedelta(microseconds=args.cam_interval_us),
              "setTimeInterval")
    if hasattr(cam, "setMIPITimeoutValue"):
        _safe(cam.setMIPITimeoutValue,
              _dt.timedelta(microseconds=args.cam_interval_us),
              "setMIPITimeoutValue")

    if args.probe_thresholds:
        probe_thresholds(cam)
        return 0

    if args.motion_probe:
        try:
            thr_list = [int(x) for x in args.motion_probe_thresholds.split(",")
                        if x.strip()]
        except ValueError as e:
            print(f"Bad --motion-probe-thresholds: {e}")
            return 1
        motion_probe(cam, thr_list, args.motion_probe_dwell_s)
        return 0

    _safe(cam.setContrastThresholdOn,  args.contrast_on,  "ContrastThresholdOn")
    _safe(cam.setContrastThresholdOff, args.contrast_off, "ContrastThresholdOff")

    state = {
        "window_ms": max(1, args.window_ms),
        "sub_level": 0,
        "roi_cx_pct": 50, "roi_cy_pct": 50,
        "roi_w_pct":  100, "roi_h_pct":  100,
        "use_hpf":    args.hot_pixel_filter,
        "roi_xywh":   (0, 0, W, H),
        "hw_roi_method": None,
        "W_full":     W,
        "H_full":     H,
    }

    if args.roi:
        try:
            cx, cy, ww, hh = [int(v.strip()) for v in args.roi.split(",")]
            state["roi_cx_pct"] = max(0, min(100, cx))
            state["roi_cy_pct"] = max(0, min(100, cy))
            state["roi_w_pct"]  = max(1, min(100, ww))
            state["roi_h_pct"]  = max(1, min(100, hh))
        except Exception as e:
            print(f"  [warn] --roi parse failed: {e}", flush=True)

    def _apply_roi():
        roi_w = max(2, W * state["roi_w_pct"] // 100)
        roi_h = max(2, H * state["roi_h_pct"] // 100)
        cx = W * state["roi_cx_pct"] // 100
        cy = H * state["roi_cy_pct"] // 100
        roi_x = max(0, min(W - roi_w, cx - roi_w // 2))
        roi_y = max(0, min(H - roi_h, cy - roi_h // 2))
        state["roi_xywh"] = (roi_x, roi_y, roi_w, roi_h)
        method = _try_set_hw_roi(cam, roi_x, roi_y, roi_w, roi_h)
        if method != state["hw_roi_method"]:
            state["hw_roi_method"] = method
            if method:
                print(f"  [info] HW ROI via {method}() = "
                      f"({roi_x},{roi_y},{roi_w},{roi_h})", flush=True)
            else:
                print("  [info] HW ROI not exposed; software mask only.",
                      flush=True)

    _apply_roi()

    hpf = HotPixelFilter(H, W)

    use_window = not args.no_window
    display_w = max(1, int(W * args.scale))
    display_h = max(1, int(H * args.scale))

    DISPLAY_WIN = "DVXplorer (ESC quits)"
    CTRL_WIN    = "Controls"

    if use_window:
        try:
            cv2.namedWindow(DISPLAY_WIN, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(CTRL_WIN,    cv2.WINDOW_NORMAL)
            cv2.resizeWindow(CTRL_WIN, CTRL_W, max(CTRL_H, 240))
            try:
                cv2.moveWindow(DISPLAY_WIN, 0, 0)
                cv2.moveWindow(CTRL_WIN, display_w + 30, 0)
            except cv2.error:
                pass

            cv2.createTrackbar("Contrast ON",  CTRL_WIN, args.contrast_on,
                               CONTRAST_MAX,
                               lambda v: _safe(cam.setContrastThresholdOn,
                                               max(1, v), "ContrastThresholdOn"))
            cv2.createTrackbar("Contrast OFF", CTRL_WIN, args.contrast_off,
                               CONTRAST_MAX,
                               lambda v: _safe(cam.setContrastThresholdOff,
                                               max(1, v), "ContrastThresholdOff"))
            cv2.createTrackbar("Subsample", CTRL_WIN,
                               state["sub_level"], len(SUBSAMPLE_LADDER) - 1,
                               lambda v: (state.update(sub_level=max(0, min(v, len(SUBSAMPLE_LADDER) - 1))),
                                          _safe(cam.setSubSampleHorizontal,
                                                SUBSAMPLE_LADDER[state["sub_level"]][1],
                                                "SubSampleHorizontal"),
                                          _safe(cam.setSubSampleVertical,
                                                SUBSAMPLE_LADDER[state["sub_level"]][1],
                                                "SubSampleVertical")))
            cv2.createTrackbar("ROI cx %", CTRL_WIN, state["roi_cx_pct"], 100,
                               lambda v: (state.update(roi_cx_pct=v), _apply_roi()))
            cv2.createTrackbar("ROI cy %", CTRL_WIN, state["roi_cy_pct"], 100,
                               lambda v: (state.update(roi_cy_pct=v), _apply_roi()))
            cv2.createTrackbar("ROI W %",  CTRL_WIN, state["roi_w_pct"], 100,
                               lambda v: (state.update(roi_w_pct=max(1, v)), _apply_roi()))
            cv2.createTrackbar("ROI H %",  CTRL_WIN, state["roi_h_pct"], 100,
                               lambda v: (state.update(roi_h_pct=max(1, v)), _apply_roi()))
            cv2.createTrackbar("HotPixFilt", CTRL_WIN,
                               1 if state["use_hpf"] else 0, 1,
                               lambda v: (state.update(use_hpf=bool(v)),
                                          hpf.reset() if not v else None))
            cv2.createTrackbar("Window ms", CTRL_WIN, state["window_ms"], 200,
                               lambda v: state.update(window_ms=max(1, v)))
            cv2.createTrackbar("ReadoutFPS", CTRL_WIN, 1,
                               len(READOUT_FPS_LADDER) - 1,
                               lambda v: _safe(cam.setReadoutFPS,
                                               READOUT_FPS_LADDER[v][1],
                                               "ReadoutFPS"))
        except cv2.error as e:
            print(f"cv2.namedWindow failed ({e}); running headless", flush=True)
            use_window = False

    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)

    stop = False

    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    drainer = EventDrainer(cam, hpf, state)
    drainer.start()

    rate_samples = collections.deque()
    bin_t0 = time.monotonic()
    bin_n_kept = 0
    bin_n_raw = 0

    log_t0 = bin_t0
    log_n_kept = 0
    log_n_raw  = 0
    log_frames = 0
    last_status = ""

    recent = None
    snap_t0 = time.monotonic()

    last_render = time.monotonic()
    render_interval = 1.0 / max(1.0, args.fps)

    try:
        while cam.isRunning() and not stop:
            now = time.monotonic()
            sleep_dur = (last_render + render_interval) - now
            if sleep_dur > 0:
                time.sleep(min(sleep_dur, 0.020))
                continue

            chunks, n_raw, n_kept = drainer.take()
            bin_n_raw  += n_raw
            bin_n_kept += n_kept
            log_n_raw  += n_raw
            log_n_kept += n_kept

            if (now - bin_t0) * 1000.0 >= RATE_BIN_MS:
                dt = now - bin_t0
                rate_samples.append((now, bin_n_kept / dt / 1e6))
                bin_n_kept = 0
                bin_n_raw = 0
                bin_t0 = now
                cutoff = now - args.rate_window_s - 1.0
                while rate_samples and rate_samples[0][0] < cutoff:
                    rate_samples.popleft()

            sub = state["sub_level"]
            step = 1 << sub
            roi_x, roi_y, roi_w, roi_h = state["roi_xywh"]
            W_eff = max(1, (roi_w + step - 1) // step)
            H_eff = max(1, (roi_h + step - 1) // step)

            if chunks:
                if recent is None or recent.size == 0:
                    recent = np.concatenate(chunks)
                else:
                    recent = np.concatenate([recent] + chunks)
            if recent is not None and recent.size > 0:
                latest_t_us = int(recent['timestamp'][-1])
                keep_us = max(state['window_ms'] * RECENT_KEEP_MULT,
                              RATE_BIN_MS * 2) * 1000
                m_keep = recent['timestamp'] > latest_t_us - keep_us
                if not m_keep.all():
                    recent = recent[m_keep]
                win_us = state["window_ms"] * 1000
                m = recent['timestamp'] > latest_t_us - win_us
                sliced = recent[m]
            else:
                sliced = None

            frame = np.zeros((H_eff, W_eff, 3), dtype=np.uint8)
            if sliced is not None and sliced.size > 0:
                ex_local = ((sliced['x'].astype(np.int32) - roi_x) >> sub)
                ey_local = ((sliced['y'].astype(np.int32) - roi_y) >> sub)
                ep = sliced['polarity']
                ok = ((ex_local >= 0) & (ex_local < W_eff) &
                      (ey_local >= 0) & (ey_local < H_eff))
                ex_local = ex_local[ok]; ey_local = ey_local[ok]; ep = ep[ok]
                pos = ep != 0
                if pos.any():
                    frame[ey_local[pos], ex_local[pos]] = (0, 0, 255)
                if (~pos).any():
                    frame[ey_local[~pos], ex_local[~pos]] = (255, 0, 0)

            if use_window:
                disp = cv2.resize(frame, (display_w, display_h),
                                  interpolation=cv2.INTER_NEAREST)
                sub_label = SUBSAMPLE_LADDER[sub][0]
                hw_tag = state["hw_roi_method"] or "sw"
                header = (f"sub={sub_label} {W_eff}x{H_eff}  "
                          f"ROI[{roi_x},{roi_y},{roi_w}x{roi_h} {hw_tag}]  "
                          f"hot={hpf.n_hot()}")
                if last_status:
                    header = header + " | " + last_status
                cv2.putText(disp, header, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)

                strip = np.zeros((STRIP_H, display_w, 3), dtype=np.uint8)
                _draw_rate_strip(strip, rate_samples, args.rate_window_s,
                                 header_extras=(f"win {args.rate_window_s:.1f}s",
                                                f"bin {RATE_BIN_MS} ms",
                                                f"fps {args.fps:.0f}",
                                                f"n={len(rate_samples)}"))
                combined = np.vstack([disp, strip])

                ctrl_canvas = np.full((CTRL_H, CTRL_W, 3), 30, dtype=np.uint8)
                cv2.putText(ctrl_canvas, "controls", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 220),
                            1, cv2.LINE_AA)
                cv2.putText(ctrl_canvas,
                            f"raw {log_n_raw / max(now - log_t0, 1e-6) / 1e6:5.2f} | "
                            f"kept {log_n_kept / max(now - log_t0, 1e-6) / 1e6:5.2f} Mev/s",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180),
                            1, cv2.LINE_AA)

                try:
                    cv2.imshow(DISPLAY_WIN, combined)
                    cv2.imshow(CTRL_WIN,    ctrl_canvas)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        stop = True
                except cv2.error as e:
                    print(f"cv2.imshow failed ({e}); switching to headless",
                          flush=True)
                    use_window = False

            if args.snapshot_dir and now - snap_t0 >= 1.0:
                cv2.imwrite(os.path.join(args.snapshot_dir,
                                         f"frame_{int(now):d}.png"), frame)
                snap_t0 = now

            log_frames += 1
            if now - log_t0 >= 1.0:
                dt = now - log_t0
                rate_kept = log_n_kept / dt / 1e6
                rate_raw  = log_n_raw  / dt / 1e6
                last_status = (f"raw {rate_raw:5.2f}  kept {rate_kept:5.2f} "
                               f"Mev/s   {log_frames:3d} fps")
                print(f"  {last_status}  hot={hpf.n_hot()}  "
                      f"sub={SUBSAMPLE_LADDER[state['sub_level']][0]}",
                      flush=True)
                log_n_kept = 0
                log_n_raw = 0
                log_frames = 0
                log_t0 = now

            last_render = now

    finally:
        drainer.stop()
        drainer.join(timeout=1.0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
