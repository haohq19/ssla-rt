"""P1 harness for the SSLA Hybrid GPU s2+s3 kernel.

Verifies the 2-block coop-per-event GPU kernel against a single-thread
CPU oracle for stages 2+3 of SSLA-S. Inputs are synthetic post-CPU-stage-1
records (t, x, y, feat1[24]) — the same shape the live pipeline will
push from libstage01_capi.so via per-block pinned rings.

Acceptance gates (HYBRID_DESIGN.md §5):
  * s2 / s3 owner-pass counts: within ±0.5 % of CPU oracle (expected
    EXACT match for offline harness — counters at owned cells see the
    same events as the oracle).
  * head-prediction max|Δ| at owned-and-passed events: ≤ 5.0
    (compared as raw s3 features for P1; head matvec lands in P2).
  * No deadlock / crash on 200 k events.

Run:
    python3 deploy/orin/bench_s2_s3_head.py
    python3 deploy/orin/bench_s2_s3_head.py --n 50000 --seed 7
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
    CudaModule, arg_ptr, arg_i32,  # noqa: E402
)
from orin.ssla_ref import LayerRef, layer_step  # noqa: E402
from orin.hybrid_common import (  # noqa: E402
    INPUT_DTYPE, OUTPUT_DTYPE,
    C1, C2, C3, N_BLOCKS,
    SSLA_S2S3_IN_OUT, SSLA_S2S3_KERNELS,
    build_random_layers, alloc_tdrop_counters, build_config,
    block_proc_range, alloc_and_fill,
)


# Geometry: aligned to even-pool dimensions for clean P1 verification.
# Live deploy at 60×80 effective resolution would give H2=15 (odd) → H3=7,
# which collides with the SSLA pipeline's `>>1` pool when evy is on the
# bottom row (evy=14 → s3_y=7, out of bounds for H3=7). The same edge
# exists in the existing cpp/methods/ssla and ssla_step.cuh — pre-existing
# concern, not something to debug here. P1 uses H2=16 (H3=8) to stay
# clear of that case; the kernel itself reads H2/W2/H3/W3 from config so
# nothing in the architecture depends on the choice.
H2, W2 = 16, 20
H3, W3 = H2 >> 1, W2 >> 1   # 8, 10

HALO_S2  = 2                # halo=2 strict at s2 res


KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


# ---- CPU oracle --------------------------------------------------------

def _tdrop(counter: np.ndarray, window: int, idx: int) -> bool:
    pre = int(counter[idx])
    counter[idx] = (pre + 1) & 0xff
    if window <= 1:
        return True
    return (pre % window) == 0


def cpu_oracle(events_x, events_y, events_feat1, cpu_layers,
               tdrop_window: int):
    """Single-thread reference: process every event through stages 2+3 in
    arrival order. Returns (pass2_count, pass3_count, s3_feats_per_event)
    where s3_feats_per_event[i] is the post-stage-3 feature for event i if
    it passed both tdrops, else None."""
    n = events_x.size
    cpu_tdrop_s2 = np.zeros(H2 * W2, dtype=np.uint8)
    cpu_tdrop_s3 = np.zeros(H3 * W3, dtype=np.uint8)
    pass2_count = 0
    pass3_count = 0
    s3_per_event = [None] * n

    L4, L5, L6, L7 = cpu_layers

    for i in range(n):
        evx, evy = int(events_x[i]), int(events_y[i])
        feat_in = events_feat1[i].astype(np.float32)
        # ---- stage 2
        buf = layer_step(feat_in, L4.H, L4.qvgIn, L4.goW, L4.input_proj,
                         L4.ln_gamma, L4.ln_beta,
                         ev_x=evx, ev_y=evy, Hl=H2, Wl=W2, K=L4.K)
        s2_out = layer_step(buf, L5.H, L5.qvgIn, L5.goW, L5.input_proj,
                            L5.ln_gamma, L5.ln_beta,
                            ev_x=evx, ev_y=evy, Hl=H2, Wl=W2, K=L5.K)
        idx2 = evy * W2 + evx
        if not _tdrop(cpu_tdrop_s2, tdrop_window, idx2):
            continue
        pass2_count += 1
        # ---- stage 3
        s3x, s3y = evx >> 1, evy >> 1
        buf = layer_step(s2_out, L6.H, L6.qvgIn, L6.goW, L6.input_proj,
                         L6.ln_gamma, L6.ln_beta,
                         ev_x=s3x, ev_y=s3y, Hl=H3, Wl=W3, K=L6.K)
        s3_out = layer_step(buf, L7.H, L7.qvgIn, L7.goW, L7.input_proj,
                            L7.ln_gamma, L7.ln_beta,
                            ev_x=s3x, ev_y=s3y, Hl=H3, Wl=W3, K=L7.K)
        idx3 = s3y * W3 + s3x
        if not _tdrop(cpu_tdrop_s3, tdrop_window, idx3):
            continue
        pass3_count += 1
        s3_per_event[i] = s3_out.copy()

    return pass2_count, pass3_count, s3_per_event


# ---- main --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-delta", type=float, default=5.0,
                    help="acceptance threshold for max|Δ| on s3 features")
    ap.add_argument("--max-drop-frac", type=float, default=0.005,
                    help="acceptance threshold for drop-count drift fraction")
    ap.add_argument("--cpu-only", action="store_true",
                    help="run only the oracle (skip GPU)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cpu_layers, gpu_layers, ka_layers = build_random_layers(rng, H2, W2, H3, W3)
    tdrop_s2_dev, tdrop_s3_dev, ka_tdrop = alloc_tdrop_counters(H2, W2, H3, W3)
    p_cfg, cfg_ka = build_config(H2, W2, H3, W3, args.tdrop,
                                  gpu_layers, tdrop_s2_dev, tdrop_s3_dev)
    keepalive = list(ka_layers) + list(ka_tdrop) + [cfg_ka]
    strip_w = W2 // N_BLOCKS

    # Generate events.
    n = args.n
    events_t   = rng.uniform(0, 1, n).astype(np.float32)
    events_x   = rng.integers(0, W2, n).astype(np.int32)
    events_y   = rng.integers(0, H2, n).astype(np.int32)
    events_f1  = rng.normal(scale=0.5, size=(n, C1)).astype(np.float32)

    # Per-block proc range (halo broadcast at producer).
    proc_lo = [max(0, 0          - HALO_S2),
               max(0, strip_w    - HALO_S2)]
    proc_hi = [min(W2, strip_w   + HALO_S2),
               min(W2, W2        + HALO_S2)]
    print(f"Strip layout: block 0 owned [0,{strip_w}) proc [{proc_lo[0]},{proc_hi[0]}); "
          f"block 1 owned [{strip_w},{W2}) proc [{proc_lo[1]},{proc_hi[1]})")

    # Filter events for each block, preserving global order.
    masks = [
        (events_x >= proc_lo[0]) & (events_x < proc_hi[0]),
        (events_x >= proc_lo[1]) & (events_x < proc_hi[1]),
    ]
    rec_block = []
    global_idx_block = []     # original event index for each per-block ring slot
    for blk in range(N_BLOCKS):
        idx = np.nonzero(masks[blk])[0]
        rec = np.zeros(idx.size, dtype=INPUT_DTYPE)
        rec["t"]     = events_t[idx]
        rec["x"]     = events_x[idx].astype(np.uint16)
        rec["y"]     = events_y[idx].astype(np.uint16)
        rec["feat1"] = events_f1[idx]
        rec_block.append(rec)
        global_idx_block.append(idx)
        print(f"  block {blk}: {idx.size} events ({idx.size / n * 100:.1f} %)")

    # CPU oracle.
    print(f"\nCPU oracle on {n} events ...", flush=True)
    t0 = time.monotonic()
    cpu_pass2, cpu_pass3, cpu_s3 = cpu_oracle(events_x, events_y, events_f1,
                                              cpu_layers, args.tdrop)
    cpu_dt = time.monotonic() - t0
    print(f"  done in {cpu_dt:.1f}s   pass_s2={cpu_pass2}   pass_s3={cpu_pass3}")

    if args.cpu_only:
        return 0

    # Allocate per-block input rings + output slots in managed memory.
    ring_dev   = []
    out_dev    = []
    out_view   = []
    for blk in range(N_BLOCKS):
        rec = rec_block[blk]
        if rec.size == 0:
            ring_dev.append(0)
            out_dev.append(0)
            out_view.append(np.empty(0, dtype=OUTPUT_DTYPE))
            continue
        # Input ring
        nbytes_in = rec.nbytes
        p_in, ka_in = cuda_util.alloc_managed(nbytes_in)
        view_in = np.frombuffer(ka_in, dtype=INPUT_DTYPE).reshape(-1)
        view_in[:] = rec
        keepalive.append(ka_in)
        ring_dev.append(p_in)
        # Output slot array
        nbytes_out = rec.size * OUTPUT_DTYPE.itemsize
        p_out, ka_out = cuda_util.alloc_managed(nbytes_out)
        ctypes.memset(p_out, 0, nbytes_out)
        view_out = np.frombuffer(ka_out, dtype=OUTPUT_DTYPE).reshape(-1)
        keepalive.append(ka_out)
        out_dev.append(p_out)
        out_view.append(view_out)

    # Compile kernel.
    src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head.cuh")).read()
    prim  = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
    layer = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()
    print(f"\nCompiling kernel ({len(src)+len(prim)+len(layer)} bytes total) ...",
          flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_s2_s3_head.cu",
                     headers={"ssla_primitives.cuh": prim,
                              "ssla_layer.cuh":      layer})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s")

    func = mod.get_function("k_ssla_s2s3_drain_n")
    SMEM = (C1 + 2 * C3 + 5 * C3) * 4   # 696 floats × 4 B = 2784 B
    print(f"GPU drain: 2 blocks × 256 threads, smem={SMEM} B "
          f"(n0={rec_block[0].size}, n1={rec_block[1].size}) ...", flush=True)
    t0 = time.monotonic()
    func.launch((2, 1, 1), (256, 1, 1), [
        arg_ptr(p_cfg),
        arg_ptr(ring_dev[0]),
        arg_ptr(ring_dev[1]),
        arg_i32(rec_block[0].size),
        arg_i32(rec_block[1].size),
        arg_ptr(out_dev[0]),
        arg_ptr(out_dev[1]),
    ], smem=SMEM)
    func.sync()
    gpu_dt = time.monotonic() - t0
    print(f"  done in {gpu_dt*1e3:.1f}ms ({n/gpu_dt/1e3:.1f} kev/s effective)")

    # ---- comparison ----
    gpu_pass2 = sum(int((out_view[b]["pass2"] == 1).sum()) for b in range(N_BLOCKS))
    gpu_pass3 = sum(int((out_view[b]["pass3"] == 1).sum()) for b in range(N_BLOCKS))
    drop_drift_2 = abs(gpu_pass2 - cpu_pass2) / max(cpu_pass2, 1)
    drop_drift_3 = abs(gpu_pass3 - cpu_pass3) / max(cpu_pass3, 1)
    print(f"\nDrop counts:  cpu_pass_s2={cpu_pass2}, gpu_pass_s2={gpu_pass2}  "
          f"(drift={drop_drift_2:.4f})")
    print(f"             cpu_pass_s3={cpu_pass3}, gpu_pass_s3={gpu_pass3}  "
          f"(drift={drop_drift_3:.4f})")

    # Build map: global event idx → GPU s3 feature (only owner-passed events).
    gpu_s3_map = {}
    for blk in range(N_BLOCKS):
        gidx = global_idx_block[blk]
        ov   = out_view[blk]
        if ov.size == 0:
            continue
        sel = (ov["pass3"] == 1)
        for slot_i in np.nonzero(sel)[0]:
            gpu_s3_map[int(gidx[slot_i])] = ov["s3_feat"][slot_i].copy()

    # Diff GPU vs CPU s3 features at each event the CPU ran through head.
    max_delta = 0.0
    n_compared = 0
    n_only_cpu = 0
    n_only_gpu = 0
    for i in range(n):
        cpu_has = cpu_s3[i] is not None
        gpu_has = i in gpu_s3_map
        if cpu_has and gpu_has:
            d = float(np.max(np.abs(cpu_s3[i] - gpu_s3_map[i])))
            if d > max_delta:
                max_delta = d
            n_compared += 1
        elif cpu_has:
            n_only_cpu += 1
        elif gpu_has:
            n_only_gpu += 1

    print(f"\ns3_feat diff: max|Δ| = {max_delta:.4g} over {n_compared} matched events")
    if n_only_cpu or n_only_gpu:
        print(f"  WARN: pass-mismatch (cpu-only={n_only_cpu}, gpu-only={n_only_gpu})")

    ok_drops = drop_drift_2 <= args.max_drop_frac and drop_drift_3 <= args.max_drop_frac
    ok_diff  = max_delta <= args.max_delta
    if ok_drops and ok_diff:
        print(f"\nP1 PASS — drop drift ≤ {args.max_drop_frac}, max|Δ| ≤ {args.max_delta}")
        return 0
    print(f"\nP1 FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
