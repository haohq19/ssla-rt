"""Throughput/latency-only harness for the celled GPU kernel.

Skips the CPU oracle (which takes 200 s+ at n=200k). Use this for fast
iteration when you only need GPU timing — NOT for correctness verification.

Output: wallclock kev/s + per-batch latency histogram (clock64) + per-event
p50/p99 derived from BATCH division.

Run P1 (bench_s2_s3_head_celled.py) separately to confirm correctness.
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import CudaModule, arg_ptr, arg_i32  # noqa: E402
from orin.hybrid_common import (  # noqa: E402
    INPUT_DTYPE, OUTPUT_DTYPE,
    C1, C2, C3, N_BLOCKS,
    build_random_layers, alloc_tdrop_counters, build_config,
)

H2, W2 = 16, 20
H3, W3 = H2 >> 1, W2 >> 1
HALO_S2 = 2
N_WARPS = 9
BATCH = 8
OUT_MAX = 96

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def _smem_size_bytes() -> int:
    C1, C2 = 24, 48
    event_slot = OUT_MAX * 4 + OUT_MAX * 4 + 5 * 4 + 4 + 8 + 8  # 5 ints + pad + 2 u64 (t_push_ns, t_emit_ns)
    smem = (
        event_slot * BATCH
        + BATCH * N_WARPS * OUT_MAX * 4
        + N_WARPS * OUT_MAX * 4
        + N_WARPS * OUT_MAX * 4
        + N_WARPS * BATCH * 4
        + N_WARPS * BATCH * 4
        + N_WARPS * 4
        + BATCH * 4
        + BATCH * 4
        + 9 * 3 * C2 * C1 * 4   # L4 qvgIn smem cache (121 KB)
    )
    return ((smem + 15) // 16) * 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cpu_layers, gpu_layers, ka_layers = build_random_layers(rng, H2, W2, H3, W3)
    tdrop_s2_dev, tdrop_s3_dev, ka_tdrop = alloc_tdrop_counters(H2, W2, H3, W3)
    p_cfg, cfg_ka = build_config(H2, W2, H3, W3, args.tdrop, gpu_layers,
                                 tdrop_s2_dev, tdrop_s3_dev)
    keepalive = list(ka_layers) + list(ka_tdrop) + [cfg_ka]
    strip_w = W2 // N_BLOCKS

    n = args.n
    events_x = rng.integers(0, W2, n).astype(np.int32)
    events_y = rng.integers(0, H2, n).astype(np.int32)
    events_f1 = rng.normal(scale=0.5, size=(n, C1)).astype(np.float32)

    proc_lo = [max(0, 0 - HALO_S2),       max(0, strip_w - HALO_S2)]
    proc_hi = [min(W2, strip_w + HALO_S2), min(W2, W2 + HALO_S2)]
    masks = [
        (events_x >= proc_lo[0]) & (events_x < proc_hi[0]),
        (events_x >= proc_lo[1]) & (events_x < proc_hi[1]),
    ]
    rec_block = []
    for blk in range(N_BLOCKS):
        idx = np.nonzero(masks[blk])[0]
        rec = np.zeros(idx.size, dtype=INPUT_DTYPE)
        if idx.size > 0:
            rec["t"] = rng.uniform(0, 1, idx.size).astype(np.float32)
            rec["x"] = events_x[idx].astype(np.uint16)
            rec["y"] = events_y[idx].astype(np.uint16)
            rec["feat1"] = events_f1[idx]
        rec_block.append(rec)

    ring_dev, out_dev = [], []
    for blk in range(N_BLOCKS):
        rec = rec_block[blk]
        if rec.size == 0:
            ring_dev.append(0); out_dev.append(0); continue
        p_in, ka_in = cuda_util.alloc_managed(rec.nbytes)
        np.frombuffer(ka_in, dtype=INPUT_DTYPE).reshape(-1)[:] = rec
        keepalive.append(ka_in); ring_dev.append(p_in)
        p_out, ka_out = cuda_util.alloc_managed(rec.size * OUTPUT_DTYPE.itemsize)
        ctypes.memset(p_out, 0, rec.size * OUTPUT_DTYPE.itemsize)
        keepalive.append(ka_out); out_dev.append(p_out)

    src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head_celled.cuh")).read()
    proto = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
    print(f"Compiling celled kernel ({len(src)+len(proto)} bytes total) ...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_s2_s3_head_celled.cu",
                     headers={"proto_layer_pair.cuh": proto})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s")

    func = mod.get_function("k_ssla_s2s3_celled_drain_n")
    threads_per_block = N_WARPS * 32
    SMEM = _smem_size_bytes()

    # Opt-in to dynamic smem >48 KB (sm_87 supports up to ~167 KB / block).
    from cuda import cuda as _cuda
    _err, = _cuda.cuFuncSetAttribute(
        func._func,
        _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SMEM)
    if int(_err) != 0:
        raise RuntimeError(f"cuFuncSetAttribute(MAX_DYNAMIC_SHARED): err={int(_err)}, smem={SMEM}")

    n_batches = [((rec_block[b].size + BATCH - 1) // BATCH) for b in range(N_BLOCKS)]
    clk_dev, clk_view = [], []
    for b in range(N_BLOCKS):
        if n_batches[b] == 0:
            clk_dev.append(0); clk_view.append(np.empty(0, dtype=np.uint64))
            continue
        nbytes = n_batches[b] * 2 * 8
        p, ka = cuda_util.alloc_managed(nbytes)
        keepalive.append(ka); clk_dev.append(p)
        clk_view.append(np.frombuffer(ka, dtype=np.uint64).reshape(n_batches[b], 2))

    SM_CLK_HZ = 918_000_000
    print(f"\nGPU drain: 2 blocks × {threads_per_block} threads, smem={SMEM} B "
          f"(n0={rec_block[0].size}, n1={rec_block[1].size})\n")
    print(f"{'run':>3} | {'wall_kev/s':>10} | "
          f"block 0: {'p50':>6} {'p99':>6} {'max':>6} mean | "
          f"block 1: {'p50':>6} {'p99':>6} {'max':>6} mean (µs/batch)")
    for run in range(args.runs):
        # Reset clk buffers
        for b in range(N_BLOCKS):
            if clk_dev[b]:
                ctypes.memset(clk_dev[b], 0, n_batches[b] * 2 * 8)
        # Reset tdrop counters (per-block list of devptrs).
        for blk in range(N_BLOCKS):
            ctypes.memset(tdrop_s2_dev[blk], 0, H2 * W2)
            ctypes.memset(tdrop_s3_dev[blk], 0, H3 * W3)
        # Reset hidden states would matter for correctness only; we leave alone

        t0 = time.monotonic()
        func.launch((2, 1, 1), (threads_per_block, 1, 1), [
            arg_ptr(p_cfg),
            arg_ptr(ring_dev[0]),
            arg_ptr(ring_dev[1]),
            arg_i32(rec_block[0].size),
            arg_i32(rec_block[1].size),
            arg_ptr(out_dev[0]),
            arg_ptr(out_dev[1]),
            arg_ptr(clk_dev[0]),
            arg_ptr(clk_dev[1]),
        ], smem=SMEM)
        func.sync()
        gpu_dt = time.monotonic() - t0
        kevs = n / gpu_dt / 1e3

        stats = []
        for b in range(N_BLOCKS):
            cv = clk_view[b]
            if cv.size == 0:
                stats.append((0.0, 0.0, 0.0, 0.0)); continue
            valid = cv[:, 1] > 0
            cv = cv[valid]
            if cv.size == 0:
                stats.append((0.0, 0.0, 0.0, 0.0)); continue
            head = max(1, len(cv) // 20)
            tail = max(1, len(cv) // 20)
            cv = cv[head:-tail] if len(cv) > head + tail else cv
            dt_us = (cv[:, 1].astype(np.int64) - cv[:, 0].astype(np.int64)) \
                    * (1e6 / SM_CLK_HZ)
            stats.append((
                float(np.percentile(dt_us, 50)),
                float(np.percentile(dt_us, 99)),
                float(dt_us.max()),
                float(dt_us.mean()),
            ))
        p50_0, p99_0, mx_0, mn_0 = stats[0]
        p50_1, p99_1, mx_1, mn_1 = stats[1]
        print(f"{run:>3} | {kevs:>10.1f} | "
              f"         {p50_0:>6.1f} {p99_0:>6.1f} {mx_0:>6.0f} {mn_0:>5.1f} | "
              f"         {p50_1:>6.1f} {p99_1:>6.1f} {mx_1:>6.0f} {mn_1:>5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
