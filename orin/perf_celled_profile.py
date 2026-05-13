"""Phase-instrumented profiler for the celled GPU kernel.

Compiles ssla_s2_s3_head_celled_profile.cuh which records clock64() at
24 phase boundaries per batch. Prints a per-phase µs breakdown
(p50 across batches) so we can see where the 303 µs per batch goes.
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
N_PHASES = 24

PHASE_NAMES = [
    "POP", "LOAD", "DISPATCH_S2",
    "L4_RES", "L4_ZERO", "L4_COMPUTE", "L4_GATHER",
    "L5_RES", "L5_ZERO", "L5_COMPUTE", "L5_GATHER",
    "TDROP_S2", "POOL", "DISPATCH_S3",
    "L6_RES", "L6_ZERO", "L6_COMPUTE", "L6_GATHER",
    "L7_RES", "L7_ZERO", "L7_COMPUTE", "L7_GATHER",
    "TDROP_S3", "OUT",
]

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
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
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

    # Build sources: profile header includes production header.
    src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head_celled_profile.cuh")).read()
    proto = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
    prod  = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head_celled.cuh")).read()
    print(f"Compiling profiling kernel ...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_s2_s3_head_celled_profile.cu",
                     headers={
                         "proto_layer_pair.cuh": proto,
                         "ssla_s2_s3_head_celled.cuh": prod,
                     })
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s")

    func = mod.get_function("k_ssla_s2s3_celled_drain_n_profile")
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
        nbytes = n_batches[b] * N_PHASES * 8
        p, ka = cuda_util.alloc_managed(nbytes)
        ctypes.memset(p, 0, nbytes)
        keepalive.append(ka); clk_dev.append(p)
        clk_view.append(np.frombuffer(ka, dtype=np.uint64).reshape(n_batches[b], N_PHASES))

    SM_CLK_HZ = 918_000_000
    print(f"\nGPU drain (profiled): 2 blocks × {threads_per_block} threads, "
          f"smem={SMEM} B (n0={rec_block[0].size}, n1={rec_block[1].size})\n")

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
    print(f"  wall: {(time.monotonic()-t0)*1e3:.1f} ms")

    # Aggregate: per-block, per-phase deltas (phase[i+1] - phase[i]).
    for b in range(N_BLOCKS):
        cv = clk_view[b]
        if cv.size == 0:
            continue
        # Skip head/tail 5% for warmup/boundary.
        head = max(1, len(cv) // 20); tail = max(1, len(cv) // 20)
        cv = cv[head:-tail] if len(cv) > head + tail else cv
        # Reject batches where any stamp is zero (skipped or partial).
        valid = (cv > 0).all(axis=1)
        cv = cv[valid]
        if cv.size == 0:
            print(f"\n[block {b}] no valid batches"); continue
        # Phase delta: cv[:, i+1] - cv[:, i] for i in [0, N_PHASES-1)
        # But our phases start with POP (= start clock). So phase i duration
        # = cv[:, i+1] - cv[:, i].
        d_clk = np.diff(cv.astype(np.int64), axis=1)        # shape (B, 23)
        d_us  = d_clk * (1e6 / SM_CLK_HZ)
        total = (cv[:, -1].astype(np.int64) - cv[:, 0].astype(np.int64)) \
                * (1e6 / SM_CLK_HZ)
        print(f"\n[block {b}] {len(cv)} valid batches, "
              f"total p50 = {np.percentile(total, 50):.1f} µs, "
              f"mean = {total.mean():.1f} µs")
        print(f"  {'phase':<14} {'p50_us':>8} {'mean_us':>8} {'share%':>8}")
        for i in range(N_PHASES - 1):
            from_n = PHASE_NAMES[i]; to_n = PHASE_NAMES[i+1]
            p50 = float(np.percentile(d_us[:, i], 50))
            mn  = float(d_us[:, i].mean())
            share = 100.0 * mn / total.mean() if total.mean() > 0 else 0.0
            print(f"  {from_n+'→'+to_n:<14} {p50:>8.2f} {mn:>8.2f} {share:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
