"""Multi-block GPU throughput sweep.

Launches the new k_ssla_s2s3_celled_drain_n_multi kernel with N blocks, one
per SM target. Each block has its own ring, hidden state, and tdrop. Events
are pre-split: block b processes events[b*M : (b+1)*M].

This is THROUGHPUT-ONLY; not a correctness test. Hidden state is per-block
and weights are shared. P1 isn't expected to pass here — that's a CPU-side
integration concern handled later.

Usage:
  python3 perf_celled_multi.py --n-blocks 4 --n 200000 --runs 3
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
    C1, C2, C3, N_BLOCKS, MAX_BLOCKS,
    CHybridS2S3Config,
    build_random_layers, build_config,
)

H2, W2 = 16, 20
H3, W3 = H2 >> 1, W2 >> 1
N_WARPS = 9
BATCH = 8
OUT_MAX = 96

KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def _smem_size_bytes() -> int:
    event_slot = OUT_MAX * 4 + OUT_MAX * 4 + 5 * 4 + 4 + 8  # 5 ints + pad + u64
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
    )
    return ((smem + 15) // 16) * 16


def build_n_block_layers(rng, n_blocks, H2, W2, H3, W3):
    """Build random weights (shared) + N hidden state buffers per layer."""
    # Re-use the standard 2-block builder for weight allocation by hijacking
    # N_BLOCKS via a quick monkeypatch? Cleaner: replicate the body but with
    # n_blocks instead.
    from orin.hybrid_common import (  # noqa
        SSLA_S2S3_IN_OUT, SSLA_S2S3_KERNELS, grid_for_layer,
        alloc_and_fill, reshape_ssla_layer, LayerRef,
    )
    cpu_layers = []
    gpu_layers = [[None] * 4 for _ in range(n_blocks)]
    keepalive = []
    for li in range(4):
        in_d, out_d = SSLA_S2S3_IN_OUT[li]
        K = SSLA_S2S3_KERNELS[li]
        A = K * K
        n_pos = A
        Win  = rng.normal(scale=0.3, size=(in_d, n_pos * out_d)).astype(np.float32)
        Wout = rng.normal(scale=0.3, size=(n_pos * out_d, out_d)).astype(np.float32)
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
        Hl, Wl = grid_for_layer(li, H2, W2, H3, W3)
        H_init = rng.normal(scale=0.05, size=(Hl * Wl, out_d)).astype(np.float32)

        cpu_layers.append(LayerRef(
            qvgIn=qvgIn.copy(), goW=goW.copy(),
            input_proj=None if input_proj is None else input_proj.copy(),
            ln_gamma=ln_gamma.copy(), ln_beta=ln_beta.copy(),
            H=H_init.copy(), K=K))

        qvgIn_t = np.ascontiguousarray(qvgIn.transpose(0, 2, 1))
        goW_t   = np.ascontiguousarray(goW.transpose(0, 2, 1))
        p_qvg, _, ka1 = alloc_and_fill(qvgIn_t); keepalive.append(ka1)
        p_go,  _, ka2 = alloc_and_fill(goW_t);   keepalive.append(ka2)
        if input_proj is not None:
            ip_t = np.ascontiguousarray(input_proj.T)
            p_ip, _, ka3 = alloc_and_fill(ip_t); keepalive.append(ka3)
        else:
            p_ip = 0
        p_lng, _, ka4 = alloc_and_fill(ln_gamma); keepalive.append(ka4)
        p_lnb, _, ka5 = alloc_and_fill(ln_beta);  keepalive.append(ka5)

        for blk in range(n_blocks):
            p_h, h_view, ka6 = alloc_and_fill(H_init); keepalive.append(ka6)
            gpu_layers[blk][li] = {
                "qvgIn": p_qvg, "goW": p_go, "input_proj": p_ip,
                "ln_gamma": p_lng, "ln_beta": p_lnb,
                "hidden": p_h, "hidden_view": h_view,
            }
    return cpu_layers, gpu_layers, keepalive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--n", type=int, default=200_000,
                    help="Total events across all blocks (split evenly).")
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if args.n_blocks > MAX_BLOCKS:
        sys.exit(f"--n-blocks {args.n_blocks} > MAX_BLOCKS {MAX_BLOCKS}")

    rng = np.random.default_rng(args.seed)
    n_blocks = args.n_blocks

    cpu_layers, gpu_layers, ka_layers = build_n_block_layers(
        rng, n_blocks, H2, W2, H3, W3)

    # Per-block tdrop counters in managed memory.
    keepalive = list(ka_layers)
    tdrop_s2_dev = []
    tdrop_s3_dev = []
    for _ in range(n_blocks):
        p, ka = cuda_util.alloc_managed(H2 * W2)
        ctypes.memset(p, 0, H2 * W2)
        keepalive.append(ka); tdrop_s2_dev.append(p)
        p, ka = cuda_util.alloc_managed(H3 * W3)
        ctypes.memset(p, 0, H3 * W3)
        keepalive.append(ka); tdrop_s3_dev.append(p)

    # Build the cfg manually since hybrid_common.build_config hard-codes N=2.
    cfg = CHybridS2S3Config()
    cfg.H2, cfg.W2 = H2, W2
    cfg.H3, cfg.W3 = H3, W3
    cfg.tdrop_window = args.tdrop
    cfg.head_out_dim = 7
    cfg.n_blocks = n_blocks
    # Strip ownership: split W2 evenly across blocks; s3 strip = strip>>1.
    strip_w = max(1, W2 // n_blocks)
    for blk in range(n_blocks):
        lo = blk * strip_w
        hi = (blk + 1) * strip_w if blk < n_blocks - 1 else W2
        cfg.strip[blk].owned_lo, cfg.strip[blk].owned_hi = lo, hi
        cfg.strip[blk].s3_owned_lo = lo >> 1
        cfg.strip[blk].s3_owned_hi = hi >> 1
        for li in range(4):
            cfg.layers[blk][li].qvgIn      = gpu_layers[blk][li]["qvgIn"]
            cfg.layers[blk][li].goW        = gpu_layers[blk][li]["goW"]
            cfg.layers[blk][li].input_proj = gpu_layers[blk][li]["input_proj"]
            cfg.layers[blk][li].ln_gamma   = gpu_layers[blk][li]["ln_gamma"]
            cfg.layers[blk][li].ln_beta    = gpu_layers[blk][li]["ln_beta"]
            cfg.hidden[blk][li]            = gpu_layers[blk][li]["hidden"]
        cfg.tdrop_s2[blk] = tdrop_s2_dev[blk]
        cfg.tdrop_s3[blk] = tdrop_s3_dev[blk]
    p_cfg, cfg_ka = cuda_util.alloc_managed(ctypes.sizeof(CHybridS2S3Config))
    ctypes.memmove(p_cfg, ctypes.addressof(cfg), ctypes.sizeof(CHybridS2S3Config))
    keepalive.append(cfg_ka)

    # Split N events across n_blocks. Each block gets its own ring.
    n_per_block = args.n // n_blocks
    print(f"Generating {args.n} events split into {n_blocks} blocks "
          f"({n_per_block}/block) ...", flush=True)
    rec_blocks = []
    for blk in range(n_blocks):
        rec = np.zeros(n_per_block, dtype=INPUT_DTYPE)
        rec["x"] = rng.integers(0, W2, n_per_block).astype(np.uint16)
        rec["y"] = rng.integers(0, H2, n_per_block).astype(np.uint16)
        rec["feat1"] = rng.normal(scale=0.5, size=(n_per_block, C1)).astype(np.float32)
        rec["t"] = rng.uniform(0, 1, n_per_block).astype(np.float32)
        rec_blocks.append(rec)

    ring_devs, out_devs = [], []
    for blk in range(n_blocks):
        rec = rec_blocks[blk]
        p_in, ka_in = cuda_util.alloc_managed(rec.nbytes)
        np.frombuffer(ka_in, dtype=INPUT_DTYPE).reshape(-1)[:] = rec
        keepalive.append(ka_in); ring_devs.append(p_in)
        p_out, ka_out = cuda_util.alloc_managed(rec.size * OUTPUT_DTYPE.itemsize)
        ctypes.memset(p_out, 0, rec.size * OUTPUT_DTYPE.itemsize)
        keepalive.append(ka_out); out_devs.append(p_out)

    n_batches_per_block = (n_per_block + BATCH - 1) // BATCH
    clk_devs, clk_views = [], []
    for _ in range(n_blocks):
        nbytes = n_batches_per_block * 2 * 8
        p, ka = cuda_util.alloc_managed(nbytes)
        ctypes.memset(p, 0, nbytes)
        keepalive.append(ka); clk_devs.append(p)
        clk_views.append(np.frombuffer(ka, dtype=np.uint64).reshape(n_batches_per_block, 2))

    # Pointer arrays: each is uint64[n_blocks] in managed memory.
    def _alloc_ptr_array(values):
        nbytes = n_blocks * 8
        p, ka = cuda_util.alloc_managed(nbytes)
        view = np.frombuffer(ka, dtype=np.uint64).reshape(-1)
        view[:] = values
        keepalive.append(ka)
        return p
    ring_ptr_arr = _alloc_ptr_array(ring_devs)
    out_ptr_arr  = _alloc_ptr_array(out_devs)
    clk_ptr_arr  = _alloc_ptr_array(clk_devs)
    ns_arr_dev, ns_ka = cuda_util.alloc_managed(n_blocks * 4)
    np.frombuffer(ns_ka, dtype=np.int32).reshape(-1)[:] = n_per_block
    keepalive.append(ns_ka)

    # Compile celled kernel.
    src   = open(os.path.join(KERNELS_DIR, "ssla_s2_s3_head_celled.cuh")).read()
    proto = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
    print(f"Compiling celled kernel ({len(src)+len(proto)} bytes) ...", flush=True)
    t0 = time.monotonic()
    mod = CudaModule(src, name="ssla_s2_s3_head_celled.cu",
                     headers={"proto_layer_pair.cuh": proto})
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s")

    func = mod.get_function("k_ssla_s2s3_celled_drain_n_multi")
    threads_per_block = N_WARPS * 32
    SMEM = _smem_size_bytes()

    print(f"\nGPU drain (multi): {n_blocks} blocks × {threads_per_block} threads, "
          f"smem={SMEM} B (n/block={n_per_block})\n")
    SM_CLK_HZ = 918_000_000  # nominal; actual is DVFS-controlled
    hdr = f"{'run':>3} | {'kev/s':>10} |"
    for blk in range(n_blocks):
        hdr += f" blk{blk} p50/p99/max |"
    print(hdr.rstrip("|"))

    for run in range(args.runs):
        # Reset clk + tdrop between runs.
        for blk in range(n_blocks):
            ctypes.memset(clk_devs[blk], 0, n_batches_per_block * 2 * 8)
            ctypes.memset(tdrop_s2_dev[blk], 0, H2 * W2)
            ctypes.memset(tdrop_s3_dev[blk], 0, H3 * W3)

        t0 = time.monotonic()
        func.launch((n_blocks, 1, 1), (threads_per_block, 1, 1), [
            arg_ptr(p_cfg),
            arg_ptr(ring_ptr_arr),
            arg_ptr(ns_arr_dev),
            arg_ptr(out_ptr_arr),
            arg_ptr(clk_ptr_arr),
        ], smem=SMEM)
        func.sync()
        gpu_dt = time.monotonic() - t0
        kevs = args.n / gpu_dt / 1e3

        line = f"{run:>3} | {kevs:>10.1f} |"
        for blk in range(n_blocks):
            cv = clk_views[blk]
            valid = cv[:, 1] > 0
            cv = cv[valid]
            if cv.size == 0:
                line += f"  (no samples) |"
                continue
            head = max(1, len(cv) // 20)
            tail = max(1, len(cv) // 20)
            cv = cv[head:-tail] if len(cv) > head + tail else cv
            dt_us = (cv[:, 1].astype(np.int64) - cv[:, 0].astype(np.int64)) \
                    * (1e6 / SM_CLK_HZ)
            line += f" {float(np.percentile(dt_us,50)):6.0f}/{float(np.percentile(dt_us,99)):6.0f}/{float(dt_us.max()):6.0f} |"
        print(line.rstrip("|"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
