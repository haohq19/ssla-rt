"""Risk-retirement prototype test for the fused warp-per-event SSLA-S
layer pair (stage 3, L6: 48→96 K=3 + L7: 96→96 K=3).

Answers the user's day-1 gate questions:
  * Registers / thread, local-memory bytes, shared-memory bytes
  * Blocks/SM and theoretical occupancy at 4 warps / block
  * Per-event kernel-side latency (single launch per event, clock64
    bracketed inside the warp) — p50 / p90 / p99 / max over a long
    sequence
  * Long-sequence correctness vs numpy-fp64 reference (drift bound)
  * Three input modes: cold-uniform synthetic, hot-strip clustered,
    DVXplorer replay (last is plumbed but skipped if no camera trace)

Run:
    python3 deploy/orin/tests/test_proto_layer_pair.py            # default 2000 events
    python3 deploy/orin/tests/test_proto_layer_pair.py --n 5000
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import (
    CudaModule, arg_ptr, arg_i32, arg_u64,  # noqa: E402
)
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402
from orin.ssla_ref import LayerRef, layer_step  # noqa: E402


KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "kernels")


def _alloc_managed_from_array(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    p, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=arr.dtype).reshape(arr.shape)
    view[...] = arr
    return p, view, ka


def _build_layer(rng, in_d, out_d, K, Hl, Wl):
    """Return (kernel-friendly tuple, ssla_ref LayerRef)."""
    A = K * K
    n_pos_in = n_pos_out = (1 if K == 1 else A)
    Win  = rng.normal(scale=0.3, size=(in_d,    n_pos_in  * out_d)).astype(np.float32)
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
    H_init = rng.normal(scale=0.05, size=(Hl * Wl, out_d)).astype(np.float32)

    # Transpose for kernel layout (IN, OUT) on the matvec axis.
    qvgIn_t = np.ascontiguousarray(qvgIn.transpose(0, 2, 1))   # (A, IN, 3*OUT)
    goW_t   = np.ascontiguousarray(goW.transpose(0, 2, 1))     # (A, OUT_in, OUT_out)
    input_proj_t = np.ascontiguousarray(input_proj.T) if input_proj is not None else None

    p_qvg,  _, ka1 = _alloc_managed_from_array(qvgIn_t)
    p_go,   _, ka2 = _alloc_managed_from_array(goW_t)
    p_ip = 0; ka3 = None
    if input_proj_t is not None:
        p_ip, _, ka3 = _alloc_managed_from_array(input_proj_t)
    p_lng, _, ka4 = _alloc_managed_from_array(ln_gamma)
    p_lnb, _, ka5 = _alloc_managed_from_array(ln_beta)
    p_h, h_view, ka6 = _alloc_managed_from_array(H_init)

    keepalive = [ka1, ka2, ka4, ka5, ka6]
    if ka3 is not None: keepalive.append(ka3)

    kernel_pkg = dict(qvgIn=p_qvg, goW=p_go, input_proj=p_ip,
                      ln_gamma=p_lng, ln_beta=p_lnb,
                      hidden_dev=p_h, hidden_view=h_view)
    ref_pkg = LayerRef(qvgIn=qvgIn.copy(), goW=goW.copy(),
                       input_proj=None if input_proj is None else input_proj.copy(),
                       ln_gamma=ln_gamma.copy(), ln_beta=ln_beta.copy(),
                       H=H_init.copy(), K=K)
    return kernel_pkg, ref_pkg, keepalive


def _query_func(func_func, name):
    from cuda import cuda
    attrs = {
        "NUM_REGS":          cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
        "SHARED_SIZE_BYTES": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        "LOCAL_SIZE_BYTES":  cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        "MAX_THREADS_PER_BLOCK": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    }
    out = {}
    for k, a in attrs.items():
        err, v = cuda.cuFuncGetAttribute(a, func_func)
        out[k] = int(v)
    print(f"  {name}:")
    for k, v in out.items():
        print(f"    {k:24s} = {v}")
    return out


def gen_events(mode: str, n: int, Hl: int, Wl: int, seed: int):
    """Generate (ev_x, ev_y) pairs.
    Modes: 'uniform' — random across full grid
           'hot' — clustered in central 1/4 area (10× density)
           'replay' — read from /tmp/dvxplorer_replay.npy if present"""
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        x = rng.integers(0, Wl, n).astype(np.int32)
        y = rng.integers(0, Hl, n).astype(np.int32)
    elif mode == "hot":
        # 10× concentrated in a 20×27 patch (~stage 3 sized cluster)
        bx0, by0 = Wl // 4, Hl // 4
        bx1, by1 = bx0 + Wl // 4, by0 + Hl // 4
        x = rng.integers(bx0, bx1, n).astype(np.int32)
        y = rng.integers(by0, by1, n).astype(np.int32)
    elif mode == "replay":
        path = "/tmp/dvxplorer_replay.npy"
        if not os.path.exists(path):
            print(f"  (no replay file at {path} — falling back to uniform)", flush=True)
            return gen_events("uniform", n, Hl, Wl, seed)
        ev = np.load(path)[:n]
        # Each (x,y) is from full-resolution; scale to stage-3 grid.
        x = (ev["x"] // 8).astype(np.int32)
        y = (ev["y"] // 8).astype(np.int32)
        x = np.clip(x, 0, Wl - 1)
        y = np.clip(y, 0, Hl - 1)
        if len(ev) < n:
            print(f"  (replay only has {len(ev)} events; padding with uniform)", flush=True)
            extra = n - len(ev)
            xx, yy = gen_events("uniform", extra, Hl, Wl, seed + 1)
            x = np.concatenate([x, xx])
            y = np.concatenate([y, yy])
    else:
        raise ValueError(f"unknown mode {mode}")
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mhz", type=float, default=918.0)
    ap.add_argument("--validate-every", type=int, default=50,
                    help="run numpy reference for every Nth event for drift tracking")
    args = ap.parse_args()

    # Stage-3 grid for SSLA-S on Gen1/DVXplorer at 480x640 sensor: 60x80.
    Hl, Wl = 60, 80
    in_dim = 48
    out_dim = 96

    src   = open(os.path.join(KERNELS_DIR, "proto_layer_pair.cuh")).read()
    print(f"=== compile proto_layer_pair.cuh ({len(src)} bytes) ===", flush=True)

    # Compile with default options to see baseline register usage.
    t0 = time.monotonic()
    mod = CudaModule(src, name="proto_layer_pair.cu")
    print(f"  compile (default): {time.monotonic()-t0:.1f}s ({mod.cache_status})", flush=True)

    # Also try forcing maxrregcount = 96 to see how the kernel behaves
    # under the user's register ceiling.
    t0 = time.monotonic()
    mod_capped = CudaModule(
        src, name="proto_layer_pair_r96.cu",
        options=(b"--gpu-architecture=compute_87", b"-std=c++17",
                 b"--use_fast_math", b"--maxrregcount=96"))
    print(f"  compile (maxrregcount=96): {time.monotonic()-t0:.1f}s "
          f"({mod_capped.cache_status})", flush=True)
    func_capped = mod_capped.get_function("k_proto_layer_pair_s3")
    print("\n=== capped kernel attributes (maxrregcount=96) ===")
    _query_func(func_capped._func, "k_proto_layer_pair_s3 (capped)")

    func = mod.get_function("k_proto_layer_pair_s3")

    print("\n=== kernel attributes ===")
    attrs = _query_func(func._func, "k_proto_layer_pair_s3")

    # Theoretical occupancy at 4 warps/block.
    THREADS_PER_BLOCK = 128  # 4 warps
    regs_per_thread = attrs["NUM_REGS"]
    regs_per_block = regs_per_thread * THREADS_PER_BLOCK
    blocks_per_sm_by_regs = 65536 // max(1, regs_per_block)
    blocks_per_sm_by_threads = 1536 // THREADS_PER_BLOCK
    blocks_per_sm = min(blocks_per_sm_by_regs, blocks_per_sm_by_threads, 16)
    warps_per_sm = blocks_per_sm * (THREADS_PER_BLOCK // 32)
    occupancy = warps_per_sm / 48.0
    print(f"\n  at {THREADS_PER_BLOCK} threads/block ({THREADS_PER_BLOCK//32} warps):")
    print(f"    regs/block = {regs_per_block} (limit 65536)")
    print(f"    blocks/SM by regs = {blocks_per_sm_by_regs}")
    print(f"    blocks/SM by threads = {blocks_per_sm_by_threads}")
    print(f"    achieved blocks/SM = {blocks_per_sm}")
    print(f"    achieved warps/SM = {warps_per_sm} of 48 ({100*occupancy:.1f}%)")

    if attrs["LOCAL_SIZE_BYTES"] > 0:
        print(f"  >>> SPILLS to local memory: {attrs['LOCAL_SIZE_BYTES']} bytes/thread")
    else:
        print(f"  >>> no register spills")

    # Build weights for L6 + L7.
    print(f"\n=== build random weights (L6: {in_dim}->{out_dim} K=3, L7: {out_dim}->{out_dim} K=3) ===", flush=True)
    rng = np.random.default_rng(args.seed)
    L6_kernel, L6_ref, ka_L6 = _build_layer(rng, in_dim, out_dim, 3, Hl, Wl)
    L7_kernel, L7_ref, ka_L7 = _build_layer(rng, out_dim, out_dim, 3, Hl, Wl)
    keepalive = ka_L6 + ka_L7

    # in_feat input for the layer pair (48 floats — single broadcast input
    # for the L6 layer's matvecs). Drawn random per-event for the test.
    p_in, in_view, ka_in = _alloc_managed_from_array(np.zeros(in_dim, dtype=np.float32))
    keepalive.append(ka_in)

    p_out, out_view, ka_out = _alloc_managed_from_array(np.zeros(out_dim, dtype=np.float32))
    keepalive.append(ka_out)

    p_t, t_ka = cuda_util.alloc_pinned(8)
    keepalive.append(t_ka)
    timing_view = np.frombuffer(t_ka, dtype=np.uint64).reshape(1)

    # Shared mem: 96 floats in_feat + 96 floats qh = 192 floats = 768 bytes.
    # (in_feat sized for the larger of L6.IN=48 vs L7.IN=96 → 96.)
    SMEM = 192 * 4

    # Three input streams.
    streams = {
        "uniform": gen_events("uniform", args.n, Hl, Wl, args.seed),
        "hot":     gen_events("hot",     args.n, Hl, Wl, args.seed + 1),
        "replay":  gen_events("replay",  args.n, Hl, Wl, args.seed + 2),
    }

    for stream_name, (xs, ys) in streams.items():
        # Reset hidden state to deterministic init (rng-derived, same each stream).
        rng_reset = np.random.default_rng(args.seed + 100)
        h6_init = rng_reset.normal(scale=0.05, size=(Hl * Wl, out_dim)).astype(np.float32)
        h7_init = rng_reset.normal(scale=0.05, size=(Hl * Wl, out_dim)).astype(np.float32)
        L6_kernel["hidden_view"][...] = h6_init
        L7_kernel["hidden_view"][...] = h7_init
        L6_ref.H = h6_init.copy().astype(np.float64)
        L7_ref.H = h7_init.copy().astype(np.float64)

        # Per-event input: random feature each event (consistent across runs).
        feat_rng = np.random.default_rng(args.seed + 200)
        feats = feat_rng.normal(scale=0.5, size=(args.n, in_dim)).astype(np.float32)

        # Warmup so GPU clock is at max (3 launches).
        for _ in range(3):
            in_view[...] = feats[0]
            func.launch((1, 1, 1), (32, 1, 1), [
                arg_i32(0), arg_i32(0), arg_i32(Hl), arg_i32(Wl),
                arg_ptr(p_in),
                arg_ptr(L6_kernel["qvgIn"]), arg_ptr(L6_kernel["goW"]),
                arg_ptr(L6_kernel["input_proj"]),
                arg_ptr(L6_kernel["ln_gamma"]), arg_ptr(L6_kernel["ln_beta"]),
                arg_ptr(L6_kernel["hidden_dev"]),
                arg_ptr(L7_kernel["qvgIn"]), arg_ptr(L7_kernel["goW"]),
                arg_ptr(L7_kernel["ln_gamma"]), arg_ptr(L7_kernel["ln_beta"]),
                arg_ptr(L7_kernel["hidden_dev"]),
                arg_ptr(p_out), arg_ptr(p_t),
            ], smem=SMEM)
            func.sync()
        # Reset state again after warmup
        L6_kernel["hidden_view"][...] = h6_init
        L7_kernel["hidden_view"][...] = h7_init
        L6_ref.H = h6_init.copy().astype(np.float64)
        L7_ref.H = h7_init.copy().astype(np.float64)

        print(f"\n=== stream={stream_name}  N={args.n}  grid {Hl}x{Wl} ===", flush=True)

        cycles = np.empty(args.n, dtype=np.uint64)
        wall_t0 = time.monotonic()
        max_drift = 0.0
        drift_samples = 0
        first_drift = -1.0

        for i in range(args.n):
            ex, ey = int(xs[i]), int(ys[i])
            in_view[...] = feats[i]

            func.launch((1, 1, 1), (32, 1, 1), [
                arg_i32(ex), arg_i32(ey), arg_i32(Hl), arg_i32(Wl),
                arg_ptr(p_in),
                arg_ptr(L6_kernel["qvgIn"]), arg_ptr(L6_kernel["goW"]),
                arg_ptr(L6_kernel["input_proj"]),
                arg_ptr(L6_kernel["ln_gamma"]), arg_ptr(L6_kernel["ln_beta"]),
                arg_ptr(L6_kernel["hidden_dev"]),
                arg_ptr(L7_kernel["qvgIn"]), arg_ptr(L7_kernel["goW"]),
                arg_ptr(L7_kernel["ln_gamma"]), arg_ptr(L7_kernel["ln_beta"]),
                arg_ptr(L7_kernel["hidden_dev"]),
                arg_ptr(p_out), arg_ptr(p_t),
            ], smem=SMEM)
            func.sync()
            cycles[i] = int(timing_view[0])

            # Periodic correctness check vs ssla_ref.
            if (i % args.validate_every) == 0:
                # Run reference with the SAME state mutation.
                # L6 step
                in_feat_arr = feats[i].astype(np.float64)
                out_L6_ref = layer_step(
                    in_feat_arr, L6_ref.H, L6_ref.qvgIn, L6_ref.goW,
                    L6_ref.input_proj, L6_ref.ln_gamma, L6_ref.ln_beta,
                    ev_x=ex, ev_y=ey, Hl=Hl, Wl=Wl, K=L6_ref.K)
                out_L7_ref = layer_step(
                    out_L6_ref, L7_ref.H, L7_ref.qvgIn, L7_ref.goW,
                    L7_ref.input_proj, L7_ref.ln_gamma, L7_ref.ln_beta,
                    ev_x=ex, ev_y=ey, Hl=Hl, Wl=Wl, K=L7_ref.K)

                kbuf = out_view.copy()
                d = float(np.max(np.abs(kbuf - out_L7_ref.astype(np.float32))))
                if first_drift < 0: first_drift = d
                if d > max_drift: max_drift = d
                drift_samples += 1
            else:
                # Mutate ref state in lockstep WITHOUT comparing — needed
                # so the next sampled comparison still sees aligned state.
                in_feat_arr = feats[i].astype(np.float64)
                out_L6_ref = layer_step(
                    in_feat_arr, L6_ref.H, L6_ref.qvgIn, L6_ref.goW,
                    L6_ref.input_proj, L6_ref.ln_gamma, L6_ref.ln_beta,
                    ev_x=ex, ev_y=ey, Hl=Hl, Wl=Wl, K=L6_ref.K)
                _ = layer_step(
                    out_L6_ref, L7_ref.H, L7_ref.qvgIn, L7_ref.goW,
                    L7_ref.input_proj, L7_ref.ln_gamma, L7_ref.ln_beta,
                    ev_x=ex, ev_y=ey, Hl=Hl, Wl=Wl, K=L7_ref.K)

        wall_dt = time.monotonic() - wall_t0
        us = cycles.astype(np.float64) / args.gpu_mhz
        # End-to-end per-event = wall_dt / N, includes Python+launch overhead.
        e2e_us = wall_dt * 1e6 / args.n

        print(f"  walltime          {wall_dt:.2f} s  ({args.n/wall_dt:.0f} ev/s overall)")
        print(f"  per-event end-to-end (incl. Python) = {e2e_us:.1f} µs")
        print(f"  kernel-only latency (clock64() inside warp):")
        print(f"    p50 = {np.percentile(us, 50):.2f} µs")
        print(f"    p90 = {np.percentile(us, 90):.2f} µs")
        print(f"    p99 = {np.percentile(us, 99):.2f} µs")
        print(f"    max = {us.max():.2f} µs")
        print(f"    min = {us.min():.2f} µs")
        print(f"  drift vs numpy fp64 ref:")
        print(f"    samples = {drift_samples}, first = {first_drift:.2e}, "
              f"max over sequence = {max_drift:.2e}")

    # Free.
    cuda_util.free_managed(p_in)
    cuda_util.free_managed(p_out)
    cuda_util.free_pinned(p_t)
    for pkg in (L6_kernel, L7_kernel):
        for k in ("qvgIn", "goW", "ln_gamma", "ln_beta", "hidden_dev"):
            cuda_util.free_managed(pkg[k])
        if pkg["input_proj"] != 0:
            cuda_util.free_managed(pkg["input_proj"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
