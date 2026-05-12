"""Compile ssla_layer.cuh under NVRTC, launch one layer-forward kernel
per (IN, OUT, K), and verify outputs match the numpy reference within
fp32 round-off tolerance.

Coverage:
    (IN= 2, OUT=12, K=1)     SSLA-S stage 0 layer 0
    (IN=12, OUT=12, K=3)     stage 0 layer 1 (smallest k=3 case)
    (IN=12, OUT=24, K=3)     stage 1 layer 0 (in/out dim mismatch → input_proj)
    (IN=96, OUT=96, K=3)     stage 3 layer 1 (largest)

For each case we test 3 event positions: interior (exercises the
explicit-9-patch unroll) and 4 boundary cases (left/right/top/bottom),
covering the bounds-checked fallback path.

Run:
    python3 deploy/orin/tests/test_layer_forward.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import CudaModule, arg_ptr, arg_i32  # noqa: E402
from orin.ssla_ref import layer_step  # noqa: E402
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402


KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "kernels")


def _read(path):
    with open(path) as f:
        return f.read()


def _alloc_and_fill(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ptr, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=np.float32).reshape(arr.shape)
    view[...] = arr
    return ptr, view


def _alloc_zero(shape):
    n = int(np.prod(shape))
    ptr, ka = cuda_util.alloc_managed(n * 4)
    view = np.frombuffer(ka, dtype=np.float32).reshape(shape)
    view[...] = 0
    return ptr, view


def _free(ptrs):
    for p in ptrs:
        cuda_util.free_managed(p)


def case(mod, IN, OUT, K, *, Hl=20, Wl=24):
    A = K * K
    rng = np.random.default_rng(IN * 1000 + OUT * 10 + K)

    # Random raw nn.Linear-shaped weights, then run through the cpp-mirror
    # reshape so the kernel sees the same fused tensors the cpp runtime
    # would build from a checkpoint.
    if K == 1:
        n_pos_in = n_pos_out = 1
    else:
        n_pos_in = n_pos_out = A
    W_in_cat  = rng.normal(scale=0.3, size=(IN,           n_pos_in  * OUT)).astype(np.float32)
    W_out_cat = rng.normal(scale=0.3, size=(n_pos_out * OUT, OUT)).astype(np.float32)
    qW = rng.normal(scale=0.3, size=(OUT, OUT)).astype(np.float32)
    vW = rng.normal(scale=0.3, size=(OUT, OUT)).astype(np.float32)
    gW = rng.normal(scale=0.3, size=(OUT, OUT)).astype(np.float32)
    oW = rng.normal(scale=0.3, size=(OUT, OUT)).astype(np.float32)
    qvgIn, goW = reshape_ssla_layer(W_in_cat, W_out_cat, qW, vW, gW, oW, IN, OUT)

    if IN != OUT:
        input_proj = rng.normal(scale=0.3, size=(OUT, IN)).astype(np.float32)
    else:
        input_proj = None
    ln_gamma = (rng.normal(scale=0.1, size=OUT) + 1.0).astype(np.float32)
    ln_beta  = rng.normal(scale=0.1, size=OUT).astype(np.float32)

    func = mod.get_function(f"k_layer_forward_{IN}_{OUT}_{K}")

    positions = [
        ("interior",   Hl // 2, Wl // 2),
        ("top-left",   0, 0),
        ("top-right",  0, Wl - 1),
        ("bot-left",   Hl - 1, 0),
        ("bot-right",  Hl - 1, Wl - 1),
    ]

    failures = 0
    for label, ev_y, ev_x in positions:
        # Fresh hidden state each position (so we know exactly what the
        # ref starts from). H_all shape (Hl*Wl, OUT_DIM).
        H_init = rng.normal(scale=0.2, size=(Hl * Wl, OUT)).astype(np.float32)

        # Reference: cpu numpy (mutates H in place — feed a copy).
        H_ref = H_init.copy()
        in_feat = rng.normal(scale=0.4, size=IN).astype(np.float32)
        out_ref = layer_step(
            in_feat, H_ref, qvgIn, goW, input_proj, ln_gamma, ln_beta,
            ev_x=ev_x, ev_y=ev_y, Hl=Hl, Wl=Wl, K=K)

        # Kernel: managed-memory copies of all inputs.
        ptrs = []
        p_in,    _ = _alloc_and_fill(in_feat);             ptrs.append(p_in)
        p_qvg,   _ = _alloc_and_fill(qvgIn);               ptrs.append(p_qvg)
        p_goW,   _ = _alloc_and_fill(goW);                 ptrs.append(p_goW)
        if input_proj is not None:
            p_ip, _ = _alloc_and_fill(input_proj);         ptrs.append(p_ip)
        else:
            p_ip = 0
        p_lng,   _ = _alloc_and_fill(ln_gamma);            ptrs.append(p_lng)
        p_lnb,   _ = _alloc_and_fill(ln_beta);             ptrs.append(p_lnb)
        p_H,    H_view = _alloc_and_fill(H_init);          ptrs.append(p_H)
        p_out, out_view = _alloc_zero((OUT,));             ptrs.append(p_out)

        func.launch((1,1,1), (1,1,1), [
            arg_i32(ev_x), arg_i32(ev_y), arg_i32(Hl), arg_i32(Wl),
            arg_ptr(p_in), arg_ptr(p_qvg), arg_ptr(p_goW),
            arg_ptr(p_ip),
            arg_ptr(p_lng), arg_ptr(p_lnb),
            arg_ptr(p_H),  arg_ptr(p_out),
        ])
        func.sync()

        err_out = float(np.max(np.abs(out_view - out_ref)))
        err_h   = float(np.max(np.abs(H_view   - H_ref)))
        # fp32 + many FMAs across 9 patches × OUT_DIM accumulations.
        # Loose-but-meaningful bound: ~16·A·OUT·ULP_32.
        atol = 16.0 * 1.2e-7 * A * OUT
        ok = err_out < atol and err_h < 5e-6
        print(f"    {label:<10s}  ev=({ev_x:>2d},{ev_y:>2d})   "
              f"max|Δout| = {err_out:.2e}   max|ΔH| = {err_h:.2e}   atol = {atol:.1e}   "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok: failures += 1
        _free(ptrs)
    return failures


def main() -> int:
    src = _read(os.path.join(KERNELS_DIR, "ssla_layer.cuh"))
    primitives = _read(os.path.join(KERNELS_DIR, "ssla_primitives.cuh"))
    import time
    t0 = time.monotonic()
    print(f"loading ssla_layer.cuh  ({len(src)}+{len(primitives)} bytes)", flush=True)
    mod = CudaModule(
        src, name="ssla_layer.cu",
        headers={"ssla_primitives.cuh": primitives},
    )
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s", flush=True)

    cases = [
        (2,  12, 1),
        (12, 12, 3),
        (12, 24, 3),
        (24, 24, 3),
        (24, 48, 3),
        (48, 48, 3),
        (48, 96, 3),
        (96, 96, 3),
    ]
    failures = 0
    for IN, OUT, K in cases:
        print(f"\n  ssla_layer_forward<IN={IN}, OUT={OUT}, K={K}>")
        failures += case(mod, IN, OUT, K)

    mod.close()
    if failures:
        print(f"\nFAIL: {failures} positions outside tolerance", flush=True)
        return 1
    print("\nAll layer-forward cases within tolerance", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
