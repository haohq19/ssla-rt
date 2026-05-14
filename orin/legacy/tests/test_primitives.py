"""Compile ssla_primitives.cuh under NVRTC and verify each primitive
matches a numpy reference within tolerance.

For each (primitive, dim) combination:
  1. Generate random fp32 inputs.
  2. Push them into managed memory.
  3. Launch the corresponding `k_*` test kernel.
  4. cuCtxSynchronize.
  5. Compare device-computed output to a numpy reference.

Tolerance choices:
  matvec_ct       : atol=2e-4  (FMA / order-of-summation drift)
  matvec_accum_ct : atol=2e-4  (same)
  lru_step_ct     : atol=2e-3  (`--use_fast_math` enables `__expf`, ~1 ULP drift)
  layernorm_ct    : atol=5e-5  (fp64 reduction in both impls)

Run:
    python3 deploy/orin/tests/test_primitives.py
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import CudaModule, arg_ptr  # noqa: E402


KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "kernels")
PRIM_HEADER = os.path.join(KERNELS_DIR, "ssla_primitives.cuh")


def _to_managed(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    ptr, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=np.float32).reshape(arr.shape)
    view[...] = arr
    return ptr, view, ka


def _alloc_managed(shape):
    n = int(np.prod(shape))
    ptr, ka = cuda_util.alloc_managed(n * 4)
    view = np.frombuffer(ka, dtype=np.float32).reshape(shape)
    view[...] = 0
    return ptr, view, ka


def _free(ptrs):
    for p in ptrs:
        cuda_util.free_managed(p)


def _ref_matvec(x, W, b):
    y = W @ x
    if b is not None:
        y = y + b
    return y


def _ref_layernorm(x, gamma, beta, eps=1e-5):
    x64 = x.astype(np.float64)
    mean = x64.mean()
    var = ((x64 - mean) ** 2).mean()
    inv = 1.0 / np.sqrt(var + eps)
    return ((x64 - mean) * inv * gamma.astype(np.float64) +
            beta.astype(np.float64)).astype(np.float32)


def _ref_lru_step(g, v, q, h):
    gc = 1.0 / (1.0 + np.exp(-g))
    h_new = gc * h + v
    y = q * h_new
    return h_new.astype(np.float32), y.astype(np.float32)


def main() -> int:
    with open(PRIM_HEADER) as f:
        src = f.read()
    print(f"compiling {PRIM_HEADER}  ({len(src)} bytes)", flush=True)
    mod = CudaModule(src, name="ssla_primitives.cu")
    print("compiled OK", flush=True)

    rng = np.random.default_rng(0)
    failures = 0

    # ---- matvec_ct ----
    for IN, OUT in [(2, 12), (12, 24), (24, 48), (48, 96), (96, 96),
                    (2, 36), (12, 72), (24, 144), (48, 288), (96, 288)]:
        x = rng.normal(scale=0.5, size=IN).astype(np.float32)
        W = rng.normal(scale=0.5, size=(OUT, IN)).astype(np.float32)
        b = rng.normal(scale=0.5, size=OUT).astype(np.float32)
        px, _,  ka_x = _to_managed(x)
        pW, _,  ka_W = _to_managed(W)
        pb, _,  ka_b = _to_managed(b)
        py, vy, ka_y = _alloc_managed((OUT,))
        f = mod.get_function(f"k_matvec_ct_{IN}_{OUT}")
        f.launch((1,1,1), (1,1,1), [arg_ptr(px), arg_ptr(pW), arg_ptr(pb), arg_ptr(py)])
        f.sync()
        ref = _ref_matvec(x, W, b)
        err = float(np.max(np.abs(vy - ref)))
        ok = err < 2e-4
        print(f"  matvec_ct<{IN:>3d},{OUT:>4d}>   max|Δ| = {err:.2e}   {'OK' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            failures += 1
        _free([px, pW, pb, py])

    # ---- matvec_accum_ct ----
    for DIM in [12, 24, 48, 96]:
        x = rng.normal(scale=0.5, size=DIM).astype(np.float32)
        W = rng.normal(scale=0.5, size=(DIM, DIM)).astype(np.float32)
        y_init = rng.normal(scale=0.5, size=DIM).astype(np.float32)
        px,  _, ka_x = _to_managed(x)
        pW,  _, ka_W = _to_managed(W)
        py, vy, ka_y = _to_managed(y_init)
        f = mod.get_function(f"k_matvec_accum_ct_{DIM}_{DIM}")
        f.launch((1,1,1), (1,1,1), [arg_ptr(px), arg_ptr(pW), arg_ptr(py)])
        f.sync()
        ref = y_init + W @ x
        err = float(np.max(np.abs(vy - ref)))
        ok = err < 2e-4
        print(f"  matvec_accum_ct<{DIM:>3d},{DIM:>3d}>   max|Δ| = {err:.2e}   "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok: failures += 1
        _free([px, pW, py])

    # ---- lru_step_ct ----
    for DIM in [12, 24, 48, 96]:
        g = rng.normal(scale=1.0, size=DIM).astype(np.float32)
        v = rng.normal(scale=0.5, size=DIM).astype(np.float32)
        q = rng.normal(scale=0.5, size=DIM).astype(np.float32)
        h = rng.normal(scale=0.5, size=DIM).astype(np.float32)
        pg, _,  ka_g = _to_managed(g)
        pv, _,  ka_v = _to_managed(v)
        pq, _,  ka_q = _to_managed(q)
        ph, vh, ka_h = _to_managed(h)
        py, vy, ka_y = _alloc_managed((DIM,))
        f = mod.get_function(f"k_lru_step_{DIM}")
        f.launch((1,1,1), (1,1,1), [arg_ptr(pg), arg_ptr(pv), arg_ptr(pq),
                                     arg_ptr(ph), arg_ptr(py)])
        f.sync()
        h_ref, y_ref = _ref_lru_step(g, v, q, h)
        err_h = float(np.max(np.abs(vh - h_ref)))
        err_y = float(np.max(np.abs(vy - y_ref)))
        ok = err_h < 2e-3 and err_y < 2e-3
        print(f"  lru_step_ct<{DIM:>3d}>      max|Δh| = {err_h:.2e}   max|Δy| = {err_y:.2e}   "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if not ok: failures += 1
        _free([pg, pv, pq, ph, py])

    # ---- layernorm_ct ----
    for DIM in [12, 24, 48, 96]:
        x     = rng.normal(scale=1.0, size=DIM).astype(np.float32)
        gamma = rng.normal(scale=0.3, size=DIM).astype(np.float32) + 1.0
        beta  = rng.normal(scale=0.3, size=DIM).astype(np.float32)
        px, vx, ka_x = _to_managed(x)
        pg, _,  ka_g = _to_managed(gamma)
        pb, _,  ka_b = _to_managed(beta)
        f = mod.get_function(f"k_layernorm_{DIM}")
        f.launch((1,1,1), (1,1,1), [arg_ptr(px), arg_ptr(pg), arg_ptr(pb)])
        f.sync()
        ref = _ref_layernorm(x, gamma, beta)
        err = float(np.max(np.abs(vx - ref)))
        ok = err < 5e-5
        print(f"  layernorm_ct<{DIM:>3d}>    max|Δ| = {err:.2e}   {'OK' if ok else 'FAIL'}",
              flush=True)
        if not ok: failures += 1
        _free([px, pg, pb])

    mod.close()
    if failures:
        print(f"\nFAIL: {failures} primitive(s) outside tolerance", flush=True)
        return 1
    print("\nAll primitives within tolerance", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
