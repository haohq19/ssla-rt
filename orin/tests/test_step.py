"""Compile ssla_step.cuh, run the SSLA-S full per-event step on a small
synthetic sensor + event sequence, and compare to the numpy reference.

Two configurations:
  tdrop_window=1   every event passes all 3 gates (full pipeline coverage).
  tdrop_window=4   ~1-in-4 events pass each gate (exercises the dropout
                    path, both kernel and ref must agree on which events
                    are filtered).

Setup uses small sensor dims (32×24) so all 8 hidden-state buffers fit
easily in unified memory and the test runs in a few seconds.

Run:
    python3 deploy/orin/tests/test_step.py
"""
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin import cuda_util  # noqa: E402
from orin.nvrtc_util import CudaModule, arg_ptr, arg_i32, arg_f32  # noqa: E402
from orin.ssla_ref import LayerRef, StepRefState, step_ref  # noqa: E402
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402


KERNELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "kernels")


# Mirrors the C++ struct in ssla_step.cuh. Layout must match exactly.
class CLayerWeights(ctypes.Structure):
    _fields_ = [
        ("qvgIn",      ctypes.c_uint64),
        ("goW",        ctypes.c_uint64),
        ("input_proj", ctypes.c_uint64),
        ("ln_gamma",   ctypes.c_uint64),
        ("ln_beta",    ctypes.c_uint64),
    ]


class CStepConfig(ctypes.Structure):
    _fields_ = [
        ("H0",           ctypes.c_int32),
        ("W0",           ctypes.c_int32),
        ("tdrop_window", ctypes.c_int32),
        ("layers",       CLayerWeights * 8),
        ("hidden",       ctypes.c_uint64 * 8),
        ("tdrop",        ctypes.c_uint64 * 3),
    ]


SSLA_S_CHANNELS = (12, 24, 48, 96)
SSLA_S_KERNELS  = (1, 3, 3, 3, 3, 3, 3, 3)
SSLA_S_IN_OUT   = (
    (2, 12),  (12, 12),    # stage 0
    (12, 24), (24, 24),    # stage 1
    (24, 48), (48, 48),    # stage 2
    (48, 96), (96, 96),    # stage 3
)


def _alloc_and_fill(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    ptr, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=arr.dtype).reshape(arr.shape)
    view[...] = arr
    return ptr, view, ka


def _stage_grid(H0, W0, stage):
    return H0 >> stage, W0 >> stage


def _build_random_layers(rng, H0, W0):
    """Return (layers_for_kernel: list of dicts with devptrs and views,
              layers_for_ref:    list of LayerRef)."""
    kernel_layers = []
    ref_layers    = []
    keepalive     = []     # hold ctypes arrays so they don't free

    for li in range(8):
        in_d, out_d = SSLA_S_IN_OUT[li]
        K = SSLA_S_KERNELS[li]
        A = K * K

        if K == 1:
            n_pos_in = n_pos_out = 1
        else:
            n_pos_in = n_pos_out = A
        Win  = rng.normal(scale=0.3, size=(in_d,         n_pos_in  * out_d)).astype(np.float32)
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

        # Hidden state grid for layer li:
        # layers 0,1 → stage 0 (H0, W0)
        # layers 2,3 → stage 1 (H0/2, W0/2)
        # layers 4,5 → stage 2 (H0/4, W0/4)
        # layers 6,7 → stage 3 (H0/8, W0/8)
        Hl, Wl = _stage_grid(H0, W0, li // 2)
        H_init = rng.normal(scale=0.05, size=(Hl * Wl, out_d)).astype(np.float32)

        # Allocate managed-memory copies for the kernel side.
        p_qvg, _,  ka_qvg = _alloc_and_fill(qvgIn);     keepalive.append(ka_qvg)
        p_go,  _,  ka_go  = _alloc_and_fill(goW);       keepalive.append(ka_go)
        if input_proj is not None:
            p_ip, _, ka_ip = _alloc_and_fill(input_proj); keepalive.append(ka_ip)
        else:
            p_ip = 0
        p_lng, _,  ka_lng = _alloc_and_fill(ln_gamma);  keepalive.append(ka_lng)
        p_lnb, _,  ka_lnb = _alloc_and_fill(ln_beta);   keepalive.append(ka_lnb)
        p_h,   H_view, ka_h = _alloc_and_fill(H_init);  keepalive.append(ka_h)

        kernel_layers.append({
            "qvgIn": p_qvg, "goW": p_go, "input_proj": p_ip,
            "ln_gamma": p_lng, "ln_beta": p_lnb,
            "hidden_dev": p_h, "hidden_view": H_view,
            "Hl": Hl, "Wl": Wl, "out_d": out_d,
        })

        # Numpy-side state — separate copy of hidden init.
        ref_layers.append(LayerRef(
            qvgIn=qvgIn.copy(), goW=goW.copy(),
            input_proj=None if input_proj is None else input_proj.copy(),
            ln_gamma=ln_gamma.copy(), ln_beta=ln_beta.copy(),
            H=H_init.copy(),
            K=K,
        ))
    return kernel_layers, ref_layers, keepalive


def run_case(name, mod, *, H0, W0, n_events, tdrop_window, seed):
    rng = np.random.default_rng(seed)
    kernel_layers, ref_layers, keepalive = _build_random_layers(rng, H0, W0)

    # Set up tdrop counters (3 stages, post-pool sizes).
    p_tdrop = []
    tdrop_views = []
    for s in range(3):
        Hl, Wl = _stage_grid(H0, W0, s + 1)
        p, ka = cuda_util.alloc_managed(Hl * Wl)
        ctypes.memset(p, 0, Hl * Wl)
        view = np.frombuffer(ka, dtype=np.uint8).reshape(Hl * Wl)
        keepalive.append(ka)
        p_tdrop.append(p)
        tdrop_views.append(view)

    ref_state = StepRefState(
        H0=H0, W0=W0, tdrop_window=tdrop_window,
        layers=ref_layers,
        tdrop=[np.zeros(_stage_grid(H0, W0, s + 1)[0]
                        * _stage_grid(H0, W0, s + 1)[1], dtype=np.uint8)
               for s in range(3)],
    )

    # Build CStepConfig in managed memory.
    cfg = CStepConfig()
    cfg.H0, cfg.W0, cfg.tdrop_window = H0, W0, tdrop_window
    for li, kl in enumerate(kernel_layers):
        cfg.layers[li].qvgIn      = kl["qvgIn"]
        cfg.layers[li].goW        = kl["goW"]
        cfg.layers[li].input_proj = kl["input_proj"]
        cfg.layers[li].ln_gamma   = kl["ln_gamma"]
        cfg.layers[li].ln_beta    = kl["ln_beta"]
        cfg.hidden[li]            = kl["hidden_dev"]
    for s in range(3):
        cfg.tdrop[s] = p_tdrop[s]
    p_cfg, cfg_ka = cuda_util.alloc_managed(ctypes.sizeof(CStepConfig))
    ctypes.memmove(p_cfg, ctypes.addressof(cfg), ctypes.sizeof(CStepConfig))
    keepalive.append(cfg_ka)

    # Per-event scratch buffers (managed-memory).
    p_s0, v_s0, ka0 = _alloc_and_fill(np.zeros(SSLA_S_CHANNELS[0], dtype=np.float32))
    p_s1, v_s1, ka1 = _alloc_and_fill(np.zeros(SSLA_S_CHANNELS[1], dtype=np.float32))
    p_s2, v_s2, ka2 = _alloc_and_fill(np.zeros(SSLA_S_CHANNELS[2], dtype=np.float32))
    p_s3, v_s3, ka3 = _alloc_and_fill(np.zeros(SSLA_S_CHANNELS[3], dtype=np.float32))
    p_tx, v_tx, kax = _alloc_and_fill(np.zeros(4, dtype=np.int32))
    p_ty, v_ty, kay = _alloc_and_fill(np.zeros(4, dtype=np.int32))
    p_pf, v_pf, kpf = _alloc_and_fill(np.zeros(1, dtype=np.int32))
    keepalive.extend([ka0, ka1, ka2, ka3, kax, kay, kpf])

    func = mod.get_function("k_ssla_step_S")

    # Synthetic events — uniform random over (H0, W0); polarity 0/1; dt random.
    ev_xs = rng.integers(0, W0, n_events).astype(np.int32)
    ev_ys = rng.integers(0, H0, n_events).astype(np.int32)
    pols  = rng.integers(0, 2,  n_events).astype(np.float32)
    dts   = rng.random(n_events).astype(np.float32)

    fail_count = 0
    pass_count = 0
    drop_match = 0
    max_err = 0.0

    for i in range(n_events):
        ex, ey = int(ev_xs[i]), int(ev_ys[i])
        dt, p = float(dts[i]), float(pols[i])

        # Reset scratch outputs.
        v_s0.fill(0); v_s1.fill(0); v_s2.fill(0); v_s3.fill(0)
        v_tx.fill(0); v_ty.fill(0); v_pf.fill(0)

        func.launch((1,1,1), (1,1,1), [
            arg_ptr(p_cfg),
            arg_i32(ex), arg_i32(ey),
            arg_f32(dt), arg_f32(p),
            arg_ptr(p_s0), arg_ptr(p_s1), arg_ptr(p_s2), arg_ptr(p_s3),
            arg_ptr(p_tx), arg_ptr(p_ty), arg_ptr(p_pf),
        ])
        func.sync()

        passed_k = bool(v_pf[0])
        passed_r, s0r, s1r, s2r, s3r, txr, tyr = step_ref(ref_state, ex, ey, dt, p)

        if passed_k != passed_r:
            print(f"    ev[{i}] FAIL: passed mismatch (k={passed_k} r={passed_r})", flush=True)
            fail_count += 1
            continue

        if not passed_r:
            drop_match += 1
            continue

        # All 4 stages reached — compare s0..s3 + touched + hidden state.
        for k_, (vk, vr, dim) in enumerate([
            (v_s0, s0r, SSLA_S_CHANNELS[0]),
            (v_s1, s1r, SSLA_S_CHANNELS[1]),
            (v_s2, s2r, SSLA_S_CHANNELS[2]),
            (v_s3, s3r, SSLA_S_CHANNELS[3]),
        ]):
            err = float(np.max(np.abs(vk - vr)))
            max_err = max(max_err, err)

        for k_ in range(4):
            if int(v_tx[k_]) != txr[k_] or int(v_ty[k_]) != tyr[k_]:
                print(f"    ev[{i}] FAIL: touched mismatch at stage {k_}: "
                      f"k=({v_tx[k_]},{v_ty[k_]})  r=({txr[k_]},{tyr[k_]})",
                      flush=True)
                fail_count += 1
        pass_count += 1

    # Compare final hidden state of every layer.
    for li, (kl, ref_l) in enumerate(zip(kernel_layers, ref_layers)):
        diff = float(np.max(np.abs(kl["hidden_view"] - ref_l.H)))
        max_err = max(max_err, diff)

    # Free everything.
    cuda_util.free_managed(p_cfg)
    for kl in kernel_layers:
        cuda_util.free_managed(kl["qvgIn"])
        cuda_util.free_managed(kl["goW"])
        if kl["input_proj"] != 0:
            cuda_util.free_managed(kl["input_proj"])
        cuda_util.free_managed(kl["ln_gamma"])
        cuda_util.free_managed(kl["ln_beta"])
        cuda_util.free_managed(kl["hidden_dev"])
    for p in p_tdrop:
        cuda_util.free_managed(p)
    for p in (p_s0, p_s1, p_s2, p_s3, p_tx, p_ty, p_pf):
        cuda_util.free_managed(p)

    # Tolerance: cumulative fp32 drift across 8 layers. Loose-but-meaningful.
    atol = 1e-2
    ok = fail_count == 0 and max_err < atol
    print(f"  {name}: events={n_events}  passed={pass_count}  dropped={drop_match}  "
          f"max|Δ|={max_err:.2e}  atol={atol:.0e}  {'OK' if ok else 'FAIL'}",
          flush=True)
    return 0 if ok else 1


def main() -> int:
    src = open(os.path.join(KERNELS_DIR, "ssla_step.cuh")).read()
    primitives = open(os.path.join(KERNELS_DIR, "ssla_primitives.cuh")).read()
    layer_h    = open(os.path.join(KERNELS_DIR, "ssla_layer.cuh")).read()
    t0 = time.monotonic()
    print(f"loading ssla_step.cuh  ({len(src)}+{len(primitives)}+{len(layer_h)} bytes)",
          flush=True)
    mod = CudaModule(
        src, name="ssla_step.cu",
        headers={
            "ssla_primitives.cuh": primitives,
            "ssla_layer.cuh":      layer_h,
        },
    )
    print(f"  cache {mod.cache_status} in {time.monotonic()-t0:.1f}s", flush=True)

    failures = 0
    failures += run_case("tdrop=1, 50ev",
                         mod, H0=24, W0=32, n_events=50, tdrop_window=1, seed=0)
    failures += run_case("tdrop=4, 50ev",
                         mod, H0=24, W0=32, n_events=50, tdrop_window=4, seed=1)

    mod.close()
    if failures:
        print(f"\nFAIL: {failures} case(s)", flush=True)
        return 1
    print("\nSSLA-S full step matches numpy reference within tolerance", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
