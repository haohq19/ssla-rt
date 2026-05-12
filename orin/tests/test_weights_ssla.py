"""Algebraic equivalence test: reshape_ssla_layer matches the Python
SSLA module's per-position forward for one event.

We don't have torch on the Orin, so we replicate the Python module's
forward semantics in numpy and compare against the cpp/streaming-kernel
formulation that uses the reshaped tensors. Same fp64 arithmetic on
both sides — so the equivalence is exact (max|Δ| == 0 expected, but
allow 1e-5 for floating-point order drift in case numpy reorders).

Coverage:
  k=1 (num_pos=1, no PAP)         : SSLA-S stage-0 layer 0
  k=3 + scatter+gather PAP (A=9)  : every other SSLA-S layer
  k=3 + scatter only              : not used by SSLA-S yaml today, but
                                     the reshape supports it
  k=3 + gather only               : same

Run:
    python3 deploy/orin/tests/test_weights_ssla.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orin.weights_ssla import reshape_ssla_layer  # noqa: E402


def python_step(in_feat, h_per_pos, W_in_cat, W_out_cat, qW, vW, gW, oW):
    """Per-event step in the Python module's per-position semantics.

    Matches `python/modules/ssla/layers/ssla_module.py:forward()` for
    the streaming case (one event, single batch sample, no padding).
    Returns out_pre_residual (D,) and h_new (A, D).
    """
    D = qW.shape[0]
    n_pos_in  = W_in_cat.shape[1] // D
    n_pos_out = W_out_cat.shape[0] // D
    A = max(n_pos_in, n_pos_out)

    out = np.zeros(D, dtype=np.float64)
    h_new = h_per_pos.astype(np.float64).copy()
    for p in range(A):
        pin  = 0 if n_pos_in  == 1 else p
        pout = 0 if n_pos_out == 1 else p
        u    = in_feat.astype(np.float64) @ W_in_cat[:, pin*D:(pin+1)*D].astype(np.float64)
        q_p  = u @ qW.astype(np.float64).T
        v_p  = u @ vW.astype(np.float64).T
        g_p  = u @ gW.astype(np.float64).T
        gc   = 1.0 / (1.0 + np.exp(-g_p))
        h_pp = gc * h_new[p] + v_p
        h_new[p] = h_pp
        qh_p = q_p * h_pp
        o_p  = qh_p @ oW.astype(np.float64).T
        out += o_p @ W_out_cat[pout*D:(pout+1)*D, :].astype(np.float64)
    return out, h_new


def cpp_step(in_feat, h_per_pos, qvgIn, goW):
    """Per-event step using the reshaped (cpp/device-kernel) layout."""
    A, three_D, _ = qvgIn.shape
    D = three_D // 3
    out = np.zeros(D, dtype=np.float64)
    h_new = h_per_pos.astype(np.float64).copy()
    for p in range(A):
        qvg = qvgIn[p].astype(np.float64) @ in_feat.astype(np.float64)
        q_p = qvg[0:D]; v_p = qvg[D:2*D]; g_p = qvg[2*D:3*D]
        gc  = 1.0 / (1.0 + np.exp(-g_p))
        h_pp = gc * h_new[p] + v_p
        h_new[p] = h_pp
        qh = q_p * h_pp
        out += goW[p].astype(np.float64) @ qh
    return out, h_new


def case(name, in_dim, D, A_in, A_out, *, dtype=np.float32, atol=None):
    """Reshape the layer at `dtype`, then compare per-position-summed
    Python forward vs the reshaped streaming-kernel forward. Both
    intermediate computations are done in fp64 so the only source of
    drift is the reshape's dtype-induced rounding."""
    rng = np.random.default_rng(hash(name) & 0x7FFFFFFF)
    W_in_cat  = rng.normal(scale=0.3, size=(in_dim,    A_in  * D)).astype(np.float32)
    W_out_cat = rng.normal(scale=0.3, size=(A_out * D, D     )).astype(np.float32)
    qW = rng.normal(scale=0.3, size=(D, D)).astype(np.float32)
    vW = rng.normal(scale=0.3, size=(D, D)).astype(np.float32)
    gW = rng.normal(scale=0.3, size=(D, D)).astype(np.float32)
    oW = rng.normal(scale=0.3, size=(D, D)).astype(np.float32)

    qvgIn, goW = reshape_ssla_layer(W_in_cat, W_out_cat, qW, vW, gW, oW,
                                     in_dim, D, dtype=dtype)
    A = max(A_in, A_out)
    assert qvgIn.shape == (A, 3 * D, in_dim), qvgIn.shape
    assert goW.shape   == (A, D, D),         goW.shape
    assert qvgIn.dtype == dtype and goW.dtype == dtype

    in_feat   = rng.normal(scale=0.5, size=in_dim).astype(np.float32)
    h_initial = rng.normal(scale=0.3, size=(A, D)).astype(np.float32)

    out_py,  h_py  = python_step(in_feat, h_initial, W_in_cat, W_out_cat, qW, vW, gW, oW)
    out_cpp, h_cpp = cpp_step(in_feat, h_initial, qvgIn, goW)

    err_out = float(np.max(np.abs(out_py - out_cpp)))
    err_h   = float(np.max(np.abs(h_py   - h_cpp)))
    if atol is None:
        # ~ N_FMA × ULP_dtype; loose-but-meaningful bound.
        ulp = 1.2e-7 if dtype == np.float32 else 2.3e-16
        atol = 16.0 * ulp * D * A
    ok = err_out < atol and err_h < atol
    tag = f"{np.dtype(dtype).name}".rjust(7)
    print(f"  {tag} {name:<48s}  max|Δout| = {err_out:.2e}   max|Δh| = {err_h:.2e}   "
          f"atol = {atol:.1e}  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    cases = [
        # (name,                                   in_dim, D, A_in, A_out)
        ("k=1, A=1, no PAP (SSLA-S L0)",          2,  12, 1, 1),
        ("k=3, A=9, scatter+gather PAP, in=2",    2,  12, 9, 9),
        ("k=3, A=9, scatter+gather PAP, D=12",    12, 12, 9, 9),
        ("k=3, A=9, scatter+gather PAP, D=24",    12, 24, 9, 9),
        ("k=3, A=9, scatter+gather PAP, D=48",    24, 48, 9, 9),
        ("k=3, A=9, scatter+gather PAP, D=96",    48, 96, 9, 9),
        ("k=3, A=9, scatter+gather PAP, in=D=96", 96, 96, 9, 9),
        ("k=3, scatter-only PAP",                 12, 12, 9, 1),
        ("k=3, gather-only PAP",                  12, 12, 1, 9),
    ]

    failures = 0
    # fp64 reshape: should be effectively bit-equivalent (4·D·A·ULP_64 ≈ 1e-13).
    print("# fp64 reshape (proves the formula matches the Python forward):")
    for c in cases:
        if not case(*c, dtype=np.float64): failures += 1
    # fp32 reshape: drifts by ~D·A·1e-7 (matches what cpp's load_layer also
    # produces — the reshape is one of the cpp's own fp32 rounding sites).
    print("\n# fp32 reshape (matches cpp/load_layer's own fp32 rounding):")
    for c in cases:
        if not case(*c, dtype=np.float32): failures += 1

    if failures:
        print(f"\nFAIL: {failures}/{2*len(cases)} cases", flush=True)
        return 1
    print("\nAll cases within tolerance", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
