"""Per-layer SSLA weight reshape — Python mirror of cpp/methods/ssla/
ssla_detection_yolox.cpp::Impl::load_layer.

The exported npz stores raw nn.Linear / nn.Parameter tensors. The
streaming step kernel (cpp + the future S9 CUDA persistent kernel)
consumes a different layout: per-position fused tensors `qvgIn[A, 3·D,
IN]` and `goW[A, D, D]`, so the per-event inner loop reduces to two
matvecs and a scalar LRU step.

This module is pure-numpy. The result tensors get pushed into managed
memory by the caller (typically the persistent-kernel runner in #11),
which knows the device-side layout it wants to bind to.

Reference:
  cpp/methods/ssla/ssla_detection_yolox.cpp:382-477  Impl::load_layer
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def reshape_ssla_layer(
    W_in_cat:  np.ndarray,
    W_out_cat: np.ndarray,
    qW: np.ndarray,
    vW: np.ndarray,
    gW: np.ndarray,
    oW: np.ndarray,
    in_dim:     int,
    hidden_dim: int,
    *,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse the four nn.Linear projections + the position-aware blocks
    of W_in_cat / W_out_cat into per-position dense tensors.

    Inputs (all float32; PyTorch nn.Linear weights have shape (out, in)):
        W_in_cat   (in_dim, A·D)  if scatter_proj else (in_dim, D)
        W_out_cat  (A·D, D)       if gather_proj  else (D, D)
        qW, vW, gW (D, D)         q/v/g_proj.weight
        oW         (D, D)         o_proj.weight

    Returns:
        qvgIn  (A, 3·D, in_dim)   row-major; per-position the rows split
                                   as [q_rows : v_rows : g_rows].
        goW    (A, D, D)          row-major; matvec-friendly per-position.

    Where A = max(W_in_cat.shape[1] // D, W_out_cat.shape[0] // D).

    See cpp source for the index derivation. Equivalence to the Python
    forward is verified by `test_weights_ssla.py` (algebraic round-trip).
    """
    D = hidden_dim
    if W_in_cat.shape[1] % D != 0:
        raise ValueError(f"W_in_cat cols {W_in_cat.shape[1]} not divisible by D={D}")
    if W_out_cat.shape[0] % D != 0:
        raise ValueError(f"W_out_cat rows {W_out_cat.shape[0]} not divisible by D={D}")
    n_pos_in  = W_in_cat.shape[1]  // D
    n_pos_out = W_out_cat.shape[0] // D
    A = max(n_pos_in, n_pos_out)

    Win  = W_in_cat.astype(dtype, copy=False)
    Wout = W_out_cat.astype(dtype, copy=False)
    qWd  = qW.astype(dtype, copy=False)
    vWd  = vW.astype(dtype, copy=False)
    gWd  = gW.astype(dtype, copy=False)
    oWd  = oW.astype(dtype, copy=False)

    qvgIn = np.empty((A, 3 * D, in_dim), dtype=dtype)
    goW   = np.empty((A, D, D),         dtype=dtype)

    for p in range(A):
        pin  = 0 if n_pos_in  == 1 else p
        pout = 0 if n_pos_out == 1 else p

        # scatter[d, c] = W_in_cat[c, pin*D + d]
        scatter = np.ascontiguousarray(Win[:, pin * D:(pin + 1) * D].T)  # (D, in)
        qvgIn[p, 0 * D:1 * D, :] = qWd @ scatter
        qvgIn[p, 1 * D:2 * D, :] = vWd @ scatter
        qvgIn[p, 2 * D:3 * D, :] = gWd @ scatter

        # gather[i, j] = W_out_cat[pout*D + j, i]
        gather = np.ascontiguousarray(Wout[pout * D:(pout + 1) * D, :].T)  # (D, D)
        goW[p] = gather @ oWd

    return qvgIn, goW
