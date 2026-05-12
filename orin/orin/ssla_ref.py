"""Numpy reference for the SSLA-S step. Mirrors the cpp/streaming-kernel
formulation (uses pre-reshaped per-position weights) — what the CUDA
kernel compares against.

`layer_step()` runs one SSLA module's per-event forward.
`StepRefState` + `step_ref()` chain 8 layers with spatial halving and
temporal_dropout, matching ssla_step.cuh::ssla_step_ct<C0,C1,C2,C3>.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def layer_step(
    in_feat: np.ndarray,        # (IN_DIM,) float32
    H_all:   np.ndarray,        # (Hl*Wl, OUT_DIM) float32 — read-AND-write
    qvgIn:   np.ndarray,        # (A, 3*OUT_DIM, IN_DIM) float32
    goW:     np.ndarray,        # (A, OUT_DIM, OUT_DIM) float32
    input_proj: Optional[np.ndarray],   # (OUT_DIM, IN_DIM) or None
    ln_gamma:   np.ndarray,     # (OUT_DIM,) float32
    ln_beta:    np.ndarray,     # (OUT_DIM,) float32
    *,
    ev_x: int, ev_y: int,
    Hl:   int, Wl:   int,
    K:    int,                  # 1 or 3
) -> np.ndarray:
    """One SSLA layer forward. Returns out_feat (OUT_DIM,) and mutates
    H_all in place at the patches the event touches.

    fp64 internally — matches `prim::layernorm`'s fp64 reduction. The
    CUDA kernel uses fp32 for matvec/lru and fp64 for layernorm; small
    differences vs this ref are bounded in test_layer_forward.py.
    """
    A_w = qvgIn.shape[0]
    OUT_DIM = qvgIn.shape[1] // 3
    IN_DIM  = qvgIn.shape[2]
    assert in_feat.shape == (IN_DIM,)
    assert goW.shape == (A_w, OUT_DIM, OUT_DIM)
    A_expected = K * K
    assert A_w == A_expected, (A_w, A_expected)

    # Residual.
    if input_proj is not None:
        residual = (input_proj.astype(np.float64) @ in_feat.astype(np.float64))
    else:
        assert IN_DIM == OUT_DIM
        residual = in_feat.astype(np.float64).copy()

    out = np.zeros(OUT_DIM, dtype=np.float64)

    def process(patch_idx, delta):
        qvg = qvgIn[delta].astype(np.float64) @ in_feat.astype(np.float64)   # (3D,)
        q = qvg[0:OUT_DIM]
        v = qvg[OUT_DIM:2*OUT_DIM]
        g = qvg[2*OUT_DIM:3*OUT_DIM]
        gc = 1.0 / (1.0 + np.exp(-g))
        h_old = H_all[patch_idx].astype(np.float64)
        h_new = gc * h_old + v
        H_all[patch_idx] = h_new.astype(np.float32)
        qh = q * h_new
        return goW[delta].astype(np.float64) @ qh

    if K == 1:
        out += process(ev_y * Wl + ev_x, 0)
    else:
        base = ev_y * Wl + ev_x
        interior = (ev_x > 0) and (ev_x + 1 < Wl) and (ev_y > 0) and (ev_y + 1 < Hl)
        if interior:
            for delta, off in [
                (8, -Wl - 1), (7, -Wl), (6, -Wl + 1),
                (5, -1),      (4, 0),    (3, 1),
                (2, Wl - 1),  (1, Wl),   (0, Wl + 1),
            ]:
                out += process(base + off, delta)
        else:
            for dy in (-1, 0, 1):
                py = ev_y + dy
                if py < 0 or py >= Hl: continue
                for dx in (-1, 0, 1):
                    px = ev_x + dx
                    if px < 0 or px >= Wl: continue
                    delta = (1 - dy) * 3 + (1 - dx)
                    out += process(py * Wl + px, delta)

    out = out + residual
    mean = out.mean()
    var  = ((out - mean) ** 2).mean()
    inv  = 1.0 / np.sqrt(var + 1e-5)
    return (((out - mean) * inv) * ln_gamma.astype(np.float64)
            + ln_beta.astype(np.float64)).astype(np.float32)


# ------------- multi-layer step (SSLA-S) -----------------------------

@dataclass
class LayerRef:
    """Per-layer host-side state used by step_ref. Mirrors LayerWeights +
    a hidden-state numpy buffer."""
    qvgIn:      np.ndarray
    goW:        np.ndarray
    input_proj: Optional[np.ndarray]
    ln_gamma:   np.ndarray
    ln_beta:    np.ndarray
    H:          np.ndarray   # (Hl·Wl, OUT_DIM) — mutated in place
    K:          int


@dataclass
class StepRefState:
    """Mirrors cuda StepConfig — 8 layers, 3 tdrop counters, sensor dims."""
    H0:           int
    W0:           int
    tdrop_window: int
    layers:       List[LayerRef]                            # length 8
    tdrop:        List[np.ndarray]   = field(default_factory=list)  # 3 × uint8


def step_ref(state: StepRefState, ev_x: int, ev_y: int,
             dt_norm: float, polarity: float
             ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray],
                        Optional[np.ndarray], Optional[np.ndarray],
                        List[int], List[int]]:
    """Full per-event step. Mutates `state.layers[*].H` and `state.tdrop[*]`
    in place. Returns (passed, s0, s1, s2, s3, touched_x, touched_y) where
    `passed` is False if the event was filtered (out-of-bounds or
    temporal_dropout). On False, s* / touched_* up to the filter point
    may have been computed but the caller should treat them as undefined."""
    if ev_x < 0 or ev_x >= state.W0 or ev_y < 0 or ev_y >= state.H0:
        return False, None, None, None, None, [-1]*4, [-1]*4

    feat_in = np.array([dt_norm, polarity], dtype=np.float32)
    touched_x, touched_y = [-1]*4, [-1]*4
    evx, evy = ev_x, ev_y
    Hl,  Wl  = state.H0, state.W0

    # ---- stage 0
    touched_x[0] = evx; touched_y[0] = evy
    L0, L1 = state.layers[0], state.layers[1]
    buf = layer_step(feat_in, L0.H, L0.qvgIn, L0.goW, L0.input_proj,
                     L0.ln_gamma, L0.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L0.K)
    s0  = layer_step(buf,     L1.H, L1.qvgIn, L1.goW, L1.input_proj,
                     L1.ln_gamma, L1.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L1.K)
    evx >>= 1; evy >>= 1; Hl >>= 1; Wl >>= 1
    if not _tdrop(state.tdrop[0], state.tdrop_window, evy * Wl + evx):
        return False, s0, None, None, None, touched_x, touched_y

    # ---- stage 1
    touched_x[1] = evx; touched_y[1] = evy
    L2, L3 = state.layers[2], state.layers[3]
    buf = layer_step(s0,  L2.H, L2.qvgIn, L2.goW, L2.input_proj,
                     L2.ln_gamma, L2.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L2.K)
    s1  = layer_step(buf, L3.H, L3.qvgIn, L3.goW, L3.input_proj,
                     L3.ln_gamma, L3.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L3.K)
    evx >>= 1; evy >>= 1; Hl >>= 1; Wl >>= 1
    if not _tdrop(state.tdrop[1], state.tdrop_window, evy * Wl + evx):
        return False, s0, s1, None, None, touched_x, touched_y

    # ---- stage 2
    touched_x[2] = evx; touched_y[2] = evy
    L4, L5 = state.layers[4], state.layers[5]
    buf = layer_step(s1,  L4.H, L4.qvgIn, L4.goW, L4.input_proj,
                     L4.ln_gamma, L4.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L4.K)
    s2  = layer_step(buf, L5.H, L5.qvgIn, L5.goW, L5.input_proj,
                     L5.ln_gamma, L5.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L5.K)
    evx >>= 1; evy >>= 1; Hl >>= 1; Wl >>= 1
    if not _tdrop(state.tdrop[2], state.tdrop_window, evy * Wl + evx):
        return False, s0, s1, s2, None, touched_x, touched_y

    # ---- stage 3 (no pool / tdrop after)
    touched_x[3] = evx; touched_y[3] = evy
    L6, L7 = state.layers[6], state.layers[7]
    buf = layer_step(s2,  L6.H, L6.qvgIn, L6.goW, L6.input_proj,
                     L6.ln_gamma, L6.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L6.K)
    s3  = layer_step(buf, L7.H, L7.qvgIn, L7.goW, L7.input_proj,
                     L7.ln_gamma, L7.ln_beta,
                     ev_x=evx, ev_y=evy, Hl=Hl, Wl=Wl, K=L7.K)
    return True, s0, s1, s2, s3, touched_x, touched_y


def _tdrop(counter: np.ndarray, window: int, idx: int) -> bool:
    pre = int(counter[idx])
    counter[idx] = (pre + 1) & 0xff
    if window <= 1:
        return True
    return (pre % window) == 0
