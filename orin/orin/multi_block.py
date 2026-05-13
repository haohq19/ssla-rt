"""Helpers for N-block multi-strip GPU live mode.

Splits the s2 grid (W2 × H2) into a grid of (Wn × Hn) sub-strips with
halo=2 overlap. Each sub-strip becomes one GPU block. Per-block resources
(hidden state, tdrop counters, predictions, rings, tails) are allocated
here so hybrid_runner can stay focused on the launch flow.

Topology presets:
  N=2:  2 W × 1 H  (legacy)
  N=4:  2 W × 2 H
  N=8:  2 W × 4 H
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import cuda_util
from .hybrid_common import (
    INPUT_DTYPE, OUTPUT_DTYPE, TIMING_DTYPE,
    C1, C2, C3, MAX_BLOCKS, HEAD_OUT_DEFAULT,
    CHybridS2S3Config, SSLA_S2S3_IN_OUT, SSLA_S2S3_KERNELS,
    grid_for_layer, alloc_and_fill, reshape_ssla_layer,
)
from .ssla_ref import LayerRef
from .weights_ssla import reshape_ssla_layer as _r  # noqa: F401


HALO = 2   # locked per CLAUDE.md §2.2; never reduce


@dataclass
class BlockTopo:
    """Per-block ownership + processing window in s2-grid coords."""
    owned_x_lo: int
    owned_x_hi: int
    owned_y_lo: int
    owned_y_hi: int
    proc_x_lo:  int    # owned ± halo, clamped
    proc_x_hi:  int
    proc_y_lo:  int
    proc_y_hi:  int


def _balanced_splits(extent: int, n_strips: int) -> List[int]:
    """Return n_strips+1 boundary indices in [0, extent] such that the
    per-strip proc width (= owned + halo on each non-boundary side) is
    equal across strips. For n_strips >= 3, this gives unequal owned
    sizes: corner strips (touching boundary) get an extra `HALO` cells
    so corner_proc == middle_proc. For n_strips <= 2, all strips touch
    a boundary on the same side (one-sided halo only) and uniform split
    is balanced.
    """
    if n_strips == 1:
        return [0, extent]
    if n_strips == 2:
        size = extent // 2
        return [0, size, extent]
    # n_strips >= 3: corners (top+bot) own `middle + HALO` extra cells so
    # all proc widths equal. Solve: 2*(middle+HALO) + (n-2)*middle = extent.
    middle = (extent - 2 * HALO) // n_strips
    if middle < 1:
        # Fall back to uniform if uneven split would give zero/negative.
        size = extent // n_strips
        return [i * size if i < n_strips else extent for i in range(n_strips + 1)]
    corner = middle + HALO
    total = 2 * corner + (n_strips - 2) * middle
    # Distribute any remainder to the corners.
    deficit = extent - total
    corner_lo_extra = deficit // 2
    corner_hi_extra = deficit - corner_lo_extra
    splits = [0]
    cur = 0
    for i in range(n_strips):
        if i == 0:
            sz = corner + corner_lo_extra
        elif i == n_strips - 1:
            sz = corner + corner_hi_extra
        else:
            sz = middle
        cur += sz
        splits.append(cur)
    assert splits[-1] == extent, f"split mismatch: {splits} for extent={extent}"
    return splits


def grid_topology(n_blocks: int, W2: int, H2: int) -> List[BlockTopo]:
    """Return per-block (owned, proc) ranges. owned cells are disjoint
    across blocks; proc cells overlap by halo with neighbours.

    For 4-strip-on-one-axis topologies (e.g. n_blocks=8 = 2W×4H), the
    middle strips would normally accumulate halo on BOTH sides, making
    them ~1.33× bigger than corner strips. We use _balanced_splits so
    middle owned ranges are smaller, equalizing proc width across all
    strips — measured to shift saturation off the middle blocks.
    """
    if n_blocks == 2:
        nw, nh = 2, 1
    elif n_blocks == 4:
        nw, nh = 2, 2
    elif n_blocks == 8:
        nw, nh = 2, 4
    else:
        raise ValueError(f"unsupported n_blocks={n_blocks}; use 2, 4, or 8")
    assert nw * nh == n_blocks

    x_splits = _balanced_splits(W2, nw)
    y_splits = _balanced_splits(H2, nh)
    out = []
    for bw in range(nw):
        for bh in range(nh):
            ox_lo, ox_hi = x_splits[bw], x_splits[bw + 1]
            oy_lo, oy_hi = y_splits[bh], y_splits[bh + 1]
            out.append(BlockTopo(
                owned_x_lo=ox_lo, owned_x_hi=ox_hi,
                owned_y_lo=oy_lo, owned_y_hi=oy_hi,
                proc_x_lo=max(0,  ox_lo - HALO),
                proc_x_hi=min(W2, ox_hi + HALO),
                proc_y_lo=max(0,  oy_lo - HALO),
                proc_y_hi=min(H2, oy_hi + HALO),
            ))
    return out


def build_n_block_random_layers(rng: np.random.Generator,
                                  n_blocks: int,
                                  H2: int, W2: int, H3: int, W3: int):
    """Build random weights (shared across blocks) plus N per-block
    hidden state buffers per layer. Returns (cpu_layers, gpu_layers,
    keepalive)."""
    cpu_layers = []
    gpu_layers = [[None] * 4 for _ in range(n_blocks)]
    keepalive = []
    for li in range(4):
        in_d, out_d = SSLA_S2S3_IN_OUT[li]
        K = SSLA_S2S3_KERNELS[li]
        n_pos = K * K
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


def alloc_tdrop_n_block(H2: int, W2: int, H3: int, W3: int, n_blocks: int):
    """Per-block tdrop counter byte arrays (managed mem)."""
    tdrop_s2_dev, tdrop_s3_dev = [], []
    keepalive = []
    for _ in range(n_blocks):
        p, ka = cuda_util.alloc_managed(H2 * W2)
        ctypes.memset(p, 0, H2 * W2)
        keepalive.append(ka)
        tdrop_s2_dev.append(p)
        p, ka = cuda_util.alloc_managed(H3 * W3)
        ctypes.memset(p, 0, H3 * W3)
        keepalive.append(ka)
        tdrop_s3_dev.append(p)
    return tdrop_s2_dev, tdrop_s3_dev, keepalive


def alloc_predictions_n_block(topo: List[BlockTopo],
                                head_out_dim: int = HEAD_OUT_DEFAULT):
    """Per-block prediction + version buffers (pinned). Each block's grid
    is sized to its (s3_owned_h × s3_owned_w)."""
    preds_dev, preds_view = [], []
    version_dev, version_view = [], []
    keepalive = []
    for t in topo:
        s3w = (t.owned_x_hi - t.owned_x_lo) >> 1
        s3h = (t.owned_y_hi - t.owned_y_lo) >> 1
        n_cells = max(1, s3w * s3h)
        nbytes = n_cells * head_out_dim * 4
        p, ka = cuda_util.alloc_pinned(nbytes)
        ctypes.memset(p, 0, nbytes)
        keepalive.append(ka)
        preds_dev.append(p)
        preds_view.append(np.frombuffer(ka, dtype=np.float32)
                            .reshape(n_cells, head_out_dim))
        nbytes_v = n_cells * 4
        pv, kav = cuda_util.alloc_pinned(nbytes_v)
        ctypes.memset(pv, 0, nbytes_v)
        keepalive.append(kav)
        version_dev.append(pv)
        version_view.append(np.frombuffer(kav, dtype=np.uint32).reshape(n_cells))
    return preds_dev, preds_view, version_dev, version_view, keepalive


def alloc_timing_n_block(capacity_pow2: int, n_blocks: int):
    """Per-block GPU timing buffers (managed)."""
    nbytes = capacity_pow2 * TIMING_DTYPE.itemsize
    devs, views, kas = [], [], []
    for _ in range(n_blocks):
        p, ka = cuda_util.alloc_managed(nbytes)
        ctypes.memset(p, 0, nbytes)
        kas.append(ka)
        devs.append(p)
        views.append(np.frombuffer(ka, dtype=TIMING_DTYPE).reshape(-1))
    return devs, views, capacity_pow2 - 1, kas


def alloc_kernel_clk_n_block(n_blocks: int):
    """Per-block kernel_start_clk + kernel_end_clk (pinned, 8 bytes each)."""
    starts_dev, starts_view = [], []
    ends_dev, ends_view = [], []
    keepalive = []
    for _ in range(n_blocks):
        p, ka = cuda_util.alloc_pinned(8)
        ctypes.memset(p, 0, 8)
        keepalive.append(ka)
        starts_dev.append(p)
        starts_view.append(np.frombuffer(ka, dtype=np.uint64).reshape(1))
        p, ka = cuda_util.alloc_pinned(8)
        ctypes.memset(p, 0, 8)
        keepalive.append(ka)
        ends_dev.append(p)
        ends_view.append(np.frombuffer(ka, dtype=np.uint64).reshape(1))
    return starts_dev, starts_view, ends_dev, ends_view, keepalive


def build_n_block_config(H2, W2, H3, W3, tdrop_window: int,
                           gpu_layers, tdrop_s2_dev, tdrop_s3_dev,
                           topo: List[BlockTopo],
                           *, head_W=0, head_b=0, head_out_dim=HEAD_OUT_DEFAULT,
                           preds_dev=None, version_dev=None,
                           timing_dev=None, timing_mask=0,
                           kernel_start_clk=None, kernel_end_clk=None):
    """Build N-block CHybridS2S3Config in managed memory.
    Returns (devptr, keepalive)."""
    n = len(topo)
    cfg = CHybridS2S3Config()
    cfg.H2, cfg.W2 = H2, W2
    cfg.H3, cfg.W3 = H3, W3
    cfg.tdrop_window = tdrop_window
    cfg.head_out_dim = head_out_dim
    cfg.n_blocks = n
    for b, t in enumerate(topo):
        cfg.strip[b].owned_lo,    cfg.strip[b].owned_hi    = t.owned_x_lo, t.owned_x_hi
        cfg.strip[b].s3_owned_lo, cfg.strip[b].s3_owned_hi = t.owned_x_lo >> 1, t.owned_x_hi >> 1
        cfg.strip[b].owned_y_lo,    cfg.strip[b].owned_y_hi    = t.owned_y_lo, t.owned_y_hi
        cfg.strip[b].s3_owned_y_lo, cfg.strip[b].s3_owned_y_hi = t.owned_y_lo >> 1, t.owned_y_hi >> 1
        for li in range(4):
            cfg.layers[b][li].qvgIn      = gpu_layers[b][li]["qvgIn"]
            cfg.layers[b][li].goW        = gpu_layers[b][li]["goW"]
            cfg.layers[b][li].input_proj = gpu_layers[b][li]["input_proj"]
            cfg.layers[b][li].ln_gamma   = gpu_layers[b][li]["ln_gamma"]
            cfg.layers[b][li].ln_beta    = gpu_layers[b][li]["ln_beta"]
            cfg.hidden[b][li]            = gpu_layers[b][li]["hidden"]
        cfg.tdrop_s2[b] = tdrop_s2_dev[b]
        cfg.tdrop_s3[b] = tdrop_s3_dev[b]
        if preds_dev   is not None: cfg.preds[b]   = preds_dev[b]
        if version_dev is not None: cfg.version[b] = version_dev[b]
        if timing_dev  is not None: cfg.timing[b]  = timing_dev[b]
        if kernel_start_clk is not None: cfg.kernel_start_clk[b] = kernel_start_clk[b]
        if kernel_end_clk   is not None: cfg.kernel_end_clk[b]   = kernel_end_clk[b]
    cfg.head_W = head_W
    cfg.head_b = head_b
    cfg.timing_mask = timing_mask
    p_cfg, cfg_ka = cuda_util.alloc_managed(ctypes.sizeof(CHybridS2S3Config))
    ctypes.memmove(p_cfg, ctypes.addressof(cfg), ctypes.sizeof(CHybridS2S3Config))
    return p_cfg, cfg_ka
