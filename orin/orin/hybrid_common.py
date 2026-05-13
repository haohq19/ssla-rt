"""Shared helpers for the SSLA Hybrid CPU+GPU pipeline.

Used by both `deploy/orin/bench_s2_s3_head.py` (offline P1 harness) and
`deploy/orin/hybrid_runner.py` (live P2 driver). Keeps the GPU-side
struct definitions, layer-build code, and weight reshape utilities in
one place so the two paths can't drift.

Layout of CUDA structs in this file MUST match
`deploy/orin/kernels/ssla_s2_s3_head.cuh`. Layout of HybridInputRec
MUST match `deploy/src/lib_stage01_to_gpu.cpp` (112 bytes).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import cuda_util
from .ssla_ref import LayerRef
from .weights_ssla import reshape_ssla_layer


# ---- SSLA-S channel sizes ---------------------------------------------------
C1, C2, C3 = 24, 48, 96


# ---- numpy dtypes (mirror the CUDA structs) --------------------------------
INPUT_DTYPE = np.dtype([
    ("seq_done",  "<u8"),
    ("t",         "<f4"),
    ("x",         "<u2"),
    ("y",         "<u2"),
    ("feat1",     "<f4", (C1,)),
    ("t_push_ns", "<u8"),
    ("t_emit_ns", "<u8"),
])
assert INPUT_DTYPE.itemsize == 128, INPUT_DTYPE.itemsize

OUTPUT_DTYPE = np.dtype([
    ("pass2",   "<i4"),
    ("pass3",   "<i4"),
    ("s3_feat", "<f4", (C3,)),
])


# ---- ctypes mirrors of the CUDA structs in ssla_s2_s3_head.cuh -------------

class CHybridStrip(ctypes.Structure):
    _fields_ = [
        ("owned_lo",       ctypes.c_int32),
        ("owned_hi",       ctypes.c_int32),
        ("s3_owned_lo",    ctypes.c_int32),
        ("s3_owned_hi",    ctypes.c_int32),
        ("owned_y_lo",     ctypes.c_int32),
        ("owned_y_hi",     ctypes.c_int32),
        ("s3_owned_y_lo",  ctypes.c_int32),
        ("s3_owned_y_hi",  ctypes.c_int32),
    ]


class CLayerWeightsS2S3(ctypes.Structure):
    _fields_ = [
        ("qvgIn",      ctypes.c_uint64),
        ("goW",        ctypes.c_uint64),
        ("input_proj", ctypes.c_uint64),
        ("ln_gamma",   ctypes.c_uint64),
        ("ln_beta",    ctypes.c_uint64),
    ]


MAX_BLOCKS = 16   # match constexpr in ssla_s2_s3_head*.cuh


class CHybridS2S3Config(ctypes.Structure):
    _fields_ = [
        ("H2", ctypes.c_int32), ("W2", ctypes.c_int32),
        ("H3", ctypes.c_int32), ("W3", ctypes.c_int32),
        ("tdrop_window", ctypes.c_int32),
        ("head_out_dim", ctypes.c_int32),
        ("n_blocks",     ctypes.c_int32),
        ("_pad_nblocks", ctypes.c_int32),
        ("strip",     CHybridStrip * MAX_BLOCKS),
        ("layers",    (CLayerWeightsS2S3 * 4) * MAX_BLOCKS),
        ("hidden",    (ctypes.c_uint64 * 4) * MAX_BLOCKS),
        ("tdrop_s2",  ctypes.c_uint64 * MAX_BLOCKS),
        ("tdrop_s3",  ctypes.c_uint64 * MAX_BLOCKS),
        ("head_W",    ctypes.c_uint64),
        ("head_b",    ctypes.c_uint64),
        ("preds",     ctypes.c_uint64 * MAX_BLOCKS),
        ("version",   ctypes.c_uint64 * MAX_BLOCKS),
        ("timing",       ctypes.c_uint64 * MAX_BLOCKS),
        ("timing_mask",  ctypes.c_uint32),
        ("_pad_timing",  ctypes.c_uint32),
        ("kernel_start_clk", ctypes.c_uint64 * MAX_BLOCKS),
        ("kernel_end_clk",   ctypes.c_uint64 * MAX_BLOCKS),
    ]


# numpy dtype mirror of GpuTimingSlot in ssla_s2_s3_head.cuh.
TIMING_DTYPE = np.dtype([
    ("t_pop_clk",  "<u8"),
    ("t_done_clk", "<u8"),
    ("seq",        "<u4"),
    ("owner",      "<u4"),
    ("t_push_ns",  "<u8"),
    ("t_emit_ns",  "<u8"),
])
assert TIMING_DTYPE.itemsize == 40


HEAD_OUT_DEFAULT = 7    # 5 box + 2 cls (Gen1)


# Layer schedule for the GPU side (post-CPU-stage-1).
# L4 = 24→48 K=3, L5 = 48→48 K=3, L6 = 48→96 K=3, L7 = 96→96 K=3.
SSLA_S2S3_KERNELS = (3, 3, 3, 3)
SSLA_S2S3_IN_OUT  = ((C1, C2), (C2, C2), (C2, C3), (C3, C3))
N_BLOCKS = 2


# ---- helpers ---------------------------------------------------------------

def alloc_and_fill(arr: np.ndarray):
    """Allocate managed mem, copy arr into it, return (devptr, view, keepalive)."""
    arr = np.ascontiguousarray(arr)
    ptr, ka = cuda_util.alloc_managed(arr.nbytes)
    view = np.frombuffer(ka, dtype=arr.dtype).reshape(arr.shape)
    view[...] = arr
    return ptr, view, ka


def grid_for_layer(li: int, H2: int, W2: int, H3: int, W3: int) -> Tuple[int, int]:
    """L4, L5 are at s2 grid; L6, L7 at s3."""
    return (H2, W2) if li < 2 else (H3, W3)


def build_random_layers(rng: np.random.Generator,
                         H2: int, W2: int, H3: int, W3: int):
    """Build random weights for L4..L7 plus per-block hidden state.

    Returns:
        cpu_layers : list[LayerRef]  — for the CPU oracle (single private hidden copy)
        gpu_layers : list[list[dict]] — gpu_layers[block][li] has dev pointers
        keepalive  : list — keep alive while devptrs are in use
    """
    cpu_layers = []
    gpu_layers = [[None] * 4 for _ in range(N_BLOCKS)]
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

        # Weights are shared read-only across blocks; hidden state is per-block.
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

        for blk in range(N_BLOCKS):
            p_h, h_view, ka6 = alloc_and_fill(H_init); keepalive.append(ka6)
            gpu_layers[blk][li] = {
                "qvgIn": p_qvg, "goW": p_go, "input_proj": p_ip,
                "ln_gamma": p_lng, "ln_beta": p_lnb,
                "hidden": p_h, "hidden_view": h_view,
            }
    return cpu_layers, gpu_layers, keepalive


def build_config(H2: int, W2: int, H3: int, W3: int, tdrop_window: int,
                  gpu_layers, tdrop_s2_dev, tdrop_s3_dev,
                  *,
                  head_W: int = 0, head_b: int = 0,
                  head_out_dim: int = HEAD_OUT_DEFAULT,
                  preds: Tuple[int, int] = (0, 0),
                  version: Tuple[int, int] = (0, 0),
                  timing: Tuple[int, int] = (0, 0),
                  timing_mask: int = 0,
                  kernel_start_clk: Tuple[int, int] = (0, 0),
                  kernel_end_clk:   Tuple[int, int] = (0, 0),
                  ) -> Tuple[int, ctypes.Array]:
    """Build CHybridS2S3Config in managed memory; return (devptr, keepalive).

    Head + predictions args default to NULL — used by the P1 oracle harness
    which compares s3 features directly. The hybrid runner passes real
    pointers to enable head matvec + per-cell prediction writes."""
    cfg = CHybridS2S3Config()
    cfg.H2, cfg.W2 = H2, W2
    cfg.H3, cfg.W3 = H3, W3
    cfg.tdrop_window = tdrop_window
    cfg.head_out_dim = head_out_dim
    cfg.n_blocks = N_BLOCKS    # existing 2-block default
    strip_w = W2 // N_BLOCKS
    cfg.strip[0].owned_lo,     cfg.strip[0].owned_hi    = 0,                strip_w
    cfg.strip[0].s3_owned_lo,  cfg.strip[0].s3_owned_hi = 0,                strip_w >> 1
    cfg.strip[0].owned_y_lo,     cfg.strip[0].owned_y_hi    = 0, H2
    cfg.strip[0].s3_owned_y_lo,  cfg.strip[0].s3_owned_y_hi = 0, H3
    cfg.strip[1].owned_lo,     cfg.strip[1].owned_hi    = strip_w,          W2
    cfg.strip[1].s3_owned_lo,  cfg.strip[1].s3_owned_hi = strip_w >> 1,     W3
    cfg.strip[1].owned_y_lo,     cfg.strip[1].owned_y_hi    = 0, H2
    cfg.strip[1].s3_owned_y_lo,  cfg.strip[1].s3_owned_y_hi = 0, H3
    for blk in range(N_BLOCKS):
        for li in range(4):
            cfg.layers[blk][li].qvgIn      = gpu_layers[blk][li]["qvgIn"]
            cfg.layers[blk][li].goW        = gpu_layers[blk][li]["goW"]
            cfg.layers[blk][li].input_proj = gpu_layers[blk][li]["input_proj"]
            cfg.layers[blk][li].ln_gamma   = gpu_layers[blk][li]["ln_gamma"]
            cfg.layers[blk][li].ln_beta    = gpu_layers[blk][li]["ln_beta"]
            cfg.hidden[blk][li]            = gpu_layers[blk][li]["hidden"]
        cfg.tdrop_s2[blk] = tdrop_s2_dev[blk]
        cfg.tdrop_s3[blk] = tdrop_s3_dev[blk]
    cfg.head_W     = head_W
    cfg.head_b     = head_b
    cfg.preds[0]   = preds[0]
    cfg.preds[1]   = preds[1]
    cfg.version[0] = version[0]
    cfg.version[1] = version[1]
    cfg.timing[0]  = timing[0]
    cfg.timing[1]  = timing[1]
    cfg.timing_mask = timing_mask
    cfg.kernel_start_clk[0] = kernel_start_clk[0]
    cfg.kernel_start_clk[1] = kernel_start_clk[1]
    cfg.kernel_end_clk[0]   = kernel_end_clk[0]
    cfg.kernel_end_clk[1]   = kernel_end_clk[1]
    p_cfg, cfg_ka = cuda_util.alloc_managed(ctypes.sizeof(CHybridS2S3Config))
    ctypes.memmove(p_cfg, ctypes.addressof(cfg), ctypes.sizeof(CHybridS2S3Config))
    return p_cfg, cfg_ka


def alloc_timing(capacity_pow2: int):
    """Per-block GPU timing-slot ring (managed mem, GPU writes / host reads
    after kernel exit).

    Returns (timing_dev[2], timing_view[2], timing_mask, keepalive)."""
    if capacity_pow2 & (capacity_pow2 - 1):
        raise ValueError("timing capacity must be power of 2")
    nbytes = capacity_pow2 * TIMING_DTYPE.itemsize
    devs, views, kas = [], [], []
    for _ in range(N_BLOCKS):
        p, ka = cuda_util.alloc_managed(nbytes)
        ctypes.memset(p, 0, nbytes)
        kas.append(ka)
        devs.append(p)
        views.append(np.frombuffer(ka, dtype=TIMING_DTYPE).reshape(-1))
    return devs, views, capacity_pow2 - 1, kas


def alloc_kernel_start_clk():
    """Per-block uint64 slot the kernel writes once at entry (SM clock cycles).
    Host reads after launch + small spin to obtain the (CPU_ns ↔ SM_cycles)
    calibration anchor.

    MUST be pinned host memory (not managed) because the CPU spin-polls this
    while the persistent kernel is running, and Orin reports
    CONCURRENT_MANAGED_ACCESS=0 — concurrent CPU/GPU access to managed memory
    is UB on Tegra. See CLAUDE.md §2.1.

    Returns (devs[2], views[2], keepalive)."""
    devs, views, kas = [], [], []
    for _ in range(N_BLOCKS):
        p, ka = cuda_util.alloc_pinned(8)
        ctypes.memset(p, 0, 8)
        kas.append(ka)
        devs.append(p)
        views.append(np.frombuffer(ka, dtype=np.uint64).reshape(1))
    return devs, views, kas


def build_head_weights(rng: np.random.Generator, head_out_dim: int = HEAD_OUT_DEFAULT):
    """Random-init head weights in managed mem.

    matvec_ct expects W layout (IN, OUT) accessed as W[i*OUT + o], so head_W
    has shape (C3, head_out_dim) row-major.
    """
    head_W = rng.normal(scale=0.1, size=(C3, head_out_dim)).astype(np.float32)
    head_b = rng.normal(scale=0.1, size=(head_out_dim,)).astype(np.float32)
    p_W, _, ka_W = alloc_and_fill(head_W)
    p_b, _, ka_b = alloc_and_fill(head_b)
    return p_W, p_b, [ka_W, ka_b]


def alloc_predictions(H3: int, W2: int, head_out_dim: int = HEAD_OUT_DEFAULT):
    """Per-block pinned-host predictions + version counters.

    Returns (preds_dev[2], preds_view[2], version_dev[2], version_view[2], keepalive).
    Layout for block b: preds_view[b][cell_idx, :] where
        cell_idx = hy * s3_owned_w + hx_local.
    """
    s3_owned_w = (W2 // N_BLOCKS) >> 1
    n_cells = H3 * s3_owned_w
    preds_dev, preds_view = [], []
    version_dev, version_view = [], []
    keepalive = []
    for b in range(N_BLOCKS):
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


def alloc_tdrop_counters(H2, W2, H3, W3):
    """Per-block tdrop counter byte arrays in managed memory.
    Returns ([tdrop_s2_dev[2]], [tdrop_s3_dev[2]], keepalive)."""
    tdrop_s2_dev, tdrop_s3_dev = [], []
    keepalive = []
    for blk in range(N_BLOCKS):
        p, ka = cuda_util.alloc_managed(H2 * W2)
        ctypes.memset(p, 0, H2 * W2)
        keepalive.append(ka)
        tdrop_s2_dev.append(p)
        p, ka = cuda_util.alloc_managed(H3 * W3)
        ctypes.memset(p, 0, H3 * W3)
        keepalive.append(ka)
        tdrop_s3_dev.append(p)
    return tdrop_s2_dev, tdrop_s3_dev, keepalive


def block_proc_range(W2: int, halo_s2: int = 2):
    """Per-block s2-x processing range with halo. Returns (proc_lo[2], proc_hi[2])."""
    strip_w = W2 // N_BLOCKS
    proc_lo = [max(0,  0       - halo_s2),
               max(0,  strip_w - halo_s2)]
    proc_hi = [min(W2, strip_w + halo_s2),
               min(W2, W2      + halo_s2)]
    return proc_lo, proc_hi


# ---- ctypes wrapper for libstage01_to_gpu.so -------------------------------

class S01gAPI:
    """Thin ctypes wrapper around lib_stage01_to_gpu.so. Manages a single
    handle returned by s01g_init."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        L = self.lib
        L.s01g_init.restype = ctypes.c_void_p
        L.s01g_init.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.s01g_init_full.restype = ctypes.c_void_p
        L.s01g_init_full.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_uint]
        L.s01g_attach_gpu_rings.restype = None
        L.s01g_attach_gpu_rings.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        L.s01g_attach_gpu_rings_multi.restype = None
        L.s01g_attach_gpu_rings_multi.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),    # ring_bufs[]
            ctypes.POINTER(ctypes.c_void_p),    # ring_heads[]
            ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),       # proc_x_los[]
            ctypes.POINTER(ctypes.c_int),       # proc_x_his[]
            ctypes.POINTER(ctypes.c_int),       # proc_y_los[]
            ctypes.POINTER(ctypes.c_int),       # proc_y_his[]
        ]
        L.s01g_submit_batch.restype = ctypes.c_int
        L.s01g_submit_batch.argtypes = [ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int]
        L.s01g_snapshot_stats.restype = None
        L.s01g_snapshot_stats.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_double)]
        L.s01g_reset_stats.restype = None
        L.s01g_reset_stats.argtypes = [ctypes.c_void_p]
        L.s01g_shutdown.restype = None
        L.s01g_shutdown.argtypes = [ctypes.c_void_p]
        L.s01g_start_synthetic.restype = None
        L.s01g_start_synthetic.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                            ctypes.c_int, ctypes.c_uint64]
        L.s01g_stop_synthetic.restype = None
        L.s01g_stop_synthetic.argtypes = [ctypes.c_void_p]
        L.s01g_synthetic_n_pushed.restype = ctypes.c_uint64
        L.s01g_synthetic_n_pushed.argtypes = [ctypes.c_void_p]
        self.handle = None

    def init(self, weights_dir: str, n_shards: int, halo: int,
             base_core: int, sample_cap: int, shard_ring_cap: int = 1 << 16):
        self.handle = self.lib.s01g_init_full(weights_dir.encode(), n_shards,
                                                halo, base_core, sample_cap,
                                                shard_ring_cap)
        if not self.handle:
            raise RuntimeError("s01g_init returned NULL "
                               "(check weights_dir / meta.json)")

    def attach(self, ring_buf_0, ring_head_0, ring_buf_1, ring_head_1,
                ring_mask, W2, H2,
                proc_lo_0, proc_hi_0, proc_lo_1, proc_hi_1):
        self.lib.s01g_attach_gpu_rings(self.handle,
            ctypes.c_void_p(int(ring_buf_0)),
            ctypes.c_void_p(int(ring_head_0)),
            ctypes.c_void_p(int(ring_buf_1)),
            ctypes.c_void_p(int(ring_head_1)),
            ring_mask, W2, H2,
            proc_lo_0, proc_hi_0, proc_lo_1, proc_hi_1)

    def attach_multi(self, ring_bufs, ring_heads, ring_mask, W2, H2,
                     proc_x_los, proc_x_his, proc_y_los, proc_y_his):
        n = len(ring_bufs)
        assert all(len(a) == n for a in
                   (ring_heads, proc_x_los, proc_x_his, proc_y_los, proc_y_his))
        BufArr   = ctypes.c_void_p * n
        IntArr   = ctypes.c_int * n
        bufs   = BufArr(*(ctypes.c_void_p(int(p)) for p in ring_bufs))
        heads  = BufArr(*(ctypes.c_void_p(int(p)) for p in ring_heads))
        xlos   = IntArr(*proc_x_los)
        xhis   = IntArr(*proc_x_his)
        ylos   = IntArr(*proc_y_los)
        yhis   = IntArr(*proc_y_his)
        # Keep arrays alive across the call.
        self._attach_keepalive = (bufs, heads, xlos, xhis, ylos, yhis)
        self.lib.s01g_attach_gpu_rings_multi(
            self.handle, ctypes.c_int(n),
            bufs, heads, ring_mask, W2, H2,
            xlos, xhis, ylos, yhis)

    def submit(self, events_array: np.ndarray) -> int:
        n = events_array.shape[0]
        if n == 0:
            return 0
        ptr = events_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self.lib.s01g_submit_batch(self.handle, ptr, n)

    def stats(self):
        # 32 slots: 0..15 legacy stats, 16..21 per-segment sum_ns,
        # 22..27 per-segment counts, 28..31 reserved. See
        # src/lib_stage01_to_gpu.cpp::s01g_snapshot_stats for the contract.
        out = (ctypes.c_double * 32)()
        self.lib.s01g_snapshot_stats(self.handle, out)
        return list(out)

    def reset(self):
        self.lib.s01g_reset_stats(self.handle)

    def start_synthetic(self, rate_mev: float, pin_core: int, seed: int):
        """Start a C++-side synthetic event generator. Uses rdtsc spin-pacing
        for ns-precision event spacing — replacement for Python SyntheticReader
        which is limited to time.sleep millisecond precision and has GIL /
        interpreter jitter that spikes MAX latency to ~700 µs even at 0.1 Mev/s.
        """
        self.lib.s01g_start_synthetic(self.handle, rate_mev, pin_core, seed)

    def stop_synthetic(self):
        self.lib.s01g_stop_synthetic(self.handle)

    def synthetic_n_pushed(self) -> int:
        return int(self.lib.s01g_synthetic_n_pushed(self.handle))

    def shutdown(self):
        if self.handle is not None:
            self.lib.s01g_shutdown(self.handle)
            self.handle = None
