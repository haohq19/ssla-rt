"""Lazy CUDA context + managed-memory allocator helpers.

The first call to `ensure_context()` initialises the driver, picks
device 0, retains its primary context, and makes it current. All later
calls are no-ops. The primary context is leaked at process exit on
purpose — releasing it can race against background threads still
holding device pointers.

`alloc_managed(nbytes)` returns `(devptr, ctypes_array)` — the device
pointer is what NVRTC-launched kernels need; the ctypes array is what
`np.frombuffer` wraps to produce a host-visible view.
"""
from __future__ import annotations

import ctypes
from typing import Tuple

_initialized = False
_dev = None
_ctx = None


def ensure_context():
    global _initialized, _dev, _ctx
    if _initialized:
        return
    from cuda import cuda
    (err,) = cuda.cuInit(0)
    _check(err, "cuInit")
    err, _dev = cuda.cuDeviceGet(0)
    _check(err, "cuDeviceGet")
    err, _ctx = cuda.cuDevicePrimaryCtxRetain(_dev)
    _check(err, "cuDevicePrimaryCtxRetain")
    (err,) = cuda.cuCtxSetCurrent(_ctx)
    _check(err, "cuCtxSetCurrent")
    _initialized = True


def alloc_managed(nbytes: int) -> Tuple[int, ctypes.Array]:
    ensure_context()
    from cuda import cuda
    err, ptr = cuda.cuMemAllocManaged(
        nbytes, cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)
    _check(err, "cuMemAllocManaged")
    devptr = int(ptr)
    arr = (ctypes.c_ubyte * nbytes).from_address(devptr)
    ctypes.memset(devptr, 0, nbytes)
    return devptr, arr


def free_managed(devptr: int):
    if not _initialized:
        return
    from cuda import cuda
    (err,) = cuda.cuMemFree(devptr)
    _check(err, "cuMemFree")


def alloc_pinned(nbytes: int) -> Tuple[int, ctypes.Array]:
    """Pinned host memory mapped into the GPU address space.

    Use this for buffers that need *concurrent* CPU+GPU access (SPSC
    ring head/tail, stop flag, output ring on Tegra). Managed memory
    can't do this on Orin (CONCURRENT_MANAGED_ACCESS=0).

    On Tegra integrated GPUs the host pointer == device pointer (unified
    address space), so the same value works for kernel arguments.
    """
    ensure_context()
    from cuda import cuda
    err, ptr = cuda.cuMemHostAlloc(nbytes, 0)
    _check(err, "cuMemHostAlloc")
    devptr = int(ptr)
    arr = (ctypes.c_ubyte * nbytes).from_address(devptr)
    ctypes.memset(devptr, 0, nbytes)
    return devptr, arr


def free_pinned(devptr: int):
    if not _initialized:
        return
    from cuda import cuda
    (err,) = cuda.cuMemFreeHost(devptr)
    _check(err, "cuMemFreeHost")


def _check(err, op: str):
    from cuda import cuda
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{op} failed: {err}")
