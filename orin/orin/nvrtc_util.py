"""NVRTC compile + kernel launch helpers for cuda-python on Orin.

Wraps the verbose NVRTC + driver-API calls in a small object model:

    mod = CudaModule(src, name="ssla_primitives.cu")
    func = mod.get_function("k_matvec_ct_12_24")
    func.launch((1,1,1), (1,1,1),
                [arg(devptr_x), arg(devptr_W), arg(devptr_b), arg(devptr_y)])

`arg(...)` constructs the ctypes object the driver expects (use
`arg_ptr(devptr)` for `void*` device pointers, `arg_i32(int)` for an
`int`, `arg_f32(float)` for a `float`). Each arg is held in a Python
list so its address stays valid until launch returns.

PTX caching — NVRTC takes minutes on the Orin's ARM CPU even for
moderate template instantiations. CudaModule hashes the (source +
headers + options + arch) tuple and stores the resulting PTX under
~/.cache/openeva-orin-ptx/. Subsequent runs with identical inputs skip
NVRTC entirely (cuModuleLoadData on cached bytes takes < 100 ms). Set
`OPENEVA_ORIN_NO_PTX_CACHE=1` to force a recompile.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from . import cuda_util


_DEFAULT_CACHE = Path(os.environ.get(
    "OPENEVA_ORIN_PTX_CACHE",
    str(Path.home() / ".cache" / "openeva-orin-ptx")))


def _check(err, op: str):
    from cuda import cuda, nvrtc
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{op} failed: {err}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{op} failed: {err}")
    else:
        raise RuntimeError(f"{op}: unexpected error type {type(err)}: {err!r}")


# ---- arg packing helpers ------------------------------------------------

def arg_ptr(devptr: int) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(devptr))


def arg_i32(v: int) -> ctypes.c_int32:
    return ctypes.c_int32(int(v))


def arg_u32(v: int) -> ctypes.c_uint32:
    return ctypes.c_uint32(int(v))


def arg_u64(v: int) -> ctypes.c_uint64:
    return ctypes.c_uint64(int(v))


def arg_f32(v: float) -> ctypes.c_float:
    return ctypes.c_float(float(v))


# ---- Module / Function --------------------------------------------------

class CudaModule:
    def __init__(self, src: str, *, name: str = "module.cu",
                 options: Sequence[bytes] = None,
                 arch: str = "compute_87",
                 headers: Optional[dict] = None,
                 cache_dir: Optional[Path] = None):
        cuda_util.ensure_context()
        from cuda import cuda
        if options is None:
            options = (
                f"--gpu-architecture={arch}".encode(),
                b"-std=c++17",
                b"--use_fast_math",
            )
        cache_disabled = bool(os.environ.get("OPENEVA_ORIN_NO_PTX_CACHE"))
        cache_root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        cache_path = None
        if not cache_disabled:
            digest = self._cache_key(src, options, headers, arch)
            cache_path = cache_root / f"{digest}.ptx"
            if cache_path.is_file():
                ptx_buf = cache_path.read_bytes()
                self._cache_status = "hit"
                err, self._mod = cuda.cuModuleLoadData(ptx_buf)
                _check(err, "cuModuleLoadData")
                return

        ptx_buf = self._compile_nvrtc(src, name, options, headers)
        self._cache_status = "miss"
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".ptx.tmp")
            tmp.write_bytes(ptx_buf)
            os.replace(tmp, cache_path)
        err, self._mod = cuda.cuModuleLoadData(ptx_buf)
        _check(err, "cuModuleLoadData")

    @staticmethod
    def _cache_key(src: str, options: Sequence[bytes],
                   headers: Optional[dict], arch: str) -> str:
        h = hashlib.sha256()
        h.update(b"openeva-orin-ptx-v1\n")
        h.update(arch.encode()); h.update(b"\n")
        h.update(src.encode());  h.update(b"\n--HDRS--\n")
        if headers:
            for k in sorted(headers):
                h.update(k.encode()); h.update(b":")
                h.update(headers[k].encode()); h.update(b"\n")
        h.update(b"--OPTS--\n")
        for opt in options:
            h.update(opt); h.update(b"\n")
        return h.hexdigest()

    @staticmethod
    def _compile_nvrtc(src: str, name: str,
                       options: Sequence[bytes],
                       headers: Optional[dict]) -> bytes:
        from cuda import nvrtc
        if headers:
            header_sources = [v.encode() for v in headers.values()]
            header_names   = [k.encode() for k in headers.keys()]
        else:
            header_sources, header_names = [], []
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(),
                                             len(header_sources),
                                             header_sources, header_names)
        _check(err, "nvrtcCreateProgram")
        try:
            err, = nvrtc.nvrtcCompileProgram(prog, len(options), list(options))
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                err_log, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
                log_buf = b" " * log_size
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                raise RuntimeError(
                    f"NVRTC compile failed:\n{log_buf.decode(errors='replace')}")
            err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
            _check(err, "nvrtcGetPTXSize")
            ptx_buf = b" " * ptx_size
            err, = nvrtc.nvrtcGetPTX(prog, ptx_buf)
            _check(err, "nvrtcGetPTX")
            return ptx_buf
        finally:
            nvrtc.nvrtcDestroyProgram(prog)

    @property
    def cache_status(self) -> str:
        """\"hit\" if the PTX was loaded from disk, \"miss\" if it was just compiled."""
        return self._cache_status

    def get_function(self, name: str) -> "CudaFunction":
        from cuda import cuda
        err, func = cuda.cuModuleGetFunction(self._mod, name.encode())
        _check(err, f"cuModuleGetFunction({name})")
        return CudaFunction(func, name)

    def close(self):
        if self._mod is None:
            return
        from cuda import cuda
        cuda.cuModuleUnload(self._mod)
        self._mod = None


class CudaFunction:
    def __init__(self, func, name: str):
        self._func = func
        self._name = name

    def launch(self,
               grid: Tuple[int, int, int],
               block: Tuple[int, int, int],
               args: Iterable[ctypes._SimpleCData],
               *,
               smem: int = 0,
               stream: int = 0):
        from cuda import cuda
        args_list: List[ctypes._SimpleCData] = list(args)
        # Driver wants an array of `void*` where each element is a pointer
        # to the host-side arg storage.
        addr_list = [ctypes.addressof(a) for a in args_list]
        addr_arr = (ctypes.c_void_p * len(addr_list))(*addr_list)

        err, = cuda.cuLaunchKernel(
            self._func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            smem, stream, addr_arr, 0)
        _check(err, f"cuLaunchKernel({self._name})")

    def sync(self):
        """Block until all prior launches on the default stream complete.
        cuda-python lacks a 1-arg cuStreamSynchronize; use cuCtxSynchronize."""
        from cuda import cuda
        err, = cuda.cuCtxSynchronize()
        _check(err, "cuCtxSynchronize")
