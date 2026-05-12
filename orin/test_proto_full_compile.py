"""Minimal compile check for proto_full.cuh — query register/spill stats
before building the full harness."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orin.nvrtc_util import CudaModule

K = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
src1 = open(os.path.join(K, "proto_layer_pair.cuh")).read()
src2 = open(os.path.join(K, "proto_full.cuh")).read()

t0 = time.monotonic()
mod = CudaModule(src2, name="proto_full.cu",
                 headers={"proto_layer_pair.cuh": src1})
print(f"compile: {time.monotonic()-t0:.1f}s ({mod.cache_status})")

func = mod.get_function("k_proto_persistent_full")
from cuda import cuda
attrs = {
    "NUM_REGS": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
    "SHARED_SIZE_BYTES": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    "LOCAL_SIZE_BYTES": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    "MAX_THREADS_PER_BLOCK": cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
}
print("k_proto_persistent_full:")
for k, a in attrs.items():
    err, v = cuda.cuFuncGetAttribute(a, func._func)
    print(f"  {k:24s} = {int(v)}")
