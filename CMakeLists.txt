cmake_minimum_required(VERSION 3.16)
project(ssla-rt LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "Real-time SSLA-S inference runtime for edge devices")

# C++20 — Orin GCC 10 max; libstdc++ on Tegra has only partial C++23.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Release flags must match the vendored openeva/prim/* template instantiation
# settings (-O3 -march=native -ffast-math) so per-event matvec kernels keep
# their auto-vectorization.
add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -funroll-loops -ffast-math
                      -fopenmp-simd -DNDEBUG)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    add_link_options(-lmvec)
  endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # public ssla-rt headers
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor    # vendored openeva/* and src/* headers
)

find_package(Threads REQUIRED)

# ---- Vendored openeva primitives + weights loader (compiled in) -------------
add_library(openeva_vendor STATIC
    vendor/src/weights_loader.cpp
    vendor/src/io.cpp
)
set_target_properties(openeva_vendor PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- SSLA-S CPU kernel port (used by all CPU pipeline libs/exes) ------------
add_library(ssla_kernels STATIC
    src/ssla_kernels.cpp
)
target_link_libraries(ssla_kernels PRIVATE openeva_vendor)
set_target_properties(ssla_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Note: src/s1_reference.cpp and src/multicore_bench.cpp depend on
# openeva::create_model() — a research-side factory spanning all methods —
# and are intentionally not built here. Keep the files for reference; if
# you want them, depend on the full openeva research tree.

# ---- Oracle diff CLI --------------------------------------------------------
add_executable(oracle_diff src/oracle_diff.cpp)
target_link_libraries(oracle_diff PRIVATE Threads::Threads)

# ---- Microbench: scalar vs hand-NEON matvec_ct<12, 36> ----------------------
# Drives the Phase-2 decision on whether to write NEON specializations for
# the IN=12 layers (stage-0 L1 hot path). See STATUS.md §8.4 for context.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  add_executable(bench_matvec12 tests/bench_matvec12.cpp)
  target_link_libraries(bench_matvec12 PRIVATE openeva_vendor)
endif()

# ---- CPU pipeline shared libs (Python callers via ctypes) -------------------
add_library(stage0_capi SHARED src/lib_stage0_capi.cpp)
target_link_libraries(stage0_capi PRIVATE ssla_kernels openeva_vendor Threads::Threads)
set_target_properties(stage0_capi PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(stage01_capi SHARED src/lib_stage01_capi.cpp)
target_link_libraries(stage01_capi PRIVATE ssla_kernels openeva_vendor Threads::Threads)
set_target_properties(stage01_capi PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Hybrid CPU→GPU pipeline (s0+s1 on CPU, ring push to GPU). Used by
# orin/hybrid_runner.py with --kernel-variant celled.
add_library(stage01_to_gpu SHARED src/lib_stage01_to_gpu.cpp)
target_link_libraries(stage01_to_gpu PRIVATE ssla_kernels openeva_vendor Threads::Threads)
set_target_properties(stage01_to_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Intermediate CPU sharding schemes (S2..S8) -----------------------------
# Historical scheme-comparison binaries from the dev log. Tested on EPYC
# (x86-64) — some use `__builtin_ia32_pause` directly and won't compile on
# aarch64 without conditional patches. Opt-in via `-DSSLA_RT_BUILD_SCHEMES=ON`.
# The production path is the SHARED libs above + the Python runtime in orin/;
# the schemes are not needed for normal use.
option(SSLA_RT_BUILD_SCHEMES "Build historical CPU sharding scheme binaries" OFF)
if(SSLA_RT_BUILD_SCHEMES)
  foreach(scheme
      s2_decomp_st s_stage0_only s_stage0_multicore s2_pipeline s3_sharded
      s4_owner_shard s5_locked s6_full_locked s7_replicated s8_hierarchical)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${scheme}.cpp)
      add_executable(${scheme} src/${scheme}.cpp)
      target_link_libraries(${scheme} PRIVATE ssla_kernels openeva_vendor Threads::Threads)
    endif()
  endforeach()
endif()
