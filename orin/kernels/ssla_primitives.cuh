// SSLA-S device-side primitives — block-cooperative variant.
//
// All primitives are *block-cooperative*: they assume the caller has
// launched a single block of SSLA_BLOCK_THREADS threads. Output dim is
// striped across threads: thread `tid` handles indices
// [tid, tid+SSLA_BLOCK_THREADS, ...). Multiple warps in flight let the
// SM hide global-memory load latency (single-warp blocks bottleneck on
// load-stall serialization).
//
// Primitives that write shared memory call `__syncthreads()` at the
// end so subsequent primitives see fresh values. LayerNorm uses a
// shared-memory partial-sum array for cross-warp reduction (one entry
// per warp) plus warp-level shuffle within each warp.
//
// Compiled by NVRTC at runtime; nothing in this file may #include host
// headers. All math is float32, except LayerNorm which uses fp64
// reductions to match the C++ runtime byte-for-byte.
#pragma once

#ifndef SSLA_BLOCK_THREADS
#define SSLA_BLOCK_THREADS 256            // 8 warps
#endif
#define SSLA_BLOCK_WARPS  ((SSLA_BLOCK_THREADS + 31) / 32)

__device__ inline int  block_tid_()  { return threadIdx.x; }
__device__ inline int  warp_lane_()  { return threadIdx.x & 31; }
__device__ inline int  warp_id_()    { return threadIdx.x >> 5; }

// ---------- matvec ---------------------------------------------------
//
// Weight layout convention: W stored as (IN, OUT), accessed W[i*OUT+o].
// Adjacent threads handle adjacent `o` → coalesced 128-byte loads.

template <int IN, int OUT>
__device__ inline void matvec_ct(const float* __restrict__ x,
                                 const float* __restrict__ W,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y) {
    const int tid = block_tid_();
    for (int o = tid; o < OUT; o += SSLA_BLOCK_THREADS) {
        float acc = (bias != 0) ? bias[o] : 0.0f;
        #pragma unroll
        for (int i = 0; i < IN; ++i) acc += W[(long long)i * OUT + o] * x[i];
        y[o] = acc;
    }
    __syncthreads();
}

template <int IN, int OUT>
__device__ inline void matvec_accum_ct(const float* __restrict__ x,
                                       const float* __restrict__ W,
                                       float* __restrict__ y) {
    const int tid = block_tid_();
    for (int o = tid; o < OUT; o += SSLA_BLOCK_THREADS) {
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < IN; ++i) acc += W[(long long)i * OUT + o] * x[i];
        y[o] += acc;
    }
    __syncthreads();
}

// ---------- elementwise + activations -------------------------------

__device__ inline float sigmoidf_(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

template <int DIM>
__device__ inline void add_inplace_ct(float* __restrict__ y,
                                      const float* __restrict__ b) {
    const int tid = block_tid_();
    for (int i = tid; i < DIM; i += SSLA_BLOCK_THREADS) y[i] += b[i];
    __syncthreads();
}

#ifndef SSLA_LRU_EXP
#define SSLA_LRU_EXP __expf
#endif
template <int DIM>
__device__ inline void lru_step_ct(const float* __restrict__ g,
                                   const float* __restrict__ v,
                                   const float* __restrict__ q,
                                   float* __restrict__ h,
                                   float* __restrict__ y) {
    const int tid = block_tid_();
    for (int c = tid; c < DIM; c += SSLA_BLOCK_THREADS) {
        const float gc = 1.0f / (1.0f + SSLA_LRU_EXP(-g[c]));
        const float hc = gc * h[c] + v[c];
        h[c]  = hc;
        y[c]  = q[c] * hc;
    }
    __syncthreads();
}

// ---------- LayerNorm (fp64 reduction across all block threads) -----

template <int DIM>
__device__ inline void layernorm_ct(float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    float eps = 1e-5f) {
    const int tid  = block_tid_();
    const int lane = warp_lane_();
    const int warp = warp_id_();
    __shared__ double warp_partials[SSLA_BLOCK_WARPS];

    // ---- pass 1: sum ----
    double sum = 0.0;
    for (int i = tid; i < DIM; i += SSLA_BLOCK_THREADS) sum += (double)x[i];
    // warp-level reduce
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (lane == 0) warp_partials[warp] = sum;
    __syncthreads();
    // cross-warp reduce (small fixed count)
    double total = 0.0;
    if (tid < SSLA_BLOCK_WARPS) total = warp_partials[tid];
    for (int off = 1; off < SSLA_BLOCK_WARPS; off <<= 1)
        total += __shfl_xor_sync(0xffffffff, total, off);
    if (tid == 0) warp_partials[0] = total;
    __syncthreads();
    const double mean = warp_partials[0] / (double)DIM;

    // ---- pass 2: var ----
    double var = 0.0;
    for (int i = tid; i < DIM; i += SSLA_BLOCK_THREADS) {
        const double d = (double)x[i] - mean;
        var += d * d;
    }
    for (int off = 16; off > 0; off >>= 1)
        var += __shfl_xor_sync(0xffffffff, var, off);
    if (lane == 0) warp_partials[warp] = var;
    __syncthreads();
    double tot_var = 0.0;
    if (tid < SSLA_BLOCK_WARPS) tot_var = warp_partials[tid];
    for (int off = 1; off < SSLA_BLOCK_WARPS; off <<= 1)
        tot_var += __shfl_xor_sync(0xffffffff, tot_var, off);
    if (tid == 0) warp_partials[0] = tot_var;
    __syncthreads();
    const double full_var = warp_partials[0] / (double)DIM;
    const double inv = 1.0 / sqrt(full_var + (double)eps);

    // ---- normalize + affine ----
    for (int i = tid; i < DIM; i += SSLA_BLOCK_THREADS) {
        const double v = ((double)x[i] - mean) * inv;
        x[i] = (float)(v * (double)gamma[i] + (double)beta[i]);
    }
    __syncthreads();
}

// =====================================================================
// Test wrappers — single-block / SSLA_BLOCK_THREADS threads.
// =====================================================================
#ifndef SSLA_PRIM_NO_TEST_WRAPPERS

#define SSLA_DEF_MATVEC_CT(IN, OUT)                                         \
extern "C" __global__ void k_matvec_ct_##IN##_##OUT(                        \
    const float* x, const float* W, const float* bias, float* y) {         \
    if (blockIdx.x == 0)                                                    \
        matvec_ct<IN, OUT>(x, W, bias, y);                                  \
}

#define SSLA_DEF_MATVEC_ACCUM_CT(IN, OUT)                                   \
extern "C" __global__ void k_matvec_accum_ct_##IN##_##OUT(                  \
    const float* x, const float* W, float* y) {                             \
    if (blockIdx.x == 0)                                                    \
        matvec_accum_ct<IN, OUT>(x, W, y);                                  \
}

#define SSLA_DEF_LRU_STEP(DIM)                                              \
extern "C" __global__ void k_lru_step_##DIM(                                \
    const float* g, const float* v, const float* q,                         \
    float* h, float* y) {                                                   \
    if (blockIdx.x == 0)                                                    \
        lru_step_ct<DIM>(g, v, q, h, y);                                    \
}

#define SSLA_DEF_LAYERNORM(DIM)                                             \
extern "C" __global__ void k_layernorm_##DIM(                               \
    float* x, const float* gamma, const float* beta) {                      \
    if (blockIdx.x == 0)                                                    \
        layernorm_ct<DIM>(x, gamma, beta);                                  \
}

SSLA_DEF_MATVEC_CT(2,  12)
SSLA_DEF_MATVEC_CT(12, 12)
SSLA_DEF_MATVEC_CT(12, 24)
SSLA_DEF_MATVEC_CT(24, 24)
SSLA_DEF_MATVEC_CT(24, 48)
SSLA_DEF_MATVEC_CT(48, 48)
SSLA_DEF_MATVEC_CT(48, 96)
SSLA_DEF_MATVEC_CT(96, 96)
SSLA_DEF_MATVEC_CT(2,  36)
SSLA_DEF_MATVEC_CT(12, 36)
SSLA_DEF_MATVEC_CT(12, 72)
SSLA_DEF_MATVEC_CT(24, 72)
SSLA_DEF_MATVEC_CT(24, 144)
SSLA_DEF_MATVEC_CT(48, 144)
SSLA_DEF_MATVEC_CT(48, 288)
SSLA_DEF_MATVEC_CT(96, 288)

SSLA_DEF_MATVEC_ACCUM_CT(12, 12)
SSLA_DEF_MATVEC_ACCUM_CT(24, 24)
SSLA_DEF_MATVEC_ACCUM_CT(48, 48)
SSLA_DEF_MATVEC_ACCUM_CT(96, 96)

SSLA_DEF_LRU_STEP(12)
SSLA_DEF_LRU_STEP(24)
SSLA_DEF_LRU_STEP(48)
SSLA_DEF_LRU_STEP(96)

SSLA_DEF_LAYERNORM(12)
SSLA_DEF_LAYERNORM(24)
SSLA_DEF_LAYERNORM(48)
SSLA_DEF_LAYERNORM(96)

#endif  // SSLA_PRIM_NO_TEST_WRAPPERS
