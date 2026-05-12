#pragma once

// Fused interior-only layer kernels for s0 L1 <12,12>, s1 L0 <12,24>, and
// s1 L1 <24,24>. Each fused kernel:
//   - holds qvg (3*OUT fp32) and qh (OUT fp32) in NEON registers across
//     matvec_qvg → lru_step → matvec_accum (no scratch-buffer round-trip)
//   - manually unrolls all 9 interior patches (no lambda / functor)
//   - uses named local vars instead of arrays for the qvg slots
//     (critical — observed >40% perf gap between qvg[18] and qvg0..qvg17
//     because GCC treats fp32x4_t qvg[18] as memory and spills)
//   - calls vendored layernorm_ct (fp64 reduction) unchanged for numerics
//
// Caller must check interior bounds (ev_x ∈ [1, Wl-2], ev_y ∈ [1, Hl-2]);
// otherwise hidden-state access at base + dy*Wl + dx is out-of-bounds.
// The boundary path falls back to layer_forward_ct's bounds-clipped loop.
//
// Microbench (tests/bench_fused.cpp, Cortex-A78AE pinned core 4):
//   s0 L1 <12,12>:  822 → 752 ns/call  (1.09×)
//   s1 L0 <12,24>: 2223 → 1712 ns/call (1.30×)
//   s1 L1 <24,24>: 4184 → 2799 ns/call (1.49×)
//   max|Δ| feat_out vs generic path: 0 / 2.4e-7 / 2.4e-7 (single call).

#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>
#include <array>
#include <cstring>
#include <vector>

#include "openeva/prim/layernorm.h"
#include "ssla_neon_lru.h"   // for neon_sigmoid_tanh_pade

namespace deploy {
namespace fused {

__attribute__((always_inline))
static inline float32x4_t fused_sigmoid(float32x4_t x) {
    return openeva::prim::neon_sigmoid_tanh_pade(x);
}

// ============================================================================
// One patch of <12, 12>: matvec_qvg<12,36> + lru_step<12> + matvec_accum<12,12>
// ============================================================================
__attribute__((always_inline))
static inline void patch_12_12(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float* __restrict__ qvgW,
    const float* __restrict__ goW,
    float*       __restrict__ h_ptr,
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2)
{
    float32x4_t qvg0, qvg1, qvg2, qvg3, qvg4, qvg5, qvg6, qvg7, qvg8;

    #define MQV(og, sink) do {                                                 \
        const float* w0 = qvgW + ((og)*4 + 0) * 12;                            \
        const float* w1 = qvgW + ((og)*4 + 1) * 12;                            \
        const float* w2 = qvgW + ((og)*4 + 2) * 12;                            \
        const float* w3 = qvgW + ((og)*4 + 3) * 12;                            \
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0 + 0));                     \
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1 + 0));                     \
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2 + 0));                     \
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3 + 0));                     \
        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 + 4));                             \
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 + 4));                             \
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 + 4));                             \
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 + 4));                             \
        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 + 8));                             \
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 + 8));                             \
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 + 8));                             \
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 + 8));                             \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (sink) = vpaddq_f32(s01, s23);                                         \
    } while (0)
    MQV(0, qvg0); MQV(1, qvg1); MQV(2, qvg2);
    MQV(3, qvg3); MQV(4, qvg4); MQV(5, qvg5);
    MQV(6, qvg6); MQV(7, qvg7); MQV(8, qvg8);
    #undef MQV

    float32x4_t qh0, qh1, qh2;
    #define LRU(b, qv, vv, gv, qhv) do {                                       \
        const float32x4_t gc = fused_sigmoid(gv);                              \
        const float32x4_t hc = vfmaq_f32(vv, gc, vld1q_f32(h_ptr + (b) * 4));  \
        vst1q_f32(h_ptr + (b) * 4, hc);                                        \
        (qhv) = vmulq_f32(qv, hc);                                             \
    } while (0)
    LRU(0, qvg0, qvg3, qvg6, qh0);
    LRU(1, qvg1, qvg4, qvg7, qh1);
    LRU(2, qvg2, qvg5, qvg8, qh2);
    #undef LRU

    #define MAC(og, outx) do {                                                 \
        const float* w0 = goW + ((og)*4 + 0) * 12;                             \
        const float* w1 = goW + ((og)*4 + 1) * 12;                             \
        const float* w2 = goW + ((og)*4 + 2) * 12;                             \
        const float* w3 = goW + ((og)*4 + 3) * 12;                             \
        float32x4_t a0 = vmulq_f32(qh0, vld1q_f32(w0 + 0));                    \
        float32x4_t a1 = vmulq_f32(qh0, vld1q_f32(w1 + 0));                    \
        float32x4_t a2 = vmulq_f32(qh0, vld1q_f32(w2 + 0));                    \
        float32x4_t a3 = vmulq_f32(qh0, vld1q_f32(w3 + 0));                    \
        a0 = vfmaq_f32(a0, qh1, vld1q_f32(w0 + 4));                            \
        a1 = vfmaq_f32(a1, qh1, vld1q_f32(w1 + 4));                            \
        a2 = vfmaq_f32(a2, qh1, vld1q_f32(w2 + 4));                            \
        a3 = vfmaq_f32(a3, qh1, vld1q_f32(w3 + 4));                            \
        a0 = vfmaq_f32(a0, qh2, vld1q_f32(w0 + 8));                            \
        a1 = vfmaq_f32(a1, qh2, vld1q_f32(w1 + 8));                            \
        a2 = vfmaq_f32(a2, qh2, vld1q_f32(w2 + 8));                            \
        a3 = vfmaq_f32(a3, qh2, vld1q_f32(w3 + 8));                            \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (outx) = vaddq_f32((outx), vpaddq_f32(s01, s23));                      \
    } while (0)
    MAC(0, out0); MAC(1, out1); MAC(2, out2);
    #undef MAC
}

// s0 L1 = <12, 12>. No input_proj (IN==OUT) → residual is feat_in.
static inline void s0_l1_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_arr,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)
{
    const float32x4_t x0 = vld1q_f32(feat_in + 0);
    const float32x4_t x1 = vld1q_f32(feat_in + 4);
    const float32x4_t x2 = vld1q_f32(feat_in + 8);

    float32x4_t out0 = vdupq_n_f32(0.0f);
    float32x4_t out1 = vdupq_n_f32(0.0f);
    float32x4_t out2 = vdupq_n_f32(0.0f);

    const int base = ev_y * Wl + ev_x;
    constexpr int dx_arr[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    constexpr int dy_arr[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    constexpr int delta_arr[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};

    for (int k = 0; k < 9; ++k) {
        const int patch_idx = base + dy_arr[k] * Wl + dx_arr[k];
        const int pos       = delta_arr[k];
        patch_12_12(x0, x1, x2,
                    qvgIn_arr[pos].data(), goW_arr[pos].data(),
                    H_all + (std::ptrdiff_t)patch_idx * 12,
                    out0, out1, out2);
    }

    out0 = vaddq_f32(out0, x0);
    out1 = vaddq_f32(out1, x1);
    out2 = vaddq_f32(out2, x2);

    alignas(16) float tmp[12];
    vst1q_f32(tmp + 0, out0);
    vst1q_f32(tmp + 4, out1);
    vst1q_f32(tmp + 8, out2);
    openeva::prim::layernorm_ct<12>(tmp, ln_gamma, ln_beta);
    std::memcpy(feat_out, tmp, sizeof(float) * 12);
}

// ============================================================================
// One patch of <12, 24>: matvec_qvg<12,72> + lru_step<24> + matvec_accum<24,24>
// Used by s1 L0.
// ============================================================================
__attribute__((always_inline))
static inline void patch_12_24(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float* __restrict__ qvgW,
    const float* __restrict__ goW,
    float*       __restrict__ h_ptr,
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2,
    float32x4_t& out3, float32x4_t& out4, float32x4_t& out5)
{
    float32x4_t qvg0, qvg1, qvg2, qvg3, qvg4, qvg5;
    float32x4_t qvg6, qvg7, qvg8, qvg9, qvg10, qvg11;
    float32x4_t qvg12, qvg13, qvg14, qvg15, qvg16, qvg17;

    #define MQV(og, sink) do {                                                 \
        const float* w0 = qvgW + ((og)*4 + 0) * 12;                            \
        const float* w1 = qvgW + ((og)*4 + 1) * 12;                            \
        const float* w2 = qvgW + ((og)*4 + 2) * 12;                            \
        const float* w3 = qvgW + ((og)*4 + 3) * 12;                            \
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0 + 0));                     \
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1 + 0));                     \
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2 + 0));                     \
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3 + 0));                     \
        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 + 4));                             \
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 + 4));                             \
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 + 4));                             \
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 + 4));                             \
        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 + 8));                             \
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 + 8));                             \
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 + 8));                             \
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 + 8));                             \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (sink) = vpaddq_f32(s01, s23);                                         \
    } while (0)
    MQV(0,  qvg0);  MQV(1,  qvg1);  MQV(2,  qvg2);
    MQV(3,  qvg3);  MQV(4,  qvg4);  MQV(5,  qvg5);
    MQV(6,  qvg6);  MQV(7,  qvg7);  MQV(8,  qvg8);
    MQV(9,  qvg9);  MQV(10, qvg10); MQV(11, qvg11);
    MQV(12, qvg12); MQV(13, qvg13); MQV(14, qvg14);
    MQV(15, qvg15); MQV(16, qvg16); MQV(17, qvg17);
    #undef MQV

    float32x4_t qh0, qh1, qh2, qh3, qh4, qh5;
    #define LRU(b, qv, vv, gv, qhv) do {                                       \
        const float32x4_t gc = fused_sigmoid(gv);                              \
        const float32x4_t hc = vfmaq_f32(vv, gc, vld1q_f32(h_ptr + (b) * 4));  \
        vst1q_f32(h_ptr + (b) * 4, hc);                                        \
        (qhv) = vmulq_f32(qv, hc);                                             \
    } while (0)
    LRU(0, qvg0,  qvg6,  qvg12, qh0);
    LRU(1, qvg1,  qvg7,  qvg13, qh1);
    LRU(2, qvg2,  qvg8,  qvg14, qh2);
    LRU(3, qvg3,  qvg9,  qvg15, qh3);
    LRU(4, qvg4,  qvg10, qvg16, qh4);
    LRU(5, qvg5,  qvg11, qvg17, qh5);
    #undef LRU

    #define MAC(og, outx) do {                                                 \
        const float* w0 = goW + ((og)*4 + 0) * 24;                             \
        const float* w1 = goW + ((og)*4 + 1) * 24;                             \
        const float* w2 = goW + ((og)*4 + 2) * 24;                             \
        const float* w3 = goW + ((og)*4 + 3) * 24;                             \
        float32x4_t a0 = vmulq_f32(qh0, vld1q_f32(w0 +  0));                   \
        float32x4_t a1 = vmulq_f32(qh0, vld1q_f32(w1 +  0));                   \
        float32x4_t a2 = vmulq_f32(qh0, vld1q_f32(w2 +  0));                   \
        float32x4_t a3 = vmulq_f32(qh0, vld1q_f32(w3 +  0));                   \
        a0 = vfmaq_f32(a0, qh1, vld1q_f32(w0 +  4));                           \
        a1 = vfmaq_f32(a1, qh1, vld1q_f32(w1 +  4));                           \
        a2 = vfmaq_f32(a2, qh1, vld1q_f32(w2 +  4));                           \
        a3 = vfmaq_f32(a3, qh1, vld1q_f32(w3 +  4));                           \
        a0 = vfmaq_f32(a0, qh2, vld1q_f32(w0 +  8));                           \
        a1 = vfmaq_f32(a1, qh2, vld1q_f32(w1 +  8));                           \
        a2 = vfmaq_f32(a2, qh2, vld1q_f32(w2 +  8));                           \
        a3 = vfmaq_f32(a3, qh2, vld1q_f32(w3 +  8));                           \
        a0 = vfmaq_f32(a0, qh3, vld1q_f32(w0 + 12));                           \
        a1 = vfmaq_f32(a1, qh3, vld1q_f32(w1 + 12));                           \
        a2 = vfmaq_f32(a2, qh3, vld1q_f32(w2 + 12));                           \
        a3 = vfmaq_f32(a3, qh3, vld1q_f32(w3 + 12));                           \
        a0 = vfmaq_f32(a0, qh4, vld1q_f32(w0 + 16));                           \
        a1 = vfmaq_f32(a1, qh4, vld1q_f32(w1 + 16));                           \
        a2 = vfmaq_f32(a2, qh4, vld1q_f32(w2 + 16));                           \
        a3 = vfmaq_f32(a3, qh4, vld1q_f32(w3 + 16));                           \
        a0 = vfmaq_f32(a0, qh5, vld1q_f32(w0 + 20));                           \
        a1 = vfmaq_f32(a1, qh5, vld1q_f32(w1 + 20));                           \
        a2 = vfmaq_f32(a2, qh5, vld1q_f32(w2 + 20));                           \
        a3 = vfmaq_f32(a3, qh5, vld1q_f32(w3 + 20));                           \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (outx) = vaddq_f32((outx), vpaddq_f32(s01, s23));                      \
    } while (0)
    MAC(0, out0); MAC(1, out1); MAC(2, out2);
    MAC(3, out3); MAC(4, out4); MAC(5, out5);
    #undef MAC
}

// s1 L0 = <12, 24>. Has input_proj for residual.
static inline void s1_l0_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,
    const float* __restrict__ input_proj,
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_arr,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)
{
    const float32x4_t x0 = vld1q_f32(feat_in + 0);
    const float32x4_t x1 = vld1q_f32(feat_in + 4);
    const float32x4_t x2 = vld1q_f32(feat_in + 8);

    float32x4_t out0, out1, out2, out3, out4, out5;
    #define RES(og, outx) do {                                                 \
        const float* w0 = input_proj + ((og)*4 + 0) * 12;                      \
        const float* w1 = input_proj + ((og)*4 + 1) * 12;                      \
        const float* w2 = input_proj + ((og)*4 + 2) * 12;                      \
        const float* w3 = input_proj + ((og)*4 + 3) * 12;                      \
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0 + 0));                     \
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1 + 0));                     \
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2 + 0));                     \
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3 + 0));                     \
        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 + 4));                             \
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 + 4));                             \
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 + 4));                             \
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 + 4));                             \
        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 + 8));                             \
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 + 8));                             \
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 + 8));                             \
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 + 8));                             \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (outx) = vpaddq_f32(s01, s23);                                         \
    } while (0)
    RES(0, out0); RES(1, out1); RES(2, out2);
    RES(3, out3); RES(4, out4); RES(5, out5);
    #undef RES

    const int base = ev_y * Wl + ev_x;
    constexpr int dx_arr[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    constexpr int dy_arr[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    constexpr int delta_arr[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
    for (int k = 0; k < 9; ++k) {
        const int patch_idx = base + dy_arr[k] * Wl + dx_arr[k];
        const int pos       = delta_arr[k];
        patch_12_24(x0, x1, x2,
                    qvgIn_arr[pos].data(), goW_arr[pos].data(),
                    H_all + (std::ptrdiff_t)patch_idx * 24,
                    out0, out1, out2, out3, out4, out5);
    }

    alignas(16) float tmp[24];
    vst1q_f32(tmp +  0, out0);
    vst1q_f32(tmp +  4, out1);
    vst1q_f32(tmp +  8, out2);
    vst1q_f32(tmp + 12, out3);
    vst1q_f32(tmp + 16, out4);
    vst1q_f32(tmp + 20, out5);
    openeva::prim::layernorm_ct<24>(tmp, ln_gamma, ln_beta);
    std::memcpy(feat_out, tmp, sizeof(float) * 24);
}

// ============================================================================
// One patch of <24, 24>: matvec_qvg<24,72> + lru_step<24> + matvec_accum<24,24>
// Used by s1 L1.
// ============================================================================
__attribute__((always_inline))
static inline void patch_24_24(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float32x4_t x3, const float32x4_t x4, const float32x4_t x5,
    const float* __restrict__ qvgW,
    const float* __restrict__ goW,
    float*       __restrict__ h_ptr,
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2,
    float32x4_t& out3, float32x4_t& out4, float32x4_t& out5)
{
    float32x4_t qvg0, qvg1, qvg2, qvg3, qvg4, qvg5;
    float32x4_t qvg6, qvg7, qvg8, qvg9, qvg10, qvg11;
    float32x4_t qvg12, qvg13, qvg14, qvg15, qvg16, qvg17;

    #define MQV(og, sink) do {                                                 \
        const float* w0 = qvgW + ((og)*4 + 0) * 24;                            \
        const float* w1 = qvgW + ((og)*4 + 1) * 24;                            \
        const float* w2 = qvgW + ((og)*4 + 2) * 24;                            \
        const float* w3 = qvgW + ((og)*4 + 3) * 24;                            \
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0 +  0));                    \
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1 +  0));                    \
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2 +  0));                    \
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3 +  0));                    \
        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 +  4));                            \
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 +  4));                            \
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 +  4));                            \
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 +  4));                            \
        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 +  8));                            \
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 +  8));                            \
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 +  8));                            \
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 +  8));                            \
        a0 = vfmaq_f32(a0, x3, vld1q_f32(w0 + 12));                            \
        a1 = vfmaq_f32(a1, x3, vld1q_f32(w1 + 12));                            \
        a2 = vfmaq_f32(a2, x3, vld1q_f32(w2 + 12));                            \
        a3 = vfmaq_f32(a3, x3, vld1q_f32(w3 + 12));                            \
        a0 = vfmaq_f32(a0, x4, vld1q_f32(w0 + 16));                            \
        a1 = vfmaq_f32(a1, x4, vld1q_f32(w1 + 16));                            \
        a2 = vfmaq_f32(a2, x4, vld1q_f32(w2 + 16));                            \
        a3 = vfmaq_f32(a3, x4, vld1q_f32(w3 + 16));                            \
        a0 = vfmaq_f32(a0, x5, vld1q_f32(w0 + 20));                            \
        a1 = vfmaq_f32(a1, x5, vld1q_f32(w1 + 20));                            \
        a2 = vfmaq_f32(a2, x5, vld1q_f32(w2 + 20));                            \
        a3 = vfmaq_f32(a3, x5, vld1q_f32(w3 + 20));                            \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (sink) = vpaddq_f32(s01, s23);                                         \
    } while (0)
    MQV(0,  qvg0);  MQV(1,  qvg1);  MQV(2,  qvg2);
    MQV(3,  qvg3);  MQV(4,  qvg4);  MQV(5,  qvg5);
    MQV(6,  qvg6);  MQV(7,  qvg7);  MQV(8,  qvg8);
    MQV(9,  qvg9);  MQV(10, qvg10); MQV(11, qvg11);
    MQV(12, qvg12); MQV(13, qvg13); MQV(14, qvg14);
    MQV(15, qvg15); MQV(16, qvg16); MQV(17, qvg17);
    #undef MQV

    float32x4_t qh0, qh1, qh2, qh3, qh4, qh5;
    #define LRU(b, qv, vv, gv, qhv) do {                                       \
        const float32x4_t gc = fused_sigmoid(gv);                              \
        const float32x4_t hc = vfmaq_f32(vv, gc, vld1q_f32(h_ptr + (b) * 4));  \
        vst1q_f32(h_ptr + (b) * 4, hc);                                        \
        (qhv) = vmulq_f32(qv, hc);                                             \
    } while (0)
    LRU(0, qvg0,  qvg6,  qvg12, qh0);
    LRU(1, qvg1,  qvg7,  qvg13, qh1);
    LRU(2, qvg2,  qvg8,  qvg14, qh2);
    LRU(3, qvg3,  qvg9,  qvg15, qh3);
    LRU(4, qvg4,  qvg10, qvg16, qh4);
    LRU(5, qvg5,  qvg11, qvg17, qh5);
    #undef LRU

    #define MAC(og, outx) do {                                                 \
        const float* w0 = goW + ((og)*4 + 0) * 24;                             \
        const float* w1 = goW + ((og)*4 + 1) * 24;                             \
        const float* w2 = goW + ((og)*4 + 2) * 24;                             \
        const float* w3 = goW + ((og)*4 + 3) * 24;                             \
        float32x4_t a0 = vmulq_f32(qh0, vld1q_f32(w0 +  0));                   \
        float32x4_t a1 = vmulq_f32(qh0, vld1q_f32(w1 +  0));                   \
        float32x4_t a2 = vmulq_f32(qh0, vld1q_f32(w2 +  0));                   \
        float32x4_t a3 = vmulq_f32(qh0, vld1q_f32(w3 +  0));                   \
        a0 = vfmaq_f32(a0, qh1, vld1q_f32(w0 +  4));                           \
        a1 = vfmaq_f32(a1, qh1, vld1q_f32(w1 +  4));                           \
        a2 = vfmaq_f32(a2, qh1, vld1q_f32(w2 +  4));                           \
        a3 = vfmaq_f32(a3, qh1, vld1q_f32(w3 +  4));                           \
        a0 = vfmaq_f32(a0, qh2, vld1q_f32(w0 +  8));                           \
        a1 = vfmaq_f32(a1, qh2, vld1q_f32(w1 +  8));                           \
        a2 = vfmaq_f32(a2, qh2, vld1q_f32(w2 +  8));                           \
        a3 = vfmaq_f32(a3, qh2, vld1q_f32(w3 +  8));                           \
        a0 = vfmaq_f32(a0, qh3, vld1q_f32(w0 + 12));                           \
        a1 = vfmaq_f32(a1, qh3, vld1q_f32(w1 + 12));                           \
        a2 = vfmaq_f32(a2, qh3, vld1q_f32(w2 + 12));                           \
        a3 = vfmaq_f32(a3, qh3, vld1q_f32(w3 + 12));                           \
        a0 = vfmaq_f32(a0, qh4, vld1q_f32(w0 + 16));                           \
        a1 = vfmaq_f32(a1, qh4, vld1q_f32(w1 + 16));                           \
        a2 = vfmaq_f32(a2, qh4, vld1q_f32(w2 + 16));                           \
        a3 = vfmaq_f32(a3, qh4, vld1q_f32(w3 + 16));                           \
        a0 = vfmaq_f32(a0, qh5, vld1q_f32(w0 + 20));                           \
        a1 = vfmaq_f32(a1, qh5, vld1q_f32(w1 + 20));                           \
        a2 = vfmaq_f32(a2, qh5, vld1q_f32(w2 + 20));                           \
        a3 = vfmaq_f32(a3, qh5, vld1q_f32(w3 + 20));                           \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (outx) = vaddq_f32((outx), vpaddq_f32(s01, s23));                      \
    } while (0)
    MAC(0, out0); MAC(1, out1); MAC(2, out2);
    MAC(3, out3); MAC(4, out4); MAC(5, out5);
    #undef MAC
}

// s1 L1 = <24, 24>. No input_proj (IN==OUT) → residual is feat_in.
static inline void s1_l1_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_arr,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)
{
    const float32x4_t x0 = vld1q_f32(feat_in +  0);
    const float32x4_t x1 = vld1q_f32(feat_in +  4);
    const float32x4_t x2 = vld1q_f32(feat_in +  8);
    const float32x4_t x3 = vld1q_f32(feat_in + 12);
    const float32x4_t x4 = vld1q_f32(feat_in + 16);
    const float32x4_t x5 = vld1q_f32(feat_in + 20);

    float32x4_t out0 = vdupq_n_f32(0.0f);
    float32x4_t out1 = vdupq_n_f32(0.0f);
    float32x4_t out2 = vdupq_n_f32(0.0f);
    float32x4_t out3 = vdupq_n_f32(0.0f);
    float32x4_t out4 = vdupq_n_f32(0.0f);
    float32x4_t out5 = vdupq_n_f32(0.0f);

    const int base = ev_y * Wl + ev_x;
    constexpr int dx_arr[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    constexpr int dy_arr[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    constexpr int delta_arr[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};

    for (int k = 0; k < 9; ++k) {
        const int patch_idx = base + dy_arr[k] * Wl + dx_arr[k];
        const int pos       = delta_arr[k];
        patch_24_24(x0, x1, x2, x3, x4, x5,
                    qvgIn_arr[pos].data(), goW_arr[pos].data(),
                    H_all + (std::ptrdiff_t)patch_idx * 24,
                    out0, out1, out2, out3, out4, out5);
    }

    out0 = vaddq_f32(out0, x0);
    out1 = vaddq_f32(out1, x1);
    out2 = vaddq_f32(out2, x2);
    out3 = vaddq_f32(out3, x3);
    out4 = vaddq_f32(out4, x4);
    out5 = vaddq_f32(out5, x5);

    alignas(16) float tmp[24];
    vst1q_f32(tmp +  0, out0);
    vst1q_f32(tmp +  4, out1);
    vst1q_f32(tmp +  8, out2);
    vst1q_f32(tmp + 12, out3);
    vst1q_f32(tmp + 16, out4);
    vst1q_f32(tmp + 20, out5);
    openeva::prim::layernorm_ct<24>(tmp, ln_gamma, ln_beta);
    std::memcpy(feat_out, tmp, sizeof(float) * 24);
}

}  // namespace fused
}  // namespace deploy
#endif  // __ARM_NEON || __aarch64__
