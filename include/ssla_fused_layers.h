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
#include <cmath>
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

// NEON in-place layernorm for DIM=12 / DIM=24, fp32 throughout.
// Vendor's layernorm_ct uses fp64 reduction for bit-equivalent PyTorch CPU
// numerics; we accept ~1e-5 per-call drift in exchange for ~30 ns/call
// savings. Cumulative drift through the recurrence is bounded the same way
// as the NEON sigmoid in ssla_neon_lru.h (cum drift ~ √N × per-call err).
__attribute__((always_inline))
static inline void neon_layernorm_12(float* __restrict__ y,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta) {
    const float32x4_t v0 = vld1q_f32(y + 0);
    const float32x4_t v1 = vld1q_f32(y + 4);
    const float32x4_t v2 = vld1q_f32(y + 8);
    const float32x4_t s  = vaddq_f32(vaddq_f32(v0, v1), v2);
    const float mean = vaddvq_f32(s) * (1.0f / 12.0f);
    const float32x4_t vmean = vdupq_n_f32(mean);
    const float32x4_t c0 = vsubq_f32(v0, vmean);
    const float32x4_t c1 = vsubq_f32(v1, vmean);
    const float32x4_t c2 = vsubq_f32(v2, vmean);
    float32x4_t sq = vmulq_f32(c0, c0);
    sq = vfmaq_f32(sq, c1, c1);
    sq = vfmaq_f32(sq, c2, c2);
    const float var = vaddvq_f32(sq) * (1.0f / 12.0f);
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    const float32x4_t vinv = vdupq_n_f32(inv);
    vst1q_f32(y + 0, vfmaq_f32(vld1q_f32(beta + 0),
                                vmulq_f32(c0, vinv), vld1q_f32(gamma + 0)));
    vst1q_f32(y + 4, vfmaq_f32(vld1q_f32(beta + 4),
                                vmulq_f32(c1, vinv), vld1q_f32(gamma + 4)));
    vst1q_f32(y + 8, vfmaq_f32(vld1q_f32(beta + 8),
                                vmulq_f32(c2, vinv), vld1q_f32(gamma + 8)));
}

__attribute__((always_inline))
static inline void neon_layernorm_24(float* __restrict__ y,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta) {
    const float32x4_t v0 = vld1q_f32(y +  0);
    const float32x4_t v1 = vld1q_f32(y +  4);
    const float32x4_t v2 = vld1q_f32(y +  8);
    const float32x4_t v3 = vld1q_f32(y + 12);
    const float32x4_t v4 = vld1q_f32(y + 16);
    const float32x4_t v5 = vld1q_f32(y + 20);
    const float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)),
                                     vaddq_f32(v4, v5));
    const float mean = vaddvq_f32(s) * (1.0f / 24.0f);
    const float32x4_t vmean = vdupq_n_f32(mean);
    const float32x4_t c0 = vsubq_f32(v0, vmean);
    const float32x4_t c1 = vsubq_f32(v1, vmean);
    const float32x4_t c2 = vsubq_f32(v2, vmean);
    const float32x4_t c3 = vsubq_f32(v3, vmean);
    const float32x4_t c4 = vsubq_f32(v4, vmean);
    const float32x4_t c5 = vsubq_f32(v5, vmean);
    float32x4_t sq = vmulq_f32(c0, c0);
    sq = vfmaq_f32(sq, c1, c1);
    sq = vfmaq_f32(sq, c2, c2);
    sq = vfmaq_f32(sq, c3, c3);
    sq = vfmaq_f32(sq, c4, c4);
    sq = vfmaq_f32(sq, c5, c5);
    const float var = vaddvq_f32(sq) * (1.0f / 24.0f);
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    const float32x4_t vinv = vdupq_n_f32(inv);
    vst1q_f32(y +  0, vfmaq_f32(vld1q_f32(beta +  0),
                                 vmulq_f32(c0, vinv), vld1q_f32(gamma +  0)));
    vst1q_f32(y +  4, vfmaq_f32(vld1q_f32(beta +  4),
                                 vmulq_f32(c1, vinv), vld1q_f32(gamma +  4)));
    vst1q_f32(y +  8, vfmaq_f32(vld1q_f32(beta +  8),
                                 vmulq_f32(c2, vinv), vld1q_f32(gamma +  8)));
    vst1q_f32(y + 12, vfmaq_f32(vld1q_f32(beta + 12),
                                 vmulq_f32(c3, vinv), vld1q_f32(gamma + 12)));
    vst1q_f32(y + 16, vfmaq_f32(vld1q_f32(beta + 16),
                                 vmulq_f32(c4, vinv), vld1q_f32(gamma + 16)));
    vst1q_f32(y + 20, vfmaq_f32(vld1q_f32(beta + 20),
                                 vmulq_f32(c5, vinv), vld1q_f32(gamma + 20)));
}

// ============================================================================
// One patch of <2, 12>: matvec_qvg<2, 36> + lru_step<12> + matvec_accum<12, 12>
// Used by s0 L0. IN=2 → broadcast each input scalar; per-output cost dominated
// by broadcast-fma rather than by inner reduce.
// ============================================================================
__attribute__((always_inline))
static inline void patch_2_12(
    const float32x4_t x0_bcast,   // = vdupq_n_f32(feat_in[0])
    const float32x4_t x1_bcast,   // = vdupq_n_f32(feat_in[1])
    const float* __restrict__ qvgW,         // (36, 2)
    const float* __restrict__ goW,          // (12, 12)
    float*       __restrict__ h_ptr,        // (12,)
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2)
{
    // matvec_qvg<2, 36>: 36 outputs, 9 groups of 4. Per group: load w[4][2]
    // (= 4 rows × 2 cols = 8 floats = 2 vec loads with deinterleave). Use
    // broadcast pattern: y[4] = w[:, 0]*x0 + w[:, 1]*x1.
    //
    // Weight layout is row-major (36, 2). Rows are contiguous. To pull
    // column 0 of 4 consecutive rows: w[og*4+0][0], w[og*4+1][0], …
    // — strided. Use ld2 (interleaved load) to deinterleave columns.
    float32x4_t qvg0, qvg1, qvg2, qvg3, qvg4, qvg5, qvg6, qvg7, qvg8;

    #define MQV(og, sink) do {                                                 \
        const float* w = qvgW + (og) * 4 * 2;                                  \
        const float32x4x2_t W2 = vld2q_f32(w);                                 \
        float32x4_t a = vmulq_f32(W2.val[0], x0_bcast);                        \
        a = vfmaq_f32(a, W2.val[1], x1_bcast);                                 \
        (sink) = a;                                                            \
    } while (0)
    MQV(0, qvg0); MQV(1, qvg1); MQV(2, qvg2);
    MQV(3, qvg3); MQV(4, qvg4); MQV(5, qvg5);
    MQV(6, qvg6); MQV(7, qvg7); MQV(8, qvg8);
    #undef MQV

    // qvg layout (36 outputs row-major): q[0..11], v[0..11], g[0..11]
    // → q = qvg0..qvg2, v = qvg3..qvg5, g = qvg6..qvg8
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

    // matvec_accum<12, 12>: 3 output groups of 4, across-IN axis.
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

// s0 L0 = <2, 12>. Has input_proj (IN != OUT). feat_in is only 2 floats.
// **NUM_POS = 1** for stage-0 L0 (K=1, single-cell update — unlike all the
// later layers which are K=3, 9-patch). Caller passes only qvgIn[0] / goW[0].
static inline void s0_l0_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,        // (2,)
    const float* __restrict__ input_proj,     // (12, 2)
    const float* __restrict__ qvgIn0,         // (36, 2) — qvgIn[0] only
    const float* __restrict__ goW0,           // (12, 12) — goW[0] only
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)        // (12,)
{
    const float32x4_t x0b = vdupq_n_f32(feat_in[0]);
    const float32x4_t x1b = vdupq_n_f32(feat_in[1]);

    // Residual: out = input_proj @ feat_in  (12 outputs from 2 inputs).
    // 3 output groups of 4. ld2 + broadcast pattern.
    float32x4_t out0, out1, out2;
    #define RES(og, outx) do {                                                 \
        const float* w = input_proj + (og) * 4 * 2;                            \
        const float32x4x2_t W2 = vld2q_f32(w);                                 \
        float32x4_t a = vmulq_f32(W2.val[0], x0b);                             \
        a = vfmaq_f32(a, W2.val[1], x1b);                                      \
        (outx) = a;                                                            \
    } while (0)
    RES(0, out0); RES(1, out1); RES(2, out2);
    #undef RES

    // Single patch (num_pos = 1). pos = 0, patch_idx = base = ev_y*Wl + ev_x.
    const int patch_idx = ev_y * Wl + ev_x;
    patch_2_12(x0b, x1b,
               qvgIn0, goW0,
               H_all + (std::ptrdiff_t)patch_idx * 12,
               out0, out1, out2);

    vst1q_f32(feat_out + 0, out0);
    vst1q_f32(feat_out + 4, out1);
    vst1q_f32(feat_out + 8, out2);
    openeva::prim::layernorm_ct<12>(feat_out, ln_gamma, ln_beta);
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

    // Direct write into feat_out (caller-supplied, 16-aligned per
    // std::vector<float> guarantee). LN works in-place — no tmp+memcpy.
    vst1q_f32(feat_out + 0, out0);
    vst1q_f32(feat_out + 4, out1);
    vst1q_f32(feat_out + 8, out2);
    openeva::prim::layernorm_ct<12>(feat_out, ln_gamma, ln_beta);
}

// ============================================================================
// One patch of <12, 24> tile-streaming variant. Mirrors patch_24_24_tiled but
// with IN=12 (3 NEON input vectors instead of 6).
// 6 tiles of 4 OUT channels each. qvgW is (72, 12), goW_T is (24, 24).
// Hidden state width remains 24 → ACCUM_TILE identical to the L1 tiled path.
// ============================================================================
__attribute__((always_inline))
static inline void patch_12_24_tiled(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float* __restrict__ qvgW,    // (72, 12) row-major
    const float* __restrict__ goW_T,   // (24, 24) input-major
    float*       __restrict__ h_ptr,   // (24,)
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2,
    float32x4_t& out3, float32x4_t& out4, float32x4_t& out5)
{
    // matvec for 4 outputs across IN=12 (3 input vectors).
    #define MV4OUT12(rowbase, sink) do {                                       \
        const float* r0 = (rowbase) + 0 * 12;                                  \
        const float* r1 = (rowbase) + 1 * 12;                                  \
        const float* r2 = (rowbase) + 2 * 12;                                  \
        const float* r3 = (rowbase) + 3 * 12;                                  \
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(r0 + 0));                     \
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(r1 + 0));                     \
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(r2 + 0));                     \
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(r3 + 0));                     \
        a0 = vfmaq_f32(a0, x1, vld1q_f32(r0 + 4));                             \
        a1 = vfmaq_f32(a1, x1, vld1q_f32(r1 + 4));                             \
        a2 = vfmaq_f32(a2, x1, vld1q_f32(r2 + 4));                             \
        a3 = vfmaq_f32(a3, x1, vld1q_f32(r3 + 4));                             \
        a0 = vfmaq_f32(a0, x2, vld1q_f32(r0 + 8));                             \
        a1 = vfmaq_f32(a1, x2, vld1q_f32(r1 + 8));                             \
        a2 = vfmaq_f32(a2, x2, vld1q_f32(r2 + 8));                             \
        a3 = vfmaq_f32(a3, x2, vld1q_f32(r3 + 8));                             \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (sink) = vpaddq_f32(s01, s23);                                         \
    } while (0)

    #define ACCUM_TILE_12(tile) do {                                           \
        const float* gT0 = goW_T + ((tile) * 4 + 0) * 24;                      \
        const float* gT1 = goW_T + ((tile) * 4 + 1) * 24;                      \
        const float* gT2 = goW_T + ((tile) * 4 + 2) * 24;                      \
        const float* gT3 = goW_T + ((tile) * 4 + 3) * 24;                      \
        const float32x4_t qhb0 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 0));      \
        const float32x4_t qhb1 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 1));      \
        const float32x4_t qhb2 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 2));      \
        const float32x4_t qhb3 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 3));      \
        out0 = vfmaq_f32(out0, vld1q_f32(gT0 +  0), qhb0);                     \
        out0 = vfmaq_f32(out0, vld1q_f32(gT1 +  0), qhb1);                     \
        out0 = vfmaq_f32(out0, vld1q_f32(gT2 +  0), qhb2);                     \
        out0 = vfmaq_f32(out0, vld1q_f32(gT3 +  0), qhb3);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT0 +  4), qhb0);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT1 +  4), qhb1);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT2 +  4), qhb2);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT3 +  4), qhb3);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT0 +  8), qhb0);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT1 +  8), qhb1);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT2 +  8), qhb2);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT3 +  8), qhb3);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT0 + 12), qhb0);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT1 + 12), qhb1);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT2 + 12), qhb2);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT3 + 12), qhb3);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT0 + 16), qhb0);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT1 + 16), qhb1);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT2 + 16), qhb2);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT3 + 16), qhb3);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT0 + 20), qhb0);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT1 + 20), qhb1);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT2 + 20), qhb2);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT3 + 20), qhb3);                     \
    } while (0)

    #define DO_TILE_12(tile) do {                                              \
        float32x4_t q_tile, v_tile, g_tile;                                    \
        MV4OUT12(qvgW + ((tile) * 4) * 12,           q_tile);                  \
        MV4OUT12(qvgW + (24 + (tile) * 4) * 12,      v_tile);                  \
        MV4OUT12(qvgW + (48 + (tile) * 4) * 12,      g_tile);                  \
        const float32x4_t gc = fused_sigmoid(g_tile);                          \
        const float32x4_t hc = vfmaq_f32(v_tile, gc,                           \
                                          vld1q_f32(h_ptr + (tile) * 4));      \
        vst1q_f32(h_ptr + (tile) * 4, hc);                                     \
        const float32x4_t qh_tile = vmulq_f32(q_tile, hc);                     \
        ACCUM_TILE_12(tile);                                                   \
    } while (0)

    DO_TILE_12(0);
    DO_TILE_12(1);
    DO_TILE_12(2);
    DO_TILE_12(3);
    DO_TILE_12(4);
    DO_TILE_12(5);

    #undef DO_TILE_12
    #undef ACCUM_TILE_12
    #undef MV4OUT12
}

// ============================================================================
// s1 L0 tile-streaming interior path. Same structure as s1_l0_interior but
// calls patch_12_24_tiled (consuming goW_T) instead of patch_12_24.
// ============================================================================
__attribute__((always_inline))
static inline void s1_l0_interior_tiled(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,
    const float* __restrict__ input_proj,
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_T_arr,
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
        patch_12_24_tiled(x0, x1, x2,
                          qvgIn_arr[pos].data(), goW_T_arr[pos].data(),
                          H_all + (std::ptrdiff_t)patch_idx * 24,
                          out0, out1, out2, out3, out4, out5);
    }

    vst1q_f32(feat_out +  0, out0);
    vst1q_f32(feat_out +  4, out1);
    vst1q_f32(feat_out +  8, out2);
    vst1q_f32(feat_out + 12, out3);
    vst1q_f32(feat_out + 16, out4);
    vst1q_f32(feat_out + 20, out5);
    openeva::prim::layernorm_ct<24>(feat_out, ln_gamma, ln_beta);
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

    // LRU two-phase (see patch_24_24 comment).
    const float32x4_t gc0 = fused_sigmoid(qvg12);
    const float32x4_t gc1 = fused_sigmoid(qvg13);
    const float32x4_t gc2 = fused_sigmoid(qvg14);
    const float32x4_t gc3 = fused_sigmoid(qvg15);
    const float32x4_t gc4 = fused_sigmoid(qvg16);
    const float32x4_t gc5 = fused_sigmoid(qvg17);

    const float32x4_t hc0 = vfmaq_f32(qvg6,  gc0, vld1q_f32(h_ptr +  0));
    const float32x4_t hc1 = vfmaq_f32(qvg7,  gc1, vld1q_f32(h_ptr +  4));
    const float32x4_t hc2 = vfmaq_f32(qvg8,  gc2, vld1q_f32(h_ptr +  8));
    const float32x4_t hc3 = vfmaq_f32(qvg9,  gc3, vld1q_f32(h_ptr + 12));
    const float32x4_t hc4 = vfmaq_f32(qvg10, gc4, vld1q_f32(h_ptr + 16));
    const float32x4_t hc5 = vfmaq_f32(qvg11, gc5, vld1q_f32(h_ptr + 20));
    vst1q_f32(h_ptr +  0, hc0);
    vst1q_f32(h_ptr +  4, hc1);
    vst1q_f32(h_ptr +  8, hc2);
    vst1q_f32(h_ptr + 12, hc3);
    vst1q_f32(h_ptr + 16, hc4);
    vst1q_f32(h_ptr + 20, hc5);
    const float32x4_t qh0 = vmulq_f32(qvg0, hc0);
    const float32x4_t qh1 = vmulq_f32(qvg1, hc1);
    const float32x4_t qh2 = vmulq_f32(qvg2, hc2);
    const float32x4_t qh3 = vmulq_f32(qvg3, hc3);
    const float32x4_t qh4 = vmulq_f32(qvg4, hc4);
    const float32x4_t qh5 = vmulq_f32(qvg5, hc5);

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

    vst1q_f32(feat_out +  0, out0);
    vst1q_f32(feat_out +  4, out1);
    vst1q_f32(feat_out +  8, out2);
    vst1q_f32(feat_out + 12, out3);
    vst1q_f32(feat_out + 16, out4);
    vst1q_f32(feat_out + 20, out5);
    openeva::prim::layernorm_ct<24>(feat_out, ln_gamma, ln_beta);
}

// ============================================================================
// One patch of <24, 24> tile-streaming variant.
// 6 tiles of 4 OUT channels each. Per tile:
//   1) compute q_tile, v_tile, g_tile (4 channels each) via 4-out ILP
//   2) lru_step on this tile → qh_tile
//   3) matvec_accum partial: out[0..23] += goW_T[tile*4..+4, :] × qh_tile
// Each tile holds only ~12 live NEON regs (qvg of one tile + qh + temps);
// the 6 persistent x[6] + 6 out[6] across tiles totals ~24 regs.
// vs. all-in-once version's ~36 regs → no spill.
//
// `goW_T` (input-major transpose of goW) lets the matvec_accum tile use
// contiguous loads. Pre-computed at load_layer().
// ============================================================================
__attribute__((always_inline))
static inline void patch_24_24_tiled(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float32x4_t x3, const float32x4_t x4, const float32x4_t x5,
    const float* __restrict__ qvgW,    // (72, 24) row-major
    const float* __restrict__ goW_T,   // (24, 24) input-major
    float*       __restrict__ h_ptr,   // (24,)
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2,
    float32x4_t& out3, float32x4_t& out4, float32x4_t& out5)
{
    // matvec for 4 outputs across IN=24 (6 input vectors).
    #define MV4OUT24(rowbase, sink) do {                                       \
        const float* r0 = (rowbase) + 0 * 24;                                  \
        const float* r1 = (rowbase) + 1 * 24;                                  \
        const float* r2 = (rowbase) + 2 * 24;                                  \
        const float* r3 = (rowbase) + 3 * 24;                                  \
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(r0 +  0));                    \
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(r1 +  0));                    \
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(r2 +  0));                    \
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(r3 +  0));                    \
        a0 = vfmaq_f32(a0, x1, vld1q_f32(r0 +  4));                            \
        a1 = vfmaq_f32(a1, x1, vld1q_f32(r1 +  4));                            \
        a2 = vfmaq_f32(a2, x1, vld1q_f32(r2 +  4));                            \
        a3 = vfmaq_f32(a3, x1, vld1q_f32(r3 +  4));                            \
        a0 = vfmaq_f32(a0, x2, vld1q_f32(r0 +  8));                            \
        a1 = vfmaq_f32(a1, x2, vld1q_f32(r1 +  8));                            \
        a2 = vfmaq_f32(a2, x2, vld1q_f32(r2 +  8));                            \
        a3 = vfmaq_f32(a3, x2, vld1q_f32(r3 +  8));                            \
        a0 = vfmaq_f32(a0, x3, vld1q_f32(r0 + 12));                            \
        a1 = vfmaq_f32(a1, x3, vld1q_f32(r1 + 12));                            \
        a2 = vfmaq_f32(a2, x3, vld1q_f32(r2 + 12));                            \
        a3 = vfmaq_f32(a3, x3, vld1q_f32(r3 + 12));                            \
        a0 = vfmaq_f32(a0, x4, vld1q_f32(r0 + 16));                            \
        a1 = vfmaq_f32(a1, x4, vld1q_f32(r1 + 16));                            \
        a2 = vfmaq_f32(a2, x4, vld1q_f32(r2 + 16));                            \
        a3 = vfmaq_f32(a3, x4, vld1q_f32(r3 + 16));                            \
        a0 = vfmaq_f32(a0, x5, vld1q_f32(r0 + 20));                            \
        a1 = vfmaq_f32(a1, x5, vld1q_f32(r1 + 20));                            \
        a2 = vfmaq_f32(a2, x5, vld1q_f32(r2 + 20));                            \
        a3 = vfmaq_f32(a3, x5, vld1q_f32(r3 + 20));                            \
        const float32x4_t s01 = vpaddq_f32(a0, a1);                            \
        const float32x4_t s23 = vpaddq_f32(a2, a3);                            \
        (sink) = vpaddq_f32(s01, s23);                                         \
    } while (0)

    #define ACCUM_TILE(tile) do {                                              \
        const float* gT0 = goW_T + ((tile) * 4 + 0) * 24;                      \
        const float* gT1 = goW_T + ((tile) * 4 + 1) * 24;                      \
        const float* gT2 = goW_T + ((tile) * 4 + 2) * 24;                      \
        const float* gT3 = goW_T + ((tile) * 4 + 3) * 24;                      \
        const float32x4_t qhb0 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 0));      \
        const float32x4_t qhb1 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 1));      \
        const float32x4_t qhb2 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 2));      \
        const float32x4_t qhb3 = vdupq_n_f32(vgetq_lane_f32(qh_tile, 3));      \
        out0 = vfmaq_f32(out0, vld1q_f32(gT0 +  0), qhb0);                     \
        out0 = vfmaq_f32(out0, vld1q_f32(gT1 +  0), qhb1);                     \
        out0 = vfmaq_f32(out0, vld1q_f32(gT2 +  0), qhb2);                     \
        out0 = vfmaq_f32(out0, vld1q_f32(gT3 +  0), qhb3);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT0 +  4), qhb0);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT1 +  4), qhb1);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT2 +  4), qhb2);                     \
        out1 = vfmaq_f32(out1, vld1q_f32(gT3 +  4), qhb3);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT0 +  8), qhb0);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT1 +  8), qhb1);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT2 +  8), qhb2);                     \
        out2 = vfmaq_f32(out2, vld1q_f32(gT3 +  8), qhb3);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT0 + 12), qhb0);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT1 + 12), qhb1);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT2 + 12), qhb2);                     \
        out3 = vfmaq_f32(out3, vld1q_f32(gT3 + 12), qhb3);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT0 + 16), qhb0);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT1 + 16), qhb1);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT2 + 16), qhb2);                     \
        out4 = vfmaq_f32(out4, vld1q_f32(gT3 + 16), qhb3);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT0 + 20), qhb0);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT1 + 20), qhb1);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT2 + 20), qhb2);                     \
        out5 = vfmaq_f32(out5, vld1q_f32(gT3 + 20), qhb3);                     \
    } while (0)

    // NEON tile-streaming. 6 tiles, each 4 channels.
    #define DO_TILE(tile) do {                                                 \
        float32x4_t q_tile, v_tile, g_tile;                                    \
        MV4OUT24(qvgW + ((tile) * 4) * 24,           q_tile);                  \
        MV4OUT24(qvgW + (24 + (tile) * 4) * 24,      v_tile);                  \
        MV4OUT24(qvgW + (48 + (tile) * 4) * 24,      g_tile);                  \
        const float32x4_t gc = fused_sigmoid(g_tile);                          \
        const float32x4_t hc = vfmaq_f32(v_tile, gc,                           \
                                          vld1q_f32(h_ptr + (tile) * 4));      \
        vst1q_f32(h_ptr + (tile) * 4, hc);                                     \
        const float32x4_t qh_tile = vmulq_f32(q_tile, hc);                     \
        ACCUM_TILE(tile);                                                      \
    } while (0)

    DO_TILE(0);
    DO_TILE(1);
    DO_TILE(2);
    DO_TILE(3);
    DO_TILE(4);
    DO_TILE(5);

    #undef DO_TILE
    #undef ACCUM_TILE
    #undef MV4OUT24
}

// ============================================================================
// Original (not-tiled) patch_24_24 kept below for back-compat / boundary path.
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

    // LRU two-phase: all 6 sigmoids issued together (independent vdiv chains
    // pipeline on A78AE's FP pipes), then fmadd+store+mul as a second pass.
    // Forces compiler to schedule the 6 sigmoid latencies in parallel rather
    // than chunk-by-chunk sequential.
    const float32x4_t gc0 = fused_sigmoid(qvg12);
    const float32x4_t gc1 = fused_sigmoid(qvg13);
    const float32x4_t gc2 = fused_sigmoid(qvg14);
    const float32x4_t gc3 = fused_sigmoid(qvg15);
    const float32x4_t gc4 = fused_sigmoid(qvg16);
    const float32x4_t gc5 = fused_sigmoid(qvg17);

    const float32x4_t hc0 = vfmaq_f32(qvg6,  gc0, vld1q_f32(h_ptr +  0));
    const float32x4_t hc1 = vfmaq_f32(qvg7,  gc1, vld1q_f32(h_ptr +  4));
    const float32x4_t hc2 = vfmaq_f32(qvg8,  gc2, vld1q_f32(h_ptr +  8));
    const float32x4_t hc3 = vfmaq_f32(qvg9,  gc3, vld1q_f32(h_ptr + 12));
    const float32x4_t hc4 = vfmaq_f32(qvg10, gc4, vld1q_f32(h_ptr + 16));
    const float32x4_t hc5 = vfmaq_f32(qvg11, gc5, vld1q_f32(h_ptr + 20));
    vst1q_f32(h_ptr +  0, hc0);
    vst1q_f32(h_ptr +  4, hc1);
    vst1q_f32(h_ptr +  8, hc2);
    vst1q_f32(h_ptr + 12, hc3);
    vst1q_f32(h_ptr + 16, hc4);
    vst1q_f32(h_ptr + 20, hc5);
    const float32x4_t qh0 = vmulq_f32(qvg0, hc0);
    const float32x4_t qh1 = vmulq_f32(qvg1, hc1);
    const float32x4_t qh2 = vmulq_f32(qvg2, hc2);
    const float32x4_t qh3 = vmulq_f32(qvg3, hc3);
    const float32x4_t qh4 = vmulq_f32(qvg4, hc4);
    const float32x4_t qh5 = vmulq_f32(qvg5, hc5);

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

    vst1q_f32(feat_out +  0, out0);
    vst1q_f32(feat_out +  4, out1);
    vst1q_f32(feat_out +  8, out2);
    vst1q_f32(feat_out + 12, out3);
    vst1q_f32(feat_out + 16, out4);
    vst1q_f32(feat_out + 20, out5);
    openeva::prim::layernorm_ct<24>(feat_out, ln_gamma, ln_beta);
}

// ============================================================================
// Tile-streaming s1 L1. Takes goW_T (pre-transposed) instead of goW.
// ============================================================================
static inline void s1_l1_interior_tiled(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_T_arr,
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
        patch_24_24_tiled(x0, x1, x2, x3, x4, x5,
                          qvgIn_arr[pos].data(), goW_T_arr[pos].data(),
                          H_all + (std::ptrdiff_t)patch_idx * 24,
                          out0, out1, out2, out3, out4, out5);
    }

    out0 = vaddq_f32(out0, x0);
    out1 = vaddq_f32(out1, x1);
    out2 = vaddq_f32(out2, x2);
    out3 = vaddq_f32(out3, x3);
    out4 = vaddq_f32(out4, x4);
    out5 = vaddq_f32(out5, x5);

    vst1q_f32(feat_out +  0, out0);
    vst1q_f32(feat_out +  4, out1);
    vst1q_f32(feat_out +  8, out2);
    vst1q_f32(feat_out + 12, out3);
    vst1q_f32(feat_out + 16, out4);
    vst1q_f32(feat_out + 20, out5);
    openeva::prim::layernorm_ct<24>(feat_out, ln_gamma, ln_beta);
}

}  // namespace fused
}  // namespace deploy
#endif  // __ARM_NEON || __aarch64__
