// tests/bench_fused.cpp — expert proposal test: fused, hand-NEON, fully
// unrolled, interior-only kernel for stage-1 L1 <24, 24> vs current generic
// path.
//
// Question this bench answers (falsifiable):
//   Does a hand-fused <24, 24> interior kernel beat the current generic
//   layer_forward_ct<24, 24> by ≥ 5% per call, on this hardware, with
//   numerics preserved (max|Δ| < 1e-3 vs scalar reference)?
//
// Expert hypothesis (`include/ssla_specialized_*.h` not authored yet — the
// fused kernel is inline below to keep the test self-contained):
//   - removing function-call boundaries between matvec_ct<24,72>,
//     lru_step<24>, matvec_accum_ct<24,24> should keep qvg/qh in NEON
//     registers across the three phases (no scratch-buffer round-trips)
//   - manually unrolling the 9 interior patches should improve scheduling
//   - skipping add_macs / scratch_*[stage].data() accesses cuts a few ns/call
//
// We run BOTH the current production path (`pipe.layer_forward(3, ...)`) and
// the fused kernel below, on the same SslaSPipeline weights and hidden state
// (cloned), at random interior coords. Compare ns/call and numeric drift.

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ssla_kernels.h"

namespace {

// NEON sigmoid (tanh-Padé[3/2]) — same impl as the production ssla_neon_lru.h
// so numeric comparison is apples-to-apples against the running lru_step<24>.
static inline float32x4_t sig_tanh_pade(float32x4_t x) {
    const float32x4_t hi   = vdupq_n_f32( 6.0f);
    const float32x4_t lo   = vdupq_n_f32(-6.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t c27  = vdupq_n_f32(27.0f);
    const float32x4_t c9   = vdupq_n_f32(9.0f);
    x = vminq_f32(vmaxq_f32(x, lo), hi);
    const float32x4_t t  = vmulq_f32(x, half);
    const float32x4_t t2 = vmulq_f32(t, t);
    const float32x4_t num = vmulq_f32(t, vaddq_f32(c27, t2));
    const float32x4_t den = vfmaq_f32(c27, c9, t2);
    return vfmaq_f32(half, half, vdivq_f32(num, den));
}

// One patch of <24, 24>: matvec_qvg → lru_step → matvec_accum, fully inline.
// qvg (72 fp32) and qh (24 fp32) live in NEON registers across the three
// phases — no scratch buffer.
__attribute__((always_inline))
static inline void patch_24_24(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float32x4_t x3, const float32x4_t x4, const float32x4_t x5,
    const float* __restrict__ qvgW,         // (72, 24) = 1728 fp32
    const float* __restrict__ goW,          // (24, 24) =  576 fp32
    float*       __restrict__ h_ptr,        // (24,)
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2,
    float32x4_t& out3, float32x4_t& out4, float32x4_t& out5)
{
    // matvec_qvg: 72 outputs, 4-out ILP. 18 output groups of 4.
    // qvg layout: q = qvg0..5, v = qvg6..11, g = qvg12..17 (each fp32x4).
    // Use 18 NAMED variables so GCC sees them as candidates for register
    // allocation rather than as a stack array.
    float32x4_t qvg0, qvg1, qvg2, qvg3, qvg4, qvg5;
    float32x4_t qvg6, qvg7, qvg8, qvg9, qvg10, qvg11;
    float32x4_t qvg12, qvg13, qvg14, qvg15, qvg16, qvg17;

    #define MATVEC_QVG_GROUP(og, sink_var) do {                                \
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
        (sink_var) = vpaddq_f32(s01, s23);                                     \
    } while (0)

    MATVEC_QVG_GROUP(0,  qvg0);  MATVEC_QVG_GROUP(1,  qvg1);  MATVEC_QVG_GROUP(2,  qvg2);
    MATVEC_QVG_GROUP(3,  qvg3);  MATVEC_QVG_GROUP(4,  qvg4);  MATVEC_QVG_GROUP(5,  qvg5);
    MATVEC_QVG_GROUP(6,  qvg6);  MATVEC_QVG_GROUP(7,  qvg7);  MATVEC_QVG_GROUP(8,  qvg8);
    MATVEC_QVG_GROUP(9,  qvg9);  MATVEC_QVG_GROUP(10, qvg10); MATVEC_QVG_GROUP(11, qvg11);
    MATVEC_QVG_GROUP(12, qvg12); MATVEC_QVG_GROUP(13, qvg13); MATVEC_QVG_GROUP(14, qvg14);
    MATVEC_QVG_GROUP(15, qvg15); MATVEC_QVG_GROUP(16, qvg16); MATVEC_QVG_GROUP(17, qvg17);
    #undef MATVEC_QVG_GROUP

    // lru_step manually unrolled (6 chunks of 4 elements). Each chunk uses
    // q, v, g from named vars (no array indirection). 6 NAMED qh vars too.
    #define LRU_CHUNK(b, qv, vv, gv, qhv) do {                                 \
        const float32x4_t gc = sig_tanh_pade(gv);                              \
        const float32x4_t hc = vfmaq_f32(vv, gc, vld1q_f32(h_ptr + (b) * 4));  \
        vst1q_f32(h_ptr + (b) * 4, hc);                                        \
        (qhv) = vmulq_f32(qv, hc);                                             \
    } while (0)

    float32x4_t qh0, qh1, qh2, qh3, qh4, qh5;
    LRU_CHUNK(0, qvg0,  qvg6,  qvg12, qh0);
    LRU_CHUNK(1, qvg1,  qvg7,  qvg13, qh1);
    LRU_CHUNK(2, qvg2,  qvg8,  qvg14, qh2);
    LRU_CHUNK(3, qvg3,  qvg9,  qvg15, qh3);
    LRU_CHUNK(4, qvg4,  qvg10, qvg16, qh4);
    LRU_CHUNK(5, qvg5,  qvg11, qvg17, qh5);
    #undef LRU_CHUNK

    // matvec_accum: feat_out += goW @ qh.   24 outputs, 4-out ILP.
    // qh0..qh5 in named registers; out0..out5 also named.
    #define MATVEC_ACCUM_GROUP(og, outx) do {                                  \
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

    MATVEC_ACCUM_GROUP(0, out0);
    MATVEC_ACCUM_GROUP(1, out1);
    MATVEC_ACCUM_GROUP(2, out2);
    MATVEC_ACCUM_GROUP(3, out3);
    MATVEC_ACCUM_GROUP(4, out4);
    MATVEC_ACCUM_GROUP(5, out5);
    #undef MATVEC_ACCUM_GROUP
}

// Specialized <24, 24> interior layer kernel. Assumes ev_x, ev_y interior
// (so all 9 patches are in bounds). No bias / input_proj path (s1 L1 has
// dims==dims so input_proj is empty). Computes:
//   - 9 patches of process_patch fused into one routine
//   - feat_out += residual (= feat_in)
//   - layernorm(feat_out, gamma, beta)  — kept calling the vendor primary so
//     fp64 reduction numerics match exactly (the vendor LN dominates the
//     residual scalar fmadds in disassembly; we leave it untouched per
//     expert's "do not spend time on layernorm" guidance).
static inline void layer_forward_s1_l1_fused_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_arr,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)
{
    // Load feat_in into 6 NEON regs (held across all 9 patches).
    const float32x4_t x0 = vld1q_f32(feat_in +  0);
    const float32x4_t x1 = vld1q_f32(feat_in +  4);
    const float32x4_t x2 = vld1q_f32(feat_in +  8);
    const float32x4_t x3 = vld1q_f32(feat_in + 12);
    const float32x4_t x4 = vld1q_f32(feat_in + 16);
    const float32x4_t x5 = vld1q_f32(feat_in + 20);

    // Accumulator: 6 NEON regs (feat_out across all 9 patch contributions).
    float32x4_t out0 = vdupq_n_f32(0.0f);
    float32x4_t out1 = vdupq_n_f32(0.0f);
    float32x4_t out2 = vdupq_n_f32(0.0f);
    float32x4_t out3 = vdupq_n_f32(0.0f);
    float32x4_t out4 = vdupq_n_f32(0.0f);
    float32x4_t out5 = vdupq_n_f32(0.0f);

    const int base = ev_y * Wl + ev_x;
    // Patch order matches generic layer_forward_ct:
    //   delta=8 at (-1,-1),  delta=7 (-1, 0), ..., delta=0 (+1, +1)
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

    // Residual + LN. Residual is feat_in (no input_proj for <24, 24>).
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

// ============================================================================
// Fused <12, 12> (s0 L1) — smaller register pressure than <24, 24>.
// ============================================================================
__attribute__((always_inline))
static inline void patch_12_12(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float* __restrict__ qvgW,         // (36, 12)
    const float* __restrict__ goW,          // (12, 12)
    float*       __restrict__ h_ptr,        // (12,)
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2)
{
    float32x4_t qvg0, qvg1, qvg2;     // q[0..11]
    float32x4_t qvg3, qvg4, qvg5;     // v[0..11]
    float32x4_t qvg6, qvg7, qvg8;     // g[0..11]

    #define MQV12(og, sink) do {                                               \
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

    MQV12(0, qvg0); MQV12(1, qvg1); MQV12(2, qvg2);
    MQV12(3, qvg3); MQV12(4, qvg4); MQV12(5, qvg5);
    MQV12(6, qvg6); MQV12(7, qvg7); MQV12(8, qvg8);
    #undef MQV12

    float32x4_t qh0, qh1, qh2;
    #define LRU12(b, qv, vv, gv, qhv) do {                                     \
        const float32x4_t gc = sig_tanh_pade(gv);                              \
        const float32x4_t hc = vfmaq_f32(vv, gc, vld1q_f32(h_ptr + (b) * 4));  \
        vst1q_f32(h_ptr + (b) * 4, hc);                                        \
        (qhv) = vmulq_f32(qv, hc);                                             \
    } while (0)
    LRU12(0, qvg0, qvg3, qvg6, qh0);
    LRU12(1, qvg1, qvg4, qvg7, qh1);
    LRU12(2, qvg2, qvg5, qvg8, qh2);
    #undef LRU12

    #define MAC12(og, outx) do {                                               \
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
    MAC12(0, out0); MAC12(1, out1); MAC12(2, out2);
    #undef MAC12
}

static inline void layer_forward_s0_l1_fused_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,        // (12,)
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_arr,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)        // (12,)
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
// Fused <12, 24> (s1 L0) — input_proj path for residual (in_dim != out_dim).
// ============================================================================
__attribute__((always_inline))
static inline void patch_12_24(
    const float32x4_t x0, const float32x4_t x1, const float32x4_t x2,
    const float* __restrict__ qvgW,         // (72, 12)
    const float* __restrict__ goW,          // (24, 24)
    float*       __restrict__ h_ptr,        // (24,)
    float32x4_t& out0, float32x4_t& out1, float32x4_t& out2,
    float32x4_t& out3, float32x4_t& out4, float32x4_t& out5)
{
    float32x4_t qvg0, qvg1, qvg2, qvg3, qvg4, qvg5;
    float32x4_t qvg6, qvg7, qvg8, qvg9, qvg10, qvg11;
    float32x4_t qvg12, qvg13, qvg14, qvg15, qvg16, qvg17;

    #define MQV12_72(og, sink) do {                                            \
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

    MQV12_72(0,  qvg0);  MQV12_72(1,  qvg1);  MQV12_72(2,  qvg2);
    MQV12_72(3,  qvg3);  MQV12_72(4,  qvg4);  MQV12_72(5,  qvg5);
    MQV12_72(6,  qvg6);  MQV12_72(7,  qvg7);  MQV12_72(8,  qvg8);
    MQV12_72(9,  qvg9);  MQV12_72(10, qvg10); MQV12_72(11, qvg11);
    MQV12_72(12, qvg12); MQV12_72(13, qvg13); MQV12_72(14, qvg14);
    MQV12_72(15, qvg15); MQV12_72(16, qvg16); MQV12_72(17, qvg17);
    #undef MQV12_72

    float32x4_t qh0, qh1, qh2, qh3, qh4, qh5;
    #define LRU24(b, qv, vv, gv, qhv) do {                                     \
        const float32x4_t gc = sig_tanh_pade(gv);                              \
        const float32x4_t hc = vfmaq_f32(vv, gc, vld1q_f32(h_ptr + (b) * 4));  \
        vst1q_f32(h_ptr + (b) * 4, hc);                                        \
        (qhv) = vmulq_f32(qv, hc);                                             \
    } while (0)
    LRU24(0, qvg0,  qvg6,  qvg12, qh0);
    LRU24(1, qvg1,  qvg7,  qvg13, qh1);
    LRU24(2, qvg2,  qvg8,  qvg14, qh2);
    LRU24(3, qvg3,  qvg9,  qvg15, qh3);
    LRU24(4, qvg4,  qvg10, qvg16, qh4);
    LRU24(5, qvg5,  qvg11, qvg17, qh5);
    #undef LRU24

    #define MAC24_24(og, outx) do {                                            \
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
    MAC24_24(0, out0); MAC24_24(1, out1); MAC24_24(2, out2);
    MAC24_24(3, out3); MAC24_24(4, out4); MAC24_24(5, out5);
    #undef MAC24_24
}

static inline void layer_forward_s1_l0_fused_interior(
    int ev_x, int ev_y, int Wl,
    const float* __restrict__ feat_in,        // (12,)
    const float* __restrict__ input_proj,     // (24, 12) — for residual
    const std::array<std::vector<float>, 9>& qvgIn_arr,
    const std::array<std::vector<float>, 9>& goW_arr,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ feat_out)        // (24,)
{
    const float32x4_t x0 = vld1q_f32(feat_in + 0);
    const float32x4_t x1 = vld1q_f32(feat_in + 4);
    const float32x4_t x2 = vld1q_f32(feat_in + 8);

    // Residual: out = input_proj @ feat_in  (24 outputs from 12 inputs).
    // 6 output groups of 4. Use Phase-1 NEON pattern (across-IN, 4-out ILP).
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

    // 9 patches.
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

}  // namespace

int main(int argc, char** argv) {
    const char* weights = (argc > 1) ? argv[1] : "/tmp/ssla_s_64x80";
    constexpr long long N_ITERS  = 200'000;
    constexpr long long N_WARMUP = 10'000;

    deploy::SslaSPipeline pipe;
    try {
        pipe.load(weights);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load failed: %s\n", e.what());
        return 1;
    }
    pipe.reset();

    const int H = pipe.H();
    const int W = pipe.W();
    const int Wl = pipe.bench_Wl(1);   // stage 1 → W/2 = 40
    const int Hl = pipe.bench_Hl(1);   // stage 1 → H/2 = 32
    std::printf("Loaded %s  H=%d W=%d   (s1 grid Wl=%d Hl=%d)\n",
                weights, H, W, Wl, Hl);

    const auto& Lw = pipe.bench_layer_weights(3);   // s1 L1 = <24, 24>
    float* H_all   = pipe.bench_hidden(3);
    std::printf("Layer 3 (s1 L1): in=%d out=%d num_pos=%d\n",
                Lw.in_dim, Lw.out_dim, Lw.num_pos);

    alignas(16) float feat_in [24];
    alignas(16) float feat_out_g[24];   // generic path
    alignas(16) float feat_out_f[24];   // fused path

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : feat_in) v = dist(rng);

    // Snapshot the layer-3 hidden state so we can rewind between paths
    // (each call mutates it).
    std::vector<float> H_snap(24 * Hl * Wl);
    std::memcpy(H_snap.data(), H_all, H_snap.size() * sizeof(float));

    // ---- equivalence: same input → max|Δ| feat_out ------------------------
    pipe.layer_forward(3, 5, 5, feat_in, feat_out_g);
    std::memcpy(H_all, H_snap.data(), H_snap.size() * sizeof(float));

    layer_forward_s1_l1_fused_interior(
        5, 5, Wl, feat_in, Lw.qvgIn, Lw.goW,
        Lw.ln_gamma.data(), Lw.ln_beta.data(),
        H_all, feat_out_f);
    std::memcpy(H_all, H_snap.data(), H_snap.size() * sizeof(float));

    float max_d = 0.f;
    for (int i = 0; i < 24; ++i) {
        const float d = std::fabs(feat_out_g[i] - feat_out_f[i]);
        if (d > max_d) max_d = d;
    }
    std::printf("\n=== correctness ===\n  max|Δ| feat_out (single call): %.3e\n",
                max_d);

    // ---- bench generic path -----------------------------------------------
    for (long long i = 0; i < N_WARMUP; ++i) {
        const int x = 2 + (int)(i % (Wl - 4));
        const int y = 2 + (int)((i / Wl) % (Hl - 4));
        pipe.layer_forward(3, x, y, feat_in, feat_out_g);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (long long i = 0; i < N_ITERS; ++i) {
        const int x = 2 + (int)(i % (Wl - 4));
        const int y = 2 + (int)((i / Wl) % (Hl - 4));
        pipe.layer_forward(3, x, y, feat_in, feat_out_g);
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ns_generic =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / N_ITERS;

    // Re-snapshot hidden state for fused path so both see same initial state.
    std::memcpy(H_all, H_snap.data(), H_snap.size() * sizeof(float));

    // ---- bench fused path -------------------------------------------------
    for (long long i = 0; i < N_WARMUP; ++i) {
        const int x = 2 + (int)(i % (Wl - 4));
        const int y = 2 + (int)((i / Wl) % (Hl - 4));
        layer_forward_s1_l1_fused_interior(
            x, y, Wl, feat_in, Lw.qvgIn, Lw.goW,
            Lw.ln_gamma.data(), Lw.ln_beta.data(),
            H_all, feat_out_f);
    }
    auto t2 = std::chrono::steady_clock::now();
    for (long long i = 0; i < N_ITERS; ++i) {
        const int x = 2 + (int)(i % (Wl - 4));
        const int y = 2 + (int)((i / Wl) % (Hl - 4));
        layer_forward_s1_l1_fused_interior(
            x, y, Wl, feat_in, Lw.qvgIn, Lw.goW,
            Lw.ln_gamma.data(), Lw.ln_beta.data(),
            H_all, feat_out_f);
    }
    auto t3 = std::chrono::steady_clock::now();
    const double ns_fused =
        std::chrono::duration<double, std::nano>(t3 - t2).count() / N_ITERS;

    std::printf("\n=== bench: s1 L1 <24, 24>, interior pixel, %lld iters ===\n",
                N_ITERS);
    std::printf("  generic (pipe.layer_forward) : %7.1f ns/call\n", ns_generic);
    std::printf("  fused hand-NEON              : %7.1f ns/call    %.2fx\n",
                ns_fused, ns_generic / ns_fused);

    // ========================================================================
    // s0 L1 <12, 12> bench
    // ========================================================================
    {
        const auto& Lw1 = pipe.bench_layer_weights(1);
        float* H1 = pipe.bench_hidden(1);
        const int Wl0 = pipe.bench_Wl(0);
        const int Hl0 = pipe.bench_Hl(0);
        alignas(16) float fi[12], fo_g[12], fo_f[12];
        for (auto& v : fi) v = dist(rng);
        std::vector<float> H1_snap(12 * Hl0 * Wl0);
        std::memcpy(H1_snap.data(), H1, H1_snap.size() * sizeof(float));

        pipe.layer_forward(1, 5, 5, fi, fo_g);
        std::memcpy(H1, H1_snap.data(), H1_snap.size() * sizeof(float));
        layer_forward_s0_l1_fused_interior(
            5, 5, Wl0, fi, Lw1.qvgIn, Lw1.goW,
            Lw1.ln_gamma.data(), Lw1.ln_beta.data(),
            H1, fo_f);
        std::memcpy(H1, H1_snap.data(), H1_snap.size() * sizeof(float));
        float md = 0.f;
        for (int i = 0; i < 12; ++i) md = std::max(md, std::fabs(fo_g[i] - fo_f[i]));

        for (long long i = 0; i < N_WARMUP; ++i) {
            const int x = 2 + (int)(i % (Wl0 - 4));
            const int y = 2 + (int)((i / Wl0) % (Hl0 - 4));
            pipe.layer_forward(1, x, y, fi, fo_g);
        }
        auto u0 = std::chrono::steady_clock::now();
        for (long long i = 0; i < N_ITERS; ++i) {
            const int x = 2 + (int)(i % (Wl0 - 4));
            const int y = 2 + (int)((i / Wl0) % (Hl0 - 4));
            pipe.layer_forward(1, x, y, fi, fo_g);
        }
        auto u1 = std::chrono::steady_clock::now();
        const double ns_g = std::chrono::duration<double, std::nano>(u1 - u0).count() / N_ITERS;
        std::memcpy(H1, H1_snap.data(), H1_snap.size() * sizeof(float));
        for (long long i = 0; i < N_WARMUP; ++i) {
            const int x = 2 + (int)(i % (Wl0 - 4));
            const int y = 2 + (int)((i / Wl0) % (Hl0 - 4));
            layer_forward_s0_l1_fused_interior(x, y, Wl0, fi, Lw1.qvgIn, Lw1.goW,
                Lw1.ln_gamma.data(), Lw1.ln_beta.data(), H1, fo_f);
        }
        auto u2 = std::chrono::steady_clock::now();
        for (long long i = 0; i < N_ITERS; ++i) {
            const int x = 2 + (int)(i % (Wl0 - 4));
            const int y = 2 + (int)((i / Wl0) % (Hl0 - 4));
            layer_forward_s0_l1_fused_interior(x, y, Wl0, fi, Lw1.qvgIn, Lw1.goW,
                Lw1.ln_gamma.data(), Lw1.ln_beta.data(), H1, fo_f);
        }
        auto u3 = std::chrono::steady_clock::now();
        const double ns_f = std::chrono::duration<double, std::nano>(u3 - u2).count() / N_ITERS;
        std::printf("\n=== bench: s0 L1 <12, 12>, interior pixel ===\n");
        std::printf("  max|Δ|: %.3e\n", md);
        std::printf("  generic : %7.1f ns/call\n", ns_g);
        std::printf("  fused   : %7.1f ns/call    %.2fx\n", ns_f, ns_g / ns_f);
    }

    // ========================================================================
    // s1 L0 <12, 24> bench
    // ========================================================================
    {
        const auto& Lw2 = pipe.bench_layer_weights(2);
        float* H2 = pipe.bench_hidden(2);
        const int Wl1 = pipe.bench_Wl(1);
        const int Hl1 = pipe.bench_Hl(1);
        if (Lw2.input_proj.empty()) {
            std::printf("\nERROR: s1 L0 input_proj is empty (in==out)? Skipping.\n");
        } else {
            alignas(16) float fi[12], fo_g[24], fo_f[24];
            for (auto& v : fi) v = dist(rng);
            std::vector<float> H2_snap(24 * Hl1 * Wl1);
            std::memcpy(H2_snap.data(), H2, H2_snap.size() * sizeof(float));

            pipe.layer_forward(2, 5, 5, fi, fo_g);
            std::memcpy(H2, H2_snap.data(), H2_snap.size() * sizeof(float));
            layer_forward_s1_l0_fused_interior(
                5, 5, Wl1, fi, Lw2.input_proj.data(),
                Lw2.qvgIn, Lw2.goW,
                Lw2.ln_gamma.data(), Lw2.ln_beta.data(),
                H2, fo_f);
            std::memcpy(H2, H2_snap.data(), H2_snap.size() * sizeof(float));
            float md = 0.f;
            for (int i = 0; i < 24; ++i) md = std::max(md, std::fabs(fo_g[i] - fo_f[i]));

            for (long long i = 0; i < N_WARMUP; ++i) {
                const int x = 2 + (int)(i % (Wl1 - 4));
                const int y = 2 + (int)((i / Wl1) % (Hl1 - 4));
                pipe.layer_forward(2, x, y, fi, fo_g);
            }
            auto u0 = std::chrono::steady_clock::now();
            for (long long i = 0; i < N_ITERS; ++i) {
                const int x = 2 + (int)(i % (Wl1 - 4));
                const int y = 2 + (int)((i / Wl1) % (Hl1 - 4));
                pipe.layer_forward(2, x, y, fi, fo_g);
            }
            auto u1 = std::chrono::steady_clock::now();
            const double ns_g = std::chrono::duration<double, std::nano>(u1 - u0).count() / N_ITERS;
            std::memcpy(H2, H2_snap.data(), H2_snap.size() * sizeof(float));
            for (long long i = 0; i < N_WARMUP; ++i) {
                const int x = 2 + (int)(i % (Wl1 - 4));
                const int y = 2 + (int)((i / Wl1) % (Hl1 - 4));
                layer_forward_s1_l0_fused_interior(x, y, Wl1, fi, Lw2.input_proj.data(),
                    Lw2.qvgIn, Lw2.goW, Lw2.ln_gamma.data(), Lw2.ln_beta.data(), H2, fo_f);
            }
            auto u2 = std::chrono::steady_clock::now();
            for (long long i = 0; i < N_ITERS; ++i) {
                const int x = 2 + (int)(i % (Wl1 - 4));
                const int y = 2 + (int)((i / Wl1) % (Hl1 - 4));
                layer_forward_s1_l0_fused_interior(x, y, Wl1, fi, Lw2.input_proj.data(),
                    Lw2.qvgIn, Lw2.goW, Lw2.ln_gamma.data(), Lw2.ln_beta.data(), H2, fo_f);
            }
            auto u3 = std::chrono::steady_clock::now();
            const double ns_f = std::chrono::duration<double, std::nano>(u3 - u2).count() / N_ITERS;
            std::printf("\n=== bench: s1 L0 <12, 24>, interior pixel ===\n");
            std::printf("  max|Δ|: %.3e\n", md);
            std::printf("  generic : %7.1f ns/call\n", ns_g);
            std::printf("  fused   : %7.1f ns/call    %.2fx\n", ns_f, ns_g / ns_f);
        }
    }

    return 0;
}
