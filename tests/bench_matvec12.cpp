// tests/bench_matvec12.cpp — microbench: scalar vs hand-NEON matvec_ct<12, 36>.
//
// Why this exists: STATUS.md §8.4 showed that GCC's auto-vectorizer rejects
// matvec_ct<12, OUT> and emits a fully-unrolled 12-fmadd scalar chain per
// output. This is the dominant compute in stage-0 L1 (`layer_forward_ct<12,12>`
// lambda — runs on every owner + halo event). This bench measures whether
// a hand-NEON intrinsic version beats the scalar code on Orin NX's
// Cortex-A78AE, and by how much. Result drives the Phase-2 NEON specialization
// decision.
//
// Methodology:
//   - Same matvec dim as in production: IN=12, OUT=36 (qvg projection in
//     stage-0 L1).
//   - 10M iterations of each implementation. Data dependency (result_sink
//     fed into x[]) prevents the compiler from hoisting / DCE.
//   - Output comparison verifies the NEON impl produces same answer modulo
//     fp32 rounding noise.
//   - Pin to one core (taskset -c <id>) before running.

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "openeva/prim/linear.h"

// NEON, across-IN-axis dot product. For IN=12 the input fits in 3 fp32x4
// registers loaded once outside the OUT loop. Per output: 1 fmul + 2 fmla
// (vector) + 2 faddp (horizontal reduce) + 1 store.
template <int OUT>
static inline void matvec_neon_in12(const float* __restrict__ x,
                                     const float* __restrict__ W,
                                     const float* __restrict__ bias,
                                     float* __restrict__ y) {
    const float32x4_t x0 = vld1q_f32(x + 0);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    for (int o = 0; o < OUT; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * 12;
        float32x4_t a = vmulq_f32(x0, vld1q_f32(w + 0));
        a = vfmaq_f32(a, x1, vld1q_f32(w + 4));
        a = vfmaq_f32(a, x2, vld1q_f32(w + 8));
        // Horizontal reduce: faddp twice — element 0 of the 2-lane result
        // holds the sum of all 4 lanes of `a`.
        float32x2_t a2 = vadd_f32(vget_low_f32(a), vget_high_f32(a));
        a2 = vpadd_f32(a2, a2);
        float acc = vget_lane_f32(a2, 0);
        if (bias) acc += bias[o];
        y[o] = acc;
    }
}

// NEON, 4-output interleaved (still across-IN axis). 4 independent FMA
// accumulator chains hide vfmaq latency via ILP across 4 FP pipelines on
// A78AE. Horizontal reduces are paired (vpaddq) so 4 outputs need only
// 3 vpaddq (vs 8 faddp in the per-output version). Requires OUT % 4 == 0.
template <int OUT>
static inline void matvec_neon_in12_4out(const float* __restrict__ x,
                                          const float* __restrict__ W,
                                          const float* __restrict__ bias,
                                          float* __restrict__ y) {
    static_assert(OUT % 4 == 0, "OUT must be multiple of 4 for 4-out variant");
    const float32x4_t x0 = vld1q_f32(x + 0);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    for (int o = 0; o < OUT; o += 4) {
        const float* w0 = W + (o + 0) * 12;
        const float* w1 = W + (o + 1) * 12;
        const float* w2 = W + (o + 2) * 12;
        const float* w3 = W + (o + 3) * 12;

        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0));
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1));
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2));
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3));

        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 + 4));
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 + 4));
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 + 4));
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 + 4));

        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 + 8));
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 + 8));
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 + 8));
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 + 8));

        // Paired horizontal reduce: 3 vpaddq → [sum0, sum1, sum2, sum3].
        const float32x4_t s01 = vpaddq_f32(a0, a1);   // [s0a, s0b, s1a, s1b]
        const float32x4_t s23 = vpaddq_f32(a2, a3);
        float32x4_t s0123 = vpaddq_f32(s01, s23);     // [s0, s1, s2, s3]
        if (bias) s0123 = vaddq_f32(s0123, vld1q_f32(bias + o));
        vst1q_f32(y + o, s0123);
    }
}

int main(int argc, char** argv) {
    constexpr int IN  = 12;
    constexpr int OUT = 36;
    constexpr long long N_ITERS  = 10'000'000;
    constexpr long long N_WARMUP =  1'000'000;
    const long long n_iters  = (argc > 1) ? std::atoll(argv[1]) : N_ITERS;
    const long long n_warmup = std::min<long long>(n_iters / 10, N_WARMUP);

    // 16-byte aligned allocations so NEON loads are aligned.
    alignas(16) float W[OUT * IN];
    alignas(16) float x_buf[IN];
    alignas(16) float y_scalar[OUT];
    alignas(16) float y_neon[OUT];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : W)     v = dist(rng);
    for (auto& v : x_buf) v = dist(rng);
    for (auto& v : y_scalar) v = 0.f;
    for (auto& v : y_neon)   v = 0.f;

    // ---- Equivalence check first (before warm-up jitter) -------------------
    alignas(16) float y_neon4[OUT];
    openeva::prim::matvec_ct<IN, OUT>(x_buf, W, nullptr, y_scalar);
    matvec_neon_in12<OUT>(x_buf, W, nullptr, y_neon);
    matvec_neon_in12_4out<OUT>(x_buf, W, nullptr, y_neon4);
    float max_diff_neon = 0.f, max_diff_4out = 0.f;
    for (int i = 0; i < OUT; ++i) {
        const float d1 = std::fabs(y_scalar[i] - y_neon[i]);
        const float d2 = std::fabs(y_scalar[i] - y_neon4[i]);
        if (d1 > max_diff_neon) max_diff_neon = d1;
        if (d2 > max_diff_4out) max_diff_4out = d2;
    }
    const float max_diff = max_diff_neon;  // keep prior name for the original report

    // ---- Warm-up -----------------------------------------------------------
    float sink_scalar = 0.f, sink_neon = 0.f;
    for (long long i = 0; i < n_warmup; ++i) {
        openeva::prim::matvec_ct<IN, OUT>(x_buf, W, nullptr, y_scalar);
        sink_scalar += y_scalar[i & (OUT - 1) > OUT - 1 ? 0 : (i & 0x1f)];
    }
    for (long long i = 0; i < n_warmup; ++i) {
        matvec_neon_in12<OUT>(x_buf, W, nullptr, y_neon);
        sink_neon += y_neon[i & 0x1f];
    }

    // ---- Time scalar -------------------------------------------------------
    sink_scalar = 0.f;
    auto t0 = std::chrono::steady_clock::now();
    for (long long i = 0; i < n_iters; ++i) {
        openeva::prim::matvec_ct<IN, OUT>(x_buf, W, nullptr, y_scalar);
        // Data dependency: feed one output back into x so the compiler
        // cannot hoist this call out of the loop.
        x_buf[i & 0x7] = y_scalar[i & 0x1f] * 1e-7f;
        sink_scalar += y_scalar[0];
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ns_scalar =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / n_iters;

    // Re-randomise x so NEON path doesn't inherit scalar's drift.
    for (auto& v : x_buf) v = dist(rng);

    // ---- Time NEON (per-output, across-IN) ---------------------------------
    sink_neon = 0.f;
    auto t2 = std::chrono::steady_clock::now();
    for (long long i = 0; i < n_iters; ++i) {
        matvec_neon_in12<OUT>(x_buf, W, nullptr, y_neon);
        x_buf[i & 0x7] = y_neon[i & 0x1f] * 1e-7f;
        sink_neon += y_neon[0];
    }
    auto t3 = std::chrono::steady_clock::now();
    const double ns_neon =
        std::chrono::duration<double, std::nano>(t3 - t2).count() / n_iters;

    // Re-randomise.
    for (auto& v : x_buf) v = dist(rng);

    // ---- Time NEON-4out (4 outputs interleaved) ----------------------------
    float sink_neon4 = 0.f;
    auto t4 = std::chrono::steady_clock::now();
    for (long long i = 0; i < n_iters; ++i) {
        matvec_neon_in12_4out<OUT>(x_buf, W, nullptr, y_neon4);
        x_buf[i & 0x7] = y_neon4[i & 0x1f] * 1e-7f;
        sink_neon4 += y_neon4[0];
    }
    auto t5 = std::chrono::steady_clock::now();
    const double ns_neon4 =
        std::chrono::duration<double, std::nano>(t5 - t4).count() / n_iters;

    // ---- Report ------------------------------------------------------------
    std::printf("matvec_ct<IN=%d, OUT=%d>  (%lld iters)\n", IN, OUT, n_iters);
    std::printf("  scalar              : %7.2f ns/call    (baseline)\n", ns_scalar);
    std::printf("  NEON  (1-out)       : %7.2f ns/call    %.2fx\n",
                ns_neon,  ns_scalar / ns_neon);
    std::printf("  NEON  (4-out ILP)   : %7.2f ns/call    %.2fx\n",
                ns_neon4, ns_scalar / ns_neon4);
    std::printf("  max|Δ| 1-out vs scalar : %.3e\n", max_diff_neon);
    std::printf("  max|Δ| 4-out vs scalar : %.3e\n", max_diff_4out);
    std::printf("  sinks (dummy)       : %g %g %g\n",
                sink_scalar, sink_neon, sink_neon4);
    return 0;
}
