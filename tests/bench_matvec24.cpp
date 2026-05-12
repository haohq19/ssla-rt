// tests/bench_matvec24.cpp — scalar/GCC-auto vs hand-NEON for IN=24 matvecs.
//
// Stage-1 L1 (the largest remaining per-event cost) is dominated by:
//   matvec_ct<24, 72>     — qvg projection, 9× / event (pass0)
//   matvec_accum_ct<24, 24> — goW gather,    9× / event (pass0)
//
// GCC's auto-vectoriser handles IN=24 (multiple of 4) and produces NEON FMA
// in the §8.4 disassembly. The question: can hand-written 4-output ILP do
// even better (like it did for IN=12 in bench_matvec12.cpp, where the
// 4-out variant was 2.93× over scalar)?

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "openeva/prim/linear.h"

// Hand NEON: 4-output ILP, across-IN. For IN=24 the input fits in 6 fp32x4
// registers. Each output group does 6 fmla per accumulator + paired vpaddq
// reduce.
template <int OUT>
static inline void matvec_neon_in24_4out(const float* __restrict__ x,
                                          const float* __restrict__ W,
                                          const float* __restrict__ bias,
                                          float* __restrict__ y) {
    static_assert(OUT % 4 == 0, "OUT must be multiple of 4");
    const float32x4_t x0 = vld1q_f32(x + 0);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    const float32x4_t x3 = vld1q_f32(x + 12);
    const float32x4_t x4 = vld1q_f32(x + 16);
    const float32x4_t x5 = vld1q_f32(x + 20);
    for (int o = 0; o < OUT; o += 4) {
        const float* w0 = W + (o + 0) * 24;
        const float* w1 = W + (o + 1) * 24;
        const float* w2 = W + (o + 2) * 24;
        const float* w3 = W + (o + 3) * 24;

        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0));
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1));
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2));
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3));

        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 +  4));
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 +  4));
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 +  4));
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 +  4));

        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 +  8));
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 +  8));
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 +  8));
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 +  8));

        a0 = vfmaq_f32(a0, x3, vld1q_f32(w0 + 12));
        a1 = vfmaq_f32(a1, x3, vld1q_f32(w1 + 12));
        a2 = vfmaq_f32(a2, x3, vld1q_f32(w2 + 12));
        a3 = vfmaq_f32(a3, x3, vld1q_f32(w3 + 12));

        a0 = vfmaq_f32(a0, x4, vld1q_f32(w0 + 16));
        a1 = vfmaq_f32(a1, x4, vld1q_f32(w1 + 16));
        a2 = vfmaq_f32(a2, x4, vld1q_f32(w2 + 16));
        a3 = vfmaq_f32(a3, x4, vld1q_f32(w3 + 16));

        a0 = vfmaq_f32(a0, x5, vld1q_f32(w0 + 20));
        a1 = vfmaq_f32(a1, x5, vld1q_f32(w1 + 20));
        a2 = vfmaq_f32(a2, x5, vld1q_f32(w2 + 20));
        a3 = vfmaq_f32(a3, x5, vld1q_f32(w3 + 20));

        const float32x4_t s01 = vpaddq_f32(a0, a1);
        const float32x4_t s23 = vpaddq_f32(a2, a3);
        float32x4_t s0123 = vpaddq_f32(s01, s23);
        if (bias) s0123 = vaddq_f32(s0123, vld1q_f32(bias + o));
        vst1q_f32(y + o, s0123);
    }
}

template <int IN, int OUT>
static void bench(const char* label) {
    constexpr long long N_ITERS = 5'000'000;
    alignas(16) float W[OUT * IN];
    alignas(16) float x[IN];
    alignas(16) float y_scalar[OUT];
    alignas(16) float y_neon[OUT];

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : W) v = dist(rng);
    for (auto& v : x) v = dist(rng);

    // Equivalence check.
    openeva::prim::matvec_ct<IN, OUT>(x, W, nullptr, y_scalar);
    matvec_neon_in24_4out<OUT>(x, W, nullptr, y_neon);
    float max_diff = 0.f;
    for (int i = 0; i < OUT; ++i) {
        const float d = std::fabs(y_scalar[i] - y_neon[i]);
        if (d > max_diff) max_diff = d;
    }

    // Warm up.
    for (int i = 0; i < 100'000; ++i) {
        openeva::prim::matvec_ct<IN, OUT>(x, W, nullptr, y_scalar);
    }

    // Time scalar (GCC auto-vec).
    float sink_s = 0.f;
    auto t0 = std::chrono::steady_clock::now();
    for (long long i = 0; i < N_ITERS; ++i) {
        openeva::prim::matvec_ct<IN, OUT>(x, W, nullptr, y_scalar);
        x[i & (IN - 1)] = y_scalar[i & (OUT - 1)] * 1e-7f;
        sink_s += y_scalar[0];
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ns_s =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / N_ITERS;

    for (auto& v : x) v = dist(rng);

    // Warm up NEON.
    for (int i = 0; i < 100'000; ++i) {
        matvec_neon_in24_4out<OUT>(x, W, nullptr, y_neon);
    }

    // Time NEON 4-out.
    float sink_n = 0.f;
    auto t2 = std::chrono::steady_clock::now();
    for (long long i = 0; i < N_ITERS; ++i) {
        matvec_neon_in24_4out<OUT>(x, W, nullptr, y_neon);
        x[i & (IN - 1)] = y_neon[i & (OUT - 1)] * 1e-7f;
        sink_n += y_neon[0];
    }
    auto t3 = std::chrono::steady_clock::now();
    const double ns_n =
        std::chrono::duration<double, std::nano>(t3 - t2).count() / N_ITERS;

    std::printf("%s\n", label);
    std::printf("  GCC auto-vec    : %7.2f ns/call\n", ns_s);
    std::printf("  NEON 4-out ILP  : %7.2f ns/call    %.2fx\n",
                ns_n, ns_s / ns_n);
    std::printf("  max|Δ|          : %.3e   (sinks: %g %g)\n\n",
                max_diff, sink_s, sink_n);
}

int main() {
    bench<24, 72>("matvec_ct<IN=24, OUT=72>  (s1 L0 qvg)");
    bench<24, 24>("matvec_ct<IN=24, OUT=24>  (s1 L1 input_proj / s2 use)");
    return 0;
}
