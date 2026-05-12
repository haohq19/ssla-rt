// tests/bench_sigmoid.cpp — sigmoid implementations: speed + accuracy.
//
// Three NEON polynomial approximations of sigmoid(x) = 1/(1+e^-x), plus the
// scalar reference (vendor/openeva/prim/activation.h's std::exp form).
//
// What this answers, before committing to a NEON lru_step rewrite:
//   1. Is libm expf already fast on Orin (≤10 ns)? Yes → polynomial NEON
//      may not be much faster.
//   2. What's max|Δ| vs scalar reference across x ∈ [-10, 10]?
//   3. What's the cumulative drift from a 200k-step lru recurrence (the
//      pattern in production: h_t+1 = sigmoid(g) * h_t + v) — i.e. does
//      sigmoid error fed back through the recurrence stay bounded enough
//      to pass P1's max|Δ| ≤ 5 gate?

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

// Scalar reference — bit-equivalent to vendor sigmoid().
static inline float sig_ref(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ---- Variant A: degree-5 odd polynomial around 0.5 -------------------------
// sigmoid(x) ≈ 0.5 + x * (a1 + a3*x^2 + a5*x^4)
// Coefficients minimax-fit on [-4, 4] with sigmoid(±4) ≈ 0.018 / 0.982.
// Outside [-4, 4] clamp to 0 or 1.
static inline float32x4_t neon_sig_poly5(float32x4_t x) {
    const float32x4_t lo = vdupq_n_f32(-4.0f);
    const float32x4_t hi = vdupq_n_f32( 4.0f);
    x = vmaxq_f32(x, lo);
    x = vminq_f32(x, hi);
    const float32x4_t x2 = vmulq_f32(x, x);
    // Coefficients for sigmoid on [-4,4] from minimax fit (degree-5 odd).
    const float32x4_t c5 = vdupq_n_f32( 5.69e-4f);
    const float32x4_t c3 = vdupq_n_f32(-1.55e-2f);
    const float32x4_t c1 = vdupq_n_f32( 2.18e-1f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t inner = vfmaq_f32(c3, c5, x2);     // c3 + c5*x^2
    inner = vfmaq_f32(c1, inner, x2);              // c1 + (c3+c5*x^2)*x^2
    return vfmaq_f32(half, x, inner);              // 0.5 + x*inner
}

// ---- Variant B: 2-piece minimax (one piece for |x|<2, one for [-4,-2]∪[2,4]),
//      saturating outside.  More accurate than poly5 on the tail. Skipped —
//      branching kills NEON throughput. Keep as a fallback if accuracy is the
//      problem.

// ---- Variant C: e^x via 2nd-order Padé(1, sat at ±5) -----------------------
// sigmoid(x) = e^x / (1 + e^x). Compute e^x via Padé[2/2] then divide. Padé
// gives ~1e-3 absolute accuracy in [-3, 3]; saturate outside.
static inline float32x4_t neon_sig_pade(float32x4_t x) {
    const float32x4_t hi  = vdupq_n_f32( 5.0f);
    const float32x4_t lo  = vdupq_n_f32(-5.0f);
    x = vmaxq_f32(x, lo);
    x = vminq_f32(x, hi);
    // Direct rational approximation of sigmoid:
    //   sigmoid(x) ≈ (1 + a*x + b*x^2) / (2 + c*x^2)  on [-5, 5].
    // Coefficients chosen so the rational matches sigmoid value + slope at
    // 0 and the boundaries. Below are placeholders; replace with fit.
    const float32x4_t a  = vdupq_n_f32(0.5f);
    const float32x4_t b  = vdupq_n_f32(0.018f);
    const float32x4_t c  = vdupq_n_f32(0.07f);
    const float32x4_t x2 = vmulq_f32(x, x);
    const float32x4_t num = vfmaq_f32(vfmaq_f32(vdupq_n_f32(1.0f), a, x), b, x2);
    const float32x4_t den = vfmaq_f32(vdupq_n_f32(2.0f), c, x2);
    return vdivq_f32(num, den);
}

// ---- Variant D: tanh-based ---------------------------------------------------
// sigmoid(x) = 0.5 * (1 + tanh(x/2))
// tanh ≈ x * (27 + x^2) / (27 + 9*x^2) [Pade] on [-3, 3]; saturate outside.
static inline float32x4_t neon_sig_tanh(float32x4_t x) {
    const float32x4_t hi  = vdupq_n_f32( 6.0f);
    const float32x4_t lo  = vdupq_n_f32(-6.0f);
    x = vmaxq_f32(x, lo);
    x = vminq_f32(x, hi);
    const float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t t = vmulq_f32(x, half);            // x/2
    const float32x4_t t2 = vmulq_f32(t, t);
    const float32x4_t c27 = vdupq_n_f32(27.0f);
    const float32x4_t c9  = vdupq_n_f32(9.0f);
    const float32x4_t num = vmulq_f32(t, vaddq_f32(c27, t2));
    const float32x4_t den = vfmaq_f32(c27, c9, t2);
    const float32x4_t tanh_t = vdivq_f32(num, den);
    return vfmaq_f32(half, half, tanh_t);          // 0.5 + 0.5 * tanh
}

// ============================================================================
// Accuracy across x ∈ [-10, 10] sampled every 0.001
// ============================================================================

template <typename VFn>
double max_abs_err(VFn vfn) {
    double max_err = 0.0;
    for (int i = -10000; i <= 10000; ++i) {
        const float x = i * 0.001f;
        const float r = sig_ref(x);
        alignas(16) float in[4]  = {x, x, x, x};
        alignas(16) float out[4];
        vst1q_f32(out, vfn(vld1q_f32(in)));
        const float err = std::fabs(out[0] - r);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ============================================================================
// Throughput: lru_step<12> with each sigmoid impl
// ============================================================================

// Reference scalar lru_step (matches vendor) — used as throughput baseline.
static inline void lru12_scalar(const float* g, const float* v, const float* q,
                                 float* h, float* y) {
    for (int c = 0; c < 12; ++c) {
        const float gc = sig_ref(g[c]);
        const float hc = gc * h[c] + v[c];
        h[c] = hc;
        y[c] = q[c] * hc;
    }
}

template <typename VFn>
static inline void lru12_neon(const float* g, const float* v, const float* q,
                               float* h, float* y, VFn vfn) {
    for (int c = 0; c < 12; c += 4) {
        const float32x4_t gc = vfn(vld1q_f32(g + c));
        const float32x4_t hc = vfmaq_f32(vld1q_f32(v + c), gc, vld1q_f32(h + c));
        vst1q_f32(h + c, hc);
        vst1q_f32(y + c, vmulq_f32(vld1q_f32(q + c), hc));
    }
}

int main() {
    constexpr long long N_ITERS = 5'000'000;

    // ---- Accuracy summary ---------------------------------------------------
    std::printf("=== sigmoid single-call accuracy, x ∈ [-10, 10] ===\n");
    std::printf("  %-24s  %12s\n", "impl", "max|Δ|");
    std::printf("  %-24s  %12.3e\n", "poly5  (clamp ±4)",  max_abs_err(neon_sig_poly5));
    std::printf("  %-24s  %12.3e\n", "pade   (clamp ±5)",  max_abs_err(neon_sig_pade));
    std::printf("  %-24s  %12.3e\n", "tanh-pade (clamp ±6)", max_abs_err(neon_sig_tanh));

    // ---- Throughput ---------------------------------------------------------
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    alignas(16) float g[12], v[12], q[12], h[12], y[12];
    for (int i = 0; i < 12; ++i) {
        g[i] = dist(rng); v[i] = dist(rng); q[i] = dist(rng); h[i] = dist(rng) * 0.1f;
    }

    // Vary g per iter to prevent the compiler from hoisting sigmoid(g[c])
    // out of the loop body (the iteration-invariant input made the
    // scalar variant look 9× faster than reality in the first revision).
    auto time_loop = [&](auto loop) -> double {
        for (int i = 0; i < 50000; ++i) loop(i);   // warm
        auto t0 = std::chrono::steady_clock::now();
        for (long long i = 0; i < N_ITERS; ++i) loop(i);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count() / N_ITERS;
    };

    std::printf("\n=== lru_step<12> throughput, %lld iters ===\n", N_ITERS);
    std::printf("  %-28s  %10s\n", "impl", "ns/call");

    double sink = 0.f;
    std::printf("  %-28s  %10.1f\n", "scalar (std::exp)", time_loop([&](long long i) {
        // Perturb every g entry — kills constant-folding of sigmoid().
        const float bias = static_cast<float>(i & 0xff) * 1e-2f - 1.27f;
        for (int c = 0; c < 12; ++c) g[c] += bias * 1e-4f;
        lru12_scalar(g, v, q, h, y);
        sink += y[0];
    }));
    std::printf("  %-28s  %10.1f\n", "NEON poly5", time_loop([&](long long i) {
        const float bias = static_cast<float>(i & 0xff) * 1e-2f - 1.27f;
        for (int c = 0; c < 12; ++c) g[c] += bias * 1e-4f;
        lru12_neon(g, v, q, h, y, [](float32x4_t x){ return neon_sig_poly5(x); });
        sink += y[0];
    }));
    std::printf("  %-28s  %10.1f\n", "NEON pade", time_loop([&](long long i) {
        const float bias = static_cast<float>(i & 0xff) * 1e-2f - 1.27f;
        for (int c = 0; c < 12; ++c) g[c] += bias * 1e-4f;
        lru12_neon(g, v, q, h, y, [](float32x4_t x){ return neon_sig_pade(x); });
        sink += y[0];
    }));
    std::printf("  %-28s  %10.1f\n", "NEON tanh-pade", time_loop([&](long long i) {
        const float bias = static_cast<float>(i & 0xff) * 1e-2f - 1.27f;
        for (int c = 0; c < 12; ++c) g[c] += bias * 1e-4f;
        lru12_neon(g, v, q, h, y, [](float32x4_t x){ return neon_sig_tanh(x); });
        sink += y[0];
    }));
    std::printf("  (sink: %g  — ignore)\n", sink);

    // ---- Cumulative drift from lru recurrence -------------------------------
    // Pattern: feed a stream of (g, v, q) and accumulate h. Compare final h
    // between scalar and each NEON variant.
    std::printf("\n=== cumulative drift over 200k steps (lru recurrence) ===\n");
    std::printf("  %-24s  %12s\n", "impl", "max|Δh| at end");
    std::mt19937 rng2(456);
    std::uniform_real_distribution<float> dgate(-3.0f, 3.0f);
    std::uniform_real_distribution<float> dval (-1.0f, 1.0f);

    auto run_recurrence = [&](auto sig_fn_scalar_or_vec, bool use_neon) {
        alignas(16) float h_acc[12] = {0};
        for (int step = 0; step < 200'000; ++step) {
            alignas(16) float gs[12], vs[12], qs[12], ys[12];
            for (int c = 0; c < 12; ++c) {
                gs[c] = dgate(rng2); vs[c] = dval(rng2); qs[c] = dval(rng2);
            }
            if (use_neon) {
                for (int c = 0; c < 12; c += 4) {
                    const float32x4_t gc = sig_fn_scalar_or_vec(vld1q_f32(gs + c));
                    const float32x4_t hc = vfmaq_f32(vld1q_f32(vs + c),
                                                      gc, vld1q_f32(h_acc + c));
                    vst1q_f32(h_acc + c, hc);
                }
            } else {
                for (int c = 0; c < 12; ++c) {
                    const float gc = sig_ref(gs[c]);
                    h_acc[c] = gc * h_acc[c] + vs[c];
                }
            }
        }
        return std::vector<float>(h_acc, h_acc + 12);
    };

    rng2.seed(456);
    auto h_ref = run_recurrence([](float32x4_t x){ return x; }, /*use_neon=*/false);

    rng2.seed(456);
    auto h_poly5 = run_recurrence([](float32x4_t x){ return neon_sig_poly5(x); }, true);
    double err_poly5 = 0.f;
    for (int c = 0; c < 12; ++c)
        err_poly5 = std::max<double>(err_poly5, std::fabs(h_poly5[c] - h_ref[c]));

    rng2.seed(456);
    auto h_pade = run_recurrence([](float32x4_t x){ return neon_sig_pade(x); }, true);
    double err_pade = 0.f;
    for (int c = 0; c < 12; ++c)
        err_pade = std::max<double>(err_pade, std::fabs(h_pade[c] - h_ref[c]));

    rng2.seed(456);
    auto h_tanh = run_recurrence([](float32x4_t x){ return neon_sig_tanh(x); }, true);
    double err_tanh = 0.f;
    for (int c = 0; c < 12; ++c)
        err_tanh = std::max<double>(err_tanh, std::fabs(h_tanh[c] - h_ref[c]));

    std::printf("  %-24s  %12.4f\n", "poly5",     err_poly5);
    std::printf("  %-24s  %12.4f\n", "pade",      err_pade);
    std::printf("  %-24s  %12.4f\n", "tanh-pade", err_tanh);
    std::printf("\n  P1 gate: max|Δ| ≤ 5  →  pass if cumulative drift < ~2.0\n");
    std::printf("  (P1 is checked on s3_feat, not h_acc, so this is conservative\n");
    std::printf("   — actual P1 drift includes attenuation through more layers.)\n");

    return 0;
}
