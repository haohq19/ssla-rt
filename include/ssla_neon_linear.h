#pragma once

// NEON specializations for matvec_ct<IN=12, OUT> and matvec_accum_ct<IN=12, OUT>.
//
// Why this file exists: STATUS.md §8.4 disassembly showed GCC's auto-vectorizer
// rejects across-IN-axis vectorization when IN=12. The resulting fully-unrolled
// scalar 12-fmadd chain dominates stage-0 L1's per-event cost (Task #1
// instrumentation: stage_forward(0) = 3.94 µs/event = 57% of CPU budget).
//
// Microbench (Task #2, tests/bench_matvec12.cpp, Cortex-A78AE @ 1.984 GHz):
//   scalar matvec_ct<12, 36>            : 148.31 ns/call
//   NEON 4-output ILP (this file's impl):  50.68 ns/call   (2.93x)
//
// Implementation: 4 outputs computed in parallel (4 independent fp32x4 FMA
// chains → 4 FP pipelines on A78AE), then 3 vpaddq instructions reduce to
// 4 scalar sums in one vector register and store. Each output requires:
//   3 vfmaq_f32 + ~0.75 vpaddq + 1 vst1q_f32 (amortised across 4 outputs).
//
// IMPORTANT — preserving P1 numerics: the scalar reference accumulates
// left-to-right (acc += w[i]*x[i] for i = 0..11). This NEON impl accumulates
// in 3 lanes (0..3, 4..7, 8..11) then horizontal-reduces via paired adds.
// The FP add order differs, producing per-call max|Δ| ≈ 1e-7. Over 200k P1
// events the accumulated drift is well below the max|Δ| ≤ 5 gate (verified
// in Task #5).
//
// NOT vendor/openeva/prim/linear.h: keeping upstream drift to zero. This
// header is included exactly once, from src/ssla_kernels.cpp, AFTER the
// primary scalar template has been seen. The explicit specializations below
// then override the scalar version at all instantiation sites in that TU,
// which is where all layer_forward_ct<…> bodies live (and thus where every
// matvec_ct<12, …> call is actually emitted into libssla_kernels.a).

#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>
#include <cstddef>

#include "openeva/prim/linear.h"   // primary templates: matvec_ct, matvec_accum_ct
#include "openeva/prim/flop.h"     // add_macs

namespace openeva::prim {

// ----- shared core: 4-output ILP NEON kernel for IN=12 ----------------------
//
// Computes y[o] = (bias ? bias[o] : 0) + Σ_i W[o, i] * x[i] for o ∈ [0, OUT),
// row-major W with stride 12. OUT must be divisible by 4.
//
// The bool template parameter `Accumulate` selects between matvec (y = ...)
// and matvec_accum (y += ...). Compile-time branch — generates two distinct
// code paths.
template <int OUT, bool Accumulate>
inline void matvec_in12_neon_core(const float* __restrict__ x,
                                   const float* __restrict__ W,
                                   const float* __restrict__ bias,
                                   float* __restrict__ y) {
    static_assert(OUT % 4 == 0, "matvec_in12_neon_core: OUT must be multiple of 4");

    const float32x4_t x0 = vld1q_f32(x + 0);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);

    for (int o = 0; o < OUT; o += 4) {
        const float* w0 = W + static_cast<std::ptrdiff_t>(o + 0) * 12;
        const float* w1 = W + static_cast<std::ptrdiff_t>(o + 1) * 12;
        const float* w2 = W + static_cast<std::ptrdiff_t>(o + 2) * 12;
        const float* w3 = W + static_cast<std::ptrdiff_t>(o + 3) * 12;

        // Initialize 4 independent accumulator chains.
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

        // Paired horizontal reduce → [sum0, sum1, sum2, sum3] in one vector.
        const float32x4_t s01   = vpaddq_f32(a0, a1);
        const float32x4_t s23   = vpaddq_f32(a2, a3);
        float32x4_t       s0123 = vpaddq_f32(s01, s23);

        if constexpr (Accumulate) {
            // y[o..o+4] += s0123
            const float32x4_t prev = vld1q_f32(y + o);
            vst1q_f32(y + o, vaddq_f32(prev, s0123));
        } else {
            // y[o..o+4] = (bias ? bias[o..o+4] : 0) + s0123
            if (bias) s0123 = vaddq_f32(s0123, vld1q_f32(bias + o));
            vst1q_f32(y + o, s0123);
        }
    }
}

// ----- Explicit specializations of the templates in linear.h ----------------
//
// Sizes covered (Task #1 + reading layer_forward_ct call graph):
//   matvec_ct<12, 24>  — stage 1 L0 input_proj for residual.  1× / pass0 event
//   matvec_ct<12, 36>  — stage 0 L1 qvg projection.           9× / event
//   matvec_ct<12, 72>  — stage 1 L0 qvg projection.           9× / pass0 event
//   matvec_accum_ct<12, 12> — stage 0 L0/L1 goW gather.       9× / patch (×2 layers)
//
// Other IN=12 sizes (matvec_ct<12, 12>): also present but only via the
// matvec_ct path when out_dim == in_dim — none such in the SSLA-S graph at
// these layer dims, so no specialization needed. If a future export hits it,
// the scalar version still works (correct, just not vectorized).

template <>
inline void matvec_ct<12, 24>(const float* x, const float* W,
                               const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(12) * 24);
    matvec_in12_neon_core<24, /*Accumulate=*/false>(x, W, bias, y);
}

template <>
inline void matvec_ct<12, 36>(const float* x, const float* W,
                               const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(12) * 36);
    matvec_in12_neon_core<36, /*Accumulate=*/false>(x, W, bias, y);
}

template <>
inline void matvec_ct<12, 72>(const float* x, const float* W,
                               const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(12) * 72);
    matvec_in12_neon_core<72, /*Accumulate=*/false>(x, W, bias, y);
}

template <>
inline void matvec_accum_ct<12, 12>(const float* x, const float* W, float* y) {
    add_macs(static_cast<std::size_t>(12) * 12);
    matvec_in12_neon_core<12, /*Accumulate=*/true>(x, W, /*bias=*/nullptr, y);
}

}  // namespace openeva::prim

#endif  // __ARM_NEON || __aarch64__
