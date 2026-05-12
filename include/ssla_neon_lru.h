#pragma once

// NEON specializations for openeva::prim::lru_step<DIM>.
//
// Why this file exists: bench_sigmoid (tests/bench_sigmoid.cpp) showed the
// vendored scalar lru_step is dominated by libm's scalar expf calls — the
// `#pragma omp declare simd` annotation on sigmoid() in
// vendor/openeva/prim/activation.h did NOT trigger libmvec vectorisation in
// the GCC 10 build (disassembly of layer_forward_ct<12, 12> lambda shows
// `bl <expf@plt>`). Result: ~80 ns per scalar lru_step<12> call, accumulating
// to ~972 ns/event in stage 0 (18 calls × 54 ns measured in-context).
//
// Replacement: 4-wide NEON sigmoid via tanh(x/2) Padé[3/2] approximation.
// The recurrence
//     h_{t+1} = sigmoid(g_t) * h_t + v_t
// is fed forward; sigmoid accuracy 1.2e-2 single-call, cumulative drift on
// h over 200k steps measured at 0.009 (bench_sigmoid). P1 (max|Δ| ≤ 5 on
// s3 feat after attenuation through s2+s3 matvecs) verified in Task #5.
//
// Microbench result (scalar vs tanh-pade NEON, lru_step<12>):
//     scalar  : 82.8 ns/call
//     NEON    : 21.4 ns/call   (3.87× faster)
//
// Not vendor/openeva/prim/rnn.h: keeping upstream drift to zero. Include
// from src/ssla_kernels.cpp AFTER vendor/openeva/prim/rnn.h (primary
// template visible first), before any layer_forward_ct<…> instantiation.

#if defined(__ARM_NEON) || defined(__aarch64__)

#include <arm_neon.h>
#include <cstddef>

#include "openeva/prim/rnn.h"      // primary template lru_step<DIM>
#include "openeva/prim/flop.h"     // add_flops, kSigmoidFlops

namespace openeva::prim {

// ----- NEON sigmoid via tanh Padé[3/2] approximation ------------------------
// sigmoid(x) = 0.5 * (1 + tanh(x/2))
// tanh(t)   ≈ t * (27 + t²) / (27 + 9t²)   — Padé[3/2] around 0, exact at 0.
// Saturate input at ±6 so tanh(x/2)'s argument stays in [-3, 3] where the
// approximation max-error is ~1e-2. Single-call max|Δ| vs std::exp form is
// 1.2e-2 across x ∈ [-10, 10] (measured in bench_sigmoid.cpp).
static inline float32x4_t neon_sigmoid_tanh_pade(float32x4_t x) {
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
    const float32x4_t tanh_t = vdivq_f32(num, den);
    return vfmaq_f32(half, half, tanh_t);
}

// ----- Core lru_step body: 4 lanes per NEON iteration -----------------------
// DIM must be a multiple of 4. Computes, in-place on h:
//   gc = sigmoid(g[c])
//   h[c] = gc * h[c] + v[c]
//   y[c] = q[c] * h[c]
template <int DIM>
static inline void lru_step_neon_core(const float* __restrict__ g,
                                       const float* __restrict__ v,
                                       const float* __restrict__ q,
                                       float* __restrict__ h,
                                       float* __restrict__ y) {
    static_assert(DIM % 4 == 0, "lru_step_neon_core: DIM must be multiple of 4");
    for (int c = 0; c < DIM; c += 4) {
        const float32x4_t gc = neon_sigmoid_tanh_pade(vld1q_f32(g + c));
        const float32x4_t hc = vfmaq_f32(vld1q_f32(v + c),
                                          gc, vld1q_f32(h + c));
        vst1q_f32(h + c, hc);
        vst1q_f32(y + c, vmulq_f32(vld1q_f32(q + c), hc));
    }
}

// ----- Explicit specializations for the SSLA-S sizes ------------------------
//
// Sizes hit by the per-event hot path:
//   lru_step<12> — stage 0 L0 + L1 (every event, 9 patches × 2 layers)
//   lru_step<24> — stage 1 L0 + L1 (~25% of events, same fanout)
//
// Stages 2 and 3 use DIM=48 and DIM=96 respectively; they're behind a much
// smaller event count after two more tdrop rounds. Specialise them too —
// the pattern is identical and the include cost is zero — so future profiles
// don't have to come back for them.

template <>
inline void lru_step<12>(const float* g, const float* v, const float* q,
                          float* h, float* y) {
    add_flops(static_cast<std::size_t>(12) * (kSigmoidFlops + 3));
    lru_step_neon_core<12>(g, v, q, h, y);
}

template <>
inline void lru_step<24>(const float* g, const float* v, const float* q,
                          float* h, float* y) {
    add_flops(static_cast<std::size_t>(24) * (kSigmoidFlops + 3));
    lru_step_neon_core<24>(g, v, q, h, y);
}

template <>
inline void lru_step<48>(const float* g, const float* v, const float* q,
                          float* h, float* y) {
    add_flops(static_cast<std::size_t>(48) * (kSigmoidFlops + 3));
    lru_step_neon_core<48>(g, v, q, h, y);
}

template <>
inline void lru_step<96>(const float* g, const float* v, const float* q,
                          float* h, float* y) {
    add_flops(static_cast<std::size_t>(96) * (kSigmoidFlops + 3));
    lru_step_neon_core<96>(g, v, q, h, y);
}

}  // namespace openeva::prim

#endif  // __ARM_NEON || __aarch64__
