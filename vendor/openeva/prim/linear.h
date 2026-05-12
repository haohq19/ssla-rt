#pragma once

#include <cstddef>

#include "openeva/prim/flop.h"

namespace openeva::prim {

/// y = W @ x + bias. Row-major W shape (out, in). `bias` may be nullptr.
/// Self-counts IN*OUT MACs.
///
/// fp32 accumulator (no double upcast) so GCC emits AVX2/AVX-512 FMA on
/// the inner loop. Matches PyTorch's CPU kernels exactly.
inline void matvec(const float* x, int in_dim,
                   const float* W, int out_dim,
                   const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(in_dim) * out_dim);
    for (int o = 0; o < out_dim; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * in_dim;
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_dim; ++i) acc += w[i] * x[i];
        y[o] = acc;
    }
}

/// y[o] += W @ x. No bias. Used by SSLA's MOS attention recurrence
/// (multiple per-position contributions accumulating into a shared output).
inline void matvec_accum(const float* x, int in_dim,
                         const float* W, int out_dim,
                         float* y) {
    add_macs(static_cast<std::size_t>(in_dim) * out_dim);
    for (int o = 0; o < out_dim; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * in_dim;
        float acc = 0.0f;
        for (int i = 0; i < in_dim; ++i) acc += w[i] * x[i];
        y[o] += acc;
    }
}

/// y[o] += W @ x + bias[o]. `bias` may be nullptr. Used by DAGr's skip path
/// where the residual (skip) projection accumulates into the main path's
/// already-populated output.
inline void matvec_accum_with_bias(const float* x, int in_dim,
                                    const float* W, int out_dim,
                                    const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(in_dim) * out_dim);
    if (bias) add_flops(static_cast<std::size_t>(out_dim));
    for (int o = 0; o < out_dim; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * in_dim;
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_dim; ++i) acc += w[i] * x[i];
        y[o] += acc;
    }
}

/// y[o] += Σ_i W[o, i] · (a[i] - b[i]). Computes the delta of a linear
/// projection without materializing the (a-b) buffer. Used by DAGr's
/// delta-path when only a small fraction of the input has changed since
/// the previous step, so we add Δy = W·Δx into a cached y rather than
/// recomputing W·a from scratch.
///
/// Counts in_dim*out_dim MACs (the FMA per (i, o)) plus in_dim*out_dim
/// FLOPs for the per-iteration (a-b) subtractions.
inline void matvec_diff_accum(const float* a, const float* b, int in_dim,
                               const float* W, int out_dim, float* y) {
    add_macs(static_cast<std::size_t>(in_dim) * out_dim);
    add_flops(static_cast<std::size_t>(in_dim) * out_dim);
    for (int o = 0; o < out_dim; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * in_dim;
        float acc = 0.0f;
        for (int i = 0; i < in_dim; ++i) acc += w[i] * (a[i] - b[i]);
        y[o] += acc;
    }
}

/// Compile-time-dim variant. Preserves SSLA-S/M's IN_DIM=12/24 unrolling
/// and SIMD-across-OUT-axis vectorization that GCC produces when both
/// dimensions are constant-folded. Calling this with a small IN often
/// produces a fully-unrolled loop with broadcast + FMA on weight columns.
template <int IN, int OUT>
inline void matvec_ct(const float* x, const float* W,
                      const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(IN) * OUT);
    for (int o = 0; o < OUT; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * IN;
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < IN; ++i) acc += w[i] * x[i];
        y[o] = acc;
    }
}

template <int IN, int OUT>
inline void matvec_accum_ct(const float* x, const float* W, float* y) {
    add_macs(static_cast<std::size_t>(IN) * OUT);
    for (int o = 0; o < OUT; ++o) {
        const float* w = W + static_cast<std::ptrdiff_t>(o) * IN;
        float acc = 0.0f;
        for (int i = 0; i < IN; ++i) acc += w[i] * x[i];
        y[o] += acc;
    }
}

}  // namespace openeva::prim
