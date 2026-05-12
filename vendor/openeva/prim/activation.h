#pragma once

#include <cmath>
#include <cstddef>

#include "openeva/prim/flop.h"

namespace openeva::prim {

// ---- scalar variants (NOT self-counting; loop callers count via array variants) ----
//
// `#pragma omp declare simd` tells the compiler to emit a SIMD clone so callers
// inside `#pragma omp simd` loops vectorize the expf / tanhf via libmvec under
// -fopenmp-simd + -ffast-math.

#pragma omp declare simd
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

#pragma omp declare simd
inline float silu(float x) { return x * sigmoid(x); }

#pragma omp declare simd
inline float tanh_f(float x) { return std::tanh(x); }

#pragma omp declare simd
inline float elu(float x) { return x > 0.0f ? x : (std::exp(x) - 1.0f); }

#pragma omp declare simd
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// ---- array-inplace variants (these self-count) ----

inline void sigmoid_inplace(float* x, int dim) {
    add_flops(static_cast<std::size_t>(dim) * kSigmoidFlops);
    #pragma omp simd
    for (int i = 0; i < dim; ++i) x[i] = sigmoid(x[i]);
}

inline void silu_inplace(float* x, int dim) {
    add_flops(static_cast<std::size_t>(dim) * (kSigmoidFlops + 1));   // sigmoid + mul
    #pragma omp simd
    for (int i = 0; i < dim; ++i) x[i] = silu(x[i]);
}

inline void tanh_inplace(float* x, int dim) {
    add_flops(static_cast<std::size_t>(dim) * kTanhFlops);
    #pragma omp simd
    for (int i = 0; i < dim; ++i) x[i] = tanh_f(x[i]);
}

inline void elu_inplace(float* x, int dim) {
    add_flops(static_cast<std::size_t>(dim) * (kExpFlops + 1));        // exp + sub
    #pragma omp simd
    for (int i = 0; i < dim; ++i) x[i] = elu(x[i]);
}

inline void relu_inplace(float* x, int dim) {
    add_flops(static_cast<std::size_t>(dim));                          // compare-select
    #pragma omp simd
    for (int i = 0; i < dim; ++i) x[i] = relu(x[i]);
}

template <int DIM> inline void sigmoid_inplace_ct(float* x) {
    add_flops(DIM * kSigmoidFlops);
    #pragma omp simd
    for (int i = 0; i < DIM; ++i) x[i] = sigmoid(x[i]);
}

template <int DIM> inline void silu_inplace_ct(float* x) {
    add_flops(DIM * (kSigmoidFlops + 1));
    #pragma omp simd
    for (int i = 0; i < DIM; ++i) x[i] = silu(x[i]);
}

template <int DIM> inline void tanh_inplace_ct(float* x) {
    add_flops(DIM * kTanhFlops);
    #pragma omp simd
    for (int i = 0; i < DIM; ++i) x[i] = tanh_f(x[i]);
}

template <int DIM> inline void elu_inplace_ct(float* x) {
    add_flops(DIM * (kExpFlops + 1));
    #pragma omp simd
    for (int i = 0; i < DIM; ++i) x[i] = elu(x[i]);
}

template <int DIM> inline void relu_inplace_ct(float* x) {
    add_flops(DIM);
    #pragma omp simd
    for (int i = 0; i < DIM; ++i) x[i] = relu(x[i]);
}

}  // namespace openeva::prim
