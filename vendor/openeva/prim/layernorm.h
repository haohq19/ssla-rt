#pragma once

#include <cmath>
#include <cstddef>

#include "openeva/prim/flop.h"

namespace openeva::prim {

/// In-place LayerNorm. fp64 mean/var, fp32 affine — matches PyTorch's CPU
/// kernel (fp64 reduction prevents catastrophic cancellation on small dims).
///
/// FLOP breakdown (transcendentals counted as 1 each):
///   mean:   DIM         (sum + 1 div)
///   var:    3*DIM + 1   (DIM subs, DIM squares, DIM sums, 1 div)
///   inv:    2           (1 sqrt, 1 div)
///   norm:   4*DIM       (sub, mul-by-inv, mul gamma, add beta)
///   total:  8*DIM + 3
inline void layernorm(float* x, int dim,
                      const float* gamma, const float* beta,
                      float eps = 1e-5f) {
    add_flops(static_cast<std::size_t>(8) * dim + 3);
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) sum += static_cast<double>(x[i]);
    const double mean = sum / static_cast<double>(dim);
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double d = static_cast<double>(x[i]) - mean;
        var += d * d;
    }
    var /= static_cast<double>(dim);
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
    for (int i = 0; i < dim; ++i) {
        const double v = (static_cast<double>(x[i]) - mean) * inv;
        x[i] = static_cast<float>(v * static_cast<double>(gamma[i])
                                  + static_cast<double>(beta[i]));
    }
}

template <int DIM>
inline void layernorm_ct(float* x,
                         const float* gamma, const float* beta,
                         float eps = 1e-5f) {
    add_flops(8 * DIM + 3);
    double sum = 0.0;
    for (int i = 0; i < DIM; ++i) sum += static_cast<double>(x[i]);
    const double mean = sum / static_cast<double>(DIM);
    double var = 0.0;
    for (int i = 0; i < DIM; ++i) {
        const double d = static_cast<double>(x[i]) - mean;
        var += d * d;
    }
    var /= static_cast<double>(DIM);
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
    for (int i = 0; i < DIM; ++i) {
        const double v = (static_cast<double>(x[i]) - mean) * inv;
        x[i] = static_cast<float>(v * static_cast<double>(gamma[i])
                                  + static_cast<double>(beta[i]));
    }
}

}  // namespace openeva::prim
