#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "openeva/prim/flop.h"

namespace openeva::prim {

/// Pre-fused BatchNorm: BN(x) = scale * x + shift, where
///   scale = gamma / sqrt(var + eps)
///   shift = beta  - gamma * mean / sqrt(var + eps)
struct FusedBN {
    std::vector<float> scale;
    std::vector<float> shift;
};

/// Compute (scale, shift) from (gamma, beta, running_mean, running_var, eps).
/// Used at weight-load time. Self-counts dim * 5 FLOPs (1 add, 1 sqrt, 1 div
/// for the inverse + 2 muls for scale and shift formulas). Per-event MAC
/// reporting uses a read-delta protocol so load-time contributions land
/// outside any event window — but instrumenting now means a future hot-path
/// caller (e.g. online BN) is correctly counted with no silent hole.
inline FusedBN fuse_bn(const float* gamma, const float* beta,
                       const float* mean,  const float* var,
                       int dim, float eps = 1e-5f) {
    add_flops(static_cast<std::size_t>(dim) * 5);
    FusedBN out;
    out.scale.resize(dim);
    out.shift.resize(dim);
    for (int i = 0; i < dim; ++i) {
        const float inv = 1.0f / std::sqrt(var[i] + eps);
        out.scale[i] = gamma[i] * inv;
        out.shift[i] = beta[i] - gamma[i] * mean[i] * inv;
    }
    return out;
}

/// Fold per-channel BN scale into a 2D row-major weight (out, in):
///   W'[o, i] = W[o, i] * scale[o]
/// Also produces fused_bias[o] = (b_conv[o] or 0) * scale[o] + shift[o].
/// `b_conv_or_null` may be empty/nullptr; treated as zero.
/// Self-counts out_dim * in_dim FLOPs for the weight muls plus 2 * out_dim
/// FLOPs for the fused_bias formula.
inline void fold_bn_into_weight_2d(float* W, int out_dim, int in_dim,
                                    const float* scale,
                                    const float* shift,
                                    const float* b_conv_or_null,
                                    float* fused_bias) {
    add_flops(static_cast<std::size_t>(out_dim) * in_dim
              + static_cast<std::size_t>(out_dim) * 2);
    for (int o = 0; o < out_dim; ++o) {
        const float s = scale[o];
        for (int i = 0; i < in_dim; ++i) {
            W[static_cast<std::ptrdiff_t>(o) * in_dim + i] *= s;
        }
        const float bc = b_conv_or_null ? b_conv_or_null[o] : 0.0f;
        fused_bias[o] = bc * s + shift[o];
    }
}

}  // namespace openeva::prim
