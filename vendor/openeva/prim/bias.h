#pragma once

#include <cstddef>

#include "openeva/prim/flop.h"

namespace openeva::prim {

/// y[i] += bias[i]. No-op if `bias` is nullptr. Self-counts dim FLOPs.
/// Used after MV when caller wants explicit control over when bias lands
/// (e.g., after residual add). Most matvec callers should pass bias to
/// `matvec(...)` directly instead.
inline void add_bias(int dim, const float* bias, float* y) {
    if (!bias) return;
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) y[i] += bias[i];
}

}  // namespace openeva::prim
