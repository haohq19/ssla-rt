#pragma once

#include <algorithm>
#include <cstddef>

#include "openeva/prim/flop.h"

namespace openeva::prim {

/// y[i] += b[i]. Used for residual / skip-connection branches.
inline void add_inplace(int dim, float* y, const float* b) {
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) y[i] += b[i];
}

/// out[i] = a[i] + b[i]. Used when caller cannot mutate a or b in place.
inline void add(int dim, const float* a, const float* b, float* out) {
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) out[i] = a[i] + b[i];
}

/// y[i] = max(y[i], b[i]). Used by SparseMaxPool's per-cell merge.
/// Comparisons are NOT counted as FLOPs (fvcore convention).
inline void max_inplace(int dim, float* y, const float* b) {
    for (int i = 0; i < dim; ++i) y[i] = std::max(y[i], b[i]);
}

/// y[i] += s for scalar s. Counts dim FLOPs.
inline void add_scalar_inplace(int dim, float* y, float s) {
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) y[i] += s;
}

/// y[i] *= s for scalar s.
inline void mul_scalar_inplace(int dim, float* y, float s) {
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) y[i] *= s;
}

/// Compute mean over `count` contributions accumulated in `sum_feat`,
/// writing to `out_feat`. Used by DAGr's mean-aggregation pool. Counts
/// `dim` FLOPs (one mul-by-inv per channel; reciprocal is computed once).
inline void mean_from_sum(int dim, const float* sum_feat, int count, float* out_feat) {
    if (count <= 0) return;
    add_flops(static_cast<std::size_t>(dim));
    const float inv = 1.0f / static_cast<float>(count);
    for (int i = 0; i < dim; ++i) out_feat[i] = sum_feat[i] * inv;
}

/// sum_feat[i] += feat[i]. Used by DAGr pool's sum aggregation.
inline void sum_accum(int dim, float* sum_feat, const float* feat) {
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) sum_feat[i] += feat[i];
}

/// sum_feat[i] -= feat[i]. Used when DAGr pool removes a member's contribution.
inline void sum_subtract(int dim, float* sum_feat, const float* feat) {
    add_flops(static_cast<std::size_t>(dim));
    for (int i = 0; i < dim; ++i) sum_feat[i] -= feat[i];
}

}  // namespace openeva::prim
