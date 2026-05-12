#pragma once

// spline.h — torch_spline_conv-equivalent B-spline basis evaluation +
// generic SplineConv message sum, parameterized over kernel size K and
// edge-attr dimensionality.
//
// Method-agnostic per CLAUDE.md §11.2: any GNN method using PyG's
// `torch_geometric.nn.conv.SplineConv` (DAGr, AEGNN, future NVS) gets
// the same basis math regardless of (K, DIM, root, bias, self-edge)
// architecture choices. Method-specific concerns — LUT pre-fusion,
// root-term handling, self-edge handling, BatchNorm/ELU fusing — stay
// in `cpp/methods/<name>/`.
//
// Convention: torch_spline_conv `open=True, degree=1`. Each axis of
// edge_attr ∈ [0, 1] is mapped to two adjacent knots of a K-knot grid
// with bilinear weights summing to 1; the kD evaluation is the tensor
// product of D 1D evaluations, yielding 2^D hits per edge.
//
// Linear index follows torch_spline_conv's "first dim innermost" stride:
//   2D: k = v0 * K + u0
//   3D: k = w0 * K² + v0 * K + u0
// Caller's W tensor must use the matching layout `(K^DIM, in, out)`
// row-major.

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "openeva/prim/flop.h"

namespace openeva::prim {

// One bilinear / trilinear hit: linear kernel index + product weight.
// At kernel boundaries some hits have w == 0 — caller skips them.
struct SplineHit { int k; float w; };

// 1D B-spline degree=1 basis at a ∈ [0, 1] over K knots.
// Returns two hits `(knot_index, weight)`.
//
//   u  = a · (K − 1)
//   u0 = clamp(floor(u), 0, K − 2)
//   fu = u − u0
//   hits = { (u0, 1 − fu), (u0 + 1, fu) }
//
// At a = 1 exactly, u = K − 1, u0 = K − 1; we clamp u0 to K − 2 so
// (u0 + 1) is always a valid kernel index. The math still gives the
// correct result because fu = 0 → the high hit's weight is 0.
template <int K>
inline std::array<std::pair<int, float>, 2> spline_basis_1d(float a) {
    static_assert(K >= 2, "spline_basis_1d requires K >= 2");
    constexpr float scale = static_cast<float>(K - 1);
    const float u  = std::clamp(a * scale, 0.0f, scale);
    const int   u0 = std::min(static_cast<int>(u), K - 2);
    const float fu = u - static_cast<float>(u0);
    return {{ {u0, 1.0f - fu}, {u0 + 1, fu} }};
}

// 2D bilinear basis at (ax, ay) over a K×K kernel.
// Linear index: k = v0 · K + u0.
template <int K>
inline std::array<SplineHit, 4> spline_basis_2d(float ax, float ay) {
    const auto hu = spline_basis_1d<K>(ax);
    const auto hv = spline_basis_1d<K>(ay);
    return {{
        { hv[0].first * K + hu[0].first, hu[0].second * hv[0].second },
        { hv[0].first * K + hu[1].first, hu[1].second * hv[0].second },
        { hv[1].first * K + hu[0].first, hu[0].second * hv[1].second },
        { hv[1].first * K + hu[1].first, hu[1].second * hv[1].second },
    }};
}

// 3D trilinear basis at (ax, ay, az) over a K³ kernel.
// Linear index: k = (w0 · K + v0) · K + u0.
template <int K>
inline std::array<SplineHit, 8> spline_basis_3d(float ax, float ay, float az) {
    const auto hu = spline_basis_1d<K>(ax);
    const auto hv = spline_basis_1d<K>(ay);
    const auto hw = spline_basis_1d<K>(az);
    std::array<SplineHit, 8> out{};
    int idx = 0;
    for (int kw = 0; kw < 2; ++kw)
    for (int kv = 0; kv < 2; ++kv)
    for (int ku = 0; ku < 2; ++ku) {
        out[idx++] = {
            (hw[kw].first * K + hv[kv].first) * K + hu[ku].first,
            hu[ku].second * hv[kv].second * hw[kw].second,
        };
    }
    return out;
}

// SplineConv message sum over a list of incoming edges (2D edge_attr).
//
//   out += Σ_e Σ_{4 hits} hit.w · W[hit.k, :, :] · feat_of(e.src)
//
// Matches PyG `SplineConv(aggr='add', degree=1)` per-message contribution
// (root + bias are caller's responsibility — they are method-specific).
//
// `EdgeT` requirements (duck-typed): `.src` (int), `.ax, .ay` (float).
// Each method defines its own EdgeT struct (§11.1), this template just
// reads the fields.
//
// `W` is `(K², in_dim, out_dim)` row-major.
// `out` is (out_dim) — ACCUMULATED into, not overwritten.
//
// Self-counts MACs per non-zero hit.
template <int K, typename EdgeT, typename SrcFeatFn>
inline void spline_message_sum_2d(
    int in_dim, int out_dim,
    const EdgeT* edges, int n_edges,
    SrcFeatFn&&  feat_of_src,
    const float* W,
    float*       out)
{
    for (int e = 0; e < n_edges; ++e) {
        const float* xs   = feat_of_src(edges[e].src);
        const auto   hits = spline_basis_2d<K>(edges[e].ax, edges[e].ay);
        for (const auto& hit : hits) {
            if (hit.w == 0.0f) continue;
            add_macs(static_cast<std::size_t>(in_dim) * out_dim);
            const float* Wk = W
                + static_cast<std::ptrdiff_t>(hit.k) * in_dim * out_dim;
            for (int i = 0; i < in_dim; ++i) {
                const float scale = hit.w * xs[i];
                for (int o = 0; o < out_dim; ++o) {
                    out[o] += scale * Wk[i * out_dim + o];
                }
            }
        }
    }
}

// 3D variant. `EdgeT` additionally requires `.az` (float).
// `W` is `(K³, in_dim, out_dim)` row-major.
template <int K, typename EdgeT, typename SrcFeatFn>
inline void spline_message_sum_3d(
    int in_dim, int out_dim,
    const EdgeT* edges, int n_edges,
    SrcFeatFn&&  feat_of_src,
    const float* W,
    float*       out)
{
    for (int e = 0; e < n_edges; ++e) {
        const float* xs   = feat_of_src(edges[e].src);
        const auto   hits = spline_basis_3d<K>(
            edges[e].ax, edges[e].ay, edges[e].az);
        for (const auto& hit : hits) {
            if (hit.w == 0.0f) continue;
            add_macs(static_cast<std::size_t>(in_dim) * out_dim);
            const float* Wk = W
                + static_cast<std::ptrdiff_t>(hit.k) * in_dim * out_dim;
            for (int i = 0; i < in_dim; ++i) {
                const float scale = hit.w * xs[i];
                for (int o = 0; o < out_dim; ++o) {
                    out[o] += scale * Wk[i * out_dim + o];
                }
            }
        }
    }
}

}  // namespace openeva::prim
