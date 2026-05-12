#pragma once

#include <cstddef>

#include "openeva/prim/activation.h"
#include "openeva/prim/flop.h"

namespace openeva::prim {

/// PyTorch LSTMCell, gate order [i, f, g, o]. Strict numerical match.
///
/// Shapes:
///   x        (in_dim,)
///   h, c     (hid_dim,)         — read AND written (recurrent state)
///   W_ih     (4*hid_dim, in_dim)
///   W_hh     (4*hid_dim, hid_dim)
///   b_ih     (4*hid_dim,)
///   b_hh     (4*hid_dim,)
///   scratch  (4*hid_dim,)        — caller-owned scratch buffer
///
/// Cost (with kSigmoid = kTanh = 1):
///   MACs:  4*hid * (in_dim + hid_dim)            (two MVs)
///   FLOPs: 4*hid                                  (b_ih + b_hh seed)
///        + hid * (3*kSigmoid + 2*kTanh + 5)       (gate activations + cell update)
inline void lstm_cell(const float* x, int in_dim, int hid_dim,
                      const float* W_ih, const float* b_ih,
                      const float* W_hh, const float* b_hh,
                      float* h, float* c,
                      float* scratch_4h) {
    const int N = 4 * hid_dim;
    add_macs(static_cast<std::size_t>(N) * (in_dim + hid_dim));
    add_flops(static_cast<std::size_t>(N)
              + static_cast<std::size_t>(hid_dim) *
                (3 * kSigmoidFlops + 2 * kTanhFlops + 5));

    // scratch = b_ih + b_hh
    for (int k = 0; k < N; ++k) scratch_4h[k] = b_ih[k] + b_hh[k];
    // scratch += W_ih @ x
    for (int o = 0; o < N; ++o) {
        const float* w = W_ih + static_cast<std::ptrdiff_t>(o) * in_dim;
        float acc = 0.0f;
        for (int i = 0; i < in_dim; ++i) acc += w[i] * x[i];
        scratch_4h[o] += acc;
    }
    // scratch += W_hh @ h
    for (int o = 0; o < N; ++o) {
        const float* w = W_hh + static_cast<std::ptrdiff_t>(o) * hid_dim;
        float acc = 0.0f;
        for (int i = 0; i < hid_dim; ++i) acc += w[i] * h[i];
        scratch_4h[o] += acc;
    }
    const float* gi = scratch_4h + 0 * hid_dim;
    const float* gf = scratch_4h + 1 * hid_dim;
    const float* gg = scratch_4h + 2 * hid_dim;
    const float* go = scratch_4h + 3 * hid_dim;
    #pragma omp simd
    for (int k = 0; k < hid_dim; ++k) {
        const float i_g = sigmoid(gi[k]);
        const float f_g = sigmoid(gf[k]);
        const float g_t = tanh_f(gg[k]);
        const float o_g = sigmoid(go[k]);
        c[k] = f_g * c[k] + i_g * g_t;
        h[k] = o_g * tanh_f(c[k]);
    }
}

/// Linear Recurrent Unit (LRU) step, elementwise.
///
/// Per-channel gated linear recurrence with sigmoid gate:
///     h_c ← σ(g_c) * h_c + v_c
///     y_c ← q_c * h_c
///
/// Generic kernel — also the inner update of GLA / RWKV / Mamba-S6
/// (without discretization) and SSLA's MOS attention. Caller supplies
/// the input and output projections.
///
/// Shapes: g, v, q, y are (DIM,); h is (DIM,) read AND written.
/// Cost: DIM * (kSigmoidFlops + 3) FLOPs (no MACs).
template <int DIM>
inline void lru_step(const float* g, const float* v, const float* q,
                     float* h, float* y) {
    add_flops(static_cast<std::size_t>(DIM) * (kSigmoidFlops + 3));
    #pragma omp simd
    for (int c = 0; c < DIM; ++c) {
        const float gc = sigmoid(g[c]);
        const float hc = gc * h[c] + v[c];
        h[c] = hc;
        y[c] = q[c] * hc;
    }
}

}  // namespace openeva::prim
