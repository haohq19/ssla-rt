#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "openeva/prim/flop.h"

namespace openeva::prim {

/// Small dense GEMM: C = B @ A. Self-counts b_out * b_in * a_in MACs.
///
/// Currently called only at weight-load time (SSLA's QVG packing). Per-event
/// MAC reporting uses a read-delta protocol (read counter before & after each
/// step()), so load-time mm contributions land outside any event window and
/// don't pollute per-event numbers — but if a future caller uses mm inside
/// step(), the cost shows up in that event's count automatically. No silent
/// holes.
///
/// Shapes: B (b_out, b_in), A (a_out=b_in, a_in), C (b_out, a_in).
inline std::vector<float>
mm(const std::vector<float>& B, int b_out, int b_in,
   const std::vector<float>& A, int a_out, int a_in) {
    if (b_in != a_out) throw std::runtime_error("prim::mm: shape mismatch");
    add_macs(static_cast<std::size_t>(b_out)
             * static_cast<std::size_t>(b_in)
             * static_cast<std::size_t>(a_in));
    std::vector<float> C(static_cast<std::size_t>(b_out) * a_in, 0.0f);
    for (int o = 0; o < b_out; ++o) {
        const float* br = B.data() + static_cast<std::ptrdiff_t>(o) * b_in;
        float*       cr = C.data() + static_cast<std::ptrdiff_t>(o) * a_in;
        for (int k = 0; k < b_in; ++k) {
            const float bv = br[k];
            const float* ar = A.data() + static_cast<std::ptrdiff_t>(k) * a_in;
            for (int i = 0; i < a_in; ++i) cr[i] += bv * ar[i];
        }
    }
    return C;
}

}  // namespace openeva::prim
