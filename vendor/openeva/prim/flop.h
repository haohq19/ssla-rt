#pragma once

#include <cstddef>

namespace openeva::prim {

/// Per-thread MAC and FLOP counters for per-event accounting.
///
/// Convention (matches fvcore / thop):
///   - `add_macs(n)`  : multiply-accumulates in matmul / conv / spline scatter.
///                      Reported as the headline "MACs" number, comparable to
///                      papers and hardware vendor specs.
///   - `add_flops(n)` : every other floating-point op (layernorm, activations,
///                      bias, residual, sqrt, div, pool reductions).
///
/// Total FLOPs = 2 * MACs + other_FLOPs (one MAC = 1 mul + 1 add).
/// Comparisons (max/min/select), integer ops, and control flow are NOT counted.
///
/// `inline thread_local` ⇒ one TLS slot per thread across the whole program
/// (C++17 inline variable rule), so multiple TUs share the counter.
inline thread_local std::size_t g_macs  = 0;
inline thread_local std::size_t g_flops = 0;

inline void add_macs(std::size_t n)  { g_macs  += n; }
inline void add_flops(std::size_t n) { g_flops += n; }

inline std::size_t read_macs()  { return g_macs;  }
inline std::size_t read_flops() { return g_flops; }
inline void reset_counters()    { g_macs = 0; g_flops = 0; }

/// Conventional FLOP cost of transcendentals (per element).
/// fvcore convention: 1 each. Override at compile-time with -D if needed.
#ifndef OPENEVA_FLOPS_SIGMOID
#define OPENEVA_FLOPS_SIGMOID 1
#endif
#ifndef OPENEVA_FLOPS_TANH
#define OPENEVA_FLOPS_TANH    1
#endif
#ifndef OPENEVA_FLOPS_EXP
#define OPENEVA_FLOPS_EXP     1
#endif

inline constexpr std::size_t kSigmoidFlops = OPENEVA_FLOPS_SIGMOID;
inline constexpr std::size_t kTanhFlops    = OPENEVA_FLOPS_TANH;
inline constexpr std::size_t kExpFlops     = OPENEVA_FLOPS_EXP;

}  // namespace openeva::prim
