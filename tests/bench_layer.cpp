// tests/bench_layer.cpp — per-layer and per-primitive microbench.
//
// Layers measured (all in stage 0 / stage 1, the per-event hot path):
//   L0_s0  = layer_forward_ct< 2, 12>  (stage 0 L0)
//   L1_s0  = layer_forward_ct<12, 12>  (stage 0 L1)
//   L0_s1  = layer_forward_ct<12, 24>  (stage 1 L0)
//   L1_s1  = layer_forward_ct<24, 24>  (stage 1 L1)
//
// Each layer is called N times at an INTERIOR pixel (so process_patch runs
// all 9 patches). Output is ns per call. Total of 4 numbers cross-check
// against §9.5 production data (stage_forward(0) = 1.83 µs, stage_forward(1)
// = 9.50 µs); per-stage = L0 + L1 + dispatch.
//
// Primitives measured in isolation:
//   lru_step<12>, lru_step<24>     — gate/value gating, has sigmoid
//   layernorm_ct<12>, layernorm_ct<24>  — mean + var + scale
//
// Pin to one core (taskset -c <id>) before running.

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "ssla_kernels.h"
#include "openeva/event.h"
#include "openeva/prim/elementwise.h"
#include "openeva/prim/layernorm.h"
#include "openeva/prim/rnn.h"

int main(int argc, char** argv) {
    const std::string weights_dir =
        (argc > 1) ? argv[1] : "/tmp/ssla_s_64x80";
    constexpr long long N_ITERS  = 200'000;
    constexpr long long N_WARMUP =  10'000;

    deploy::SslaSPipeline pipe;
    try {
        pipe.load(weights_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load failed: %s\n  use --weights-dir or pass path "
                              "as argv[1]\n", e.what());
        return 1;
    }
    pipe.reset();

    const int H = pipe.H();
    const int W = pipe.W();
    std::printf("Loaded weights from %s  (H=%d, W=%d)\n",
                weights_dir.c_str(), H, W);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> coord_x(2, W - 3);
    std::uniform_int_distribution<int> coord_y(2, H - 3);

    // Per-layer buffers — sized by the layer's IN/OUT.
    alignas(16) float buf_2 [ 2] = {0.1f, -0.5f};
    alignas(16) float buf_12[12], buf_12b[12];
    alignas(16) float buf_24[24], buf_24b[24];

    // Fill with deterministic non-zero data so lru_step / layernorm don't
    // hit degenerate paths.
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : buf_12)  v = dist(rng);
    for (auto& v : buf_12b) v = dist(rng);
    for (auto& v : buf_24)  v = dist(rng);
    for (auto& v : buf_24b) v = dist(rng);

    // Helper: time a callable across N iters, return ns/call.
    auto time_fn = [&](auto fn, long long n) -> double {
        auto t0 = std::chrono::steady_clock::now();
        for (long long i = 0; i < n; ++i) fn(i);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count() / n;
    };

    // ---- Layer-level bench --------------------------------------------------
    // Each layer accesses hidden state at the layer's stage resolution.
    // For SSLA-S: stage 0 = H × W, stage 1 = H/2 × W/2, etc. We must pass
    // coords AT THE STAGE RESOLUTION so the 3×3 patch loop sees interior
    // pixels (otherwise the bounds-clipping fallback runs 0 patches and the
    // measurement becomes meaningless).
    struct LayerInfo {
        int  idx;
        int  in_dim;
        int  out_dim;
        int  hl;             // hidden-state H at this layer's stage
        int  wl;             // hidden-state W at this layer's stage
        const float* feat_in;
        float* feat_out;
        const char* name;
    } layers[] = {
        {0,  2, 12, H,     W,     buf_2,   buf_12,  "L0 s0  <2 -> 12>"},
        {1, 12, 12, H,     W,     buf_12,  buf_12b, "L1 s0  <12->12>"},
        {2, 12, 24, H / 2, W / 2, buf_12,  buf_24,  "L0 s1  <12->24>"},
        {3, 24, 24, H / 2, W / 2, buf_24,  buf_24b, "L1 s1  <24->24>"},
    };

    std::printf("\n=== per-layer (interior pixel, 9 patches each call) ===\n");
    std::printf("  %-18s  %5s %5s  %10s\n", "layer", "Wl", "Hl", "ns/call");
    for (auto& L : layers) {
        // Warm up.
        for (long long i = 0; i < N_WARMUP; ++i) {
            const int x = 2 + (int)(i % (L.wl - 4));
            const int y = 2 + (int)((i / L.wl) % (L.hl - 4));
            pipe.layer_forward(L.idx, x, y, L.feat_in, L.feat_out);
        }
        const double ns = time_fn([&](long long i) {
            // Vary coords each call so we touch fresh hidden-state cells —
            // avoids unrealistically hot single-cell access pattern.
            const int x = 2 + (int)(i % (L.wl - 4));
            const int y = 2 + (int)((i / L.wl) % (L.hl - 4));
            pipe.layer_forward(L.idx, x, y, L.feat_in, L.feat_out);
        }, N_ITERS);
        std::printf("  %-18s  %5d %5d  %10.1f\n", L.name, L.wl, L.hl, ns);
    }

    // ---- Primitive bench ----------------------------------------------------
    // lru_step<OUT>(g, v, q, h_state, qh_out). Inputs are pre-projected gate,
    // value, query; h_state is the recurrent cell. We measure the per-call
    // cost on dummy data at the production OUT=12 and OUT=24 sizes.
    std::printf("\n=== primitives ===\n");
    std::printf("  %-22s  %10s\n", "primitive", "ns/call");

    alignas(16) float g12[12], v12[12], q12[12], h12[12], qh12[12];
    alignas(16) float g24[24], v24[24], q24[24], h24[24], qh24[24];
    for (auto& v : g12) v = dist(rng);
    for (auto& v : v12) v = dist(rng);
    for (auto& v : q12) v = dist(rng);
    for (auto& v : h12) v = dist(rng) * 0.1f;
    for (auto& v : g24) v = dist(rng);
    for (auto& v : v24) v = dist(rng);
    for (auto& v : q24) v = dist(rng);
    for (auto& v : h24) v = dist(rng) * 0.1f;

    {
        for (long long i = 0; i < N_WARMUP; ++i)
            openeva::prim::lru_step<12>(g12, v12, q12, h12, qh12);
        const double ns = time_fn([&](long long i) {
            openeva::prim::lru_step<12>(g12, v12, q12, h12, qh12);
            (void)i;
        }, N_ITERS * 10);
        std::printf("  %-22s  %10.1f\n", "lru_step<12>", ns);
    }
    {
        for (long long i = 0; i < N_WARMUP; ++i)
            openeva::prim::lru_step<24>(g24, v24, q24, h24, qh24);
        const double ns = time_fn([&](long long i) {
            openeva::prim::lru_step<24>(g24, v24, q24, h24, qh24);
            (void)i;
        }, N_ITERS * 10);
        std::printf("  %-22s  %10.1f\n", "lru_step<24>", ns);
    }

    // layernorm_ct<DIM>(y, gamma, beta) — in-place; needs gamma/beta buffers.
    alignas(16) float ln_g12[12], ln_b12[12], ln_y12[12];
    alignas(16) float ln_g24[24], ln_b24[24], ln_y24[24];
    for (auto& v : ln_g12) v = 1.0f + dist(rng) * 0.1f;
    for (auto& v : ln_b12) v = dist(rng) * 0.1f;
    for (auto& v : ln_g24) v = 1.0f + dist(rng) * 0.1f;
    for (auto& v : ln_b24) v = dist(rng) * 0.1f;
    // layernorm needs a sink to prevent DCE.
    float sink_ln = 0.f;
    {
        for (long long i = 0; i < N_WARMUP; ++i) {
            for (int j = 0; j < 12; ++j) ln_y12[j] = dist(rng);
            openeva::prim::layernorm_ct<12>(ln_y12, ln_g12, ln_b12);
        }
        const double ns = time_fn([&](long long i) {
            // Re-randomise y each call so the layernorm doesn't see all-equal
            // input (where var = 0 → degenerate path). Read after to sink.
            ln_y12[i & 0xf] += 0.001f;
            openeva::prim::layernorm_ct<12>(ln_y12, ln_g12, ln_b12);
            sink_ln += ln_y12[i & 0xf];
        }, N_ITERS * 10);
        std::printf("  %-22s  %10.1f\n", "layernorm_ct<12>", ns);
    }
    {
        for (long long i = 0; i < N_WARMUP; ++i) {
            for (int j = 0; j < 24; ++j) ln_y24[j] = dist(rng);
            openeva::prim::layernorm_ct<24>(ln_y24, ln_g24, ln_b24);
        }
        const double ns = time_fn([&](long long i) {
            ln_y24[i & 0x1f] += 0.001f;
            openeva::prim::layernorm_ct<24>(ln_y24, ln_g24, ln_b24);
            sink_ln += ln_y24[i & 0x1f];
        }, N_ITERS * 10);
        std::printf("  %-22s  %10.1f\n", "layernorm_ct<24>", ns);
    }
    std::printf("\n(sink: %g  — ignore)\n", sink_ln);

    return 0;
}
