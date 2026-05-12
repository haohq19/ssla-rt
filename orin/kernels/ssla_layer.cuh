// SSLA-S layer forward — device-side per-event step for one SSLA module.
//
// Mirrors openeva/cpp/methods/ssla/ssla_detection_yolox.cpp::Impl::
// ssla_layer_forward_ct exactly, modulo (a) FMA ordering inside the
// matvec primitives, and (b) sigmoidf using __expf when --use_fast_math
// is on (still under 1-ULP per #10's tolerance).
//
// Inputs (device pointers, all float32 row-major):
//   in_feat     (IN_DIM,)           per-event input features
//                                    (any addressable space)
//   qvgIn       (A, 3·OUT_DIM, IN)  reshaped per #17 (weights_ssla)
//   goW         (A, OUT_DIM, OUT_DIM)
//   input_proj  (OUT_DIM, IN_DIM)   may be NULL (devptr == 0) — passes
//                                    in_feat through if IN==OUT
//   ln_gamma    (OUT_DIM,)
//   ln_beta     (OUT_DIM,)
//   H_all       (Hl·Wl, OUT_DIM)    per-(layer, pixel) LRU hidden state
//                                    (managed memory)
//   scratch     (≥ 5·OUT_DIM,)      shared-memory scratch — the layer
//                                    uses 2·OUT_DIM (residual, qh) plus
//                                    3·OUT_DIM (qvg). Caller supplies a
//                                    single shared buffer of width
//                                    SSLA_LAYER_SCRATCH_FLOATS(OUT_DIM).
// Outputs:
//   out_feat    (OUT_DIM,) shared   post-LayerNorm per-event feature
//
// The implementation expects a single warp (32 threads). Every primitive
// is warp-cooperative (see ssla_primitives.cuh). Patches in the 9-cell
// neighborhood are processed serially within the warp — order matches
// cpp's reference exactly so accumulation is bit-stable.

#pragma once

#define SSLA_PRIM_NO_TEST_WRAPPERS
#include "ssla_primitives.cuh"

// scratch layout: residual[OUT] | qh[OUT] | qvg[3*OUT]
#define SSLA_LAYER_SCRATCH_FLOATS(OUT) (5 * (OUT))

template <int IN_DIM, int OUT_DIM, int K>
__device__ inline void ssla_layer_forward_ct(
    int ev_x, int ev_y,
    int Hl,   int Wl,
    const float* __restrict__ in_feat,
    const float* __restrict__ qvgIn,
    const float* __restrict__ goW,
    const float* __restrict__ input_proj,
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float*       __restrict__ H_all,
    float*       __restrict__ out_feat,
    float*       __restrict__ scratch)
{
    constexpr int A          = K * K;
    constexpr int QVG_STRIDE = 3 * OUT_DIM * IN_DIM;
    constexpr int GOW_STRIDE = OUT_DIM * OUT_DIM;

    float* residual = scratch;
    float* qh       = scratch + OUT_DIM;
    float* qvg      = scratch + 2 * OUT_DIM;   // size 3·OUT_DIM

    // Residual: matvec when input_proj is provided; otherwise pass
    // through (only valid at runtime when IN_DIM == OUT_DIM).
    if constexpr (IN_DIM == OUT_DIM) {
        if (input_proj != 0) {
            matvec_ct<IN_DIM, OUT_DIM>(in_feat, input_proj, (const float*)0, residual);
        } else {
            for (int i = block_tid_(); i < OUT_DIM; i += SSLA_BLOCK_THREADS)
                residual[i] = in_feat[i];
            __syncthreads();
        }
    } else {
        matvec_ct<IN_DIM, OUT_DIM>(in_feat, input_proj, (const float*)0, residual);
    }

    // Zero output before per-patch accumulation.
    for (int i = block_tid_(); i < OUT_DIM; i += SSLA_BLOCK_THREADS)
        out_feat[i] = 0.0f;
    __syncthreads();

    // process_patch: matvec(qvgIn[delta], in_feat) → qvg → split q/v/g
    //                LRU step on H_all + patch_idx*OUT_DIM
    //                out_feat += matvec(goW[delta], qh)
    auto process_patch = [&] (int patch_idx, int delta) {
        const float* qvg_w = qvgIn + (long long)delta * QVG_STRIDE;
        const float* go_w  = goW   + (long long)delta * GOW_STRIDE;
        matvec_ct<IN_DIM, 3 * OUT_DIM>(in_feat, qvg_w, (const float*)0, qvg);
        const float* q = qvg;
        const float* v = qvg +     OUT_DIM;
        const float* g = qvg + 2 * OUT_DIM;
        float* h_ptr = H_all + (long long)patch_idx * OUT_DIM;
        lru_step_ct<OUT_DIM>(g, v, q, h_ptr, qh);
        matvec_accum_ct<OUT_DIM, OUT_DIM>(qh, go_w, out_feat);
    };

    if (A == 1) {
        process_patch(ev_y * Wl + ev_x, 0);
    } else {
        const int base = ev_y * Wl + ev_x;
        const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                           && (ev_y > 0) && (ev_y + 1 < Hl);
        if (interior) {
            // Identical delta ordering to cpp (delta=8 first, ..., delta=0 last).
            process_patch(base - Wl - 1, 8);
            process_patch(base - Wl    , 7);
            process_patch(base - Wl + 1, 6);
            process_patch(base      - 1, 5);
            process_patch(base         , 4);
            process_patch(base      + 1, 3);
            process_patch(base + Wl - 1, 2);
            process_patch(base + Wl    , 1);
            process_patch(base + Wl + 1, 0);
        } else {
            for (int dy = -1; dy <= 1; ++dy) {
                const int py = ev_y + dy;
                if (py < 0 || py >= Hl) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int px = ev_x + dx;
                    if (px < 0 || px >= Wl) continue;
                    const int delta = (1 - dy) * 3 + (1 - dx);
                    process_patch(py * Wl + px, delta);
                }
            }
        }
    }

    add_inplace_ct<OUT_DIM>(out_feat, residual);
    layernorm_ct<OUT_DIM>(out_feat, ln_gamma, ln_beta);
}

// Test wrappers — single-block, 32 threads (one warp); pass devptrs as int.
// Allocates scratch + out_feat in dynamic shared memory.

#define SSLA_DEF_LAYER_FWD(IN, OUT, K)                                       \
extern "C" __global__ void k_layer_forward_##IN##_##OUT##_##K(               \
    int ev_x, int ev_y, int Hl, int Wl,                                      \
    const float* in_feat,                                                    \
    const float* qvgIn,                                                      \
    const float* goW,                                                        \
    const float* input_proj,                                                 \
    const float* ln_gamma,                                                   \
    const float* ln_beta,                                                    \
    float* H_all,                                                            \
    float* out_feat_global) {                                                \
    if (blockIdx.x != 0) return;                                             \
    extern __shared__ float smem[];                                          \
    float* out_sm   = smem;                                                  \
    float* scratch  = smem + OUT;                                            \
    ssla_layer_forward_ct<IN, OUT, K>(                                       \
        ev_x, ev_y, Hl, Wl, in_feat, qvgIn, goW, input_proj,                 \
        ln_gamma, ln_beta, H_all, out_sm, scratch);                          \
    for (int i = threadIdx.x; i < OUT; i += SSLA_BLOCK_THREADS)              \
        out_feat_global[i] = out_sm[i];                                      \
}

// SSLA-S layer schedule
SSLA_DEF_LAYER_FWD(2,  12, 1)
SSLA_DEF_LAYER_FWD(12, 12, 3)
SSLA_DEF_LAYER_FWD(12, 24, 3)
SSLA_DEF_LAYER_FWD(24, 24, 3)
SSLA_DEF_LAYER_FWD(24, 48, 3)
SSLA_DEF_LAYER_FWD(48, 48, 3)
SSLA_DEF_LAYER_FWD(48, 96, 3)
SSLA_DEF_LAYER_FWD(96, 96, 3)
