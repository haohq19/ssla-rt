// Risk-retirement prototype: fused warp-per-event SSLA-S layer pair.
//
// Design constraints (from user):
//   * one warp = one event, scratch in registers (lane-striped)
//   * NO __syncthreads anywhere (only __syncwarp / warp shuffles)
//   * NO atomic spinlocks (caller guarantees one event per cell)
//   * weights stored (IN, OUT) layout for coalesced loads
//
// Lane-striped tensor convention:
//   Tensor of dim D is stored across the 32 warp lanes such that lane `l`
//   holds slots [l, l+32, l+64, ...]. Per-lane register count is
//   ceil(D/32). Lanes with l >= D simply hold unused values.
//
// Scope of THIS file: a single fused stage-3 layer pair (L6: 48→96 K=3,
// L7: 96→96 K=3), the worst representative deep path. If this fits in
// register budget and meets latency, the smaller stages compose.

#pragma once

__device__ inline float sigmoidf_(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------
// matvec_w<IN, OUT, OUT_STRIDE>:
//   y[k] = Σ_i W[i * OUT_STRIDE + (lane + 32*k)] * x[i]
// for k in [0, ceil(OUT/32)). The OUT_STRIDE is the row pitch of W in
// the (IN, OUT_STRIDE) layout — for combined Q|V|G weights with
// 3*OUT total cols, set OUT_STRIDE = 3*OUT and offset W by 0/OUT/2*OUT
// to get the q/v/g sub-matrices.
//
// `x` lives in shared memory (broadcast load by all 32 lanes), `W` in
// global memory (coalesced reads since adjacent lanes hit adjacent
// addresses). Output `y` is lane-striped registers.
//
// add_bias=false → y = matvec; add_bias=true → y += matvec_accum
// (the goW path needs accumulation across patches).
// ---------------------------------------------------------------
template <int IN, int OUT, int OUT_STRIDE = OUT, bool ACCUM = false>
__device__ inline void matvec_w(
    const float* __restrict__ x,        // shared mem, IN floats
    const float* __restrict__ W,        // global, (IN, OUT_STRIDE) row-major
    float (&y)[(OUT + 31) / 32]
) {
    constexpr int K = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) {
            float acc = ACCUM ? y[k] : 0.0f;
            #pragma unroll
            for (int i = 0; i < IN; ++i) {
                acc += W[(long long)i * OUT_STRIDE + o] * x[i];
            }
            y[k] = acc;
        }
    }
    __syncwarp();
}

// ---------------------------------------------------------------
// lru_step_w<OUT>:
//   gc[k] = sigmoid(g[k])
//   h_cell[o]  =  gc[k] * h_cell[o] + v[k]
//   qh[k]      =  q[k] * h_cell_new
// `h_cell` is global memory (managed); 32-wide coalesced read+write.
// All scalars (g, v, q, qh) are lane-striped registers.
// ---------------------------------------------------------------
template <int OUT>
__device__ inline void lru_step_w(
    const float (&g)[(OUT + 31) / 32],
    const float (&v)[(OUT + 31) / 32],
    const float (&q)[(OUT + 31) / 32],
    float* __restrict__ h_cell,         // (OUT,) global
    float (&qh)[(OUT + 31) / 32]
) {
    constexpr int K = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) {
            const float gc = sigmoidf_(g[k]);
            const float hc = gc * h_cell[o] + v[k];
            h_cell[o] = hc;
            qh[k] = q[k] * hc;
        }
    }
    __syncwarp();
}

// ---------------------------------------------------------------
// layernorm_w<OUT>: in-place LayerNorm of lane-striped y.
// fp64 partial sums, warp-shuffle butterfly reduction.
// ---------------------------------------------------------------
template <int OUT>
__device__ inline void layernorm_w(
    float (&y)[(OUT + 31) / 32],
    const float* __restrict__ gamma,    // global (OUT,)
    const float* __restrict__ beta,     // global (OUT,)
    float eps = 1e-5f
) {
    constexpr int K = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;

    // sum
    double sum = 0.0;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) sum += (double)y[k];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    const double mean = sum / (double)OUT;

    // var
    double var = 0.0;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) {
            const double d = (double)y[k] - mean;
            var += d * d;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        var += __shfl_xor_sync(0xffffffff, var, off);
    }
    var /= (double)OUT;
    const double inv = 1.0 / sqrt(var + (double)eps);

    // normalize + affine
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) {
            const double v_ = ((double)y[k] - mean) * inv;
            y[k] = (float)(v_ * (double)gamma[o] + (double)beta[o]);
        }
    }
}

// ---------------------------------------------------------------
// process_patch_w: one of 9 (or 1) per-cell SSLA contributions.
//   1) qvg matvec — produces q, v, g (each OUT-wide lane-striped)
//      from a single (IN, 3*OUT) qvgIn[delta] block. We do it as 3
//      separate matvecs with OUT_STRIDE=3*OUT to keep q/v/g register-
//      aligned (no cross-lane shuffle needed to split).
//   2) lru_step on H_all + patch_idx*OUT
//   3) goW matvec ACCUMULATING into out_local
//
// goW[delta] is (IN_OUT, OUT_OUT) where IN_OUT = OUT_OUT = OUT.
// ---------------------------------------------------------------
template <int IN, int OUT>
__device__ inline void process_patch_w(
    int delta, int patch_idx,
    const float* __restrict__ in_feat_sm,
    const float* __restrict__ qvgIn,    // (A, IN, 3*OUT) flattened
    const float* __restrict__ goW,      // (A, OUT, OUT) flattened — (in, out) layout, OUT_STRIDE=OUT
    float* __restrict__ H_all,          // (Hl*Wl, OUT) global
    float (&out_local)[(OUT + 31) / 32]
) {
    constexpr int QVG_STRIDE = 3 * OUT * IN;
    constexpr int GOW_STRIDE = OUT * OUT;
    const float* qvg_w = qvgIn + (long long)delta * QVG_STRIDE;
    const float* go_w  = goW   + (long long)delta * GOW_STRIDE;

    float q_local[(OUT + 31) / 32];
    float v_local[(OUT + 31) / 32];
    float g_local[(OUT + 31) / 32];
    matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvg_w + 0 * OUT, q_local);
    matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvg_w + 1 * OUT, v_local);
    matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvg_w + 2 * OUT, g_local);

    float qh_local[(OUT + 31) / 32];
    float* h_cell = H_all + (long long)patch_idx * OUT;
    lru_step_w<OUT>(g_local, v_local, q_local, h_cell, qh_local);

    // qh in registers; need it in shared mem for the goW matvec input.
    // We materialize qh through shared mem region (the same one that
    // holds in_feat — we can overwrite it after this layer's q/v/g/lru
    // is done since in_feat won't be read again until next layer).
    // *** Actually, in_feat IS still needed for OTHER patches' qvg
    // matvecs in this layer! So we must NOT overwrite. We need a
    // separate shared region for qh.
    //
    // Caller (ssla_layer_w) provides a second shared region for qh.
    // For modularity, the goW accumulation is split out of process_patch_w
    // into ssla_layer_w directly (see below).
    //
    // This function signature exposes qh_local for the caller to handle.
    // But since C++ device fns can't return arrays easily, we'll inline
    // process_patch_w into ssla_layer_w instead. Keep this function as a
    // documentation reference; the actual call sites inline the body.

    // Goal of declaring this fn was modularity — but with the qh→sm→goW
    // dependency it's cleaner to inline. So this template is not used.
}

// ---------------------------------------------------------------
// ssla_layer_w<IN, OUT, K>:
//   one full SSLA-S layer for one event in a single warp.
//   All scratch lane-striped in registers; uses 2 shared-mem
//   regions for in_feat broadcast and per-patch qh.
// ---------------------------------------------------------------
template <int IN, int OUT, int K>
__device__ inline void ssla_layer_w(
    int ev_x, int ev_y, int Hl, int Wl,
    const float (&in_local)[(IN + 31) / 32],
    const float* __restrict__ qvgIn,
    const float* __restrict__ goW,
    const float* __restrict__ input_proj,   // (IN, OUT) layout, may be null
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float* __restrict__ H_all,
    float (&out_local)[(OUT + 31) / 32],
    float* __restrict__ in_feat_sm,         // shared, IN floats
    float* __restrict__ qh_sm               // shared, OUT floats
) {
    constexpr int A         = K * K;
    constexpr int IN_K      = (IN + 31) / 32;
    constexpr int OUT_K     = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;

    // Materialize in_feat into shared memory for broadcast access.
    #pragma unroll
    for (int k = 0; k < IN_K; ++k) {
        const int i = lane + 32 * k;
        if (i < IN) in_feat_sm[i] = in_local[k];
    }
    __syncwarp();

    // Residual.
    float residual_local[OUT_K];
    if (input_proj != 0) {
        matvec_w<IN, OUT>(in_feat_sm, input_proj, residual_local);
    } else {
        // IN == OUT passthrough: copy in_local into residual_local
        // (same lane-stripe layout when IN == OUT).
        #pragma unroll
        for (int k = 0; k < OUT_K; ++k) {
            residual_local[k] = (k < IN_K) ? in_local[k] : 0.0f;
        }
    }

    // Zero out_local.
    #pragma unroll
    for (int k = 0; k < OUT_K; ++k) out_local[k] = 0.0f;

    // 9-patch loop (or 1 if K=1) — inlined process_patch.
    auto run_patch = [&] (int delta, int patch_idx) {
        constexpr int QVG_STRIDE = 3 * OUT * IN;
        constexpr int GOW_STRIDE = OUT * OUT;
        const float* qvg_w = qvgIn + (long long)delta * QVG_STRIDE;
        const float* go_w  = goW   + (long long)delta * GOW_STRIDE;

        float q_local[OUT_K], v_local[OUT_K], g_local[OUT_K];
        matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvg_w + 0 * OUT, q_local);
        matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvg_w + 1 * OUT, v_local);
        matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvg_w + 2 * OUT, g_local);

        float qh_local[OUT_K];
        float* h_cell = H_all + (long long)patch_idx * OUT;
        lru_step_w<OUT>(g_local, v_local, q_local, h_cell, qh_local);

        // Materialize qh to shared mem so all lanes can broadcast it
        // for the goW matvec.
        #pragma unroll
        for (int k = 0; k < OUT_K; ++k) {
            const int o = lane + 32 * k;
            if (o < OUT) qh_sm[o] = qh_local[k];
        }
        __syncwarp();

        matvec_w<OUT, OUT, OUT, /*ACCUM=*/true>(qh_sm, go_w, out_local);
    };

    if (A == 1) {
        run_patch(0, ev_y * Wl + ev_x);
    } else {
        const int base = ev_y * Wl + ev_x;
        const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                           && (ev_y > 0) && (ev_y + 1 < Hl);
        if (interior) {
            run_patch(8, base - Wl - 1);
            run_patch(7, base - Wl);
            run_patch(6, base - Wl + 1);
            run_patch(5, base - 1);
            run_patch(4, base);
            run_patch(3, base + 1);
            run_patch(2, base + Wl - 1);
            run_patch(1, base + Wl);
            run_patch(0, base + Wl + 1);
        } else {
            for (int dy = -1; dy <= 1; ++dy) {
                const int py = ev_y + dy;
                if (py < 0 || py >= Hl) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int px = ev_x + dx;
                    if (px < 0 || px >= Wl) continue;
                    const int delta = (1 - dy) * 3 + (1 - dx);
                    run_patch(delta, py * Wl + px);
                }
            }
        }
    }

    // Add residual + LayerNorm.
    #pragma unroll
    for (int k = 0; k < OUT_K; ++k) out_local[k] += residual_local[k];
    layernorm_w<OUT>(out_local, ln_gamma, ln_beta);
}

// ---------------------------------------------------------------
// Test wrapper: stage-3 layer pair (L6: 48→96 K=3, L7: 96→96 K=3).
// Single event, single warp, returns final 96-wide feature.
//
// Shared memory layout (per warp):
//   in_feat_sm  : 96 floats   (largest IN seen by either layer)
//   qh_sm       : 96 floats
// total: 192 floats = 768 bytes / warp.
// ---------------------------------------------------------------
// Note on launch_bounds: __launch_bounds__(128, 4) was tested — it caps
// regs at 128/thread but introduces 8 B local-memory spill.  Default
// (no hint) gives 142 regs / 0 spill, which is healthier given the
// user's "0 spill" preference.  Kept default below.
extern "C" __global__ void k_proto_layer_pair_s3(
    int ev_x, int ev_y, int Hl, int Wl,
    const float* __restrict__ in_feat_global,    // (48,)
    const float* __restrict__ qvgIn_L6,          // (9, 48, 288)
    const float* __restrict__ goW_L6,            // (9, 96, 96)
    const float* __restrict__ input_proj_L6,     // (48, 96)
    const float* __restrict__ ln_gamma_L6,       // (96,)
    const float* __restrict__ ln_beta_L6,        // (96,)
    float*       __restrict__ H_L6,              // (Hl*Wl, 96)
    const float* __restrict__ qvgIn_L7,          // (9, 96, 288)
    const float* __restrict__ goW_L7,            // (9, 96, 96)
    const float* __restrict__ ln_gamma_L7,       // (96,)
    const float* __restrict__ ln_beta_L7,        // (96,)
    float*       __restrict__ H_L7,              // (Hl*Wl, 96)
    float*       __restrict__ s3_out_global,     // (96,) output
    unsigned long long* __restrict__ timing_cycles    // (1,)
)
{
    if (blockIdx.x != 0 || threadIdx.x >= 32) return;
    extern __shared__ float smem[];
    float* in_feat_sm = smem;            // up to 96 floats
    float* qh_sm      = smem + 96;       // 96 floats

    const int lane = threadIdx.x & 31;

    unsigned long long t0;
    if (lane == 0) t0 = clock64();
    __syncwarp();

    // Load in_feat_global (48) → lane-striped registers.
    float in_L6_local[(48 + 31) / 32];   // = 2
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = lane + 32 * k;
        if (i < 48) in_L6_local[k] = in_feat_global[i];
    }

    float out_L6_local[(96 + 31) / 32];   // = 3
    ssla_layer_w<48, 96, 3>(
        ev_x, ev_y, Hl, Wl,
        in_L6_local,
        qvgIn_L6, goW_L6, input_proj_L6,
        ln_gamma_L6, ln_beta_L6,
        H_L6, out_L6_local,
        in_feat_sm, qh_sm);

    float out_L7_local[(96 + 31) / 32];   // = 3
    ssla_layer_w<96, 96, 3>(
        ev_x, ev_y, Hl, Wl,
        out_L6_local,
        qvgIn_L7, goW_L7, /*input_proj=*/(const float*)0,   // IN==OUT passthrough
        ln_gamma_L7, ln_beta_L7,
        H_L7, out_L7_local,
        in_feat_sm, qh_sm);

    // Write s3_out_global (96 floats) from lane-striped out_L7_local.
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int o = lane + 32 * k;
        if (o < 96) s3_out_global[o] = out_L7_local[k];
    }

    __syncwarp();
    if (lane == 0) timing_cycles[0] = clock64() - t0;
}
