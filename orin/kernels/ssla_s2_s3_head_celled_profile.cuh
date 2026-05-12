// Profile-instrumented copy of ssla_s2_s3_head_celled.cuh::k_ssla_s2s3_celled_drain_n.
// Records clock64() at every phase boundary into a per-batch array of NPHASE u64.
// All __device__ helpers and templates are pulled from the production header.
//
// USAGE: NVRTC-compile both headers together (same translation unit). Launch
// k_ssla_s2s3_celled_drain_n_profile with the per-block clk buffer sized
// at least n_batches * NPHASE * sizeof(unsigned long long).

#pragma once

// NB: include the production kernel header for type/template defs. The two
// __global__ entry points are distinct symbols so no conflict.
#include "ssla_s2_s3_head_celled.cuh"

// ---- Profiling phase indices ---------------------------------------------
// Keep names short; Python side decodes indices.
constexpr int PH_POP             = 0;   // t_pop (clock at batch start)
constexpr int PH_LOAD            = 1;   // after batch record load + sync
constexpr int PH_DISPATCH_S2     = 2;
constexpr int PH_L4_RES          = 3;
constexpr int PH_L4_ZERO         = 4;
constexpr int PH_L4_COMPUTE      = 5;
constexpr int PH_L4_GATHER       = 6;
constexpr int PH_L5_RES          = 7;
constexpr int PH_L5_ZERO         = 8;
constexpr int PH_L5_COMPUTE      = 9;
constexpr int PH_L5_GATHER       = 10;
constexpr int PH_TDROP_S2        = 11;
constexpr int PH_POOL            = 12;
constexpr int PH_DISPATCH_S3     = 13;
constexpr int PH_L6_RES          = 14;
constexpr int PH_L6_ZERO         = 15;
constexpr int PH_L6_COMPUTE      = 16;
constexpr int PH_L6_GATHER       = 17;
constexpr int PH_L7_RES          = 18;
constexpr int PH_L7_ZERO         = 19;
constexpr int PH_L7_COMPUTE      = 20;
constexpr int PH_L7_GATHER       = 21;
constexpr int PH_TDROP_S3        = 22;
constexpr int PH_OUT             = 23;
constexpr int N_PHASES           = 24;


#define STAMP(idx) do {                                                       \
    __syncthreads();                                                          \
    if (threadIdx.x == 0 && phase_clk != 0) {                                 \
        phase_clk[(long long)batch_idx * N_PHASES + (idx)] = clock64();       \
    }                                                                         \
} while (0)


// Re-expanded run_layer with phase stamps. Same body as run_layer_celled
// but with STAMP(...) between phases. RES/ZERO/COMPUTE/GATHER -> 4 stamps.
template <int IN, int OUT>
__device__ inline void run_layer_celled_profiled(
    int batch_size,
    int Hl, int Wl,
    EventSlot*               event_slots,
    const int*               task_event,
    const int*               task_delta,
    const int*               task_count,
    const LayerWeightsS2S3&  layer,
    float*                   hidden,
    float*                   contrib,
    float*                   per_warp_in_feat_sm,
    float*                   per_warp_qh_sm,
    unsigned long long*      phase_clk,
    int                      batch_idx,
    int                      ph_res, int ph_zero, int ph_compute, int ph_gather)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    // RESIDUAL
    if (warp_id < batch_size) {
        const int e = warp_id;
        compute_residual<IN, OUT>(event_slots[e].in_feat, layer.input_proj,
                                  event_slots[e].residual);
    }
    STAMP(ph_res);

    // CONTRIB ZERO
    const int n_contrib_floats = batch_size * N_WARPS * OUT;
    for (int i = threadIdx.x; i < n_contrib_floats; i += blockDim.x) {
        contrib[i] = 0.0f;
    }
    STAMP(ph_zero);

    // COMPUTE
    float* my_in_feat_sm = per_warp_in_feat_sm + warp_id * IN;
    float* my_qh_sm      = per_warp_qh_sm + warp_id * OUT;
    const int* my_tasks_event = task_event + warp_id * BATCH;
    const int* my_tasks_delta = task_delta + warp_id * BATCH;
    const int  n_tasks        = task_count[warp_id];

    for (int t = 0; t < n_tasks; ++t) {
        const int e     = my_tasks_event[t];
        const int delta = my_tasks_delta[t];
        for (int i = lane; i < IN; i += 32) {
            my_in_feat_sm[i] = event_slots[e].in_feat[i];
        }
        __syncwarp();
        const int dy = 1 - (delta / 3);
        const int dx = 1 - (delta % 3);
        const int py = event_slots[e].evy + dy;
        const int px = event_slots[e].evx + dx;
        if (py < 0 || py >= Hl || px < 0 || px >= Wl) continue;
        const int patch_idx = py * Wl + px;
        float* contrib_slot = contrib + ((long long)e * N_WARPS + warp_id) * OUT;
        constexpr int QVG_STRIDE = 3 * OUT * IN;
        constexpr int GOW_STRIDE = OUT * OUT;
        const float* qvg_w = layer.qvgIn + (long long)delta * QVG_STRIDE;
        const float* go_w  = layer.goW   + (long long)delta * GOW_STRIDE;
        process_patch_cell<IN, OUT>(
            my_in_feat_sm, qvg_w, go_w,
            hidden + (long long)patch_idx * OUT,
            my_qh_sm, contrib_slot);
    }
    STAMP(ph_compute);

    // GATHER + LN
    if (warp_id < batch_size) {
        const int e = warp_id;
        float* contrib_event = contrib + (long long)e * N_WARPS * OUT;
        gather_residual_ln<OUT>(
            contrib_event,
            event_slots[e].residual,
            layer.ln_gamma, layer.ln_beta,
            event_slots[e].in_feat);
    }
    STAMP(ph_gather);
}


extern "C" __global__
__launch_bounds__(N_WARPS * 32, 1)
void k_ssla_s2s3_celled_drain_n_profile(
    const HybridS2S3Config*   cfg_in,
    const HybridInputRec*     ring0,
    const HybridInputRec*     ring1,
    int                       n0,
    int                       n1,
    HybridS2S3OutputSlot*     out0,
    HybridS2S3OutputSlot*     out1,
    unsigned long long*       phase_clk0,
    unsigned long long*       phase_clk1)
{
    if (blockIdx.x >= 2) return;
    extern __shared__ float smem_raw[];
    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg = *cfg_in;
    const HybridInputRec*   ring = (blk == 0) ? ring0 : ring1;
    HybridS2S3OutputSlot*   out  = (blk == 0) ? out0  : out1;
    const int               n    = (blk == 0) ? n0    : n1;
    unsigned long long*     phase_clk = (blk == 0) ? phase_clk0 : phase_clk1;

    char* smem = (char*)smem_raw;
    EventSlot* event_slots = (EventSlot*)smem;
    char* p = smem + sizeof(EventSlot) * BATCH;
    float* contrib = (float*)p;
    p += BATCH * N_WARPS * OUT_MAX * sizeof(float);
    float* per_warp_in_feat_sm = (float*)p;
    p += N_WARPS * OUT_MAX * sizeof(float);
    float* per_warp_qh_sm = (float*)p;
    p += N_WARPS * OUT_MAX * sizeof(float);
    int* task_event = (int*)p;
    p += N_WARPS * BATCH * sizeof(int);
    int* task_delta = (int*)p;
    p += N_WARPS * BATCH * sizeof(int);
    int* task_count = (int*)p;
    p += N_WARPS * sizeof(int);
    int* pass2 = (int*)p;
    p += BATCH * sizeof(int);
    int* pass3 = (int*)p;
    p += BATCH * sizeof(int);

    const HybridStrip& strip = cfg.strip[blk];

    int batch_idx = 0;
    for (int batch_base = 0; batch_base < n; batch_base += BATCH, ++batch_idx) {
        const int batch_size = min(BATCH, n - batch_base);
        STAMP(PH_POP);

        // LOAD
        for (int e = 0; e < batch_size; ++e) {
            const HybridInputRec& rec = ring[batch_base + e];
            for (int i = threadIdx.x; i < OUT_MAX; i += blockDim.x) {
                event_slots[e].in_feat[i] = (i < C1) ? rec.feat1[i] : 0.0f;
            }
            if (threadIdx.x == 0) {
                event_slots[e].evx = (int)rec.x;
                event_slots[e].evy = (int)rec.y;
                event_slots[e].is_owner =
                    ((int)rec.x >= strip.owned_lo &&
                     (int)rec.x <  strip.owned_hi) ? 1 : 0;
                event_slots[e].pass2 = 0;
                event_slots[e].pass3 = 0;
            }
        }
        STAMP(PH_LOAD);

        // STAGE 2
        dispatch_tasks(batch_size, cfg.H2, cfg.W2, event_slots, 0,
                       task_event, task_delta, task_count);
        STAMP(PH_DISPATCH_S2);
        run_layer_celled_profiled<C1, C2>(
            batch_size, cfg.H2, cfg.W2,
            event_slots, task_event, task_delta, task_count,
            cfg.layers[blk][0], cfg.hidden[blk][0],
            contrib, per_warp_in_feat_sm, per_warp_qh_sm,
            phase_clk, batch_idx,
            PH_L4_RES, PH_L4_ZERO, PH_L4_COMPUTE, PH_L4_GATHER);
        run_layer_celled_profiled<C2, C2>(
            batch_size, cfg.H2, cfg.W2,
            event_slots, task_event, task_delta, task_count,
            cfg.layers[blk][1], cfg.hidden[blk][1],
            contrib, per_warp_in_feat_sm, per_warp_qh_sm,
            phase_clk, batch_idx,
            PH_L5_RES, PH_L5_ZERO, PH_L5_COMPUTE, PH_L5_GATHER);
        serial_tdrop(batch_size, cfg.tdrop_window,
                     cfg.tdrop_s2[blk], event_slots, cfg.W2, 0,
                     pass2, false);
        STAMP(PH_TDROP_S2);

        if (threadIdx.x == 0) {
            for (int e = 0; e < batch_size; ++e) {
                event_slots[e].evx >>= 1;
                event_slots[e].evy >>= 1;
            }
        }
        STAMP(PH_POOL);

        // STAGE 3
        dispatch_tasks(batch_size, cfg.H3, cfg.W3, event_slots, pass2,
                       task_event, task_delta, task_count);
        STAMP(PH_DISPATCH_S3);
        run_layer_celled_profiled<C2, C3>(
            batch_size, cfg.H3, cfg.W3,
            event_slots, task_event, task_delta, task_count,
            cfg.layers[blk][2], cfg.hidden[blk][2],
            contrib, per_warp_in_feat_sm, per_warp_qh_sm,
            phase_clk, batch_idx,
            PH_L6_RES, PH_L6_ZERO, PH_L6_COMPUTE, PH_L6_GATHER);
        run_layer_celled_profiled<C3, C3>(
            batch_size, cfg.H3, cfg.W3,
            event_slots, task_event, task_delta, task_count,
            cfg.layers[blk][3], cfg.hidden[blk][3],
            contrib, per_warp_in_feat_sm, per_warp_qh_sm,
            phase_clk, batch_idx,
            PH_L7_RES, PH_L7_ZERO, PH_L7_COMPUTE, PH_L7_GATHER);
        serial_tdrop(batch_size, cfg.tdrop_window,
                     cfg.tdrop_s3[blk], event_slots, cfg.W3, pass2,
                     pass3, true);
        STAMP(PH_TDROP_S3);

        if (out != 0) {
            for (int e = 0; e < batch_size; ++e) {
                if (threadIdx.x == 0) {
                    int p2 = event_slots[e].is_owner ? pass2[e] : -1;
                    int p3 = event_slots[e].is_owner
                                ? (pass2[e] ? pass3[e] : 0)
                                : -1;
                    out[batch_base + e].pass2 = p2;
                    out[batch_base + e].pass3 = p3;
                }
                for (int i = threadIdx.x; i < C3; i += blockDim.x) {
                    out[batch_base + e].s3_feat[i] = event_slots[e].in_feat[i];
                }
            }
        }
        STAMP(PH_OUT);
    }
}

#undef STAMP
