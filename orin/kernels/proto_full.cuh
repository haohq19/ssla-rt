// Full-pipeline multi-warp persistent kernel with strict spatial
// sharding for SSLA-S on Orin NX.
//
// Constraints (from user):
//   * full pipeline (all 4 stages, 8 layers, tdrop gating)
//   * multi-warp per block: each warp processes one event
//   * strict spatial sharding: blockIdx.x indexes into per-strip
//     hidden state + ring; no cross-block writes
//   * pinned host ring (no managed concurrent CPU/GPU writes)
//   * no atomics anywhere on the device side
//   * fully fused single-event path inside one warp (lane-striped
//     scratch in registers, only __syncwarp / warp shuffles)
//   * no __syncthreads except the once-per-batch ring tail update

#pragma once
#include "proto_layer_pair.cuh"   // matvec_w, lru_step_w, layernorm_w, ssla_layer_w

struct LayerWeightsW {
    const float* qvgIn;
    const float* goW;
    const float* input_proj;   // may be null when IN==OUT
    const float* ln_gamma;
    const float* ln_beta;
};

// Per-strip configuration. Each block reads cfg[blockIdx.x].
struct StripConfig {
    int Hs0, Ws0;              // strip dims at stage 0 (e.g. Hs0=480, Ws0=80)
    int strip_x_base;          // global x of strip's first column (unused by kernel; for host)
    int tdrop_window;
    LayerWeightsW layers[8];
    float*         hidden[8];  // (Hl_strip × Wl_strip × D,) per layer
    unsigned char* tdrop[3];   // bytes per stage post-pool, sized (Hl/2 × Wl/2)
};

struct EventRecW {
    float t;     // dt_norm (host preprocessed)
    float x;     // STRIP-LOCAL x (host translated: x_global - strip_x_base)
    float y;
    float p;
};

struct OutputSlotW {
    int   passed;
    int   touched_x[4];
    int   touched_y[4];
    float s0[12];
    float s1[24];
    float s2[48];
    float s3[96];
};

// Per-warp shared scratch. Layout:
//   in_feat_sm : 96 floats   (max IN seen across layers; L7 uses 96)
//   qh_sm      : 96 floats
// Total: 192 floats / 768 bytes / warp.
// With W warps/block: W * 768 bytes dynamic shared mem.
constexpr int SCRATCH_PER_WARP_FLOATS = 192;

// Returns true if the event reached stage 3 (else early-out by bounds
// or tdrop). On true, writes per-stage features to slot.
template <int C0, int C1, int C2, int C3>
__device__ inline bool ssla_event_w(
    const StripConfig& cfg,
    const EventRecW& ev,
    OutputSlotW* slot,
    float* in_feat_sm,
    float* qh_sm)
{
    const int H0 = cfg.Hs0;
    const int W0 = cfg.Ws0;
    int evx = (int)ev.x;
    int evy = (int)ev.y;
    if (evx < 0 || evx >= W0 || evy < 0 || evy >= H0) return false;

    const int lane = threadIdx.x & 31;

    // ---------- Stage 0: L0 (2->C0 K=1) + L1 (C0->C0 K=3) ----------
    if (lane == 0) { slot->touched_x[0] = evx; slot->touched_y[0] = evy; }

    // L0 input is just (dt_norm, polarity). Materialize directly to shared mem.
    if (lane == 0) in_feat_sm[0] = ev.t;
    if (lane == 1) in_feat_sm[1] = ev.p;
    __syncwarp();

    // Construct lane-striped 2-element input for ssla_layer_w
    float feat_local[(2 + 31) / 32];   // = 1
    feat_local[0] = (lane < 2) ? in_feat_sm[lane] : 0.0f;

    int Hl = H0, Wl = W0;
    float buf_a[(C0 + 31) / 32];       // = 1 for C0=12
    ssla_layer_w<2, C0, 1>(
        evx, evy, Hl, Wl, feat_local,
        cfg.layers[0].qvgIn, cfg.layers[0].goW, cfg.layers[0].input_proj,
        cfg.layers[0].ln_gamma, cfg.layers[0].ln_beta,
        cfg.hidden[0], buf_a, in_feat_sm, qh_sm);

    float s0_local[(C0 + 31) / 32];    // = 1
    ssla_layer_w<C0, C0, 3>(
        evx, evy, Hl, Wl, buf_a,
        cfg.layers[1].qvgIn, cfg.layers[1].goW, cfg.layers[1].input_proj,
        cfg.layers[1].ln_gamma, cfg.layers[1].ln_beta,
        cfg.hidden[1], s0_local, in_feat_sm, qh_sm);

    // Write s0 to slot.
    #pragma unroll
    for (int k = 0; k < (C0 + 31) / 32; ++k) {
        const int o = lane + 32 * k;
        if (o < C0) slot->s0[o] = s0_local[k];
    }

    // Pool, then tdrop[0] gate.
    evx >>= 1; evy >>= 1; Hl >>= 1; Wl >>= 1;
    {
        unsigned char pre = 0;
        unsigned char* td = cfg.tdrop[0];
        const int idx = evy * Wl + evx;
        if (lane == 0) {
            pre = td[idx];
            td[idx] = (unsigned char)(pre + 1u);
        }
        pre = (unsigned char)__shfl_sync(0xffffffff, pre, 0);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }

    // ---------- Stage 1: L2 (C0->C1 K=3) + L3 (C1->C1 K=3) ----------
    if (lane == 0) { slot->touched_x[1] = evx; slot->touched_y[1] = evy; }

    float buf1[(C1 + 31) / 32];
    ssla_layer_w<C0, C1, 3>(
        evx, evy, Hl, Wl, s0_local,
        cfg.layers[2].qvgIn, cfg.layers[2].goW, cfg.layers[2].input_proj,
        cfg.layers[2].ln_gamma, cfg.layers[2].ln_beta,
        cfg.hidden[2], buf1, in_feat_sm, qh_sm);

    float s1_local[(C1 + 31) / 32];
    ssla_layer_w<C1, C1, 3>(
        evx, evy, Hl, Wl, buf1,
        cfg.layers[3].qvgIn, cfg.layers[3].goW, cfg.layers[3].input_proj,
        cfg.layers[3].ln_gamma, cfg.layers[3].ln_beta,
        cfg.hidden[3], s1_local, in_feat_sm, qh_sm);

    #pragma unroll
    for (int k = 0; k < (C1 + 31) / 32; ++k) {
        const int o = lane + 32 * k;
        if (o < C1) slot->s1[o] = s1_local[k];
    }

    evx >>= 1; evy >>= 1; Hl >>= 1; Wl >>= 1;
    {
        unsigned char pre = 0;
        unsigned char* td = cfg.tdrop[1];
        const int idx = evy * Wl + evx;
        if (lane == 0) {
            pre = td[idx];
            td[idx] = (unsigned char)(pre + 1u);
        }
        pre = (unsigned char)__shfl_sync(0xffffffff, pre, 0);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }

    // ---------- Stage 2: L4 (C1->C2 K=3) + L5 (C2->C2 K=3) ----------
    if (lane == 0) { slot->touched_x[2] = evx; slot->touched_y[2] = evy; }

    float buf2[(C2 + 31) / 32];
    ssla_layer_w<C1, C2, 3>(
        evx, evy, Hl, Wl, s1_local,
        cfg.layers[4].qvgIn, cfg.layers[4].goW, cfg.layers[4].input_proj,
        cfg.layers[4].ln_gamma, cfg.layers[4].ln_beta,
        cfg.hidden[4], buf2, in_feat_sm, qh_sm);

    float s2_local[(C2 + 31) / 32];
    ssla_layer_w<C2, C2, 3>(
        evx, evy, Hl, Wl, buf2,
        cfg.layers[5].qvgIn, cfg.layers[5].goW, cfg.layers[5].input_proj,
        cfg.layers[5].ln_gamma, cfg.layers[5].ln_beta,
        cfg.hidden[5], s2_local, in_feat_sm, qh_sm);

    #pragma unroll
    for (int k = 0; k < (C2 + 31) / 32; ++k) {
        const int o = lane + 32 * k;
        if (o < C2) slot->s2[o] = s2_local[k];
    }

    evx >>= 1; evy >>= 1; Hl >>= 1; Wl >>= 1;
    {
        unsigned char pre = 0;
        unsigned char* td = cfg.tdrop[2];
        const int idx = evy * Wl + evx;
        if (lane == 0) {
            pre = td[idx];
            td[idx] = (unsigned char)(pre + 1u);
        }
        pre = (unsigned char)__shfl_sync(0xffffffff, pre, 0);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }

    // ---------- Stage 3: L6 (C2->C3 K=3) + L7 (C3->C3 K=3) ----------
    if (lane == 0) { slot->touched_x[3] = evx; slot->touched_y[3] = evy; }

    float buf3[(C3 + 31) / 32];
    ssla_layer_w<C2, C3, 3>(
        evx, evy, Hl, Wl, s2_local,
        cfg.layers[6].qvgIn, cfg.layers[6].goW, cfg.layers[6].input_proj,
        cfg.layers[6].ln_gamma, cfg.layers[6].ln_beta,
        cfg.hidden[6], buf3, in_feat_sm, qh_sm);

    float s3_local[(C3 + 31) / 32];
    ssla_layer_w<C3, C3, 3>(
        evx, evy, Hl, Wl, buf3,
        cfg.layers[7].qvgIn, cfg.layers[7].goW, cfg.layers[7].input_proj,
        cfg.layers[7].ln_gamma, cfg.layers[7].ln_beta,
        cfg.hidden[7], s3_local, in_feat_sm, qh_sm);

    #pragma unroll
    for (int k = 0; k < (C3 + 31) / 32; ++k) {
        const int o = lane + 32 * k;
        if (o < C3) slot->s3[o] = s3_local[k];
    }
    return true;
}

// Persistent kernel: one block per strip; each block has W warps; each
// warp processes one event end-to-end.
//
// Multi-warp coordination: __syncthreads twice per W-event batch.
// Ring tail / out_head / events_done updated by thread 0 each batch.
//
// Note: the per-block sharded `cfg`, ring, and out_buf are passed as
// arrays; blockIdx.x indexes them.
extern "C" __global__ void k_proto_persistent_full(
    const StripConfig*     cfgs,                 // (n_blocks,)
    const EventRecW* const* ring_bufs,           // (n_blocks,) host-side pinned ptrs
    const unsigned long long* ring_masks,
    volatile unsigned long long* const* ring_heads,
    volatile unsigned long long* const* ring_tails,
    OutputSlotW* const* out_bufs,
    const unsigned long long* out_masks,
    volatile unsigned long long* const* out_heads,
    volatile int*    stop_flag,                  // single shared
    volatile unsigned long long* const* events_done,
    unsigned int     spin_ns,
    unsigned long long* const* timing_cycles     // per-block per-event clock64 deltas (size = out_buf)
) {
    const int b = blockIdx.x;
    extern __shared__ float smem[];
    const int n_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    float* my_smem    = smem + warp_id * SCRATCH_PER_WARP_FLOATS;
    float* in_feat_sm = my_smem;
    float* qh_sm      = my_smem + 96;

    const StripConfig& cfg = cfgs[b];
    const EventRecW* ring_buf = ring_bufs[b];
    const unsigned long long ring_mask = ring_masks[b];
    OutputSlotW* out_buf = out_bufs[b];
    const unsigned long long out_mask = out_masks[b];
    volatile unsigned long long* ring_head_p = ring_heads[b];
    volatile unsigned long long* ring_tail_p = ring_tails[b];
    volatile unsigned long long* out_head_p  = out_heads[b];
    volatile unsigned long long* events_done_p = events_done[b];
    unsigned long long* timing_p = timing_cycles[b];

    unsigned long long t_local = *ring_tail_p;
    unsigned long long o_local = *out_head_p;

    while (true) {
        const int  stop = *stop_flag;
        const unsigned long long head = *ring_head_p;
        if (stop != 0) return;

        unsigned long long avail = head - t_local;
        if (avail < (unsigned long long)n_warps) {
            __nanosleep(spin_ns);
            continue;
        }

        // Each warp processes one event from this batch.
        const unsigned long long my_idx = t_local + warp_id;
        const EventRecW ev = ring_buf[my_idx & ring_mask];
        const unsigned long long my_out_idx = o_local + warp_id;
        OutputSlotW* slot = &out_buf[my_out_idx & out_mask];

        unsigned long long t_event_start;
        if (lane == 0) t_event_start = clock64();

        const bool passed = ssla_event_w<12, 24, 48, 96>(
            cfg, ev, slot, in_feat_sm, qh_sm);
        if (lane == 0) {
            slot->passed = passed ? 1 : 0;
            // Record per-event GPU cycles (clock64 wraps at 2^63 — irrelevant here).
            timing_p[my_out_idx & out_mask] = clock64() - t_event_start;
        }

        __syncthreads();   // wait for all warps to complete their event
        __threadfence_system();

        t_local += n_warps;
        o_local += n_warps;
        if (threadIdx.x == 0) {
            *ring_tail_p   = t_local;
            *out_head_p    = o_local;
            *events_done_p = o_local;
            __threadfence_system();
        }
        __syncthreads();
    }
}
