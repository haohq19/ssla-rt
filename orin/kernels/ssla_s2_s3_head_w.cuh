// SSLA-S stages 2 + 3 + head per-event step — warp-per-event variant.
//
// Drop-in replacement for ssla_s2_s3_head.cuh: same HybridS2S3Config,
// HybridInputRec, HybridStrip, GpuTimingSlot, predictions / version
// contract. The only differences:
//
//   * Each warp processes one event end-to-end (instead of 256 threads
//     cooperating on a single event in the coop kernel).
//   * Layer primitives are warp-cooperative (proto_layer_pair.cuh):
//     lane-striped scratch in registers, __syncwarp / warp shuffle only.
//   * Each block runs N warps in parallel (blockDim.x = 32 * n_warps);
//     each warp claims one event from the per-block ring per "batch".
//
// Spatial sharding is still strict at the block level (halo=2 in s2
// coords, per-block private hidden state + tdrop counters). Within a
// block, the N warps share the block's hidden state and tdrop counter
// arrays. Two events in the same batch landing on the same s2 cell
// have a small race window on the cell's hidden state + tdrop byte —
// for randomly-distributed events at W2=20 with 4 warps/block the
// collision probability is ~1/20 per batch. P1 measures the resulting
// drift; if it grows beyond the documented 4.40 baseline, a dispatch
// stage routing same-cell events to the same warp would be the fix.
//
// SMEM layout: per-warp scratch (192 floats = 768 B) × n_warps.
// At n_warps=4: 3 KB shared per block.
//
// NO __syncthreads inside the per-event step. Block-wide synchronization
// is only at ring tail advance (one __syncthreads per batch).

#pragma once
#include "proto_layer_pair.cuh"     // matvec_w, lru_step_w, layernorm_w, ssla_layer_w

// Match struct definitions from ssla_s2_s3_head.cuh — copied verbatim so
// the warp kernel can be compiled standalone without including the coop
// kernel header (avoids two competing definitions of the drain_n /
// persistent symbols).
struct LayerWeightsS2S3 {
    const float* qvgIn;
    const float* goW;
    const float* input_proj;
    const float* ln_gamma;
    const float* ln_beta;
};

struct HybridStrip {
    int owned_lo;
    int owned_hi;
    int s3_owned_lo;
    int s3_owned_hi;
    int owned_y_lo;
    int owned_y_hi;
    int s3_owned_y_lo;
    int s3_owned_y_hi;
};

struct GpuTimingSlot {
    unsigned long long t_pop_clk;
    unsigned long long t_done_clk;
    unsigned int       seq;
    unsigned int       owner;
    unsigned long long t_push_ns;   // copied from HybridInputRec at process time
};

// Must match MAX_BLOCKS in ssla_s2_s3_head_celled.cuh.
constexpr int MAX_BLOCKS = 16;

struct HybridS2S3Config {
    int H2, W2;
    int H3, W3;
    int tdrop_window;
    int head_out_dim;
    int n_blocks;
    int _pad_nblocks;
    HybridStrip      strip[MAX_BLOCKS];
    LayerWeightsS2S3 layers[MAX_BLOCKS][4];
    float*           hidden[MAX_BLOCKS][4];
    unsigned char*   tdrop_s2[MAX_BLOCKS];
    unsigned char*   tdrop_s3[MAX_BLOCKS];
    const float*     head_W;
    const float*     head_b;
    float*           preds[MAX_BLOCKS];
    unsigned int*    version[MAX_BLOCKS];
    GpuTimingSlot*   timing[MAX_BLOCKS];
    unsigned int     timing_mask;
    unsigned int     _pad_timing;
    unsigned long long* kernel_start_clk[MAX_BLOCKS];
    unsigned long long* kernel_end_clk[MAX_BLOCKS];
};

struct HybridInputRec {
    unsigned long long seq_done;
    float              t;
    unsigned short     x;
    unsigned short     y;
    float              feat1[24];
    unsigned long long t_push_ns;    // CPU CLOCK_MONOTONIC_RAW ns at ring publish
};

struct HybridS2S3OutputSlot {
    int   pass2;
    int   pass3;
    float s3_feat[96];
};


// SMEM scratch per warp = in_feat_sm (max-IN = 96) + qh_sm (OUT = 96).
constexpr int W_SCRATCH_PER_WARP_FLOATS = 192;


// Per-event step inside a single warp. Returns true iff event was
// owner+pass3 (i.e. wrote a prediction).
template <int C1, int C2, int C3, int HEAD_OUT>
__device__ inline bool ssla_s2s3_step_warp(
    const HybridS2S3Config& cfg, int blk,
    const HybridInputRec& rec_warp,
    HybridS2S3OutputSlot* out_slot,
    float* __restrict__ in_feat_sm,
    float* __restrict__ qh_sm)
{
    const int lane = threadIdx.x & 31;

    const int evx = (int)rec_warp.x;
    const int evy = (int)rec_warp.y;
    const HybridStrip& strip = cfg.strip[blk];
    const bool is_owner = (evx >= strip.owned_lo) && (evx < strip.owned_hi);

    if (out_slot != 0 && lane == 0) {
        out_slot->pass2 = is_owner ? 0 : -1;
        out_slot->pass3 = is_owner ? 0 : -1;
    }

    // feat1 → lane-striped registers (C1=24 → 1 element per lane).
    constexpr int IN1_K = (C1 + 31) / 32;
    float feat1_local[IN1_K];
    #pragma unroll
    for (int k = 0; k < IN1_K; ++k) {
        const int i = lane + 32 * k;
        feat1_local[k] = (i < C1) ? rec_warp.feat1[i] : 0.0f;
    }

    // ---- Stage 2: L4 (C1→C2 K=3), L5 (C2→C2 K=3) ----
    constexpr int OUT2_K = (C2 + 31) / 32;     // 2
    float buf_a_local[OUT2_K];
    ssla_layer_w<C1, C2, 3>(
        evx, evy, cfg.H2, cfg.W2, feat1_local,
        cfg.layers[blk][0].qvgIn, cfg.layers[blk][0].goW,
        cfg.layers[blk][0].input_proj,
        cfg.layers[blk][0].ln_gamma, cfg.layers[blk][0].ln_beta,
        cfg.hidden[blk][0], buf_a_local, in_feat_sm, qh_sm);

    float buf_b_local[OUT2_K];
    ssla_layer_w<C2, C2, 3>(
        evx, evy, cfg.H2, cfg.W2, buf_a_local,
        cfg.layers[blk][1].qvgIn, cfg.layers[blk][1].goW,
        cfg.layers[blk][1].input_proj,
        cfg.layers[blk][1].ln_gamma, cfg.layers[blk][1].ln_beta,
        cfg.hidden[blk][1], buf_b_local, in_feat_sm, qh_sm);

    // s2 tdrop counter (private per-block; warps in the block race on
    // the same byte if two pick same cell — bounded drift per design).
    {
        unsigned char pre = 0;
        unsigned char* td = cfg.tdrop_s2[blk];
        const int idx = evy * cfg.W2 + evx;
        if (lane == 0) {
            pre = td[idx];
            td[idx] = (unsigned char)(pre + 1u);
        }
        pre = (unsigned char)__shfl_sync(0xffffffff, pre, 0);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }
    if (out_slot != 0 && is_owner && lane == 0) out_slot->pass2 = 1;

    // ---- Stage 3: L6 (C2→C3 K=3), L7 (C3→C3 K=3) ----
    const int s3x = evx >> 1;
    const int s3y = evy >> 1;

    constexpr int OUT3_K = (C3 + 31) / 32;     // 3
    float buf_c_local[OUT3_K];
    ssla_layer_w<C2, C3, 3>(
        s3x, s3y, cfg.H3, cfg.W3, buf_b_local,
        cfg.layers[blk][2].qvgIn, cfg.layers[blk][2].goW,
        cfg.layers[blk][2].input_proj,
        cfg.layers[blk][2].ln_gamma, cfg.layers[blk][2].ln_beta,
        cfg.hidden[blk][2], buf_c_local, in_feat_sm, qh_sm);

    float s3_out_local[OUT3_K];
    ssla_layer_w<C3, C3, 3>(
        s3x, s3y, cfg.H3, cfg.W3, buf_c_local,
        cfg.layers[blk][3].qvgIn, cfg.layers[blk][3].goW,
        cfg.layers[blk][3].input_proj,
        cfg.layers[blk][3].ln_gamma, cfg.layers[blk][3].ln_beta,
        cfg.hidden[blk][3], s3_out_local, in_feat_sm, qh_sm);

    // Materialize s3 features to out_slot for the P1 oracle diff.
    if (out_slot != 0 && is_owner) {
        #pragma unroll
        for (int k = 0; k < OUT3_K; ++k) {
            const int o = lane + 32 * k;
            if (o < C3) out_slot->s3_feat[o] = s3_out_local[k];
        }
        __syncwarp();
    }

    // s3 tdrop.
    {
        unsigned char pre = 0;
        unsigned char* td = cfg.tdrop_s3[blk];
        const int idx = s3y * cfg.W3 + s3x;
        if (lane == 0) {
            pre = td[idx];
            td[idx] = (unsigned char)(pre + 1u);
        }
        pre = (unsigned char)__shfl_sync(0xffffffff, pre, 0);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }
    if (out_slot != 0 && is_owner && lane == 0) out_slot->pass3 = 1;
    if (!is_owner) return false;

    // ---- Head (owner only, only if head_W provided) ----
    if (cfg.head_W != 0) {
        // Materialize s3 features → in_feat_sm for the head matvec input.
        #pragma unroll
        for (int k = 0; k < OUT3_K; ++k) {
            const int o = lane + 32 * k;
            if (o < C3) in_feat_sm[o] = s3_out_local[k];
        }
        __syncwarp();

        constexpr int HEAD_K = (HEAD_OUT + 31) / 32;    // = 1 for HEAD_OUT=7
        float head_local[HEAD_K];
        matvec_w<C3, HEAD_OUT>(in_feat_sm, cfg.head_W, head_local);
        if (lane < HEAD_OUT) head_local[0] += cfg.head_b[lane];

        const HybridStrip& s = cfg.strip[blk];
        const int hx_local   = s3x - s.s3_owned_lo;
        const int s3_owned_w = s.s3_owned_hi - s.s3_owned_lo;
        if (hx_local >= 0 && hx_local < s3_owned_w) {
            const int cell_idx = s3y * s3_owned_w + hx_local;
            const long long off = (long long)cell_idx * HEAD_OUT;
            if (lane < HEAD_OUT) {
                cfg.preds[blk][off + lane] = head_local[0];
            }
            __syncwarp();
            if (lane == 0) cfg.version[blk][cell_idx] += 1u;
        }
    }
    return true;
}


// ============================================================================
// drain_n kernel — offline P1 harness. gridDim.x = 2, blockDim.x = n_warps*32.
//
// Block 0 reads ring0[0..n0); block 1 reads ring1[0..n1). Within each
// block, the n_warps warps each pull one event at a time from the ring
// and process it independently. Round-robin: warp w handles slot
// (t_local + w) for the current batch; t_local advances by n_warps per
// batch.
//
// Output slot layout matches the coop kernel: out[i] holds the result
// of processing ring[i] (per block).
// ============================================================================
extern "C" __global__ void k_ssla_s2s3_w_drain_n(
    const HybridS2S3Config*   cfg_in,
    const HybridInputRec*     ring0,
    const HybridInputRec*     ring1,
    int                       n0,
    int                       n1,
    HybridS2S3OutputSlot*     out0,
    HybridS2S3OutputSlot*     out1)
{
    if (blockIdx.x >= 2) return;
    extern __shared__ float smem[];
    const int n_warps = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;

    float* my_smem    = smem + warp_id * W_SCRATCH_PER_WARP_FLOATS;
    float* in_feat_sm = my_smem;
    float* qh_sm      = my_smem + 96;

    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg = *cfg_in;
    const HybridInputRec*   ring = (blk == 0) ? ring0 : ring1;
    HybridS2S3OutputSlot*   out  = (blk == 0) ? out0  : out1;
    const int               n    = (blk == 0) ? n0    : n1;

    for (int i_base = 0; i_base < n; i_base += n_warps) {
        const int i = i_base + warp_id;
        if (i >= n) break;
        HybridS2S3OutputSlot* slot = (out != 0) ? &out[i] : (HybridS2S3OutputSlot*)0;
        ssla_s2s3_step_warp<24, 48, 96, 7>(cfg, blk, ring[i], slot,
                                             in_feat_sm, qh_sm);
        __syncwarp();
    }
}


// ============================================================================
// persistent kernel — live runner.
//
// Each block polls its per-block ring. Per batch, n_warps warps each
// grab one event at offsets t_local + 0 .. t_local + n_warps − 1.
// Wait until ALL n_warps events have seq_done published before
// processing (simpler than partial-batch handling; CPU-side admits at
// far above the GPU drain so the ring fills fast enough).
// ============================================================================
extern "C" __global__ void k_ssla_s2s3_w_persistent(
    const HybridS2S3Config*       cfg_in,
    HybridInputRec*               ring0,
    HybridInputRec*               ring1,
    unsigned long long            ring_mask,
    volatile unsigned long long*  tail0,
    volatile unsigned long long*  tail1,
    volatile int*                 stop_flag,
    volatile unsigned long long*  events_done0,
    volatile unsigned long long*  events_done1,
    unsigned int                  spin_ns)
{
    if (blockIdx.x >= 2) return;
    extern __shared__ float smem[];
    const int n_warps = blockDim.x >> 5;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    float* my_smem    = smem + warp_id * W_SCRATCH_PER_WARP_FLOATS;
    float* in_feat_sm = my_smem;
    float* qh_sm      = my_smem + 96;

    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg = *cfg_in;
    HybridInputRec* ring                  = (blk == 0) ? ring0 : ring1;
    volatile unsigned long long* tail     = (blk == 0) ? tail0 : tail1;
    volatile unsigned long long* events_done = (blk == 0) ? events_done0 : events_done1;

    unsigned long long t_local = *tail;
    __shared__ int want_stop;
    __shared__ int batch_ready;

    while (true) {
        if (threadIdx.x == 0) want_stop = *stop_flag;
        __syncthreads();
        if (want_stop) return;

        // Wait for the slowest of n_warps slots in the next batch.
        if (threadIdx.x == 0) {
            int ready_count = 0;
            for (int w = 0; w < n_warps; ++w) {
                volatile unsigned long long* seq_p =
                    &ring[(t_local + w) & ring_mask].seq_done;
                if (*seq_p == t_local + w + 1ull) ready_count++;
                else break;
            }
            batch_ready = (ready_count == n_warps) ? 1 : 0;
            if (!batch_ready) __nanosleep(spin_ns);
        }
        __syncthreads();
        if (!batch_ready) continue;

        if (threadIdx.x == 0) __threadfence_system();
        __syncthreads();

        // Per-warp event = ring[t_local + warp_id].
        const unsigned long long my_idx = t_local + (unsigned long long)warp_id;
        const HybridInputRec& src = ring[my_idx & ring_mask];

        unsigned long long t_pop = 0;
        if (lane == 0 && cfg.timing[blk] != 0) t_pop = clock64();

        const bool owner_pass = ssla_s2s3_step_warp<24, 48, 96, 7>(
            cfg, blk, src, (HybridS2S3OutputSlot*)0, in_feat_sm, qh_sm);

        if (lane == 0 && cfg.timing[blk] != 0) {
            const unsigned long long t_done = clock64();
            const unsigned int idx = (unsigned int)(my_idx & cfg.timing_mask);
            GpuTimingSlot* ts = &cfg.timing[blk][idx];
            ts->t_pop_clk  = t_pop;
            ts->t_done_clk = t_done;
            ts->seq        = (unsigned int)my_idx;
            ts->owner      = owner_pass ? 1u : 0u;
        }

        __syncthreads();
        t_local += (unsigned long long)n_warps;
        if (threadIdx.x == 0) {
            *tail        = t_local;
            *events_done = t_local;
            __threadfence_system();
        }
        __syncthreads();
    }
}
