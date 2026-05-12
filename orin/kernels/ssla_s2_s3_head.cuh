// SSLA-S stages 2 + 3 + (P2: head) per-event step — hybrid CPU/GPU deploy.
//
// CPU runs stages 0+1 (libstage01_capi.so). GPU receives post-stage-1
// records (t, x, y, feat1[24]) via per-block pinned rings. This kernel
// implements the GPU side: 2 persistent blocks × strict spatial sharding
// + halo=2 at s2 res, S9-spatial style (per-block private hidden state,
// no atomics on device). Coop-per-event within each block (256 threads
// work on one event at a time; events FIFO from per-block ring).
//
// Per the design in deploy/orin/HYBRID_DESIGN.md:
//   * Both halo and owner run s2 forward.
//   * Both halo and owner update their PRIVATE s2 tdrop counter and
//     decide pass independently. (Halo's private-counter drift at
//     halo cells is the bounded drift documented for S8.)
//   * Both halo and owner run s3 forward (gated by their pass2).
//   * Both halo and owner update their PRIVATE s3 tdrop counter.
//   * Owner alone writes head / predictions. (Head matvec lands in P2;
//     this header writes raw s3 features to a debug output slot for
//     P1 oracle diff.)
//
// No atomics, no spinlocks. Only __syncthreads inside the layer
// primitives' reductions. No __syncwarp shenanigans — coop-per-event
// is the lock-free invariant.

#pragma once
#define SSLA_PRIM_NO_TEST_WRAPPERS
#include "ssla_primitives.cuh"
#include "ssla_layer.cuh"

// Match LayerWeights from ssla_step.cuh so we can reuse export weights.
struct LayerWeightsS2S3 {
    const float* qvgIn;        // (A, IN, 3*OUT) — already transposed for coalesced load
    const float* goW;          // (A, OUT, OUT)
    const float* input_proj;   // (IN, OUT) — may be NULL when IN==OUT
    const float* ln_gamma;     // (OUT,)
    const float* ln_beta;      // (OUT,)
};

// Per-block strip configuration.
struct HybridStrip {
    int owned_lo;       // s2-x range owned by this block (inclusive)
    int owned_hi;       // (exclusive)
    int s3_owned_lo;    // s3-x range owned (= owned_lo >> 1, set by host)
    int s3_owned_hi;    // (= owned_hi >> 1)
};

// Per-event GPU timing slot. The persistent kernel records clock64() at
// slot-pop (after threadfence) and at end-of-event (post head write or
// drop). Indexed by `slot_idx & timing_mask` with timing_mask sized to
// keep the last N events. t_done_clk == 0 means the slot was never
// written (rolling buffer head).
struct GpuTimingSlot {
    unsigned long long t_pop_clk;
    unsigned long long t_done_clk;
    unsigned int       seq;     // event seq (logical pop position)
    unsigned int       owner;   // 1 if owner-pass-3 reached head; 0 otherwise
    unsigned long long t_push_ns;   // copied from HybridInputRec at process time
};

// Shared config readable by both blocks.
struct HybridS2S3Config {
    int H2, W2;                 // s2 grid (post-CPU-stage-1 pool resolution)
    int H3, W3;                 // s3 grid (post-stage-2 pool)
    int tdrop_window;
    int head_out_dim;           // = 5 + N_classes (e.g. 7 for Gen1's 2 classes).
                                // Runtime field is informational; the kernel
                                // uses HEAD_OUT as a template constant.
    HybridStrip      strip[2];
    LayerWeightsS2S3 layers[2][4];     // [blockIdx.x][L4..L7]
    float*           hidden[2][4];      // per-block, full-grid hidden state
    unsigned char*   tdrop_s2[2];       // per-block (H2*W2 bytes)
    unsigned char*   tdrop_s3[2];       // per-block (H3*W3 bytes)
    // Head (shared across blocks; NULL = skip head matvec, used by the
    // P1 oracle harness which compares s3 features directly).
    const float*     head_W;            // (C3, HEAD_OUT) row-major
    const float*     head_b;            // (HEAD_OUT,)
    // Per-block predictions (pinned host alloc on Tegra). Layout for block b:
    //   preds[b][hy * s3_owned_w + hx_local][HEAD_OUT]
    // where s3_owned_w = strip[b].s3_owned_hi - strip[b].s3_owned_lo
    //       hx_local  = s3x - strip[b].s3_owned_lo.
    float*           preds[2];
    unsigned int*    version[2];        // sized H3 * s3_owned_w per block
    // GPU timing — NULL = disabled. timing_mask = capacity − 1.
    GpuTimingSlot*   timing[2];
    unsigned int     timing_mask;
    unsigned int     _pad_timing;
    // Per-block calibration: see ssla_s2_s3_head_celled.cuh for details.
    unsigned long long* kernel_start_clk[2];
    unsigned long long* kernel_end_clk[2];
};

// Per-event input record (CPU pushes; GPU pops). 112 bytes.
//
// `seq_done` is the MPSC publication tag. Producer at logical position N
// writes the record fields, then publishes seq_done = N+1 with release
// semantics. Consumer at logical tail T spins until ring[T & mask].seq_done
// == T+1 before reading the record. Initial / empty slots have seq_done = 0.
//
// Slots wrap every `capacity` writes; on each lap the seq_done value
// advances by `capacity`. uint64 covers ~1.8e19 slots — no wrap concern.
//
// The drain_n offline harness pre-fills the ring sequentially and does
// not check seq_done; the field is only consulted by the persistent
// kernel.
struct HybridInputRec {
    unsigned long long seq_done;     // logical_pos + 1 when ready
    float              t;
    unsigned short     x;            // s2-res x
    unsigned short     y;            // s2-res y
    float              feat1[24];    // C1 = 24 features post-CPU-stage-1
    unsigned long long t_push_ns;    // CPU CLOCK_MONOTONIC_RAW ns at ring publish
};

// Per-event debug output (P1 diff). The kernel writes it for both
// owner and halo events. Halo events have pass2 / pass3 = -1.
struct HybridS2S3OutputSlot {
    int   pass2;            // 1 = owner+passed; 0 = owner+dropped; -1 = halo
    int   pass3;
    float s3_feat[96];      // C3 = 96 post-stage-3 features
};


// Returns true iff event reached s3 head (i.e. owner & passed both s2 + s3).
// HEAD_OUT is the head's output dim (5 box + N_classes). When
// `cfg.head_W == nullptr`, the head matvec is skipped — used by the P1
// oracle harness which only diffs s3 features.
template <int C1, int C2, int C3, int HEAD_OUT>
__device__ inline bool ssla_s2s3_step_block(
    const HybridS2S3Config& cfg, int blk,
    const HybridInputRec& rec,
    HybridS2S3OutputSlot* out_slot,
    float* __restrict__ smem)
{
    const int tid = threadIdx.x;
    float* feat_in   = smem;                       // C1 floats
    float* buf_a     = smem + C1;                  // C3 floats (ping-pong)
    float* buf_b     = smem + C1 + C3;             // C3 floats
    float* layer_sm  = smem + C1 + 2 * C3;         // 5*C3 floats

    // Load feat1 from rec into shared. This is the only host-memory
    // dependency in the per-event critical path.
    for (int i = tid; i < C1; i += SSLA_BLOCK_THREADS) feat_in[i] = rec.feat1[i];
    __syncthreads();

    int evx = (int)rec.x;
    int evy = (int)rec.y;
    const HybridStrip& strip = cfg.strip[blk];
    const bool is_owner = (evx >= strip.owned_lo) && (evx < strip.owned_hi);

    if (out_slot != 0 && tid == 0) {
        out_slot->pass2 = is_owner ? 0 : -1;
        out_slot->pass3 = is_owner ? 0 : -1;
    }

    // ---- Stage 2 forward (layers 4 + 5) — both halo and owner run ----
    // Layer 4 : C1=24 → C2=48 (K=3)
    ssla_layer_forward_ct<C1, C2, 3>(
        evx, evy, cfg.H2, cfg.W2, feat_in,
        cfg.layers[blk][0].qvgIn, cfg.layers[blk][0].goW,
        cfg.layers[blk][0].input_proj,
        cfg.layers[blk][0].ln_gamma, cfg.layers[blk][0].ln_beta,
        cfg.hidden[blk][0], buf_a, layer_sm);

    // Layer 5 : C2=48 → C2=48 (K=3, no input_proj)
    ssla_layer_forward_ct<C2, C2, 3>(
        evx, evy, cfg.H2, cfg.W2, buf_a,
        cfg.layers[blk][1].qvgIn, cfg.layers[blk][1].goW,
        cfg.layers[blk][1].input_proj,
        cfg.layers[blk][1].ln_gamma, cfg.layers[blk][1].ln_beta,
        cfg.hidden[blk][1], buf_b, layer_sm);

    // ---- Private s2 tdrop counter — both halo and owner run their own ----
    // Halo's counter at boundary cells drifts vs oracle (halo doesn't see
    // events outside its proc range that touch boundary cells); this is
    // the bounded drift documented for S8.
    bool my_pass2 = true;
    {
        const int idx = evy * cfg.W2 + evx;
        unsigned char* td = cfg.tdrop_s2[blk];
        if (tid == 0) td[idx] = (unsigned char)(td[idx] + 1u);
        __syncthreads();
        const unsigned char pre = (unsigned char)(td[idx] - 1u);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) my_pass2 = false;
    }
    if (out_slot != 0 && is_owner && tid == 0) out_slot->pass2 = my_pass2 ? 1 : 0;
    if (!my_pass2) return false;

    int s3x = evx >> 1;
    int s3y = evy >> 1;

    // ---- Stage 3 forward (layers 6 + 7) ----
    // Layer 6 : C2=48 → C3=96 (K=3)
    ssla_layer_forward_ct<C2, C3, 3>(
        s3x, s3y, cfg.H3, cfg.W3, buf_b,
        cfg.layers[blk][2].qvgIn, cfg.layers[blk][2].goW,
        cfg.layers[blk][2].input_proj,
        cfg.layers[blk][2].ln_gamma, cfg.layers[blk][2].ln_beta,
        cfg.hidden[blk][2], buf_a, layer_sm);

    // Layer 7 : C3=96 → C3=96 (K=3, no input_proj)
    ssla_layer_forward_ct<C3, C3, 3>(
        s3x, s3y, cfg.H3, cfg.W3, buf_a,
        cfg.layers[blk][3].qvgIn, cfg.layers[blk][3].goW,
        cfg.layers[blk][3].input_proj,
        cfg.layers[blk][3].ln_gamma, cfg.layers[blk][3].ln_beta,
        cfg.hidden[blk][3], buf_b, layer_sm);

    if (out_slot != 0 && is_owner) {
        for (int i = tid; i < C3; i += SSLA_BLOCK_THREADS) {
            out_slot->s3_feat[i] = buf_b[i];
        }
        __syncthreads();
    }

    // ---- Private s3 tdrop counter ----
    bool my_pass3 = true;
    {
        const int idx = s3y * cfg.W3 + s3x;
        unsigned char* td = cfg.tdrop_s3[blk];
        if (tid == 0) td[idx] = (unsigned char)(td[idx] + 1u);
        __syncthreads();
        const unsigned char pre = (unsigned char)(td[idx] - 1u);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) my_pass3 = false;
    }
    if (out_slot != 0 && is_owner && tid == 0) out_slot->pass3 = my_pass3 ? 1 : 0;
    if (!my_pass3) return false;

    // ---- Head — owner only, only if config has head weights ----
    // Halo blocks do not write predictions (they hit the !is_owner return
    // path earlier). For owner-pass-3, run a simple linear head:
    //   y[o] = sum_i head_W[i, o] * s3_feat[i] + head_b[o]
    // and write y[..] to cfg.preds[blk] at the cell's local index. Bump
    // the version counter so host can detect freshness.
    if (is_owner && cfg.head_W != 0) {
        // Reuse buf_a (size C3 ≥ HEAD_OUT for any reasonable HEAD_OUT)
        // as the head output buffer. buf_a was last written by L7 input
        // which is no longer needed.
        matvec_ct<C3, HEAD_OUT>(buf_b, cfg.head_W, cfg.head_b, buf_a);

        const HybridStrip& s = cfg.strip[blk];
        const int hx_local   = s3x - s.s3_owned_lo;
        const int s3_owned_w = s.s3_owned_hi - s.s3_owned_lo;
        if (hx_local >= 0 && hx_local < s3_owned_w) {
            const int cell_idx = s3y * s3_owned_w + hx_local;
            const long long off = (long long)cell_idx * HEAD_OUT;
            for (int o = tid; o < HEAD_OUT; o += SSLA_BLOCK_THREADS) {
                cfg.preds[blk][off + o] = buf_a[o];
            }
            __syncthreads();
            if (tid == 0) {
                cfg.version[blk][cell_idx] = cfg.version[blk][cell_idx] + 1u;
            }
            __syncthreads();
        }
    }
    return is_owner;
}

// ===========================================================================
// Drain-N kernel (gridDim.x = 2).
//
// Block 0 reads ring0[0..n0); Block 1 reads ring1[0..n1). Each block
// processes its ring sequentially (FIFO). Used by the offline P1
// harness to verify equivalence vs CPU oracle.
// ===========================================================================
extern "C" __global__ void k_ssla_s2s3_drain_n(
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
    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg = *cfg_in;
    const HybridInputRec*   ring = (blk == 0) ? ring0 : ring1;
    HybridS2S3OutputSlot*   out  = (blk == 0) ? out0  : out1;
    const int               n    = (blk == 0) ? n0    : n1;

    for (int i = 0; i < n; ++i) {
        ssla_s2s3_step_block<24, 48, 96, 7>(cfg, blk, ring[i],
                                          (out != 0) ? &out[i] : (HybridS2S3OutputSlot*)0,
                                          smem);
    }
}

// ===========================================================================
// Persistent kernel (gridDim.x = 2). Polls per-block input rings forever
// until *stop_flag != 0. Each block is single-consumer for its own ring;
// CPU side has multiple producers per ring (4 lib_stage01_to_gpu shard
// threads). Producer ↔ consumer sync via seq_done tag + threadfence
// (Tegra pinned-host memory is coherent — GPU L1 bypassed for these).
//
// `tail[b]` is the logical tail (monotonic, host-visible for debug).
// `events_done[b]` mirrors tail[b] and is what the host watches.
// ===========================================================================
extern "C" __global__ void k_ssla_s2s3_persistent(
    const HybridS2S3Config*       cfg_in,
    HybridInputRec*               ring0,
    HybridInputRec*               ring1,
    unsigned long long            ring_mask,           // capacity − 1
    volatile unsigned long long*  tail0,
    volatile unsigned long long*  tail1,
    volatile int*                 stop_flag,
    volatile unsigned long long*  events_done0,
    volatile unsigned long long*  events_done1,
    unsigned int                  spin_ns)
{
    if (blockIdx.x >= 2) return;
    extern __shared__ float smem[];
    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg  = *cfg_in;
    HybridInputRec*  ring        = (blk == 0) ? ring0 : ring1;
    volatile unsigned long long* tail        = (blk == 0) ? tail0        : tail1;
    volatile unsigned long long* events_done = (blk == 0) ? events_done0 : events_done1;

    unsigned long long t_local = *tail;
    __shared__ int     want_stop;
    __shared__ int     slot_ready;
    __shared__ HybridInputRec rec_sm;

    while (true) {
        // Stop-flag check (lane 0 reads, broadcast via shared mem).
        if (threadIdx.x == 0) want_stop = *stop_flag;
        __syncthreads();
        if (want_stop) return;

        // Wait for slot at t_local to be published. The producer publishes
        // by writing seq_done = t_local + 1 after the record is filled.
        if (threadIdx.x == 0) {
            volatile unsigned long long* seq_p = &ring[t_local & ring_mask].seq_done;
            const unsigned long long expected = t_local + 1ull;
            slot_ready = (*seq_p == expected) ? 1 : 0;
            if (!slot_ready) __nanosleep(spin_ns);
        }
        __syncthreads();
        if (!slot_ready) continue;

        // Slot ready — fence to order record reads after seq_done observation.
        // Tegra pinned host memory is coherent so a system-scope fence is
        // sufficient.
        if (threadIdx.x == 0) __threadfence_system();
        __syncthreads();

        // Cooperative copy of the record into shared memory so all 256
        // threads have it for the per-event step (avoids each thread
        // re-loading from host memory through the unified bus).
        const HybridInputRec* src = &ring[t_local & ring_mask];
        const int rec_words = (int)(sizeof(HybridInputRec) / 4);
        for (int i = threadIdx.x; i < rec_words; i += SSLA_BLOCK_THREADS) {
            ((unsigned int*)&rec_sm)[i] = ((const unsigned int*)src)[i];
        }
        __syncthreads();

        // Per-event timing: capture clock at start of work.
        unsigned long long t_pop = 0;
        if (threadIdx.x == 0 && cfg.timing[blk] != 0) t_pop = clock64();

        const bool owner_pass = ssla_s2s3_step_block<24, 48, 96, 7>(
            cfg, blk, rec_sm, (HybridS2S3OutputSlot*)0, smem);

        if (threadIdx.x == 0 && cfg.timing[blk] != 0) {
            const unsigned long long t_done = clock64();
            const unsigned int idx = (unsigned int)(t_local & cfg.timing_mask);
            GpuTimingSlot* ts = &cfg.timing[blk][idx];
            ts->t_pop_clk  = t_pop;
            ts->t_done_clk = t_done;
            ts->seq        = (unsigned int)t_local;
            ts->owner      = owner_pass ? 1u : 0u;
        }

        ++t_local;
        if (threadIdx.x == 0) {
            *tail        = t_local;
            *events_done = t_local;
            __threadfence_system();
        }
        __syncthreads();
    }
}
