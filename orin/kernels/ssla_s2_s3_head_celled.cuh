// SSLA-S stages 2 + 3 + head — CELL-OWNER WARP design.
//
// User-proposed alternative to "1 warp = 1 event":
//   * Each block has N_WARPS = 9 warps (= K*K patch positions).
//   * Each hidden state cell (cy, cx) is OWNED by exactly one warp:
//       warp_id = (cy % 3) * 3 + (cx % 3)
//   * Within an event's 3×3 patch update, the 9 lru_step writes naturally
//     land on 9 cells owned by 9 different warps → race-free hidden state.
//   * tdrop counter for an event center cell is also owned by one warp →
//     same-cell events serialize within that warp → race-free tdrop.
//
// Per-layer phases (4 layers L4..L7 per block):
//   1. DISPATCH: thread 0 builds per-warp task lists from BATCH events ×
//      9 deltas each.
//   2. RESIDUAL: one warp per event computes input_proj @ in_feat (or
//      passthrough when IN==OUT).
//   3. COMPUTE: all 9 warps process their task queues in parallel; each
//      task does qvg matvec, lru_step on warp's OWNED cell, goW matvec
//      writing contribution to contrib[event][warp].
//   4. GATHER + RES + LN: one warp per event sums 9 contributions, adds
//      residual, applies LayerNorm, writes next-layer in_feat.
//   5. (After L5) TDROP_S2: thread 0 serializes counter increments in
//      arrival order, decides pass2 per event.
//   6. (After L7) TDROP_S3: same for s3.
//   7. HEAD: one warp per surviving owner event runs C3→7 matvec, writes
//      prediction.
//
// SMEM layout per block (BATCH=8):
//   batch_state[8]              event metadata, in_feat, residual    ~6 KB
//   contrib[8][9][96]           per-event per-warp staging          27 KB
//   per_warp_scratch[9][192]    in_feat_sm + qh_sm per warp          7 KB
//   task lists, pass flags                                         ~1 KB
//   Total                                                          ~41 KB
//
// Note: this kernel re-uses the struct definitions from
// ssla_s2_s3_head.cuh (HybridS2S3Config, HybridInputRec, etc.) since the
// CPU side / hybrid_common.py ctypes layouts are stable. The new kernel
// is a separate file so it can be compiled standalone.

#pragma once
#include "proto_layer_pair.cuh"     // matvec_w, lru_step_w, layernorm_w (reused)

// ---- Reuse struct layouts identical to ssla_s2_s3_head.cuh ----------------

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
};

struct GpuTimingSlot {
    unsigned long long t_pop_clk;
    unsigned long long t_done_clk;
    unsigned int       seq;
    unsigned int       owner;
};

struct HybridS2S3Config {
    int H2, W2;
    int H3, W3;
    int tdrop_window;
    int head_out_dim;
    HybridStrip      strip[2];
    LayerWeightsS2S3 layers[2][4];
    float*           hidden[2][4];
    unsigned char*   tdrop_s2[2];
    unsigned char*   tdrop_s3[2];
    const float*     head_W;
    const float*     head_b;
    float*           preds[2];
    unsigned int*    version[2];
    GpuTimingSlot*   timing[2];
    unsigned int     timing_mask;
};

struct HybridInputRec {
    unsigned long long seq_done;
    float              t;
    unsigned short     x;
    unsigned short     y;
    float              feat1[24];
};

struct HybridS2S3OutputSlot {
    int   pass2;
    int   pass3;
    float s3_feat[96];
};

// ---- Architecture constants ------------------------------------------------

constexpr int C1 = 24;
constexpr int C2 = 48;
constexpr int C3 = 96;
constexpr int HEAD_OUT_DEFAULT = 7;
constexpr int N_WARPS = 9;          // exactly K*K = 9 for K=3
constexpr int BATCH = 8;            // events per batch
constexpr int OUT_MAX = C3;         // largest layer output dim

// Per-warp scratch: in_feat_sm (OUT_MAX) + qh_sm (OUT_MAX) = 192 floats
constexpr int W_SCRATCH_PER_WARP = 2 * OUT_MAX;


// ---- Single-patch primitive: factored out of ssla_layer_w -----------------
//
// Processes ONE patch position for ONE event:
//   1. qvg matvec from event's in_feat using qvgIn[delta]
//   2. lru_step on h_cell (cell owned by this warp — no race)
//   3. goW matvec from qh, written to contribution slot (NOT accumulated;
//      caller assembles per-event sum via gather).
//
// in_feat_sm:  shared, IN floats (event's input feature broadcast to warp)
// qh_sm:       shared, OUT floats (temporary within one patch)
// contrib_sm:  shared, OUT floats (per-warp per-event slot for gather)
// qvgIn:       weights for this delta, (IN, 3*OUT) row-major
// goW:         weights for this delta, (OUT, OUT) row-major
// h_cell:      hidden state at the warp-owned patch cell, (OUT,) global
template <int IN, int OUT>
__device__ inline void process_patch_cell(
    const float* __restrict__ in_feat_sm,
    const float* __restrict__ qvgIn,    // (IN, 3*OUT) for this delta
    const float* __restrict__ goW,      // (OUT, OUT) for this delta
    float*       __restrict__ h_cell,   // global, owned by this warp
    float*       __restrict__ qh_sm,    // shared per-warp
    float*       __restrict__ contrib_sm) // shared per-event per-warp
{
    constexpr int OUT_K = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;

    // qvg matvec: 3 separate matvecs aliasing the same (IN, 3*OUT) layout
    float q_local[OUT_K], v_local[OUT_K], g_local[OUT_K];
    matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvgIn + 0 * OUT, q_local);
    matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvgIn + 1 * OUT, v_local);
    matvec_w<IN, OUT, 3 * OUT>(in_feat_sm, qvgIn + 2 * OUT, g_local);

    // lru_step on owned cell — only this warp writes to h_cell, no race.
    float qh_local[OUT_K];
    lru_step_w<OUT>(g_local, v_local, q_local, h_cell, qh_local);

    // Materialize qh into shared so goW matvec can broadcast it across lanes.
    #pragma unroll
    for (int k = 0; k < OUT_K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) qh_sm[o] = qh_local[k];
    }
    __syncwarp();

    // goW matvec: contribution = goW @ qh. NOT accumulated — caller's
    // gather phase sums across all 9 warps' contribs per event.
    float contrib_local[OUT_K];
    matvec_w<OUT, OUT, OUT, /*ACCUM=*/false>(qh_sm, goW, contrib_local);

    // Write to per-event per-warp contrib slot.
    #pragma unroll
    for (int k = 0; k < OUT_K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) contrib_sm[o] = contrib_local[k];
    }
    __syncwarp();
}


// ---- Helpers for gather + LN ---------------------------------------------

// Per-event gather: sum contrib[event][0..8][:] + residual, then LN.
// Output written to out_feat_sm.
template <int OUT>
__device__ inline void gather_residual_ln(
    float* __restrict__ contrib_event,   // [9][OUT], shared
    float* __restrict__ residual_event,  // [OUT], shared
    const float* __restrict__ ln_gamma,
    const float* __restrict__ ln_beta,
    float* __restrict__ out_feat_sm)     // [OUT], shared
{
    constexpr int OUT_K = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;

    // Load into lane-striped registers, sum 9 warps + residual.
    float sum_local[OUT_K];
    #pragma unroll
    for (int k = 0; k < OUT_K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) {
            float s = residual_event[o];
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                s += contrib_event[w * OUT + o];
            }
            sum_local[k] = s;
        } else {
            sum_local[k] = 0.0f;
        }
    }
    __syncwarp();

    // LayerNorm in lane-striped form.
    layernorm_w<OUT>(sum_local, ln_gamma, ln_beta);

    // Write back to out_feat_sm.
    #pragma unroll
    for (int k = 0; k < OUT_K; ++k) {
        const int o = lane + 32 * k;
        if (o < OUT) out_feat_sm[o] = sum_local[k];
    }
    __syncwarp();
}


// Compute residual (input_proj @ in_feat) into residual_event, or
// passthrough if input_proj == nullptr (only valid when IN == OUT).
template <int IN, int OUT>
__device__ inline void compute_residual(
    const float* __restrict__ in_feat_sm,       // [IN], shared
    const float* __restrict__ input_proj,       // [IN, OUT] or null
    float*       __restrict__ residual_event)   // [OUT], shared
{
    constexpr int OUT_K = (OUT + 31) / 32;
    const int lane = threadIdx.x & 31;

    if (input_proj != 0) {
        float r_local[OUT_K];
        matvec_w<IN, OUT>(in_feat_sm, input_proj, r_local);
        #pragma unroll
        for (int k = 0; k < OUT_K; ++k) {
            const int o = lane + 32 * k;
            if (o < OUT) residual_event[o] = r_local[k];
        }
    } else {
        // IN==OUT passthrough.
        if constexpr (IN == OUT) {
            #pragma unroll
            for (int k = 0; k < (IN + 31) / 32; ++k) {
                const int o = lane + 32 * k;
                if (o < IN) residual_event[o] = in_feat_sm[o];
            }
        }
    }
    __syncwarp();
}


// ---- Per-event slot in SMEM -----------------------------------------------
//
// One per batch slot: holds the current layer's input feat + residual,
// plus per-event metadata. Reused across layers (size = C3 = max).

struct EventSlot {
    float in_feat[OUT_MAX];      // current layer input (overwritten per layer)
    float residual[OUT_MAX];
    int   evx;                   // s2 cell coords at current grid res
    int   evy;
    int   is_owner;
    int   pass2;
    int   pass3;
};


// ---- One full layer in the celled design ----------------------------------
//
// `task_event[w][.]` and `task_delta[w][.]` were filled by a prior dispatch
// pass. The dispatch is reused across L4+L5 (s2 grid) and re-built for
// L6+L7 (s3 grid).
//
// Each task represents: process event e's patch at offset delta. The
// warp that owns the patch cell handles it.
//
// `contrib_sm` is the shared per-event per-warp staging buffer.
// `event_slots` holds in_feat (read) and is overwritten with the layer's
// output (gather + LN result) at the end of this function.
template <int IN, int OUT>
__device__ inline void run_layer_celled(
    int batch_size,
    int Hl, int Wl,
    EventSlot*               event_slots,            // [BATCH]
    const int*               task_event,             // [N_WARPS][BATCH]
    const int*               task_delta,             // [N_WARPS][BATCH]
    const int*               task_count,             // [N_WARPS]
    const LayerWeightsS2S3&  layer,
    float*                   hidden,                 // global, (Hl*Wl, OUT)
    float*                   contrib,                // shared, [BATCH][N_WARPS][OUT]
    float*                   per_warp_in_feat_sm,    // shared, [N_WARPS][IN] (broadcast slot)
    float*                   per_warp_qh_sm)         // shared, [N_WARPS][OUT]
{
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    // ----- RESIDUAL phase: one warp per event in parallel -----
    // Warp w handles event e where e = w (if w < batch_size).
    if (warp_id < batch_size) {
        const int e = warp_id;
        float* in_feat_sm = event_slots[e].in_feat;
        // copy/broadcast input proj path: residual lives in event_slots[e].residual
        compute_residual<IN, OUT>(in_feat_sm, layer.input_proj,
                                  event_slots[e].residual);
    }
    __syncthreads();

    // Zero contrib for all (event, warp) slots that won't be written. This
    // ensures gather sums 0's for missing patches (boundary cells).
    // 9 warps × BATCH × OUT floats; each thread zeros some.
    const int n_contrib_floats = batch_size * N_WARPS * OUT;
    for (int i = threadIdx.x; i < n_contrib_floats; i += blockDim.x) {
        contrib[i] = 0.0f;
    }
    __syncthreads();

    // ----- COMPUTE phase: warps process their task queues in parallel -----
    float* my_in_feat_sm = per_warp_in_feat_sm + warp_id * IN;
    float* my_qh_sm      = per_warp_qh_sm + warp_id * OUT;
    const int* my_tasks_event = task_event + warp_id * BATCH;
    const int* my_tasks_delta = task_delta + warp_id * BATCH;
    const int  n_tasks        = task_count[warp_id];

    for (int t = 0; t < n_tasks; ++t) {
        const int e     = my_tasks_event[t];
        const int delta = my_tasks_delta[t];

        // Load event e's in_feat into this warp's broadcast slot.
        for (int i = lane; i < IN; i += 32) {
            my_in_feat_sm[i] = event_slots[e].in_feat[i];
        }
        __syncwarp();

        // Compute patch_idx from event coords + delta.
        // delta layout matches ssla_layer.cuh: dy = 1 - delta/3, dx = 1 - delta%3.
        const int dy = 1 - (delta / 3);
        const int dx = 1 - (delta % 3);
        const int py = event_slots[e].evy + dy;
        const int px = event_slots[e].evx + dx;
        // Boundary check: skip if patch cell is OOB (dispatcher should have
        // already filtered these, but defensive).
        if (py < 0 || py >= Hl || px < 0 || px >= Wl) continue;
        const int patch_idx = py * Wl + px;

        // Per-(event, warp) contrib slot.
        float* contrib_slot = contrib + ((long long)e * N_WARPS + warp_id) * OUT;

        // qvgIn / goW pointers for this delta.
        constexpr int QVG_STRIDE = 3 * OUT * IN;
        constexpr int GOW_STRIDE = OUT * OUT;
        const float* qvg_w = layer.qvgIn + (long long)delta * QVG_STRIDE;
        const float* go_w  = layer.goW   + (long long)delta * GOW_STRIDE;

        process_patch_cell<IN, OUT>(
            my_in_feat_sm, qvg_w, go_w,
            hidden + (long long)patch_idx * OUT,
            my_qh_sm, contrib_slot);
    }
    __syncthreads();

    // ----- GATHER + RES + LN: one warp per event -----
    if (warp_id < batch_size) {
        const int e = warp_id;
        float* contrib_event = contrib + (long long)e * N_WARPS * OUT;
        gather_residual_ln<OUT>(
            contrib_event,
            event_slots[e].residual,
            layer.ln_gamma, layer.ln_beta,
            event_slots[e].in_feat);    // write back into in_feat for next layer
    }
    __syncthreads();
}


// ---- Dispatch: build per-warp task lists from BATCH events × 9 deltas ----
//
// Thread 0 only. Fills task_event[N_WARPS][BATCH], task_delta[N_WARPS][BATCH],
// task_count[N_WARPS]. Uses event coords at the current grid (s2 or s3 res).
// Skips deltas where the patch cell is OOB.
__device__ inline void dispatch_tasks(
    int batch_size, int Hl, int Wl,
    const EventSlot*  event_slots,
    const int*  mask,           // optional: skip event e if mask[e] == 0. NULL = include all.
    int*  task_event,    // [N_WARPS][BATCH], shared
    int*  task_delta,    // [N_WARPS][BATCH], shared
    int*  task_count)    // [N_WARPS], shared
{
    if (threadIdx.x != 0) return;
    // Zero counts.
    #pragma unroll
    for (int w = 0; w < N_WARPS; ++w) task_count[w] = 0;
    // For each event × 9 deltas, route to owner warp by patch cell.
    for (int e = 0; e < batch_size; ++e) {
        if (mask != 0 && mask[e] == 0) continue;
        const int evx = event_slots[e].evx;
        const int evy = event_slots[e].evy;
        for (int delta = 0; delta < 9; ++delta) {
            const int dy = 1 - (delta / 3);
            const int dx = 1 - (delta % 3);
            const int py = evy + dy;
            const int px = evx + dx;
            if (py < 0 || py >= Hl || px < 0 || px >= Wl) continue;
            const int owner = (py % 3) * 3 + (px % 3);
            const int t = task_count[owner]++;
            // Task list is BATCH-wide per warp; in worst case all 9 deltas
            // of all BATCH events land on one warp (theoretically possible
            // but extremely unlikely with K=3 mod-3 hash). Cap at BATCH-1.
            // (Realistic max per warp: BATCH, since each event contributes
            // at most one task per warp due to mod-3 partition.)
            task_event[owner * BATCH + t] = e;
            task_delta[owner * BATCH + t] = delta;
        }
    }
}


// ---- Serialized tdrop check (thread 0) -----------------------------------
//
// Walks batch in arrival order, increments per-cell counter, decides pass.
// `event_cell_idx` is the s2 or s3 cell index where the event "is" (for
// counter location).
__device__ inline void serial_tdrop(
    int batch_size, int window,
    unsigned char* td,
    const EventSlot* event_slots,
    int Wl,
    const int* mask,    // optional: skip event e if mask[e] == 0. NULL = process all.
    int* pass_flags,    // [BATCH], shared (written for every e; 0 if masked out)
    bool is_s3)
{
    if (threadIdx.x != 0) return;
    for (int e = 0; e < batch_size; ++e) {
        if (mask != 0 && mask[e] == 0) {
            pass_flags[e] = 0;
            continue;
        }
        // s2_tdrop check is at the EVENT center cell (in s2 coords).
        // s3_tdrop check is at the s3 cell (event center after pool).
        // event_slots[e].evx/evy have already been pool-shifted by the caller
        // before calling tdrop_s3, so we just read them directly here.
        const int evx = event_slots[e].evx;
        const int evy = event_slots[e].evy;
        const int idx = evy * Wl + evx;
        const unsigned char pre = td[idx];
        td[idx] = (unsigned char)(pre + 1u);
        if (window > 1 && (pre % window) != 0u) {
            pass_flags[e] = 0;
        } else {
            pass_flags[e] = 1;
        }
    }
    (void)is_s3;
}


// ---- Drain-N kernel for P1 oracle harness --------------------------------
//
// Same interface as ssla_s2_s3_head_w.cuh's drain_n: process up to n0/n1
// events from ring0/ring1, write per-event output (pass2, pass3, s3_feat)
// to out0/out1.
// drain_n with optional per-batch clock64 timing.
// `batch_clk0/1`: optional device buffer sized ≥ ceil(n/BATCH) per block.
//                 If non-null, each batch writes its (t_pop, t_done) pair.
//                 NULL = no instrumentation (used by P1 oracle harness).
extern "C" __global__
__launch_bounds__(N_WARPS * 32, 1)   // 288 threads/block; allow MAX_THREADS=288
void k_ssla_s2s3_celled_drain_n(
    const HybridS2S3Config*   cfg_in,
    const HybridInputRec*     ring0,
    const HybridInputRec*     ring1,
    int                       n0,
    int                       n1,
    HybridS2S3OutputSlot*     out0,
    HybridS2S3OutputSlot*     out1,
    unsigned long long*       batch_clk0,
    unsigned long long*       batch_clk1)
{
    if (blockIdx.x >= 2) return;
    extern __shared__ float smem_raw[];
    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg = *cfg_in;
    const HybridInputRec*   ring = (blk == 0) ? ring0 : ring1;
    HybridS2S3OutputSlot*   out  = (blk == 0) ? out0  : out1;
    const int               n    = (blk == 0) ? n0    : n1;

    // ---- SMEM layout (offsets in floats / ints) ----
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

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const HybridStrip& strip = cfg.strip[blk];

    unsigned long long* batch_clk = (blk == 0) ? batch_clk0 : batch_clk1;

    // Process the input ring in chunks of BATCH.
    int batch_idx = 0;
    for (int batch_base = 0; batch_base < n; batch_base += BATCH, ++batch_idx) {
        const int batch_size = min(BATCH, n - batch_base);
        unsigned long long t_batch_pop_local = 0;
        if (threadIdx.x == 0 && batch_clk != 0) t_batch_pop_local = clock64();

        // Load batch records into event_slots, cooperatively.
        for (int e = 0; e < batch_size; ++e) {
            const HybridInputRec& rec = ring[batch_base + e];
            // feat1 → event_slots[e].in_feat (24 floats); zero rest up to OUT_MAX
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
        __syncthreads();

        // ---- Stage 2 ----
        // Dispatch at s2 grid (events use their s2 coords). All events
        // pass through L4+L5; pass2 mask only used to gate s3 work.
        dispatch_tasks(batch_size, cfg.H2, cfg.W2, event_slots, /*mask=*/0,
                       task_event, task_delta, task_count);
        __syncthreads();
        // Run L4 (C1 → C2)
        run_layer_celled<C1, C2>(batch_size, cfg.H2, cfg.W2,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][0], cfg.hidden[blk][0],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        // Run L5 (C2 → C2) — reuse same dispatch (same s2 grid).
        run_layer_celled<C2, C2>(batch_size, cfg.H2, cfg.W2,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][1], cfg.hidden[blk][1],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        // Serialized tdrop_s2 (at event center cell, in s2 coords).
        serial_tdrop(batch_size, cfg.tdrop_window,
                      cfg.tdrop_s2[blk], event_slots, cfg.W2, /*mask=*/0,
                      pass2, /*is_s3=*/false);
        __syncthreads();

        // Pool s2 coords → s3 coords (in-place in event_slots).
        if (threadIdx.x == 0) {
            for (int e = 0; e < batch_size; ++e) {
                event_slots[e].evx >>= 1;
                event_slots[e].evy >>= 1;
            }
        }
        __syncthreads();

        // ---- Stage 3 ----
        // Only pass2 events contribute to s3 hidden state (matches CPU oracle).
        dispatch_tasks(batch_size, cfg.H3, cfg.W3, event_slots, /*mask=*/pass2,
                       task_event, task_delta, task_count);
        __syncthreads();
        run_layer_celled<C2, C3>(batch_size, cfg.H3, cfg.W3,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][2], cfg.hidden[blk][2],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        run_layer_celled<C3, C3>(batch_size, cfg.H3, cfg.W3,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][3], cfg.hidden[blk][3],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        // Serialized tdrop_s3 — only for pass2 events; non-pass2 events
        // get pass3=0 automatically.
        serial_tdrop(batch_size, cfg.tdrop_window,
                      cfg.tdrop_s3[blk], event_slots, cfg.W3, /*mask=*/pass2,
                      pass3, /*is_s3=*/true);
        __syncthreads();

        // ---- Write output slots (P1 oracle harness) ----
        // Output slot follows ssla_s2_s3_head conventions:
        //   pass2 ∈ {1 = owner+passed, 0 = owner+dropped, -1 = halo}
        //   pass3 same
        //   s3_feat[96] = post-L7 output for owner-passed events
        // event_slots[e].in_feat NOW holds L7 output.
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
                // Always materialize s3_feat (oracle compares only owner-passed).
                for (int i = threadIdx.x; i < C3; i += blockDim.x) {
                    out[batch_base + e].s3_feat[i] = event_slots[e].in_feat[i];
                }
            }
        }
        if (threadIdx.x == 0 && batch_clk != 0) {
            const unsigned long long t_done = clock64();
            batch_clk[batch_idx * 2 + 0] = t_batch_pop_local;
            batch_clk[batch_idx * 2 + 1] = t_done;
        }
        __syncthreads();
    }
}


// ============================================================================
// Persistent kernel — live runner.
//
// Each block polls its per-block ring. When `avail >= 1`, processes a batch
// of min(avail, BATCH) events. Producer publishes via seq_done as in
// ssla_s2_s3_head_w.cuh; consumer spins on seq_done before processing.
//
// Per-batch the GPU records two clock64 timestamps in cfg.timing for EACH
// event in the batch (all events in one batch share the same t_pop/t_done —
// the batch is processed as a unit, gather barriers are at batch granularity).
//
// Head matvec + predictions write happens for events with is_owner AND pass3.
// ============================================================================
extern "C" __global__
__launch_bounds__(N_WARPS * 32, 1)
void k_ssla_s2s3_celled_persistent(
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
    extern __shared__ float smem_raw[];
    const int blk = blockIdx.x;
    const HybridS2S3Config& cfg  = *cfg_in;
    HybridInputRec*  ring        = (blk == 0) ? ring0 : ring1;
    volatile unsigned long long* tail        = (blk == 0) ? tail0        : tail1;
    volatile unsigned long long* events_done = (blk == 0) ? events_done0 : events_done1;

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

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const HybridStrip& strip = cfg.strip[blk];

    unsigned long long t_local = *tail;
    __shared__ int want_stop;
    __shared__ int batch_ready;
    __shared__ int batch_size_sm;
    __shared__ unsigned long long t_batch_pop;

    while (true) {
        // Stop check.
        if (threadIdx.x == 0) want_stop = *stop_flag;
        __syncthreads();
        if (want_stop) return;

        // Probe ring: wait for at least 1 event, take up to BATCH.
        if (threadIdx.x == 0) {
            // Sleep-poll until at least one slot is published.
            // For simplicity probe slot t_local and if not yet ready,
            // backoff. (Producer publishes in order, so slot t_local
            // is the next one to fill.)
            volatile unsigned long long* seq_p0 =
                &ring[t_local & ring_mask].seq_done;
            if (*seq_p0 != t_local + 1ull) {
                batch_ready = 0;
                __nanosleep(spin_ns);
            } else {
                // Determine how many consecutive slots are ready (up to BATCH).
                int n_ready = 1;
                for (int i = 1; i < BATCH; ++i) {
                    volatile unsigned long long* seq_pi =
                        &ring[(t_local + i) & ring_mask].seq_done;
                    if (*seq_pi == t_local + i + 1ull) n_ready++;
                    else break;
                }
                batch_size_sm = n_ready;
                batch_ready   = 1;
            }
        }
        __syncthreads();
        if (!batch_ready) continue;

        // System fence so subsequent record reads see published data.
        if (threadIdx.x == 0) {
            __threadfence_system();
            t_batch_pop = clock64();
        }
        __syncthreads();

        const int batch_size = batch_size_sm;

        // ---- Load batch ----
        for (int e = 0; e < batch_size; ++e) {
            const HybridInputRec& rec = ring[(t_local + e) & ring_mask];
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
        __syncthreads();

        // ---- Stage 2 ----
        dispatch_tasks(batch_size, cfg.H2, cfg.W2, event_slots, /*mask=*/0,
                       task_event, task_delta, task_count);
        __syncthreads();
        run_layer_celled<C1, C2>(batch_size, cfg.H2, cfg.W2,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][0], cfg.hidden[blk][0],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        run_layer_celled<C2, C2>(batch_size, cfg.H2, cfg.W2,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][1], cfg.hidden[blk][1],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        serial_tdrop(batch_size, cfg.tdrop_window,
                      cfg.tdrop_s2[blk], event_slots, cfg.W2, /*mask=*/0,
                      pass2, /*is_s3=*/false);
        __syncthreads();

        // Pool s2 coords → s3 coords.
        if (threadIdx.x == 0) {
            for (int e = 0; e < batch_size; ++e) {
                event_slots[e].evx >>= 1;
                event_slots[e].evy >>= 1;
            }
        }
        __syncthreads();

        // ---- Stage 3 ----
        dispatch_tasks(batch_size, cfg.H3, cfg.W3, event_slots, /*mask=*/pass2,
                       task_event, task_delta, task_count);
        __syncthreads();
        run_layer_celled<C2, C3>(batch_size, cfg.H3, cfg.W3,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][2], cfg.hidden[blk][2],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        run_layer_celled<C3, C3>(batch_size, cfg.H3, cfg.W3,
                                  event_slots, task_event, task_delta, task_count,
                                  cfg.layers[blk][3], cfg.hidden[blk][3],
                                  contrib, per_warp_in_feat_sm, per_warp_qh_sm);
        serial_tdrop(batch_size, cfg.tdrop_window,
                      cfg.tdrop_s3[blk], event_slots, cfg.W3, /*mask=*/pass2,
                      pass3, /*is_s3=*/true);
        __syncthreads();

        // ---- Head matvec + predictions write for owner+pass3 events ----
        // One warp per event (warp_id < batch_size). event_slots[e].in_feat
        // currently holds L7 output (C3=96 floats).
        constexpr int HEAD_OUT = HEAD_OUT_DEFAULT;
        if (cfg.head_W != 0 && warp_id < batch_size) {
            const int e = warp_id;
            const bool owner_pass3 = (event_slots[e].is_owner != 0)
                                    && (pass2[e] != 0)
                                    && (pass3[e] != 0);
            if (owner_pass3) {
                constexpr int HEAD_K = (HEAD_OUT + 31) / 32;
                float head_local[HEAD_K];
                matvec_w<C3, HEAD_OUT>(event_slots[e].in_feat,
                                       cfg.head_W, head_local);
                if (lane < HEAD_OUT) head_local[0] += cfg.head_b[lane];

                // event_slots[e].evx / evy are in s3 coords already.
                const int s3x = event_slots[e].evx;
                const int s3y = event_slots[e].evy;
                const int hx_local   = s3x - strip.s3_owned_lo;
                const int s3_owned_w = strip.s3_owned_hi - strip.s3_owned_lo;
                if (hx_local >= 0 && hx_local < s3_owned_w &&
                    s3y >= 0 && s3y < cfg.H3) {
                    const int cell_idx = s3y * s3_owned_w + hx_local;
                    const long long off = (long long)cell_idx * HEAD_OUT;
                    if (lane < HEAD_OUT) {
                        cfg.preds[blk][off + lane] = head_local[0];
                    }
                    __syncwarp();
                    if (lane == 0) cfg.version[blk][cell_idx] += 1u;
                }
            }
        }
        __syncthreads();

        // ---- Record per-event timing ----
        if (threadIdx.x == 0 && cfg.timing[blk] != 0) {
            const unsigned long long t_done = clock64();
            for (int e = 0; e < batch_size; ++e) {
                const unsigned long long seq = t_local + (unsigned long long)e;
                const unsigned int idx = (unsigned int)(seq & cfg.timing_mask);
                GpuTimingSlot* ts = &cfg.timing[blk][idx];
                ts->t_pop_clk  = t_batch_pop;
                ts->t_done_clk = t_done;
                ts->seq        = (unsigned int)seq;
                const bool owner_pass3 = (event_slots[e].is_owner != 0)
                                        && (pass2[e] != 0)
                                        && (pass3[e] != 0);
                ts->owner = owner_pass3 ? 1u : 0u;
            }
        }
        __syncthreads();

        // ---- Advance tail ----
        t_local += (unsigned long long)batch_size;
        if (threadIdx.x == 0) {
            *tail        = t_local;
            *events_done = t_local;
            __threadfence_system();
        }
        __syncthreads();
    }
}
