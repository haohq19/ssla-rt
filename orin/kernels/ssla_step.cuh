// SSLA-S full per-event step — chains 8 SSLA layers + 3 spatial halvings
// + 3 temporal-dropout gates + 4 stage-feature outputs. Mirrors
// cpp/methods/ssla/ssla_detection_yolox.cpp::Impl::step_ct<C0,C1,C2,C3>.
//
// Warp-cooperative: every kernel below launches with 32 threads / 1 block.
// All scratch lives in dynamic shared memory; per-warp scratch is what
// makes the layer warp-cooperative (otherwise lanes can't see each
// other's matvec outputs).
//
// What is NOT in this header:
//   * preprocess (dt_norm = clamp((t-last)/1e5, 0, 1)) — handled by the
//     persistent kernel before calling ssla_step_ct.
//   * YOLOX head decode — separate header / future task.
//
// LayerWeights and StepConfig hold pointers to managed-memory tensors
// that the runner allocates once at startup. Kernel reads cfg by ref;
// nothing on the device side ever frees these — the runner owns them.

#pragma once

#define SSLA_PRIM_NO_TEST_WRAPPERS
#include "ssla_primitives.cuh"
#include "ssla_layer.cuh"

struct LayerWeights {
    const float* qvgIn;        // (A, 3·D, IN)
    const float* goW;          // (A, D, D)
    const float* input_proj;   // (D, IN)  may be null if IN==D
    const float* ln_gamma;     // (D,)
    const float* ln_beta;      // (D,)
};

struct StepConfig {
    int H0, W0;                // sensor dims (e.g. 240, 304 for Gen1)
    int tdrop_window;          // shared across stages
    LayerWeights layers[8];
    float*         hidden[8];  // (Hl·Wl·D,) per layer
    unsigned char* tdrop[3];   // tdrop[s] is for the gate after stage s
};

// Worst-case shared scratch needed by ssla_step_ct<C0,C1,C2,C3>:
//   feat_in[2]                   (initial input)
//   buf_a[C3]                    (ping-pong layer output)
//   layer scratch 5·C3           (residual + qh + qvg)
//   per-stage outputs s0/s1/s2/s3 written to global, no shared cost
// Sized for SSLA-S {12,24,48,96}: 2 + 96 + 5*96 = 578 floats = 2312 B.
#define SSLA_STEP_SCRATCH_FLOATS(C3) (2 + (C3) + SSLA_LAYER_SCRATCH_FLOATS(C3))

// Returns true if the event reached stage 3 (otherwise was filtered by
// bounds or temporal_dropout). On true, s0_out..s3_out hold the
// per-stage feature for the (touched_x[k], touched_y[k]) cell.
//
// Caller passes a shared-memory scratch of SSLA_STEP_SCRATCH_FLOATS(C3)
// floats. The s0..s3 output pointers may live in any addressable space.
template <int C0, int C1, int C2, int C3>
__device__ inline bool ssla_step_ct(
    const StepConfig& cfg,
    int ev_x, int ev_y,
    float dt_norm, float polarity,
    float* __restrict__ s0_out,    // (C0,)
    float* __restrict__ s1_out,    // (C1,)
    float* __restrict__ s2_out,    // (C2,)
    float* __restrict__ s3_out,    // (C3,)
    int*   __restrict__ touched_x, // (4,)
    int*   __restrict__ touched_y, // (4,)
    float* __restrict__ smem)      // shared scratch
{
    const int H0   = cfg.H0;
    const int W0   = cfg.W0;
    if (ev_x < 0 || ev_x >= W0 || ev_y < 0 || ev_y >= H0) return false;

    const int tid = block_tid_();

    float* feat_in   = smem;             // 2 floats
    float* buf_a     = smem + 2;         // C3 floats (ping-pong)
    float* layer_sm  = smem + 2 + C3;    // 5·C3 floats (layer scratch)

    if (tid == 0) {
        feat_in[0] = dt_norm;
        feat_in[1] = polarity;
    }
    __syncthreads();

    int evx = ev_x, evy = ev_y;
    int Hl  = H0,   Wl  = W0;

    // ------ Stage 0  (in=2 → C0, layers 0,1) ------
    if (tid == 0) { touched_x[0] = evx; touched_y[0] = evy; }
    ssla_layer_forward_ct<2, C0, 1>(
        evx, evy, Hl, Wl, feat_in,
        cfg.layers[0].qvgIn, cfg.layers[0].goW, cfg.layers[0].input_proj,
        cfg.layers[0].ln_gamma, cfg.layers[0].ln_beta,
        cfg.hidden[0], buf_a, layer_sm);
    ssla_layer_forward_ct<C0, C0, 3>(
        evx, evy, Hl, Wl, buf_a,
        cfg.layers[1].qvgIn, cfg.layers[1].goW, cfg.layers[1].input_proj,
        cfg.layers[1].ln_gamma, cfg.layers[1].ln_beta,
        cfg.hidden[1], s0_out, layer_sm);

    evx >>= 1; evy >>= 1;
    Hl  >>= 1; Wl  >>= 1;
    {
        const int idx = evy * Wl + evx;
        unsigned char* td = cfg.tdrop[0];
        if (tid == 0) td[idx] = (unsigned char)(td[idx] + 1u);
        __syncthreads();
        const unsigned char pre = (unsigned char)(td[idx] - 1u);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }

    // ------ Stage 1  (in=C0 → C1, layers 2,3) ------
    if (tid == 0) { touched_x[1] = evx; touched_y[1] = evy; }
    ssla_layer_forward_ct<C0, C1, 3>(
        evx, evy, Hl, Wl, s0_out,
        cfg.layers[2].qvgIn, cfg.layers[2].goW, cfg.layers[2].input_proj,
        cfg.layers[2].ln_gamma, cfg.layers[2].ln_beta,
        cfg.hidden[2], buf_a, layer_sm);
    ssla_layer_forward_ct<C1, C1, 3>(
        evx, evy, Hl, Wl, buf_a,
        cfg.layers[3].qvgIn, cfg.layers[3].goW, cfg.layers[3].input_proj,
        cfg.layers[3].ln_gamma, cfg.layers[3].ln_beta,
        cfg.hidden[3], s1_out, layer_sm);

    evx >>= 1; evy >>= 1;
    Hl  >>= 1; Wl  >>= 1;
    {
        const int idx = evy * Wl + evx;
        unsigned char* td = cfg.tdrop[1];
        if (tid == 0) td[idx] = (unsigned char)(td[idx] + 1u);
        __syncthreads();
        const unsigned char pre = (unsigned char)(td[idx] - 1u);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }

    // ------ Stage 2  (in=C1 → C2, layers 4,5) ------
    if (tid == 0) { touched_x[2] = evx; touched_y[2] = evy; }
    ssla_layer_forward_ct<C1, C2, 3>(
        evx, evy, Hl, Wl, s1_out,
        cfg.layers[4].qvgIn, cfg.layers[4].goW, cfg.layers[4].input_proj,
        cfg.layers[4].ln_gamma, cfg.layers[4].ln_beta,
        cfg.hidden[4], buf_a, layer_sm);
    ssla_layer_forward_ct<C2, C2, 3>(
        evx, evy, Hl, Wl, buf_a,
        cfg.layers[5].qvgIn, cfg.layers[5].goW, cfg.layers[5].input_proj,
        cfg.layers[5].ln_gamma, cfg.layers[5].ln_beta,
        cfg.hidden[5], s2_out, layer_sm);

    evx >>= 1; evy >>= 1;
    Hl  >>= 1; Wl  >>= 1;
    {
        const int idx = evy * Wl + evx;
        unsigned char* td = cfg.tdrop[2];
        if (tid == 0) td[idx] = (unsigned char)(td[idx] + 1u);
        __syncthreads();
        const unsigned char pre = (unsigned char)(td[idx] - 1u);
        if (cfg.tdrop_window > 1 && (pre % cfg.tdrop_window) != 0u) return false;
    }

    // ------ Stage 3  (in=C2 → C3, layers 6,7) — no pool/tdrop after ------
    if (tid == 0) { touched_x[3] = evx; touched_y[3] = evy; }
    ssla_layer_forward_ct<C2, C3, 3>(
        evx, evy, Hl, Wl, s2_out,
        cfg.layers[6].qvgIn, cfg.layers[6].goW, cfg.layers[6].input_proj,
        cfg.layers[6].ln_gamma, cfg.layers[6].ln_beta,
        cfg.hidden[6], buf_a, layer_sm);
    ssla_layer_forward_ct<C3, C3, 3>(
        evx, evy, Hl, Wl, buf_a,
        cfg.layers[7].qvgIn, cfg.layers[7].goW, cfg.layers[7].input_proj,
        cfg.layers[7].ln_gamma, cfg.layers[7].ln_beta,
        cfg.hidden[7], s3_out, layer_sm);
    return true;
}

// Test wrapper — single SSLA-S instantiation.
extern "C" __global__ void k_ssla_step_S(
    const StepConfig* cfg,
    int ev_x, int ev_y, float dt_norm, float polarity,
    float* s0_out, float* s1_out, float* s2_out, float* s3_out,
    int* touched_x, int* touched_y, int* passed_flag)
{
    if (blockIdx.x != 0) return;
    extern __shared__ float smem[];
    bool passed = ssla_step_ct<12, 24, 48, 96>(
        *cfg, ev_x, ev_y, dt_norm, polarity,
        s0_out, s1_out, s2_out, s3_out, touched_x, touched_y, smem);
    if (threadIdx.x == 0) passed_flag[0] = passed ? 1 : 0;
}


// Per-event input record — matches deploy ring buffer (orin/ring.py
// EVENT_DTYPE: t, x, y, p as four float32). The kernel reads `dt_norm`
// from the t channel since per-pixel preprocessing happens host-side
// for now.
struct EventRec {
    float t;
    float x;
    float y;
    float p;
};

// Per-event output record. Layout fields match the Python ctypes mirror.
struct OutputSlot {
    int   passed;
    int   touched_x[4];
    int   touched_y[4];
    float s0[12];
    float s1[24];
    float s2[48];
    float s3[96];
};

// `drain_n`: read events `[start, start+n)` from a power-of-two ring,
// run ssla_step_ct on each, write into out[i]. No polling — the host
// has fully populated the ring before launch.
//
// Single-block / single warp (32 threads). Events strictly serial; the
// warp parallelizes work *within* each event.
extern "C" __global__ void k_ssla_drain_n(
    const StepConfig*    cfg,
    const EventRec*      ring,
    unsigned long long   ring_mask,
    unsigned long long   start,
    int                  n,
    OutputSlot*          out)
{
    if (blockIdx.x != 0) return;
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    for (int i = 0; i < n; ++i) {
        const EventRec& ev = ring[(start + (unsigned long long)i) & ring_mask];
        OutputSlot* slot = &out[i];
        bool passed = ssla_step_ct<12, 24, 48, 96>(
            *cfg, (int)ev.x, (int)ev.y, ev.t, ev.p,
            slot->s0, slot->s1, slot->s2, slot->s3,
            slot->touched_x, slot->touched_y, smem);
        if (tid == 0) slot->passed = passed ? 1 : 0;
        __syncthreads();
    }
}


// `persistent_loop`: launched once and runs until *stop_flag != 0.
// Polls the SPSC ring for new events; processes them in FIFO order;
// writes each event's OutputSlot to a separate output ring; advances
// ring_tail (consumer head) and out_head (producer head on the output
// side) so the host can both back-pressure the producer and read
// outputs as they become available.
//
// Memory model on Tegra: pinned/managed memory is coherent at
// synchronization points. We mark cross-side counters `volatile` so the
// GPU L1 is bypassed each load, and call __threadfence_system() before
// storing counters that the host watches. ring_buf and out_buf are NOT
// marked volatile because each event is fully written by one party
// before its counter is advanced.
//
// Single-block / single warp. Per-event work is parallelized across
// the 32 lanes; the loop is owned by lane 0 (which polls) but every
// lane participates in ssla_step_ct.
extern "C" __global__ void k_ssla_persistent_loop(
    const StepConfig*       cfg,
    const EventRec*         ring_buf,
    unsigned long long      ring_mask,
    volatile unsigned long long* ring_head,    // host writes
    volatile unsigned long long* ring_tail,    // kernel writes
    OutputSlot*             out_buf,
    unsigned long long      out_mask,
    volatile unsigned long long* out_head,     // kernel writes
    volatile int*           stop_flag,         // host writes 1 to halt
    volatile unsigned long long* events_done,  // kernel writes
    unsigned int            spin_ns)
{
    if (blockIdx.x != 0) return;
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    unsigned long long t_local = *ring_tail;
    unsigned long long o_local = *out_head;

    while (true) {
        const int  stop = *stop_flag;
        const unsigned long long h = *ring_head;
        if (stop != 0) return;
        if (t_local >= h) {
            __nanosleep(spin_ns);
            continue;
        }
        const EventRec ev = ring_buf[t_local & ring_mask];
        OutputSlot* slot = &out_buf[o_local & out_mask];
        const bool passed = ssla_step_ct<12, 24, 48, 96>(
            *cfg, (int)ev.x, (int)ev.y, ev.t, ev.p,
            slot->s0, slot->s1, slot->s2, slot->s3,
            slot->touched_x, slot->touched_y, smem);
        if (tid == 0) slot->passed = passed ? 1 : 0;
        __syncthreads();
        __threadfence_system();

        ++t_local;
        ++o_local;
        if (tid == 0) {
            *ring_tail   = t_local;
            *out_head    = o_local;
            *events_done = o_local;
            __threadfence_system();
        }
        __syncthreads();
    }
}
