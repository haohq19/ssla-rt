// lib_stage01_to_gpu.cpp — fork of lib_stage01_capi.cpp with a hybrid
// CPU→GPU ring-push hook after stage 1 succeeds.
//
// Identical CPU side: N shard threads, halo=2 routing, owner runs s0 +
// s1 + tdrop+pool, halo runs s0 only (state sync). On owner's pass-1
// success, we now ALSO push the (t, s2-x, s2-y, feat1[24]) record into
// per-GPU-block pinned-host rings allocated by the Python driver.
//
// Ring sync model: lock-free MPSC.
//   - 4 CPU shard producers may push concurrently to the same per-block
//     ring; each producer claims a slot via __atomic_fetch_add(ring_head).
//   - GPU is the single consumer per ring; sees ready slots via the
//     `seq_done` tag that the producer publishes after the record write.
//   - Layout of HybridInputRec must match deploy/orin/kernels/
//     ssla_s2_s3_head.cuh.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>
#endif

#include "openeva/event.h"

#include "lat_stats.h"
#include "spsc.h"
#include "ssla_kernels.h"
#include "timed.h"

namespace {

constexpr int N_GPU_BLOCKS = 2;

// Must match HybridInputRec in ssla_s2_s3_head.cuh (120 bytes).
struct HybridInputRec {
    std::uint64_t  seq_done;
    float          t;
    std::uint16_t  x;
    std::uint16_t  y;
    float          feat1[24];   // C1 = 24
    std::uint64_t  t_push_ns;   // CPU CLOCK_MONOTONIC_RAW ns at ring publish
};
static_assert(sizeof(HybridInputRec) == 120, "HybridInputRec layout drift");

// Wallclock at ring-publish moment, used as the per-event arrival timestamp.
// Same clock domain (CLOCK_MONOTONIC_RAW) used to calibrate the GPU SM-clock
// offset at kernel-launch time, so latency = t_done_ns - t_push_ns is well
// defined across CPU/GPU.
static inline std::uint64_t mono_raw_ns() {
#if defined(__linux__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull
         + static_cast<std::uint64_t>(ts.tv_nsec);
#else
    return 0ull;
#endif
}

void pin_to_core(int core_id) {
#if defined(__linux__)
    if (core_id < 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#else
    (void)core_id;
#endif
}

struct ShardMsg {
    std::uint64_t  t_arr_tsc;
    openeva::Event ev;
    bool           is_owner;
    bool           eof;
};

using ShardRing = deploy::SpscRing<ShardMsg>;

struct GpuBlockCfg {
    HybridInputRec*     ring_buf;        // pinned host alloc, indexed mod ring_capacity
    std::uint64_t*      ring_head;       // pinned host atomic counter
    int                 proc_lo;         // s2-x processing range (inclusive)
    int                 proc_hi;         // exclusive
    std::atomic<std::uint64_t> n_pushed{0};
};

// Per-event segment ids — index into ShardCtx::seg_sum_ticks / seg_count.
enum Seg : int {
    SEG_PREPROCESS = 0,
    SEG_S0         = 1,   // pipe.stage_forward(0)  — runs on every event
    SEG_TDROP0     = 2,   // owner only
    SEG_S1         = 3,   // owner only, after pass0
    SEG_TDROP1     = 4,   // owner only, after pass0
    SEG_PUSH       = 5,   // owner only, after pass1; GPU ring write
    NUM_SEGS       = 6,
};

struct ShardCtx {
    int  shard_id;
    int  core_id;
    bool pin;
    std::unique_ptr<deploy::SslaSPipeline> pipe;
    std::unique_ptr<ShardRing> q_in;

    std::vector<double> sample_buf;
    std::atomic<std::size_t> sample_idx{0};
    std::size_t sample_cap = 0;

    std::atomic<std::uint64_t> total_owner{0};
    std::atomic<std::uint64_t> total_halo{0};
    std::atomic<std::uint64_t> pass_stage0{0};
    std::atomic<std::uint64_t> pass_stage1{0};

    // Per-segment cumulative timing — sum of rdtsc_now() deltas (raw ticks;
    // converted to ns at snapshot time via TscClock). Each shard writes only
    // its own counters; snapshot reads them across shards via relaxed atomics.
    std::atomic<std::uint64_t> seg_sum_ticks[NUM_SEGS]{};
    std::atomic<std::uint64_t> seg_count[NUM_SEGS]{};
};

// Define SSLA_RT_NO_SEG_TIMING at compile time to compile per-segment
// instrumentation away. Cost of instrumentation measured at ~250 ns / event
// (7 rdtsc + 6 atomic adds). Production builds that don't need the
// breakdown gain ~10% admit by disabling.
inline void seg_record(ShardCtx* ctx, int seg, std::uint64_t dt_ticks) {
#ifndef SSLA_RT_NO_SEG_TIMING
    ctx->seg_sum_ticks[seg].fetch_add(dt_ticks, std::memory_order_relaxed);
    ctx->seg_count[seg].fetch_add(1, std::memory_order_relaxed);
#else
    (void)ctx; (void)seg; (void)dt_ticks;
#endif
}

}  // namespace

// ============================================================================
// Handle
// ============================================================================

struct S01gHandle {
    int N;
    int halo;
    int strip_w;
    int W_full;
    int H_full;
    bool pin_cores;
    std::vector<std::unique_ptr<ShardCtx>> ctxs;
    std::vector<std::thread> threads;
    std::atomic<std::uint64_t> n_oob{0};
    std::chrono::steady_clock::time_point t_start_or_reset;

    // GPU side — populated by s01g_attach_gpu_rings(). Until attached,
    // workers do not push to any GPU ring (pure CPU mode).
    std::atomic<bool> gpu_attached{false};
    GpuBlockCfg gpu_block[N_GPU_BLOCKS];
    int             gpu_W2 = 0;
    int             gpu_H2 = 0;
    std::uint64_t   gpu_ring_mask = 0;

    // Synthetic dispatcher (C++ side) — replaces Python SyntheticReader so
    // we get tsc-precision pacing and no Python interpreter / GIL interference.
    std::atomic<bool>         synth_running{false};
    std::thread               synth_thread;
    std::atomic<std::uint64_t> synth_n_pushed{0};
};

namespace {

void shard_worker(S01gHandle* h, ShardCtx* ctx) {
    if (ctx->pin) pin_to_core(ctx->core_id);

    deploy::SslaSPipeline& pipe = *ctx->pipe;
    float feat_in[deploy::kInDim];
    std::vector<float> feat0(deploy::kC0);
    std::vector<float> feat1(deploy::kC1);

    while (true) {
        ShardMsg m;
        if (!ctx->q_in->try_pop(m)) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
            continue;
        }
        if (m.eof) break;

        const int ex0 = static_cast<int>(m.ev.x);
        const int ey0 = static_cast<int>(m.ev.y);
        if (ex0 < 0 || ex0 >= pipe.W() || ey0 < 0 || ey0 >= pipe.H()) {
            continue;
        }

#ifdef SSLA_RT_NO_PROFILING
        #define TICK() ((std::uint64_t)0)
#else
        #define TICK() (deploy::rdtsc_now())
#endif
        const std::uint64_t t_seg0 = TICK();
        pipe.preprocess(m.ev, feat_in);
        const std::uint64_t t_seg1 = TICK();
        seg_record(ctx, SEG_PREPROCESS, t_seg1 - t_seg0);

        int x = ex0, y = ey0;

        // ---- A1 OPTIMISATION ----
        // Decide whether stage 0's OUTPUT (feat0) will be consumed before
        // running stage_forward(0): only owners that pass tdrop_0 will
        // proceed to stage 1 and need feat0. Everyone else (halo events,
        // and owners that fail tdrop_0) can run stage 0 in state-only
        // mode — s0_L0 full (its tmp feeds s0_L1 qvg input), s0_L1
        // skips goW + Res + LN.
        bool pass0 = false;
        const int x1 = x / 2, y1 = y / 2;   // stage 1 grid coords
        if (m.is_owner) {
            pass0 = pipe.tdrop_check_at(0, x1, y1);
        }
        const std::uint64_t t_seg2_tdrop = TICK();
        // Note: SEG_TDROP0 originally measured tdrop time AFTER s0; we
        // measure it BEFORE s0 in the A1 path. Same physical work.
        seg_record(ctx, SEG_TDROP0, t_seg2_tdrop - t_seg1);

        if (m.is_owner && pass0) {
            pipe.stage_forward(0, x, y, feat_in, feat0.data());
        } else {
            pipe.stage_forward_state_only(0, x, y, feat_in);
        }
        const std::uint64_t t_seg2 = TICK();
        seg_record(ctx, SEG_S0, t_seg2 - t_seg2_tdrop);

        if (m.is_owner) {
            // x/y were not modified by tdrop_check_at; pool them now
            // (the original tdrop_and_pool also pooled in-place).
            x = x1; y = y1;
            bool pass1 = false;
            std::uint64_t t_after_s1_block = t_seg2;
            if (pass0) {
                // For stage 1: check tdrop_1 first to know if feat1
                // will be consumed (i.e. pushed to GPU).
                const int x2 = x / 2, y2 = y / 2;
                pass1 = pipe.tdrop_check_at(1, x2, y2);
                const std::uint64_t t_seg3 = TICK();
                seg_record(ctx, SEG_TDROP1, t_seg3 - t_seg2);

                if (pass1) {
                    pipe.stage_forward(1, x, y, feat0.data(), feat1.data());
                } else {
                    pipe.stage_forward_state_only(1, x, y, feat0.data());
                }
                const std::uint64_t t_seg4 = TICK();
                seg_record(ctx, SEG_S1, t_seg4 - t_seg3);
                t_after_s1_block = t_seg4;
                x = x2; y = y2;
            }

            // ---- HYBRID HOOK: push to GPU per-block ring(s) ----
            if (pass1 && h->gpu_attached.load(std::memory_order_acquire)) {
                const int s2x = x;
                const int s2y = y;
                if (s2x >= 0 && s2x < h->gpu_W2 &&
                    s2y >= 0 && s2y < h->gpu_H2) {
                    for (int b = 0; b < N_GPU_BLOCKS; ++b) {
                        GpuBlockCfg& gb = h->gpu_block[b];
                        if (s2x < gb.proc_lo || s2x >= gb.proc_hi) continue;
                        const std::uint64_t slot = __atomic_fetch_add(
                            gb.ring_head, 1ull, __ATOMIC_RELAXED);
                        HybridInputRec* dst = gb.ring_buf
                                            + (slot & h->gpu_ring_mask);
                        dst->t = m.ev.t;
                        dst->x = static_cast<std::uint16_t>(s2x);
                        dst->y = static_cast<std::uint16_t>(s2y);
                        std::memcpy(dst->feat1, feat1.data(),
                                    sizeof(float) * 24);
                        // Stamp arrival time immediately before publish so
                        // latency = (GPU done time) - t_push_ns captures the
                        // full ring-wait + GPU compute window.
                        dst->t_push_ns = mono_raw_ns();
                        // Release-store seq_done; ensures record fields
                        // (including t_push_ns) are visible before consumer
                        // observes the tag.
                        __atomic_store_n(&dst->seq_done, slot + 1ull,
                                          __ATOMIC_RELEASE);
                        gb.n_pushed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                const std::uint64_t t_seg6 = TICK();
                seg_record(ctx, SEG_PUSH, t_seg6 - t_after_s1_block);
            }
#undef TICK

#ifndef SSLA_RT_NO_PROFILING
            const std::uint64_t t1 = deploy::rdtsc_now();
            const double lat_ns = deploy::TscClock::instance().tsc_to_ns(
                t1 - m.t_arr_tsc);
            const std::size_t idx = ctx->sample_idx.fetch_add(
                1, std::memory_order_relaxed);
            ctx->sample_buf[idx % ctx->sample_cap] = lat_ns;
#endif
            ctx->total_owner.fetch_add(1, std::memory_order_relaxed);
            if (pass0) ctx->pass_stage0.fetch_add(1, std::memory_order_relaxed);
            if (pass1) ctx->pass_stage1.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctx->total_halo.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

}  // namespace

// ============================================================================
// C API
// ============================================================================

extern "C" {

// Same semantics as s0_init in lib_stage01_capi.cpp.
S01gHandle* s01g_init_full(const char* weights_dir, int n_shards, int halo,
                            int base_core, int sample_cap,
                            unsigned int shard_ring_cap);

S01gHandle* s01g_init(const char* weights_dir, int n_shards, int halo,
                       int base_core, int sample_cap) {
    return s01g_init_full(weights_dir, n_shards, halo, base_core, sample_cap,
                           1u << 16);
}

S01gHandle* s01g_init_full(const char* weights_dir, int n_shards, int halo,
                            int base_core, int sample_cap,
                            unsigned int shard_ring_cap) {
    if (n_shards < 1) n_shards = 1;
    if (sample_cap < 1024) sample_cap = 1024;
    // Validate ring cap is power of two and ≥ 4.
    if (shard_ring_cap < 4) shard_ring_cap = 4;
    if (shard_ring_cap & (shard_ring_cap - 1)) {
        std::fprintf(stderr,
                     "[s01g_init] shard_ring_cap %u not power of 2; rounding\n",
                     shard_ring_cap);
        unsigned p = 1;
        while (p < shard_ring_cap) p <<= 1;
        shard_ring_cap = p;
    }

#if defined(__linux__)
    // Lock all current and future memory to prevent page faults in the
    // shard worker hot path. Cuts MAX latency spikes that come from
    // demand paging / minor faults on hidden state / weights. Requires
    // RLIMIT_MEMLOCK ≥ pipeline memory (~50 MB for stub geom). Falls
    // back silently if not permitted.
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::fprintf(stderr, "[s01g_init] mlockall failed (errno %d) — "
                     "running without memory locking, MAX latency may spike\n",
                     errno);
    }
#endif

    deploy::TscClock::instance().tsc_to_ns(0);

    auto h = std::make_unique<S01gHandle>();
    h->N = n_shards;
    h->halo = halo;
    h->pin_cores = (base_core >= 0);

    auto template_pipe = std::make_unique<deploy::SslaSPipeline>();
    try {
        template_pipe->load(weights_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[s01g_init] load failed: %s\n", e.what());
        return nullptr;
    }
    template_pipe->reset();
    h->W_full = template_pipe->W();
    h->H_full = template_pipe->H();
    h->strip_w = (h->W_full + h->N - 1) / h->N;

    for (int k = 0; k < h->N; ++k) {
        auto c = std::make_unique<ShardCtx>();
        c->shard_id = k;
        c->core_id  = h->pin_cores ? (base_core + k) : -1;
        c->pin      = h->pin_cores;
        c->pipe     = std::make_unique<deploy::SslaSPipeline>();
        c->pipe->load(weights_dir);
        c->pipe->reset();
        c->q_in     = std::make_unique<ShardRing>(shard_ring_cap);
        c->sample_cap = static_cast<std::size_t>(sample_cap);
        c->sample_buf.assign(c->sample_cap, 0.0);
        h->ctxs.push_back(std::move(c));
    }
    auto* hp = h.release();
    for (int k = 0; k < hp->N; ++k) {
        hp->threads.emplace_back(shard_worker, hp, hp->ctxs[k].get());
    }
    hp->t_start_or_reset = std::chrono::steady_clock::now();
    return hp;
}

// Attach GPU per-block ring buffers + atomic head counters. Must be
// called BEFORE the workers start receiving events that survive stage
// 1 (in practice: right after s01g_init, before the first
// s01g_submit_batch).
//
// Fields:
//   ring_buf_b[2]  : pinned-host pointer to HybridInputRec[capacity]
//   ring_head_b[2] : pinned-host pointer to a single uint64_t
//                    (multi-producer head counter; bumped via
//                    __atomic_fetch_add by workers)
//   ring_mask      : capacity − 1 (capacity must be power of 2)
//   W2 / H2        : s2 grid dims (post-stage-0+1-pool resolution)
//   proc_lo/hi[2]  : per-block s2-x processing range
void s01g_attach_gpu_rings(S01gHandle* h,
                            void* ring_buf_0, void* ring_head_0,
                            void* ring_buf_1, void* ring_head_1,
                            unsigned long long ring_mask,
                            int W2, int H2,
                            int proc_lo_0, int proc_hi_0,
                            int proc_lo_1, int proc_hi_1) {
    if (!h) return;
    h->gpu_block[0].ring_buf  = static_cast<HybridInputRec*>(ring_buf_0);
    h->gpu_block[0].ring_head = static_cast<std::uint64_t*>(ring_head_0);
    h->gpu_block[0].proc_lo   = proc_lo_0;
    h->gpu_block[0].proc_hi   = proc_hi_0;
    h->gpu_block[0].n_pushed.store(0, std::memory_order_relaxed);
    h->gpu_block[1].ring_buf  = static_cast<HybridInputRec*>(ring_buf_1);
    h->gpu_block[1].ring_head = static_cast<std::uint64_t*>(ring_head_1);
    h->gpu_block[1].proc_lo   = proc_lo_1;
    h->gpu_block[1].proc_hi   = proc_hi_1;
    h->gpu_block[1].n_pushed.store(0, std::memory_order_relaxed);
    h->gpu_ring_mask = ring_mask;
    h->gpu_W2 = W2;
    h->gpu_H2 = H2;
    std::atomic_thread_fence(std::memory_order_release);
    h->gpu_attached.store(true, std::memory_order_release);
}

int s01g_submit_batch(S01gHandle* h, const float* events_packed, int n) {
    if (!h || n <= 0) return 0;
    int accepted = 0;
    for (int i = 0; i < n; ++i) {
        const float t = events_packed[4 * i + 0];
        const float x = events_packed[4 * i + 1];
        const float y = events_packed[4 * i + 2];
        const float p = events_packed[4 * i + 3];
        const int ex = static_cast<int>(x);
        const int ey = static_cast<int>(y);
        if (ex < 0 || ex >= h->W_full || ey < 0 || ey >= h->H_full) {
            h->n_oob.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const int strip = h->strip_w;
        const int owner = std::min(ex / strip, h->N - 1);
        const std::uint64_t t_arr = deploy::rdtsc_now();
        for (int dk = -1; dk <= 1; ++dk) {
            const int k = ex / strip + dk;
            if (k < 0 || k >= h->N) continue;
            const int lo = k * strip - h->halo;
            const int hi = (k + 1) * strip + h->halo;
            if (ex < lo || ex >= hi) continue;
            ShardMsg m;
            m.t_arr_tsc = t_arr;
            m.ev.t = t;
            m.ev.x = x;
            m.ev.y = y;
            m.ev.p = p;
            m.is_owner = (k == owner);
            m.eof = false;
            h->ctxs[k]->q_in->push(m);
        }
        ++accepted;
    }
    return accepted;
}

// Layout (32 slots total; slots 0..15 unchanged from prior contract):
//   [0..15] — original stats (see prior comment in lib_stage01_capi.cpp)
//   [14] gpu_pushed_block_0
//   [15] gpu_pushed_block_1
//   [16..21] per-segment sum_ns:  preprocess, s0, tdrop0, s1, tdrop1, push
//   [22..27] per-segment count:   preprocess, s0, tdrop0, s1, tdrop1, push
//   [28..31] reserved.
void s01g_snapshot_stats(S01gHandle* h, double* out) {
    for (int i = 0; i < 32; ++i) out[i] = 0.0;
    if (!h) return;

    const auto now = std::chrono::steady_clock::now();
    const double dt_s = std::chrono::duration<double>(
        now - h->t_start_or_reset).count();

    std::uint64_t totalo = 0, totalh = 0, pass0 = 0, pass1 = 0;
    std::size_t  ring_occ = 0;
    std::vector<double> samples;
    samples.reserve(h->ctxs.size() * h->ctxs[0]->sample_cap);

    for (auto& c : h->ctxs) {
        totalo += c->total_owner.load(std::memory_order_relaxed);
        totalh += c->total_halo.load(std::memory_order_relaxed);
        pass0  += c->pass_stage0.load(std::memory_order_relaxed);
        pass1  += c->pass_stage1.load(std::memory_order_relaxed);
        ring_occ += c->q_in->size();
        const std::size_t cur = c->sample_idx.load(std::memory_order_relaxed);
        const std::size_t valid = std::min(cur, c->sample_cap);
        for (std::size_t i = 0; i < valid; ++i) {
            samples.push_back(c->sample_buf[i]);
        }
    }
    out[0] = static_cast<double>(totalo);
    out[1] = static_cast<double>(totalh);
    out[2] = static_cast<double>(pass0);
    out[3] = static_cast<double>(h->n_oob.load(std::memory_order_relaxed));
    out[4] = dt_s;
    out[5] = static_cast<double>(ring_occ);
    out[13] = static_cast<double>(pass1);
    out[14] = static_cast<double>(h->gpu_block[0].n_pushed.load(
        std::memory_order_relaxed));
    out[15] = static_cast<double>(h->gpu_block[1].n_pushed.load(
        std::memory_order_relaxed));

    if (!samples.empty()) {
        auto stats = deploy::LatStats::from_samples(samples);
        out[6]  = stats.p50_ns   * 1e-3;
        out[7]  = stats.p90_ns   * 1e-3;
        out[8]  = stats.p99_ns   * 1e-3;
        out[9]  = stats.p99_9_ns * 1e-3;
        out[10] = stats.max_ns   * 1e-3;
        out[11] = stats.mean_ns  * 1e-3;
        out[12] = static_cast<double>(samples.size());
    }

    // Per-segment timing: aggregate sum_ticks and count across shards.
    std::uint64_t seg_sum_ticks_tot[NUM_SEGS] = {0};
    std::uint64_t seg_count_tot[NUM_SEGS]     = {0};
    for (auto& c : h->ctxs) {
        for (int s = 0; s < NUM_SEGS; ++s) {
            seg_sum_ticks_tot[s] += c->seg_sum_ticks[s].load(
                std::memory_order_relaxed);
            seg_count_tot[s] += c->seg_count[s].load(
                std::memory_order_relaxed);
        }
    }
    const auto& clk = deploy::TscClock::instance();
    for (int s = 0; s < NUM_SEGS; ++s) {
        out[16 + s] = clk.tsc_to_ns(seg_sum_ticks_tot[s]);
        out[22 + s] = static_cast<double>(seg_count_tot[s]);
    }
}

// ============================================================================
// Synthetic dispatcher — C++ replacement for Python's SyntheticReader.
// Uses rdtsc-based spin-pacing for ns-precision per-event spacing (Python
// time.sleep is ~1 ms precision on stock Linux). Single producer thread,
// pinable to a dedicated core via pin_core (≥0); when negative, the thread
// runs un-pinned.
//
// Event generation: xorshift64 RNG. Distributes events uniformly over the
// (W_full × H_full) grid. Polarity = 1 bit of state.
// ============================================================================
namespace {
void synth_thread_main(S01gHandle* h, double rate_mev, int pin_core,
                       std::uint64_t seed) {
    if (pin_core >= 0) pin_to_core(pin_core);

    // ns_per_event = 1e9 / (rate_mev × 1e6). At 2 Mev/s → 500 ns.
    const double tsc_to_ns = deploy::TscClock::instance().tsc_to_ns(1);
    const std::uint64_t ns_per_event =
        (rate_mev > 0.0) ? static_cast<std::uint64_t>(1e9 / (rate_mev * 1e6)) : 0;
    // Convert ns to tsc ticks. On aarch64 tsc_to_ns(1) returns ≈ 1 so ticks≈ns,
    // but stay portable by dividing.
    const std::uint64_t ticks_per_event = (ns_per_event > 0)
        ? static_cast<std::uint64_t>(ns_per_event / tsc_to_ns) : 0;

    std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto xorshift = [&]() -> std::uint64_t {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };

    const int W_full = h->W_full;
    const int H_full = h->H_full;
    const int strip  = h->strip_w;
    const int halo   = h->halo;
    const int n_shards = h->N;

    std::uint64_t next_tick = deploy::rdtsc_now();
    float ev_t_us = 0.0f;
    const float dt_per_event_us =
        (rate_mev > 0.0) ? static_cast<float>(1.0 / rate_mev) : 1.0f;

    while (h->synth_running.load(std::memory_order_relaxed)) {
        // Spin-wait until the next scheduled event time. tsc-precision.
        while (true) {
            const std::uint64_t now = deploy::rdtsc_now();
            if (now >= next_tick) break;
#if defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
        next_tick += ticks_per_event;

        // Generate one event.
        const std::uint64_t r = xorshift();
        const int ex = static_cast<int>(r % static_cast<std::uint64_t>(W_full));
        const int ey = static_cast<int>((r >> 16) %
                                         static_cast<std::uint64_t>(H_full));
        const float p = static_cast<float>((r >> 32) & 1ULL);
        ev_t_us += dt_per_event_us;

        // Route to owner + halo neighbour shard(s).
        const int owner = std::min(ex / strip, n_shards - 1);
        const std::uint64_t t_arr = deploy::rdtsc_now();
        for (int dk = -1; dk <= 1; ++dk) {
            const int k = ex / strip + dk;
            if (k < 0 || k >= n_shards) continue;
            const int lo = k * strip - halo;
            const int hi = (k + 1) * strip + halo;
            if (ex < lo || ex >= hi) continue;
            ShardMsg m;
            m.t_arr_tsc = t_arr;
            m.ev.t = ev_t_us;
            m.ev.x = static_cast<float>(ex);
            m.ev.y = static_cast<float>(ey);
            m.ev.p = p;
            m.is_owner = (k == owner);
            m.eof = false;
            h->ctxs[k]->q_in->push(m);
        }
        h->synth_n_pushed.fetch_add(1, std::memory_order_relaxed);
    }
}
}  // namespace

void s01g_start_synthetic(S01gHandle* h, double rate_mev, int pin_core,
                           std::uint64_t seed) {
    if (!h) return;
    if (h->synth_running.load()) return;   // already running
    h->synth_running.store(true, std::memory_order_release);
    h->synth_n_pushed.store(0, std::memory_order_relaxed);
    h->synth_thread = std::thread(synth_thread_main, h, rate_mev, pin_core, seed);
}

void s01g_stop_synthetic(S01gHandle* h) {
    if (!h) return;
    h->synth_running.store(false, std::memory_order_release);
    if (h->synth_thread.joinable()) h->synth_thread.join();
}

std::uint64_t s01g_synthetic_n_pushed(S01gHandle* h) {
    if (!h) return 0;
    return h->synth_n_pushed.load(std::memory_order_relaxed);
}

void s01g_reset_stats(S01gHandle* h) {
    if (!h) return;
    for (auto& c : h->ctxs) {
        c->total_owner.store(0, std::memory_order_relaxed);
        c->total_halo.store(0, std::memory_order_relaxed);
        c->pass_stage0.store(0, std::memory_order_relaxed);
        c->pass_stage1.store(0, std::memory_order_relaxed);
        c->sample_idx.store(0, std::memory_order_relaxed);
        for (int s = 0; s < NUM_SEGS; ++s) {
            c->seg_sum_ticks[s].store(0, std::memory_order_relaxed);
            c->seg_count[s].store(0, std::memory_order_relaxed);
        }
    }
    for (int b = 0; b < N_GPU_BLOCKS; ++b) {
        h->gpu_block[b].n_pushed.store(0, std::memory_order_relaxed);
    }
    h->n_oob.store(0, std::memory_order_relaxed);
    h->t_start_or_reset = std::chrono::steady_clock::now();
}

void s01g_shutdown(S01gHandle* h) {
    if (!h) return;
    // Stop synth dispatcher first so it doesn't push to a shutting-down ring.
    if (h->synth_running.load()) {
        h->synth_running.store(false, std::memory_order_release);
        if (h->synth_thread.joinable()) h->synth_thread.join();
    }
    for (auto& c : h->ctxs) {
        ShardMsg m{}; m.eof = true;
        c->q_in->push(m);
    }
    for (auto& t : h->threads) if (t.joinable()) t.join();
    delete h;
}

}  // extern "C"
