// lib_stage01_capi.cpp — C API shared library running stage 0 + stage 1
// of SSLA-S on the CPU with multi-core spatial sharding (halo=2).
//
// Architecture: N shard threads, per-shard SPSC ring, halo=2 routing.
// Each shard owns a private SslaSPipeline (full grid; only stage-0/1
// state is touched). The owner shard runs stage 0; if stage-0 tdrop
// passes, runs stage 1 (which halves x/y to the stage-1 grid). Halo
// shards run stage 0 only for state sync.
//
// Halo=2 at stage 0 maps to halo=1 at stage 1's half-resolution grid
// (since stage-0 cols [k*Sw0 - 2, (k+1)*Sw0 + 2) → stage-1 cols
// [k*Sw1 - 1, (k+1)*Sw1 + 1) after //2), which is exactly what stage
// 1's 3×3 patch reads at the primary boundary need.
//
// Latency is measured per OWNER event from t_arr (set in submit_batch
// just before push) to t_done (when the shard finishes the work for
// that event — stage 0 only if dropped, or both stages if survived).

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
#endif

#include "openeva/event.h"

#include "lat_stats.h"
#include "spsc.h"
#include "ssla_kernels.h"
#include "timed.h"

namespace {

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

// Per-shard latency samples: ring buffer of recent samples for snapshot stats.
// We don't grow unbounded — instead keep last N samples (rolling window).
struct ShardCtx {
    int  shard_id;
    int  core_id;
    bool pin;
    std::unique_ptr<deploy::SslaSPipeline> pipe;
    std::unique_ptr<ShardRing> q_in;

    // Rolling latency-sample buffer (mod-indexed, capacity sample_cap).
    std::vector<double> sample_buf;
    std::atomic<std::size_t> sample_idx{0};   // next slot to write (monotonic)
    std::size_t sample_cap = 0;

    std::atomic<std::uint64_t> total_owner{0};
    std::atomic<std::uint64_t> total_halo{0};
    std::atomic<std::uint64_t> pass_stage0{0};
    std::atomic<std::uint64_t> pass_stage1{0};   // events surviving both tdrops
};

void shard_worker(ShardCtx* ctx) {
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

        pipe.preprocess(m.ev, feat_in);
        int x = ex0, y = ey0;
        pipe.stage_forward(0, x, y, feat_in, feat0.data());

        if (m.is_owner) {
            const bool pass0 = pipe.tdrop_and_pool(0, x, y);
            bool pass1 = false;
            if (pass0) {
                // Stage 1 — x and y are now at half-resolution (set by
                // tdrop_and_pool). Stage-1 hidden state lives in the same
                // SslaSPipeline; halo=2 at stage 0 → halo=1 at stage 1
                // is sufficient for primary-boundary 3×3 patch reads.
                pipe.stage_forward(1, x, y, feat0.data(), feat1.data());
                pass1 = pipe.tdrop_and_pool(1, x, y);
            }
            const std::uint64_t t1 = deploy::rdtsc_now();
            const double lat_ns = deploy::TscClock::instance().tsc_to_ns(t1 - m.t_arr_tsc);
            const std::size_t idx = ctx->sample_idx.fetch_add(1, std::memory_order_relaxed);
            ctx->sample_buf[idx % ctx->sample_cap] = lat_ns;
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

struct S0Handle {
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
};

extern "C" {

// Initialize the stage-0 multi-core pipeline. Returns opaque handle (NULL on
// failure). Caller must eventually call s0_shutdown.
//
// weights_dir   — path to /tmp/ssla_s_random style export
// n_shards      — number of shard threads
// halo          — column halo (default 2)
// base_core     — first shard CPU core; shard k pinned to core base_core + k
//                 (use -1 to disable pinning)
// sample_cap    — per-shard rolling sample buffer capacity (e.g. 65536)
S0Handle* s0_init(const char* weights_dir, int n_shards, int halo,
                   int base_core, int sample_cap) {
    if (n_shards < 1) n_shards = 1;
    if (sample_cap < 1024) sample_cap = 1024;

    deploy::TscClock::instance().tsc_to_ns(0);  // calibrate once

    auto h = std::make_unique<S0Handle>();
    h->N = n_shards;
    h->halo = halo;
    h->pin_cores = (base_core >= 0);

    auto template_pipe = std::make_unique<deploy::SslaSPipeline>();
    try {
        template_pipe->load(weights_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[s0_init] load failed: %s\n", e.what());
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
        c->q_in     = std::make_unique<ShardRing>(1u << 16);
        c->sample_cap = static_cast<std::size_t>(sample_cap);
        c->sample_buf.assign(c->sample_cap, 0.0);
        h->ctxs.push_back(std::move(c));
    }
    for (int k = 0; k < h->N; ++k) {
        h->threads.emplace_back(shard_worker, h->ctxs[k].get());
    }
    h->t_start_or_reset = std::chrono::steady_clock::now();
    return h.release();
}

// Submit a batch of N events. Each event is 4 floats: t_us, x, y, polarity.
// Returns # events accepted into shard rings (events at out-of-bounds pixels
// are counted toward oob and not pushed).
int s0_submit_batch(S0Handle* h, const float* events_packed, int n) {
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
        // Compute covering shards (halo=h->halo, halo at full-res is on x only)
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

// Snapshot current stats. Output layout (16 doubles):
//   [0]  total_owner (events processed by owner)
//   [1]  total_halo  (events processed by halo shards)
//   [2]  pass_stage0 (events surviving stage-0 tdrop)
//   [3]  n_oob       (events dropped at submit due to out-of-bounds pixel)
//   [4]  wall_seconds_since_reset
//   [5]  avg ring occupancy (sum across shards)
//   [6]  p50_us
//   [7]  p90_us
//   [8]  p99_us
//   [9]  p99.9_us
//   [10] max_us
//   [11] mean_us
//   [12] # samples in window
//   [13] pass_stage1 (events surviving stage-0 AND stage-1 tdrops)
//   [14..15] reserved 0
//
// Stats are computed over the rolling sample window (last sample_cap × N events
// across all shards). Counters are cumulative since last reset.
void s0_snapshot_stats(S0Handle* h, double* out) {
    for (int i = 0; i < 16; ++i) out[i] = 0.0;
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

    if (!samples.empty()) {
        auto stats = deploy::LatStats::from_samples(samples);
        out[6]  = stats.p50_ns   * 1e-3;   // ns -> µs
        out[7]  = stats.p90_ns   * 1e-3;
        out[8]  = stats.p99_ns   * 1e-3;
        out[9]  = stats.p99_9_ns * 1e-3;
        out[10] = stats.max_ns   * 1e-3;
        out[11] = stats.mean_ns  * 1e-3;
        out[12] = static_cast<double>(samples.size());
    }
    out[13] = static_cast<double>(pass1);
}

// Reset cumulative counters and the time anchor (lat samples are NOT cleared —
// they're a rolling window).
void s0_reset_stats(S0Handle* h) {
    if (!h) return;
    for (auto& c : h->ctxs) {
        c->total_owner.store(0, std::memory_order_relaxed);
        c->total_halo.store(0, std::memory_order_relaxed);
        c->pass_stage0.store(0, std::memory_order_relaxed);
        c->pass_stage1.store(0, std::memory_order_relaxed);
        c->sample_idx.store(0, std::memory_order_relaxed);
    }
    h->n_oob.store(0, std::memory_order_relaxed);
    h->t_start_or_reset = std::chrono::steady_clock::now();
}

// Shutdown — push EOF to all shards, join, free.
void s0_shutdown(S0Handle* h) {
    if (!h) return;
    for (auto& c : h->ctxs) {
        ShardMsg m{}; m.eof = true;
        c->q_in->push(m);
    }
    for (auto& t : h->threads) if (t.joinable()) t.join();
    delete h;
}

}  // extern "C"
