// s_stage0_multicore.cpp — Stage-0-only with multi-core spatial sharding.
//
// Architecture (mirrors S8's stage-0 slice, with halo=2 default):
//
//   producer ──┬─→ q_in[0] ─→ shard 0 (cols [0*S - h, 1*S + h))
//              ├─→ q_in[1] ─→ shard 1 (cols [1*S - h, 2*S + h))
//              └─→ ...
//
// Each shard owns a PRIVATE SslaSPipeline (stage-0 layers' hidden state).
// Per event, the producer broadcasts to every shard covering its column
// (owner + 0–2 halo neighbors). Each shard runs stage_forward(0) to keep
// its hidden state synced. Only the OWNER shard runs tdrop_and_pool(0)
// and counts pass/drop. State is lock-free (each shard's grid is private).
//
// Per-event latency = t_done(owner) - t_arr(producer push), measured by
// the OWNER shard only (halo work is bounded but isn't part of any
// downstream consumer's wait — the event is "done" when the owner finishes).
//
// Usage:
//   s_stage0_multicore --weights <dir> [--shards N] [--halo H]
//                      [--random-n K] [--target-mev R] [--base-core C]
//                      [--no-pin] [--warmup K]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "openeva/event.h"
#include "src/io.h"

#include "lat_stats.h"
#include "spsc.h"
#include "ssla_kernels.h"
#include "timed.h"

namespace {

struct Args {
    std::string weights_dir;
    std::string events_path;
    int  warmup       = 10000;
    int  random_n     = 200000;
    int  random_h     = 240;
    int  random_w     = 304;
    std::uint32_t random_seed = 1;
    int    num_shards = 4;
    int    halo       = 2;
    bool   pin_cores  = true;
    int    base_core  = 1;     // shard 0 on base, shard k on base+k, producer on base-1 (or 0 if base==0)
    double target_rate_mev = 0.0;
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir>\n"
        "         [--events <npy> | --random-n N --random-hw H W --random-seed S]\n"
        "         [--shards N] [--halo H]\n"
        "         [--target-mev R] [--no-pin] [--base-core C] [--warmup K]\n", argv0);
    std::exit(EXIT_FAILURE);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto eat = [&](const char* k) -> const char* {
            if (std::strcmp(argv[i], k) != 0) return nullptr;
            if (i + 1 >= argc) die_usage(argv[0]);
            return argv[++i];
        };
        if (auto v = eat("--weights"))      { a.weights_dir = v; continue; }
        if (auto v = eat("--events"))       { a.events_path = v; continue; }
        if (auto v = eat("--warmup"))       { a.warmup = std::atoi(v); continue; }
        if (auto v = eat("--random-n"))     { a.random_n = std::atoi(v); continue; }
        if (auto v = eat("--random-seed"))  { a.random_seed = static_cast<std::uint32_t>(std::atoi(v)); continue; }
        if (auto v = eat("--shards"))       { a.num_shards = std::atoi(v); continue; }
        if (auto v = eat("--halo"))         { a.halo = std::atoi(v); continue; }
        if (auto v = eat("--target-mev"))   { a.target_rate_mev = std::atof(v); continue; }
        if (auto v = eat("--base-core"))    { a.base_core = std::atoi(v); continue; }
        if (std::strcmp(argv[i], "--no-pin") == 0) { a.pin_cores = false; continue; }
        if (std::strcmp(argv[i], "--random-hw") == 0) {
            if (i + 2 >= argc) die_usage(argv[0]);
            a.random_h = std::atoi(argv[++i]);
            a.random_w = std::atoi(argv[++i]);
            continue;
        }
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) die_usage(argv[0]);
        std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
        die_usage(argv[0]);
    }
    if (a.weights_dir.empty()) die_usage(argv[0]);
    if (a.num_shards < 1) a.num_shards = 1;
    return a;
}

std::vector<openeva::Event>
make_random_events(int n, int h, int w, std::uint32_t seed) {
    std::vector<openeva::Event> ev(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dx(0, w - 1);
    std::uniform_int_distribution<int> dy(0, h - 1);
    std::uniform_int_distribution<int> dp(0, 1);
    for (int i = 0; i < n; ++i) {
        ev[i].t = static_cast<float>(i);
        ev[i].x = static_cast<float>(dx(rng));
        ev[i].y = static_cast<float>(dy(rng));
        ev[i].p = static_cast<float>(dp(rng));
    }
    return ev;
}

void pin_to_core(int core_id) {
#if defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#else
    (void)core_id;
#endif
}

// ---- per-shard message ------------------------------------------------------
struct ShardMsg {
    std::uint64_t  t_arr_tsc;     // producer-side push timestamp
    openeva::Event ev;
    bool           is_owner;       // true if this shard is the OWNER for this event
    bool           eof;
};

// ---- ring type --------------------------------------------------------------
using ShardRing = deploy::SpscRing<ShardMsg>;

// ---- shard worker -----------------------------------------------------------
struct ShardCtx {
    int  shard_id;
    int  core_id;
    bool pin;
    std::unique_ptr<deploy::SslaSPipeline> pipe;
    std::unique_ptr<ShardRing> q_in;
    // Stats (only set by OWNER events; non-owner events do work but their
    // latency is not the user-facing latency).
    std::vector<double> per_event_ns;
    std::size_t pass_owner = 0;
    std::size_t total_owner = 0;
    std::size_t total_halo  = 0;
};

void shard_worker(ShardCtx* ctx) {
    if (ctx->pin) pin_to_core(ctx->core_id);

    deploy::SslaSPipeline& pipe = *ctx->pipe;
    float feat_in[deploy::kInDim];
    std::vector<float> feat0(deploy::kC0);

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
            const bool pass = pipe.tdrop_and_pool(0, x, y);
            const std::uint64_t t1 = deploy::rdtsc_now();
            ctx->per_event_ns.push_back(
                deploy::TscClock::instance().tsc_to_ns(t1 - m.t_arr_tsc));
            ctx->total_owner += 1;
            if (pass) ctx->pass_owner += 1;
        } else {
            ctx->total_halo += 1;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    deploy::TscClock::instance().tsc_to_ns(0);

    std::vector<openeva::Event> events;
    if (!args.events_path.empty()) events = openeva::load_events_npy(args.events_path);
    else events = make_random_events(args.random_n, args.random_h, args.random_w,
                                      args.random_seed);
    if (events.empty()) { std::fprintf(stderr, "no events\n"); return EXIT_FAILURE; }

    const int N = args.num_shards;
    const int H = args.halo;

    // ---- One template pipe to learn geometry, then one private pipe per shard
    auto template_pipe = std::make_unique<deploy::SslaSPipeline>();
    template_pipe->load(args.weights_dir);
    template_pipe->reset();
    const int W_full = template_pipe->W();
    const int H_full = template_pipe->H();
    const int strip_w = (W_full + N - 1) / N;

    std::fprintf(stderr,
                 "[stage0-mc] N=%d halo=%d W=%d strip_w=%d  events=%zu warmup=%d "
                 "target=%.3f Mev/s pin=%s base_core=%d\n",
                 N, H, W_full, strip_w, events.size(), args.warmup,
                 args.target_rate_mev, args.pin_cores ? "yes" : "no",
                 args.base_core);

    // ---- Per-shard contexts
    std::vector<std::unique_ptr<ShardCtx>> ctxs(N);
    for (int k = 0; k < N; ++k) {
        ctxs[k] = std::make_unique<ShardCtx>();
        ctxs[k]->shard_id = k;
        ctxs[k]->core_id  = args.base_core + k;
        ctxs[k]->pin      = args.pin_cores;
        ctxs[k]->pipe     = std::make_unique<deploy::SslaSPipeline>();
        ctxs[k]->pipe->load(args.weights_dir);
        ctxs[k]->pipe->reset();
        ctxs[k]->q_in     = std::make_unique<ShardRing>(1u << 16);   // 64k slots
        ctxs[k]->per_event_ns.reserve(events.size() / std::max(1, N) + 256);
    }

    // ---- Producer routing helper: for col x at full res, return shards covering
    auto cover = [&](int x, int out[4]) -> int {
        int n = 0;
        // shard k owns [k*strip_w, (k+1)*strip_w); covers [k*strip_w - H, (k+1)*strip_w + H)
        for (int dk = -1; dk <= 1; ++dk) {
            const int k = x / strip_w + dk;
            if (k < 0 || k >= N) continue;
            const int lo = k * strip_w - H;
            const int hi = (k + 1) * strip_w + H;
            if (x >= lo && x < hi) out[n++] = k;
        }
        return n;
    };

    // ---- Spawn shard workers
    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int k = 0; k < N; ++k) threads.emplace_back(shard_worker, ctxs[k].get());

    // ---- Producer thread runs in main; pin to producer core (base_core - 1, or
    //      base_core + N if base_core == 0).
    if (args.pin_cores) {
        int prod_core = (args.base_core > 0) ? args.base_core - 1
                                              : args.base_core + N;
        pin_to_core(prod_core);
    }

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);

    // Warmup (still routes through queues, but stats are reset afterward)
    auto push_event = [&](const openeva::Event& e, std::uint64_t t_arr_tsc) {
        const int ex = static_cast<int>(e.x);
        if (ex < 0 || ex >= W_full) return;
        int targets[4];
        const int n_targets = cover(ex, targets);
        if (n_targets == 0) return;
        const int owner = ex / strip_w;
        for (int t = 0; t < n_targets; ++t) {
            ShardMsg m;
            m.t_arr_tsc = t_arr_tsc;
            m.ev = e;
            m.is_owner = (targets[t] == owner);
            m.eof = false;
            ctxs[targets[t]]->q_in->push(m);
        }
    };

    // Warmup pass
    for (std::size_t i = 0; i < warmup; ++i) {
        push_event(events[i], deploy::rdtsc_now());
    }

    // Drain warmup, then reset stats. Ensure all queues empty before resetting.
    for (int k = 0; k < N; ++k) {
        while (ctxs[k]->q_in->size() > 0) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
    }
    // Brief settle wait so per_event_ns flushes for in-flight events.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (int k = 0; k < N; ++k) {
        ctxs[k]->per_event_ns.clear();
        ctxs[k]->pass_owner = 0;
        ctxs[k]->total_owner = 0;
        ctxs[k]->total_halo  = 0;
    }

    // ---- Timed loop with optional rate pacing
    const double ns_per_tick = deploy::TscClock::instance().tsc_to_ns(1);
    const double ticks_per_event = (args.target_rate_mev > 0.0)
        ? (1e3 / args.target_rate_mev) / ns_per_tick    // ns/event = 1000 / Mev/s
        : 0.0;

    const auto wall_start = std::chrono::steady_clock::now();
    const std::uint64_t base_tsc = deploy::rdtsc_now();
    const std::size_t measured_first = warmup;
    for (std::size_t i = measured_first; i < total; ++i) {
        if (ticks_per_event > 0.0) {
            const std::uint64_t deadline = base_tsc + static_cast<std::uint64_t>(
                ticks_per_event * static_cast<double>(i - measured_first));
            while (deploy::rdtsc_now() < deadline) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }
        }
        push_event(events[i], deploy::rdtsc_now());
    }
    const auto wall_end_push = std::chrono::steady_clock::now();

    // ---- Send EOF, wait for shards to drain
    for (int k = 0; k < N; ++k) {
        ShardMsg m{}; m.eof = true;
        ctxs[k]->q_in->push(m);
    }
    for (auto& t : threads) t.join();
    const auto wall_end = std::chrono::steady_clock::now();

    const double push_ns = std::chrono::duration<double, std::nano>(
        wall_end_push - wall_start).count();
    const double drain_ns = std::chrono::duration<double, std::nano>(
        wall_end - wall_end_push).count();
    // Total wall = push + drain. Sustained throughput must use this; if
    // we used push_ns alone, a producer that dumps into oversized rings
    // looks artificially fast.
    const double wall_ns = push_ns + drain_ns;

    // ---- Aggregate latency across all shards
    std::vector<double> all_lat;
    std::size_t total_owner = 0, total_halo = 0, pass_owner = 0;
    for (int k = 0; k < N; ++k) {
        all_lat.insert(all_lat.end(),
                       ctxs[k]->per_event_ns.begin(),
                       ctxs[k]->per_event_ns.end());
        total_owner += ctxs[k]->total_owner;
        total_halo  += ctxs[k]->total_halo;
        pass_owner  += ctxs[k]->pass_owner;
    }

    if (all_lat.empty()) {
        std::fprintf(stderr, "no measured events\n");
        return EXIT_FAILURE;
    }

    // No time-order trim across shards (per-shard time order is preserved
    // within each shard's per_event_ns; the cross-shard concatenation has no
    // meaningful global time order, so trimming "first 5% / last 5%" doesn't
    // map cleanly). The warmup pass + 50 ms settle reset already removed
    // queue-priming bleed; rate-paced runs have no end-of-stream tail
    // amplification, and saturation runs have a small one we accept.
    auto stats = deploy::LatStats::from_samples(all_lat);
    auto& ss = all_lat;

    const double thru_meps = (wall_ns > 0.0)
        ? static_cast<double>(total_owner) * 1e9 / wall_ns * 1e-6 : 0.0;

    const double pass_rate = total_owner > 0
        ? 100.0 * static_cast<double>(pass_owner) / static_cast<double>(total_owner) : 0.0;
    const double halo_overhead = total_owner > 0
        ? 100.0 * static_cast<double>(total_halo) / static_cast<double>(total_owner) : 0.0;

    std::fprintf(stdout, "\n=== Stage-0 multi-core (N=%d, halo=%d) ===\n", N, H);
    std::fprintf(stdout, "  total events      : %zu\n", total);
    std::fprintf(stdout, "  warmup events     : %zu\n", warmup);
    std::fprintf(stdout, "  measured (owner)  : %zu\n", total_owner);
    std::fprintf(stdout, "  halo events done  : %zu (%.1f%% overhead)\n",
                 total_halo, halo_overhead);
    std::fprintf(stdout, "  measured samples  : %zu\n", all_lat.size());
    std::fprintf(stdout, "  push wall time    : %.2f ms\n", push_ns * 1e-6);
    std::fprintf(stdout, "  drain wall time   : %.2f ms\n", drain_ns * 1e-6);
    std::fprintf(stdout, "  total wall (p+d)  : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput        : %.3f Mev/s  (steady = total events / total wall)\n", thru_meps);
    if (args.target_rate_mev > 0.0) {
        std::fprintf(stdout, "  target rate       : %.3f Mev/s%s\n",
                     args.target_rate_mev,
                     thru_meps < args.target_rate_mev * 0.95 ? "  ** UNDER **" : "");
    }
    std::fprintf(stdout, "  stage-0 pass rate : %.2f%% (%zu / %zu owner)\n",
                 pass_rate, pass_owner, total_owner);
    std::fprintf(stdout, "\n  per-event end-to-end latency (push→owner-done):\n");
    stats.print("    ");
    std::fprintf(stdout, "\n  per-event distribution (ns log bins):\n");
    deploy::print_histogram(ss, "    ");

    std::fprintf(stdout, "\n  per-shard counts:\n");
    for (int k = 0; k < N; ++k) {
        std::fprintf(stdout, "    shard %2d (core %2d): owner=%zu halo=%zu pass=%zu (%.1f%%)\n",
                     k, ctxs[k]->core_id,
                     ctxs[k]->total_owner, ctxs[k]->total_halo,
                     ctxs[k]->pass_owner,
                     ctxs[k]->total_owner > 0
                         ? 100.0 * ctxs[k]->pass_owner / ctxs[k]->total_owner : 0.0);
    }

    return EXIT_SUCCESS;
}
