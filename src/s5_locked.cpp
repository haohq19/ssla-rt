// s5_locked.cpp — Scheme 5: shared-memory + per-patch lock.
//
// Key design simplification vs S3/S4:
//
//   Each event is fully owned by ONE shard (the shard owning the event's
//   center column). That shard does ALL the work for the event:
//     - All 9 patch updates for layer 0
//     - All 9 patch updates for layer 1
//     - Residual + layernorm
//     - Forward to stage thread
//
//   Hidden states for layer 0 / layer 1 (both at stage-0 spatial resolution)
//   are SHARED across shards. Race protection via per-patch spinlocks.
//   At boundary patches multiple shards may concurrently update; locks
//   serialize them but do NOT enforce producer-seq order — bounded
//   reordering is the user-accepted relaxation.
//
//   No partial aggregation. No broadcast. No central sum thread. Each
//   shard runs independently.
//
// Architecture:
//
//   producer  ──┬─→  shard 0  ──┐
//               ├─→  shard 1  ──┤   per-shard q_to_stage  ──→ stage thread
//               └─→  shard k  ──┘   (merger)                  (stages 1-3 + head)
//
// Producer pushes each event to exactly ONE shard's q_own. Each shard
// processes its own events, writing directly to shared hidden state under
// per-patch locks. Stage thread merges per-shard q_to_stage by seq.

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

#include <pthread.h>
#include <sched.h>

#include "openeva/event.h"
#include "src/io.h"

#include "lat_stats.h"
#include "oracle.h"
#include "spsc.h"
#include "ssla_kernels.h"
#include "timed.h"

namespace {

struct Args {
    std::string weights_dir;
    std::string events_path;
    int  warmup    = 10000;
    int  random_n  = 200000;
    int  random_h  = 240;
    int  random_w  = 304;
    std::uint32_t random_seed = 1;
    std::string oracle_dump;
    std::size_t oracle_every = 10000;
    int  q_capacity_pow2 = 16;
    bool pin_cores = true;
    int  base_core = 0;
    int  num_shards = 4;
    double target_rate_mev = 0.0;
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir>\n"
        "          [--events <npy> | --random-n N --random-hw H W --random-seed S]\n"
        "          [--shards N] [--warmup K]\n"
        "          [--oracle-dump P] [--oracle-every K]\n"
        "          [--target-mev R] [--no-pin] [--base-core N]\n", argv0);
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
        if (auto v = eat("--oracle-dump"))  { a.oracle_dump = v; continue; }
        if (auto v = eat("--oracle-every")) { a.oracle_every = static_cast<std::size_t>(std::atoll(v)); continue; }
        if (auto v = eat("--shards"))       { a.num_shards = std::atoi(v); continue; }
        if (auto v = eat("--target-mev"))   { a.target_rate_mev = std::atof(v); continue; }
        if (auto v = eat("--base-core"))    { a.base_core = std::atoi(v); continue; }
        if (auto v = eat("--q-cap-pow2"))   { a.q_capacity_pow2 = std::atoi(v); continue; }
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
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

inline int shard_of_col(int c, int W, int N) {
    if (c < 0) c = 0;
    if (c >= W) c = W - 1;
    const int strip_w = (W + N - 1) / N;
    int s = c / strip_w;
    if (s >= N) s = N - 1;
    return s;
}

// Stride a flat std::atomic_flag array. Need to be heap-allocated since
// atomic_flag isn't movable/copyable.

// =====================================================================
// Message types
// =====================================================================

enum class MsgType : std::uint8_t { Event = 0, Marker = 1, Eof = 2 };

struct OwnMsg {
    MsgType        type;
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x, y;
    float          feat_in[deploy::kInDim];
};

struct ToStageMsg {
    MsgType        type;
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x, y;
    float          feat[deploy::kC0];
};

// =====================================================================
// Pipeline
// =====================================================================

struct Pipeline {
    deploy::SslaSPipeline pipe;
    int num_shards = 0;
    int W = 0, H = 0;

    // Per-patch spin locks for L0 and L1 (both share stage-0 grid dims).
    // atomic_flag is non-movable/non-copyable; heap-allocate.
    std::unique_ptr<std::atomic_flag[]> patch_locks_l0;
    std::unique_ptr<std::atomic_flag[]> patch_locks_l1;
    int lock_grid_size = 0;

    std::vector<std::unique_ptr<deploy::SpscRing<OwnMsg>>>     q_own;
    std::vector<std::unique_ptr<deploy::SpscRing<ToStageMsg>>> q_to_stage;
    // committed_seq[k] = highest seq shard k has finished. Once shard k
    // observes producer's `pushed_seq_max >= S` AND its q_own is empty,
    // it advances committed_seq[k] to S — meaning "no push from me for
    // seq <= S". Stage thread uses this to know when it's safe to emit.
    std::vector<std::unique_ptr<std::atomic<std::uint64_t>>>   committed_seq;
    // Producer's high-water-mark — last pushed event seq + 1 (so 0 = nothing pushed yet).
    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> pushed_seq_max{0};

    std::vector<std::vector<double>> lat_drop_s0, lat_drop_s1, lat_drop_s2, lat_emit_s3;

    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done_stages{0};
    alignas(deploy::kCacheLine) std::atomic<bool>          done{false};
};

// =====================================================================
// Shard thread
// =====================================================================

void thread_shard(Pipeline& p, int k, int core, bool pin) {
    if (pin) pin_to_core(core);

    std::vector<float> qvg_scratch(3 * deploy::kC0);
    std::vector<float> qh_scratch (deploy::kC0);
    std::vector<float> residual_scratch(deploy::kC0);
    std::vector<float> l0_out(deploy::kC0);
    std::vector<float> l1_out(deploy::kC0);

    bool got_eof = false;
    std::uint64_t last_seq = 0;  // highest seq we've finished
    while (true) {
        bool did_work = false;
        OwnMsg om;
        while (p.q_own[k]->try_pop(om)) {
            did_work = true;
            if (om.type == MsgType::Eof) {
                got_eof = true;
                continue;
            }
            if (om.type == MsgType::Marker) {
                ToStageMsg sm{};
                sm.type = MsgType::Marker;
                sm.seq = om.seq;
                while (!p.q_to_stage[k]->try_push_nb(sm)) __builtin_ia32_pause();
                last_seq = om.seq;
                p.committed_seq[k]->store(last_seq, std::memory_order_release);
                continue;
            }
            // Real event — full stage 0 forward (L0 + L1) under per-patch locks.
            p.pipe.stage_forward_locked(
                0, om.x, om.y, om.feat_in,
                p.patch_locks_l0.get(), p.patch_locks_l1.get(),
                qvg_scratch.data(), qh_scratch.data(),
                residual_scratch.data(),
                l0_out.data(), l1_out.data());

            ToStageMsg sm{};
            sm.type      = MsgType::Event;
            sm.seq       = om.seq;
            sm.t_arr_tsc = om.t_arr_tsc;
            sm.x = om.x; sm.y = om.y;
            std::memcpy(sm.feat, l1_out.data(), sizeof(sm.feat));
            while (!p.q_to_stage[k]->try_push_nb(sm)) __builtin_ia32_pause();
            last_seq = om.seq;
            p.committed_seq[k]->store(last_seq, std::memory_order_release);
        }
        // If q_own is empty, we can advance committed_seq up to producer's
        // pushed_seq_max - 1 (the highest seq producer has pushed to ANY
        // shard so far, so we know we won't be receiving anything <= that
        // that we haven't already seen).
        if (!did_work) {
            const std::uint64_t prod_max = p.pushed_seq_max.load(std::memory_order_acquire);
            // prod_max = (last pushed seq) + 1. So shards have seen all events with seq < prod_max.
            // committed_seq advances to prod_max - 1 (= max seq routed, regardless of shard).
            if (prod_max > 0 && last_seq + 1 < prod_max) {
                last_seq = prod_max - 1;
                p.committed_seq[k]->store(last_seq, std::memory_order_release);
                did_work = true;
            }
            if (!did_work) {
                if (got_eof) break;
                __builtin_ia32_pause();
            }
        }
    }
    p.committed_seq[k]->store(UINT64_MAX, std::memory_order_release);
}

// =====================================================================
// Stage thread (merger + stages 1-3 + head)
// =====================================================================

void thread_stages(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    const int N = p.num_shards;
    std::vector<float> feat1(deploy::kC1), feat2(deploy::kC2), feat3(deploy::kC3);

    struct CachedFront {
        bool        have = false;
        ToStageMsg  msg;
    };
    std::vector<CachedFront> cache(N);

    auto try_fill = [&](int k) {
        if (!cache[k].have && p.q_to_stage[k]->try_pop(cache[k].msg)) {
            cache[k].have = true;
        }
    };

    while (true) {
        for (int k = 0; k < N; ++k) try_fill(k);

        // Pick the shard with the smallest cached seq.
        std::uint64_t min_cached = UINT64_MAX;
        int min_k = -1;
        for (int k = 0; k < N; ++k) {
            if (cache[k].have && cache[k].msg.seq < min_cached) {
                min_cached = cache[k].msg.seq;
                min_k = k;
            }
        }

        // Check global progress: are all shards done?
        if (min_k == -1) {
            std::uint64_t min_committed = UINT64_MAX;
            for (int k = 0; k < N; ++k) {
                std::uint64_t c = p.committed_seq[k]->load(std::memory_order_acquire);
                if (c < min_committed) min_committed = c;
            }
            if (min_committed == UINT64_MAX) break;  // all shards exited
            __builtin_ia32_pause();
            continue;
        }

        // We have a cached candidate seq=min_cached. Safe to emit iff
        // every OTHER shard has committed_seq >= min_cached (i.e., they
        // have finished processing everything up to and including
        // min_cached, so they won't push anything earlier).
        bool can_emit = true;
        for (int k = 0; k < N; ++k) {
            if (k == min_k) continue;
            const std::uint64_t c = p.committed_seq[k]->load(std::memory_order_acquire);
            if (c < min_cached) { can_emit = false; break; }
        }
        if (!can_emit) {
            __builtin_ia32_pause();
            continue;
        }

        ToStageMsg in = cache[min_k].msg;
        cache[min_k].have = false;

        if (in.type == MsgType::Marker) {
            p.marker_done_stages.store(in.seq, std::memory_order_release);
            continue;
        }

        int x = in.x, y = in.y;
        if (!p.pipe.tdrop_and_pool(0, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat_drop_s0[min_k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        p.pipe.stage_forward(1, x, y, in.feat, feat1.data());
        if (!p.pipe.tdrop_and_pool(1, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat_drop_s1[min_k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        p.pipe.stage_forward(2, x, y, feat1.data(), feat2.data());
        if (!p.pipe.tdrop_and_pool(2, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat_drop_s2[min_k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        p.pipe.stage_forward(3, x, y, feat2.data(), feat3.data());
        p.pipe.head_decode_cell(3, x, y, feat3.data());
        const auto t1 = deploy::rdtsc_now();
        p.lat_emit_s3[min_k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    deploy::TscClock::instance().tsc_to_ns(0);

    std::vector<openeva::Event> events;
    if (!args.events_path.empty()) events = openeva::load_events_npy(args.events_path);
    else events = make_random_events(args.random_n, args.random_h, args.random_w, args.random_seed);
    if (events.empty()) { std::fprintf(stderr, "no events\n"); return EXIT_FAILURE; }

    Pipeline pl;
    pl.num_shards = args.num_shards;
    pl.pipe.load(args.weights_dir);
    pl.pipe.reset();
    pl.W = pl.pipe.W();
    pl.H = pl.pipe.H();
    pl.lock_grid_size = pl.pipe.stage0_lock_grid_size();
    // atomic_flag is default-initialized to indeterminate value pre-C++20;
    // explicitly clear all locks.
    pl.patch_locks_l0.reset(new std::atomic_flag[pl.lock_grid_size]);
    pl.patch_locks_l1.reset(new std::atomic_flag[pl.lock_grid_size]);
    for (int i = 0; i < pl.lock_grid_size; ++i) {
        pl.patch_locks_l0[i].clear();
        pl.patch_locks_l1[i].clear();
    }

    const std::size_t qcap = std::size_t{1} << args.q_capacity_pow2;
    const int N = args.num_shards;
    pl.q_own.resize(N);
    pl.q_to_stage.resize(N);
    pl.committed_seq.resize(N);
    for (int k = 0; k < N; ++k) {
        pl.q_own[k]      = std::make_unique<deploy::SpscRing<OwnMsg>>(qcap);
        pl.q_to_stage[k] = std::make_unique<deploy::SpscRing<ToStageMsg>>(qcap);
        pl.committed_seq[k] = std::make_unique<std::atomic<std::uint64_t>>(0);
    }
    pl.lat_drop_s0.assign(N, {});
    pl.lat_drop_s1.assign(N, {});
    pl.lat_drop_s2.assign(N, {});
    pl.lat_emit_s3.assign(N, {});
    for (int k = 0; k < N; ++k) {
        pl.lat_drop_s0[k].reserve(events.size() / N + 1024);
        pl.lat_drop_s1[k].reserve(events.size() / N + 1024);
        pl.lat_drop_s2[k].reserve(events.size() / N + 1024);
        pl.lat_emit_s3[k].reserve(events.size() / N + 1024);
    }

    std::fprintf(stderr, "[s5] events=%zu shards=%d warmup=%d\n",
                 events.size(), N, args.warmup);

    std::vector<std::thread> shards;
    for (int k = 0; k < N; ++k) {
        shards.emplace_back(thread_shard, std::ref(pl), k,
                            args.base_core + k, args.pin_cores);
    }
    std::thread t_stages(thread_stages, std::ref(pl),
                         args.base_core + N, args.pin_cores);
    if (args.pin_cores) pin_to_core(args.base_core + N + 1);

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);

    auto push_with_yield = [&](auto* q, const auto& msg) {
        int spins = 0;
        while (!q->try_push_nb(msg)) {
            if (++spins > 4096) { std::this_thread::yield(); spins = 0; }
            else __builtin_ia32_pause();
        }
    };

    auto push_one = [&](std::uint64_t seq, const openeva::Event& e) {
        const int ex = static_cast<int>(e.x);
        const int ey = static_cast<int>(e.y);
        if (ex < 0 || ex >= pl.W || ey < 0 || ey >= pl.H) {
            // Even out-of-bounds events get accounted for in pushed_seq_max
            // so shards' committed_seq can advance past them.
            pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
            return;
        }
        // Preprocess on producer (single writer to last_t_us).
        float feat[deploy::kInDim];
        pl.pipe.preprocess(e, feat);
        const std::uint64_t t = deploy::rdtsc_now();
        const int owner_k = shard_of_col(ex, pl.W, N);
        OwnMsg om{};
        om.type = MsgType::Event;
        om.seq = seq;
        om.t_arr_tsc = t;
        om.x = ex; om.y = ey;
        om.feat_in[0] = feat[0]; om.feat_in[1] = feat[1];
        push_with_yield(pl.q_own[owner_k].get(), om);
        pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
    };
    auto push_marker = [&](std::uint64_t seq) {
        for (int k = 0; k < N; ++k) {
            OwnMsg om{};
            om.type = MsgType::Marker;
            om.seq = seq;
            push_with_yield(pl.q_own[k].get(), om);
        }
        pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
    };
    auto wait_marker = [&](std::uint64_t seq) {
        while (pl.marker_done_stages.load(std::memory_order_acquire) < seq) {
            __builtin_ia32_pause();
        }
    };

    for (std::size_t i = 0; i < warmup; ++i) push_one(static_cast<std::uint64_t>(i), events[i]);
    push_marker(static_cast<std::uint64_t>(warmup));
    wait_marker(static_cast<std::uint64_t>(warmup));

    std::unique_ptr<deploy::OracleRecorder> oracle;
    if (!args.oracle_dump.empty()) {
        oracle = std::make_unique<deploy::OracleRecorder>(
            pl.pipe.num_anchors_total(), pl.pipe.cols(),
            args.oracle_every, args.oracle_dump);
    }

    const double ns_per_tick = deploy::TscClock::instance().tsc_to_ns(1);
    std::uint64_t pace_ticks = 0;
    if (args.target_rate_mev > 0.0) {
        const double ns_per_event = 1000.0 / args.target_rate_mev;
        pace_ticks = static_cast<std::uint64_t>(ns_per_event / ns_per_tick);
    }

    const auto wall_start = std::chrono::steady_clock::now();
    std::uint64_t next_push_tsc = deploy::rdtsc_now();
    for (std::size_t i = warmup; i < total; ++i) {
        if (pace_ticks > 0) {
            while (deploy::rdtsc_now() < next_push_tsc) __builtin_ia32_pause();
            next_push_tsc += pace_ticks;
        }
        push_one(static_cast<std::uint64_t>(i), events[i]);
        if (oracle) {
            const std::size_t mi = i - warmup;
            if (mi % args.oracle_every == 0) {
                const std::uint64_t mseq = static_cast<std::uint64_t>(i + 1);
                push_marker(mseq);
                wait_marker(mseq);
                oracle->record(static_cast<std::uint64_t>(mi), pl.pipe.last_output());
            }
        }
    }
    push_marker(static_cast<std::uint64_t>(total));
    wait_marker(static_cast<std::uint64_t>(total));
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start).count();

    for (int k = 0; k < N; ++k) {
        OwnMsg eof{}; eof.type = MsgType::Eof;
        push_with_yield(pl.q_own[k].get(), eof);
    }
    pl.done.store(true, std::memory_order_release);
    for (auto& t : shards) t.join();
    t_stages.join();

    std::vector<double> all;
    std::size_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < N; ++k) {
        for (auto v : pl.lat_drop_s0[k]) all.push_back(v);
        for (auto v : pl.lat_drop_s1[k]) all.push_back(v);
        for (auto v : pl.lat_drop_s2[k]) all.push_back(v);
        for (auto v : pl.lat_emit_s3[k]) all.push_back(v);
        s0 += pl.lat_drop_s0[k].size();
        s1 += pl.lat_drop_s1[k].size();
        s2 += pl.lat_drop_s2[k].size();
        s3 += pl.lat_emit_s3[k].size();
    }
    auto stats = deploy::LatStats::from_samples(all);
    const double thru = wall_ns > 0 ? all.size() * 1e9 / wall_ns * 1e-6 : 0.0;

    std::fprintf(stdout, "\n=== Scheme 5: shared-memory + per-patch lock (%d shards) ===\n", N);
    std::fprintf(stdout, "  total events       : %zu\n", total);
    std::fprintf(stdout, "  measured events    : %zu\n", all.size());
    std::fprintf(stdout, "  wall (timed)       : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput         : %.3f Mev/s\n", thru);
    std::fprintf(stdout, "  drops              : s0=%zu s1=%zu s2=%zu emit=%zu\n",
                 s0, s1, s2, s3);
    std::fprintf(stdout, "\n  end-to-end latency:\n");
    stats.print("    ");
    if (oracle) {
        oracle->save();
        std::fprintf(stdout, "\n  oracle dump        : %s (%zu checkpoints)\n",
                     args.oracle_dump.c_str(), oracle->num_checkpoints());
    }
    return EXIT_SUCCESS;
}
