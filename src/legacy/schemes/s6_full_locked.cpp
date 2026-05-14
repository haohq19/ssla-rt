// s6_full_locked.cpp — Scheme 6: full-pipeline shared-memory + per-patch lock.
//
// Extension of S5: each shard runs the FULL pipeline (stages 0..3 + head)
// for its own events. NO central stage thread. All stages' hidden states
// are shared across shards via per-patch spinlocks. tdrop counters are
// atomic. Head prediction cells are per-cell locked.
//
// Architecture:
//
//   producer  ──┬─→  shard 0 (stage 0..3 + head)  ──→ predictions tensor
//               ├─→  shard 1 (stage 0..3 + head)  ──→ predictions tensor
//               └─→  shard k (stage 0..3 + head)  ──→ predictions tensor
//
// Shared state (single allocation in pl.pipe):
//   - hidden_[0..7]               (under per-layer per-patch spinlocks)
//   - tdrop_counter_ (atomic)     (under fetch_add; replaces internal uint8_t)
//   - last_predictions_           (under per-cell spinlock)
//
// Producer publishes pushed_seq_max atomic. Shards advance committed_seq
// when their q_own is empty. Producer waits for ALL committed_seq >=
// marker.seq for oracle drain.
//
// Equivalence: same as S5 — bounded reorder at boundary patches and at
// concurrent updates to the same head cell. NO info loss (every event
// is applied exactly once to each patch state in its 9-patch window).

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
    int  num_shards = 8;
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

enum class MsgType : std::uint8_t { Event = 0, Marker = 1, Eof = 2 };

struct OwnMsg {
    MsgType        type;
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x, y;
    float          feat_in[deploy::kInDim];
};

struct Pipeline {
    deploy::SslaSPipeline pipe;
    int num_shards = 0;
    int W = 0, H = 0;

    // Per-layer patch locks. Layers 0/1 share stage-0 grid, etc., but
    // we allocate separate arrays per layer for cleanest isolation.
    std::array<std::unique_ptr<std::atomic_flag[]>, deploy::kNumLayers> patch_locks;
    std::array<int, deploy::kNumLayers> layer_lock_size{};

    // Atomic tdrop counters (one per pooled cell, per stage 0..2).
    std::array<std::unique_ptr<std::atomic<std::uint8_t>[]>, 3> tdrop_atomic;
    std::array<int, 3> tdrop_size{};

    // Per-cell head locks (sized to total_anchors).
    std::unique_ptr<std::atomic_flag[]> head_cell_locks;
    int head_lock_size = 0;

    std::vector<std::unique_ptr<deploy::SpscRing<OwnMsg>>>     q_own;
    std::vector<std::unique_ptr<std::atomic<std::uint64_t>>>   committed_seq;
    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> pushed_seq_max{0};

    std::vector<std::vector<double>> lat_drop_s0, lat_drop_s1, lat_drop_s2, lat_emit_s3;
};

// =====================================================================
// Shard thread: full pipeline per event
// =====================================================================

void thread_shard(Pipeline& p, int k, int core, bool pin) {
    if (pin) pin_to_core(core);

    // Per-thread scratch sized to maximum dim.
    constexpr int kMax = deploy::kC3;
    std::vector<float> qvg_scratch(3 * kMax);
    std::vector<float> qh_scratch (kMax);
    std::vector<float> residual_scratch(kMax);
    std::vector<float> l0_out(kMax);
    std::vector<float> stage_out[deploy::kNumStages] = {
        std::vector<float>(kMax), std::vector<float>(kMax),
        std::vector<float>(kMax), std::vector<float>(kMax)
    };
    // Head decode scratch.
    std::vector<float> cls_f(kMax), reg_f(kMax), cls_logits(p.pipe.num_classes());

    bool got_eof = false;
    std::uint64_t last_seq = 0;
    while (true) {
        bool did_work = false;
        OwnMsg om;
        while (p.q_own[k]->try_pop(om)) {
            did_work = true;
            if (om.type == MsgType::Eof) { got_eof = true; continue; }
            if (om.type == MsgType::Marker) {
                last_seq = om.seq;
                p.committed_seq[k]->store(last_seq, std::memory_order_release);
                continue;
            }
            // Real event — run full pipeline locally.
            int x = om.x, y = om.y;

            // Stage 0
            p.pipe.stage_forward_locked(
                0, x, y, om.feat_in,
                p.patch_locks[0].get(), p.patch_locks[1].get(),
                qvg_scratch.data(), qh_scratch.data(), residual_scratch.data(),
                l0_out.data(), stage_out[0].data());
            if (!p.pipe.tdrop_and_pool_atomic(0, x, y, p.tdrop_atomic[0].get())) {
                const auto t1 = deploy::rdtsc_now();
                p.lat_drop_s0[k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - om.t_arr_tsc));
                last_seq = om.seq;
                p.committed_seq[k]->store(last_seq, std::memory_order_release);
                continue;
            }

            // Stage 1
            p.pipe.stage_forward_locked(
                1, x, y, stage_out[0].data(),
                p.patch_locks[2].get(), p.patch_locks[3].get(),
                qvg_scratch.data(), qh_scratch.data(), residual_scratch.data(),
                l0_out.data(), stage_out[1].data());
            if (!p.pipe.tdrop_and_pool_atomic(1, x, y, p.tdrop_atomic[1].get())) {
                const auto t1 = deploy::rdtsc_now();
                p.lat_drop_s1[k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - om.t_arr_tsc));
                last_seq = om.seq;
                p.committed_seq[k]->store(last_seq, std::memory_order_release);
                continue;
            }

            // Stage 2
            p.pipe.stage_forward_locked(
                2, x, y, stage_out[1].data(),
                p.patch_locks[4].get(), p.patch_locks[5].get(),
                qvg_scratch.data(), qh_scratch.data(), residual_scratch.data(),
                l0_out.data(), stage_out[2].data());
            if (!p.pipe.tdrop_and_pool_atomic(2, x, y, p.tdrop_atomic[2].get())) {
                const auto t1 = deploy::rdtsc_now();
                p.lat_drop_s2[k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - om.t_arr_tsc));
                last_seq = om.seq;
                p.committed_seq[k]->store(last_seq, std::memory_order_release);
                continue;
            }

            // Stage 3
            p.pipe.stage_forward_locked(
                3, x, y, stage_out[2].data(),
                p.patch_locks[6].get(), p.patch_locks[7].get(),
                qvg_scratch.data(), qh_scratch.data(), residual_scratch.data(),
                l0_out.data(), stage_out[3].data());
            // Head decode (writes one row of last_predictions_ under cell lock)
            p.pipe.head_decode_cell_locked(
                3, x, y, stage_out[3].data(),
                p.head_cell_locks.get(),
                cls_f, reg_f, cls_logits);

            const auto t1 = deploy::rdtsc_now();
            p.lat_emit_s3[k].push_back(deploy::TscClock::instance().tsc_to_ns(t1 - om.t_arr_tsc));
            last_seq = om.seq;
            p.committed_seq[k]->store(last_seq, std::memory_order_release);
        }

        // q_own empty: catch up to producer's pushed_seq.
        if (!did_work) {
            const std::uint64_t prod_max = p.pushed_seq_max.load(std::memory_order_acquire);
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

    // Allocate per-layer patch locks (8 layers, sized to their stage's grid)
    for (int l = 0; l < deploy::kNumLayers; ++l) {
        const int stage = l / deploy::kLayersPerStage;
        const int sz = pl.pipe.stage_lock_grid_size(stage);
        pl.layer_lock_size[l] = sz;
        pl.patch_locks[l].reset(new std::atomic_flag[sz]);
        for (int i = 0; i < sz; ++i) pl.patch_locks[l][i].clear();
    }
    // Atomic tdrop counters for stages 0..2 (sized to next stage's grid)
    for (int s = 0; s < 3; ++s) {
        const int sz = pl.pipe.stage_lock_grid_size(s + 1);
        pl.tdrop_size[s] = sz;
        pl.tdrop_atomic[s].reset(new std::atomic<std::uint8_t>[sz]);
        for (int i = 0; i < sz; ++i) pl.tdrop_atomic[s][i].store(0);
    }
    // Head cell locks
    pl.head_lock_size = pl.pipe.num_anchors_total();
    pl.head_cell_locks.reset(new std::atomic_flag[pl.head_lock_size]);
    for (int i = 0; i < pl.head_lock_size; ++i) pl.head_cell_locks[i].clear();

    const std::size_t qcap = std::size_t{1} << args.q_capacity_pow2;
    const int N = args.num_shards;
    pl.q_own.resize(N);
    pl.committed_seq.resize(N);
    for (int k = 0; k < N; ++k) {
        pl.q_own[k]         = std::make_unique<deploy::SpscRing<OwnMsg>>(qcap);
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

    std::fprintf(stderr,
        "[s6] events=%zu shards=%d warmup=%d  (full-pipeline shards, no stage thread)\n",
        events.size(), N, args.warmup);

    std::vector<std::thread> shards;
    for (int k = 0; k < N; ++k) {
        shards.emplace_back(thread_shard, std::ref(pl), k,
                            args.base_core + k, args.pin_cores);
    }
    if (args.pin_cores) pin_to_core(args.base_core + N);

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
            pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
            return;
        }
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
    auto wait_all_committed = [&](std::uint64_t seq) {
        // Wait until every shard's committed_seq >= seq.
        while (true) {
            bool ok = true;
            for (int k = 0; k < N; ++k) {
                if (pl.committed_seq[k]->load(std::memory_order_acquire) < seq) {
                    ok = false; break;
                }
            }
            if (ok) return;
            __builtin_ia32_pause();
        }
    };

    for (std::size_t i = 0; i < warmup; ++i) push_one(static_cast<std::uint64_t>(i), events[i]);
    push_marker(static_cast<std::uint64_t>(warmup));
    wait_all_committed(static_cast<std::uint64_t>(warmup));

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
                wait_all_committed(mseq);
                pl.pipe.publish_output_now();
                oracle->record(static_cast<std::uint64_t>(mi), pl.pipe.last_output());
            }
        }
    }
    push_marker(static_cast<std::uint64_t>(total));
    wait_all_committed(static_cast<std::uint64_t>(total));
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start).count();

    for (int k = 0; k < N; ++k) {
        OwnMsg eof{}; eof.type = MsgType::Eof;
        push_with_yield(pl.q_own[k].get(), eof);
    }
    for (auto& t : shards) t.join();

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

    std::fprintf(stdout, "\n=== Scheme 6: full-pipeline shards (%d shards, no stage thread) ===\n", N);
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
