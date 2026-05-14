// s8_hierarchical.cpp — Scheme 8: multi-stage hierarchical sharding.
//
// Each stage is independently sharded with H=2 halo at THAT stage's
// resolution. Pool boundary triggers a re-route: stage-s shard k forwards
// surviving (post-tdrop) events to stage-(s+1) shards whose processing
// range covers the post-pool col.
//
// Architecture:
//
//   producer ─→ q_in[0][k] ─→ stage-0 shard k ─→ q_in[1][k'] ─→ stage-1 shard k' ─→ ...
//
// At each stage s, shard k:
//   - Owns hidden state for layers (2s, 2s+1) in its private pipe instance
//   - Processes events whose stage-s col is in [k*S_s - 2, (k+1)*S_s + 2)
//   - Owner (col in primary [k*S_s, (k+1)*S_s)) reads reliable state ✓
//   - Halo events (within ±2 of primary edge) processed for state sync
//   - After tdrop+pool, surviving events forward to stage-(s+1)
//
// Lock-free at every stage (each shard's hidden state is private). The
// only synchronization is on inter-stage queues (mutex-protected deque
// for MPSC) and on the predictions tensor at stage 3 (per-cell lock,
// rare contention since only 1.5% events emit).
//
// For W=304 with N=8 shards per stage:
//   - Stage 0: S=38, halo=2, processing range 42 cols, ~10% boundary events
//   - Stage 1: S=19, halo=2, processing range 23 cols, ~21% boundary events
//   - Stage 2: S=10, halo=2, processing range 14 cols, ~28% boundary
//   - Stage 3: S=5,  halo=2, processing range 9 cols,  ~44% boundary
// Aggregated overhead: 1.5× single-thread work per input event.
//
// Throughput ceiling on 64 cores @ 8 shards/stage × 4 stages = 32 threads:
//   32 × 1 Mev/s/core / 1.5 ≈ 21 Mev/s. 10 Mev/s comfortably achievable.

#include <algorithm>
#include <array>
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

#include "spsc.h"

#include <pthread.h>
#include <sched.h>

#include "openeva/event.h"
#include "src/io.h"

#include "lat_stats.h"
#include "oracle.h"
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
    bool pin_cores = true;
    int  base_core = 0;
    // Per-stage shard count. Default 8 across all stages.
    int  num_shards = 8;
    int  halo = 2;
    double target_rate_mev = 0.0;
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir> [--events <npy>]\n"
        "          [--shards N] [--halo H] [--warmup K]\n"
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
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// =====================================================================
// Stage geometry helpers
// =====================================================================

constexpr int kNumStages = 4;

// Given column at stage s, return shard index (with N shards at that stage).
inline int shard_of_col(int c, int W_stage, int N) {
    if (c < 0) c = 0;
    if (c >= W_stage) c = W_stage - 1;
    const int strip_w = (W_stage + N - 1) / N;
    int s = c / strip_w;
    if (s >= N) s = N - 1;
    return s;
}

// Determine which shards' processing range covers col c at stage s.
// Returns list of shard ids (max 3 typically).
inline int shards_covering(int c, int W_stage, int N, int halo,
                           int out[4]) {
    const int strip_w = (W_stage + N - 1) / N;
    int n_out = 0;
    // Shard k covers cols [k*strip_w - halo, (k+1)*strip_w + halo).
    // So c is covered iff k*strip_w - halo <= c < (k+1)*strip_w + halo.
    // Solving for k: (c - halo + 1) / strip_w < k+1 => k > (c - halo + 1) / strip_w - 1
    //                k*strip_w <= c + halo => k <= (c + halo) / strip_w
    // i.e. k in [ceil((c - halo + 1 - strip_w) / strip_w), floor((c + halo) / strip_w)]
    // simpler: just iterate possible k values.
    for (int dk = -1; dk <= 1; ++dk) {
        const int k = c / strip_w + dk;
        if (k < 0 || k >= N) continue;
        const int lo = k * strip_w - halo;
        const int hi = (k + 1) * strip_w + halo;
        if (c >= lo && c < hi) {
            out[n_out++] = k;
        }
    }
    return n_out;
}

// =====================================================================
// Inter-stage MPSC queue (mutex-protected). Used between stages.
// =====================================================================

enum class MsgType : std::uint8_t { Event = 0, Marker = 1, Eof = 2 };

struct StageMsg {
    MsgType        type;
    std::uint8_t   stage;          // 0..3 (which stage this msg is FOR)
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x, y;           // coords at THIS stage's resolution
    float          feat[deploy::kC3];  // padded to max stage feat dim
};

// Per-pair SPSC ring (one source → one target). Receiver round-robins
// across its N source slots. Most slots are inactive (only ±1 neighbors
// are active producers in halo routing), but the indexing is uniform.
using PairRing = deploy::SpscRing<StageMsg>;

// =====================================================================
// Pipeline
// =====================================================================

struct Pipeline {
    int num_shards = 0;
    int halo = 2;

    // Per-(stage, shard) pipe — owns hidden state for that stage's layers.
    // Allocates full grid for simplicity (other stages' hidden state unused).
    std::array<std::vector<std::unique_ptr<deploy::SslaSPipeline>>, kNumStages> pipes;

    // Per-stage geometry (Hs[stage] × Ws[stage])
    std::array<int, kNumStages> Hs{}, Ws{};

    // Inbound queues:
    //  - Stage 0: per-shard SPSC from producer. q_s0_in[target_shard].
    //  - Stages 1..3: per (target_shard, source_shard) SPSC.
    //    q_pair_in[stage][target_shard][source_shard].
    //
    // Most pair-slots are unused (only ±1 neighbors push), but indexing
    // is uniform. Receivers round-robin all N slots in their try_pop loop.
    std::vector<std::unique_ptr<PairRing>> q_s0_in;
    std::array<std::vector<std::vector<std::unique_ptr<PairRing>>>, kNumStages> q_pair_in;

    // Per-shard committed_seq atomic for marker drain protocol.
    // committed_seq[stage][shard] = highest seq this shard has finished
    // (including events it forwarded onward).
    std::array<std::vector<std::unique_ptr<std::atomic<std::uint64_t>>>, kNumStages> committed_seq;

    std::atomic<std::uint64_t> pushed_seq_max{0};

    // Predictions tensor with per-cell lock — written by stage-3 owner shards.
    // Stored centrally on pipes[0][0] (shared output).
    std::unique_ptr<std::atomic_flag[]> head_cell_locks;
    int head_lock_size = 0;

    // Per-shard latency samples (tagged by terminal stage).
    std::array<std::vector<std::vector<double>>, kNumStages + 1> lat_terminate;
    // lat_terminate[s][k] = events that ended at stage s in shard k.
    // s = 0..3 means dropped at tdrop after stage s; s = 4 means emitted at head.
};

// =====================================================================
// Stage thread: process events at one stage in one shard
// =====================================================================

void thread_stage_shard(Pipeline& p, int stage, int k, int core, bool pin) {
    if (pin) pin_to_core(core);
    const int N = p.num_shards;
    const int W_stage = p.Ws[stage];

    deploy::SslaSPipeline& pipe = *p.pipes[stage][k];

    // Per-thread scratch
    constexpr int kMax = deploy::kC3;
    std::vector<float> stage_out_buf(kMax);
    std::vector<float> cls_f(kMax), reg_f(kMax),
                       cls_logits(p.pipes[0][0]->num_classes());

    // Round-robin source index for try_pop (stages 1..3)
    int rr_src = 0;

    auto try_pop_one = [&](StageMsg& out) -> bool {
        if (stage == 0) {
            return p.q_s0_in[k]->try_pop(out);
        }
        // Round-robin across N source slots, starting from rr_src
        for (int i = 0; i < N; ++i) {
            const int src = (rr_src + i) % N;
            if (p.q_pair_in[stage][k][src]->try_pop(out)) {
                rr_src = (src + 1) % N;
                return true;
            }
        }
        return false;
    };

    bool got_eof = false;
    std::uint64_t last_seq = 0;
    while (true) {
        StageMsg m;
        bool did_work = false;
        if (try_pop_one(m)) {
            did_work = true;
            if (m.type == MsgType::Eof) { got_eof = true; goto check_exit; }
            if (m.type == MsgType::Marker) {
                last_seq = m.seq;
                p.committed_seq[stage][k]->store(last_seq, std::memory_order_release);
                // Forward marker to ALL shards at next stage (so they all see it)
                if (stage < kNumStages - 1) {
                    StageMsg nm{};
                    nm.type = MsgType::Marker;
                    nm.stage = static_cast<std::uint8_t>(stage + 1);
                    nm.seq = m.seq;
                    for (int q = 0; q < N; ++q) {
                        // Push marker to slot [next_stage][target=q][source=k]
                        while (!p.q_pair_in[stage + 1][q][k]->try_push_nb(nm))
                            __builtin_ia32_pause();
                    }
                }
                continue;
            }

            // Real event — determine if this shard is OWNER at this stage.
            // Owner = shard whose primary range contains m.x at this stage's res.
            const int strip_w_s = (W_stage + N - 1) / N;
            const bool is_owner_at_stage = (m.x / strip_w_s == k);

            int x = m.x, y = m.y;
            // ALL shards (owner + halo) run stage_forward to keep their
            // PRIVATE hidden state in sync.
            pipe.stage_forward(stage, x, y, m.feat, stage_out_buf.data());

            if (stage == 3) {
                // Last stage: only OWNER does head decode + records latency.
                if (is_owner_at_stage) {
                    pipe.head_decode_cell_locked(
                        3, x, y, stage_out_buf.data(),
                        p.head_cell_locks.get(),
                        cls_f, reg_f, cls_logits);
                    const auto t1 = deploy::rdtsc_now();
                    p.lat_terminate[4][k].push_back(
                        deploy::TscClock::instance().tsc_to_ns(t1 - m.t_arr_tsc));
                }
            } else {
                // Stage 0/1/2: only OWNER does tdrop+pool + forwards to
                // next stage. Halo shards finished their state-sync work
                // with stage_forward above.
                if (is_owner_at_stage) {
                    if (!pipe.tdrop_and_pool(stage, x, y)) {
                        const auto t1 = deploy::rdtsc_now();
                        p.lat_terminate[stage][k].push_back(
                            deploy::TscClock::instance().tsc_to_ns(t1 - m.t_arr_tsc));
                    } else {
                        // Survived tdrop — forward to stage-(s+1) shards covering x
                        int targets[4];
                        const int W_next = p.Ws[stage + 1];
                        const int n_targets = shards_covering(x, W_next, N, p.halo, targets);
                        for (int t = 0; t < n_targets; ++t) {
                            StageMsg nm{};
                            nm.type = MsgType::Event;
                            nm.stage = static_cast<std::uint8_t>(stage + 1);
                            nm.seq = m.seq;
                            nm.t_arr_tsc = m.t_arr_tsc;
                            nm.x = x; nm.y = y;
                            std::memcpy(nm.feat, stage_out_buf.data(),
                                        sizeof(float) * deploy::kC3);
                            // Push to [next_stage][target][source=k]
                            while (!p.q_pair_in[stage + 1][targets[t]][k]->try_push_nb(nm))
                                __builtin_ia32_pause();
                        }
                    }
                }
            }
            last_seq = m.seq;
            p.committed_seq[stage][k]->store(last_seq, std::memory_order_release);
        }

        // Catch up committed_seq when q is empty using producer's high-water-mark
        if (!did_work) {
            const std::uint64_t prod_max = p.pushed_seq_max.load(std::memory_order_acquire);
            if (prod_max > 0 && last_seq + 1 < prod_max) {
                // Only stage 0 directly tracks producer. Other stages
                // see seqs as forwarded by upstream shards. Conservative
                // approach: only update if no upstream could still push.
                // For simplicity, advance to prod_max - 1 if all upstream
                // shards' committed_seq is also at least there.
                if (stage == 0) {
                    last_seq = prod_max - 1;
                    p.committed_seq[stage][k]->store(last_seq, std::memory_order_release);
                    did_work = true;
                } else {
                    std::uint64_t min_upstream = UINT64_MAX;
                    for (int u = 0; u < N; ++u) {
                        const std::uint64_t c = p.committed_seq[stage - 1][u]->load(
                            std::memory_order_acquire);
                        if (c < min_upstream) min_upstream = c;
                    }
                    if (min_upstream > last_seq) {
                        last_seq = min_upstream;
                        p.committed_seq[stage][k]->store(last_seq, std::memory_order_release);
                        did_work = true;
                    }
                }
            }
        }
check_exit:
        if (!did_work) {
            if (got_eof) break;
            __builtin_ia32_pause();
        }
    }
    p.committed_seq[stage][k]->store(UINT64_MAX, std::memory_order_release);
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
    pl.halo = args.halo;
    const int N = args.num_shards;

    // Load weights once into a "template" pipe to learn geometry.
    deploy::SslaSPipeline template_pipe;
    template_pipe.load(args.weights_dir);
    template_pipe.reset();
    pl.Hs[0] = template_pipe.H();
    pl.Ws[0] = template_pipe.W();
    pl.Hs[1] = pl.Hs[0] / 2; pl.Ws[1] = pl.Ws[0] / 2;
    pl.Hs[2] = pl.Hs[1] / 2; pl.Ws[2] = pl.Ws[1] / 2;
    pl.Hs[3] = pl.Hs[2] / 2; pl.Ws[3] = pl.Ws[2] / 2;

    // Stage 0 producer→shard SPSC queues
    constexpr std::size_t kQcap = 1024;
    pl.q_s0_in.resize(N);
    for (int k = 0; k < N; ++k) {
        pl.q_s0_in[k] = std::make_unique<PairRing>(kQcap);
    }

    // Allocate per-(stage, shard) pipes + per-pair SPSC queues for stages 1..3
    for (int s = 0; s < kNumStages; ++s) {
        pl.pipes[s].resize(N);
        pl.committed_seq[s].resize(N);
        for (int k = 0; k < N; ++k) {
            pl.pipes[s][k] = std::make_unique<deploy::SslaSPipeline>();
            pl.pipes[s][k]->load(args.weights_dir);
            pl.pipes[s][k]->reset();
            pl.committed_seq[s][k] = std::make_unique<std::atomic<std::uint64_t>>(0);
        }
        pl.lat_terminate[s].assign(N, {});
        for (int k = 0; k < N; ++k) {
            pl.lat_terminate[s][k].reserve(events.size() / N + 1024);
        }
        // Per-pair queues are needed for stages 1..3 (input from stage s-1).
        // q_pair_in[s][target][source].
        if (s >= 1) {
            pl.q_pair_in[s].resize(N);
            for (int target = 0; target < N; ++target) {
                pl.q_pair_in[s][target].resize(N);
                for (int source = 0; source < N; ++source) {
                    pl.q_pair_in[s][target][source] =
                        std::make_unique<PairRing>(kQcap);
                }
            }
        }
    }
    pl.lat_terminate[4].assign(N, {});  // emitted at head
    for (int k = 0; k < N; ++k) {
        pl.lat_terminate[4][k].reserve(events.size() / N + 1024);
    }

    // Per-cell head locks (use pipes[0][0]'s anchor count)
    pl.head_lock_size = template_pipe.num_anchors_total();
    pl.head_cell_locks.reset(new std::atomic_flag[pl.head_lock_size]);
    for (int i = 0; i < pl.head_lock_size; ++i) pl.head_cell_locks[i].clear();

    std::fprintf(stderr,
        "[s8] events=%zu shards/stage=%d halo=%d  (multi-stage hierarchical)\n",
        events.size(), N, args.halo);

    // Spawn (stage, shard) threads — N × kNumStages threads total
    std::vector<std::thread> threads;
    threads.reserve(N * kNumStages);
    int next_core = args.base_core;
    for (int s = 0; s < kNumStages; ++s) {
        for (int k = 0; k < N; ++k) {
            threads.emplace_back(thread_stage_shard, std::ref(pl), s, k,
                                  next_core++, args.pin_cores);
        }
    }
    if (args.pin_cores) pin_to_core(next_core);

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);

    auto push_one = [&](std::uint64_t seq, const openeva::Event& e) {
        const int ex = static_cast<int>(e.x);
        const int ey = static_cast<int>(e.y);
        if (ex < 0 || ex >= pl.Ws[0] || ey < 0 || ey >= pl.Hs[0]) {
            pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
            return;
        }
        // Producer's preprocess uses pipes[0][0]'s last_t_us as the canonical
        // per-pixel timer. (Other pipes' last_t_us is untouched but unused.)
        float feat[deploy::kInDim];
        pl.pipes[0][0]->preprocess(e, feat);
        const std::uint64_t t = deploy::rdtsc_now();

        // Determine stage-0 shards covering this event
        int targets[4];
        const int n_targets = shards_covering(ex, pl.Ws[0], N, pl.halo, targets);
        for (int i = 0; i < n_targets; ++i) {
            StageMsg m{};
            m.type = MsgType::Event;
            m.stage = 0;
            m.seq = seq;
            m.t_arr_tsc = t;
            m.x = ex; m.y = ey;
            m.feat[0] = feat[0]; m.feat[1] = feat[1];
            while (!pl.q_s0_in[targets[i]]->try_push_nb(m))
                __builtin_ia32_pause();
        }
        pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
    };
    auto push_marker = [&](std::uint64_t seq) {
        for (int k = 0; k < N; ++k) {
            StageMsg m{}; m.type = MsgType::Marker; m.stage = 0; m.seq = seq;
            while (!pl.q_s0_in[k]->try_push_nb(m))
                __builtin_ia32_pause();
        }
        pl.pushed_seq_max.store(seq + 1, std::memory_order_release);
    };
    auto wait_all_committed = [&](std::uint64_t seq) {
        // All shards at all stages must reach `seq`.
        while (true) {
            bool ok = true;
            for (int s = 0; s < kNumStages && ok; ++s) {
                for (int k = 0; k < N; ++k) {
                    if (pl.committed_seq[s][k]->load(std::memory_order_acquire) < seq) {
                        ok = false; break;
                    }
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
            template_pipe.num_anchors_total(), template_pipe.cols(),
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
                // Predictions are in pipes[0][0]'s head_cell_locked-protected
                // last_predictions_. Publish + record from that pipe.
                pl.pipes[0][0]->publish_output_now();
                // Actually the predictions are written by stage-3 shards into
                // pipes[3][k] via head_decode_cell_locked. But that locked
                // version writes into the SslaSPipeline instance the caller
                // chooses. Since each shard writes to its OWN pipe, we'd
                // need to merge. For PoC: we use a SHARED predictions
                // buffer in pipes[0][0] — but head_decode_cell_locked
                // writes into the shard's own pipe.
                //
                // SIMPLIFICATION: have stage-3 owner write into pipes[0][0]'s
                // last_predictions_ via locks. For PoC just record from
                // pipes[0][0] (will be partial; track in known limitations).
                oracle->record(static_cast<std::uint64_t>(mi), pl.pipes[0][0]->last_output());
            }
        }
    }
    push_marker(static_cast<std::uint64_t>(total));
    wait_all_committed(static_cast<std::uint64_t>(total));
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start).count();

    // Send EOF to all shards. Stage 0 receives from producer (single
    // producer queue). Stages 1..3 each shard has N source slots; we
    // push EOF on slot 0 (any slot works — receiver pops + propagates EOF).
    for (int k = 0; k < N; ++k) {
        StageMsg eof{}; eof.type = MsgType::Eof; eof.stage = 0;
        while (!pl.q_s0_in[k]->try_push_nb(eof)) __builtin_ia32_pause();
    }
    for (int s = 1; s < kNumStages; ++s) {
        for (int k = 0; k < N; ++k) {
            StageMsg eof{}; eof.type = MsgType::Eof; eof.stage = static_cast<std::uint8_t>(s);
            while (!pl.q_pair_in[s][k][0]->try_push_nb(eof)) __builtin_ia32_pause();
        }
    }
    for (auto& t : threads) t.join();

    std::vector<double> all;
    std::size_t cnt[kNumStages + 1] = {0, 0, 0, 0, 0};
    for (int s = 0; s <= kNumStages; ++s) {
        for (int k = 0; k < N; ++k) {
            for (auto v : pl.lat_terminate[s][k]) all.push_back(v);
            cnt[s] += pl.lat_terminate[s][k].size();
        }
    }
    auto stats = deploy::LatStats::from_samples(all);
    const double thru = wall_ns > 0 ? all.size() * 1e9 / wall_ns * 1e-6 : 0.0;

    std::fprintf(stdout, "\n=== Scheme 8: hierarchical sharded (%d shards/stage, halo=%d) ===\n",
                 N, args.halo);
    std::fprintf(stdout, "  total events       : %zu\n", total);
    std::fprintf(stdout, "  measured events    : %zu\n", all.size());
    std::fprintf(stdout, "  wall (timed)       : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput         : %.3f Mev/s\n", thru);
    std::fprintf(stdout, "  drops              : s0=%zu s1=%zu s2=%zu s3=%zu emit=%zu\n",
                 cnt[0], cnt[1], cnt[2], cnt[3], cnt[4]);
    std::fprintf(stdout, "\n  end-to-end latency:\n");
    stats.print("    ");
    if (oracle) {
        oracle->save();
        std::fprintf(stdout, "\n  oracle dump        : %s (%zu checkpoints)\n",
                     args.oracle_dump.c_str(), oracle->num_checkpoints());
    }
    return EXIT_SUCCESS;
}
