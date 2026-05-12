// s3_sharded.cpp — Scheme 3: stage-0 spatial-sharded with halo, stages 1-3 single-thread.
//
// Architecture:
//
//   producer  ──┬─→  S0 shard 0  ──┐
//               ├─→  S0 shard 1  ──┤    sum_thread       stage1+2+3 thread
//               ├─→  S0 shard 2  ──┼──→ (per-event   ──→ (full forward
//               └─→  S0 shard k  ──┘    partial sum,       through stages
//                                        residual + LN)    1, 2, 3 + head)
//
// Patch ownership for stage 0 (both layer 0 and layer 1):
//   shard k owns patches at columns [k*W/N, (k+1)*W/N)
//   For an event at pixel (x, y), patches affected are at columns
//   x-1, x, x+1. Producer routes the event to all shards that own at
//   least one of those columns (1, 2, or 3 shards). Each shard runs
//   shard_layer_forward() for its owned patches and pushes a partial
//   feat_out (zero-initialised + accumulated) to the sum thread.
//
// Sum thread:
//   - knows from producer how many shards will contribute per seq
//     (event_meta queue carries: seq, ev_x, ev_y, expected_count, feat_in)
//   - pops partial contributions from each shard's outbound (FIFO per shard)
//   - sums by seq; once expected_count partials gathered, calls
//     layer_finalize() to apply residual + LN, then forwards to layer 1
//     (which is itself sharded the same way — 2nd round of shard +
//     accumulate; for SSLA-S layer 1 keeps in_dim==out_dim so it's a
//     direct accumulator, no input_proj)
//
// For simplicity v1: layer 0 + layer 1 are BOTH sharded (events flow
// through layer 0 shards → sum → layer 1 shards → sum → stages 1-3).
// Within stage 0, the same N shards run BOTH layer 0 and layer 1 work
// (each shard owns the same pixel column for both layers).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <list>
#include <unordered_map>
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

// =====================================================================
// Message types
// =====================================================================

enum class MsgType : std::uint8_t { Event = 0, Marker = 1, Eof = 2 };

// Producer → S0-shard. Carries the input feature (for layer 0) at first
// dispatch; for layer 1 work, the sum thread refills feat_in and re-routes.
template <int FEAT_DIM>
struct ShardMsg {
    MsgType        type;
    std::uint8_t   layer;          // 0 or 1 (which layer this work is for)
    std::uint16_t  owned_mask;     // which patches this shard owns for this event
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x;              // event x at this layer's resolution (==ev_x0 for stage 0)
    int            y;
    float          feat_in[FEAT_DIM];
};

// Producer → sum_thread (per-event meta). Carries the input feature kept
// for residual application in layer_finalize, and how many shards will
// contribute. Same struct used twice per event (once for L0 finalize,
// once for L1 finalize) with feat_in carrying the L0/L1 input.
template <int FEAT_DIM>
struct EventMeta {
    MsgType        type;
    std::uint8_t   layer;          // 0 or 1
    std::uint8_t   expected_count; // how many shards will contribute for this seq+layer
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x;              // for forwarding to next layer / next stage
    int            y;
    float          feat_in[FEAT_DIM];
};

// Shard → sum_thread (per-shard partial feat_out for one (seq, layer)).
template <int FEAT_DIM>
struct PartialMsg {
    MsgType        type;
    std::uint8_t   layer;
    std::uint64_t  seq;
    float          partial[FEAT_DIM];
};

using SM_L0 = ShardMsg<deploy::kInDim>;
using SM_L1 = ShardMsg<deploy::kC0>;
using EM    = EventMeta<deploy::kC0>;        // both layers' residual is at C0 dim or input dim
using EM_L0 = EventMeta<deploy::kInDim>;
using EM_L1 = EventMeta<deploy::kC0>;
using PM    = PartialMsg<deploy::kC0>;

// Sum → stage1+2+3. Carries layer-1 finalised feat_out + ev coords + seq.
struct StageInMsg {
    MsgType        type;
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x, y;
    float          feat[deploy::kC0];
};

// =====================================================================
// Pipeline state
// =====================================================================

struct Pipeline {
    deploy::SslaSPipeline pipe;
    int num_shards = 0;

    // q_in_shard[k]: producer → shard k (layer 0 work)
    std::vector<std::unique_ptr<deploy::SpscRing<SM_L0>>> q_l0_in;
    // q_l1_in_shard[k]: sum thread → shard k (layer 1 work)
    std::vector<std::unique_ptr<deploy::SpscRing<SM_L1>>> q_l1_in;
    // q_partial_l0[k]: shard k → sum (layer 0 partial)
    std::vector<std::unique_ptr<deploy::SpscRing<PM>>> q_partial_l0;
    // q_partial_l1[k]: shard k → sum (layer 1 partial)
    std::vector<std::unique_ptr<deploy::SpscRing<PM>>> q_partial_l1;

    // Producer → sum: per-event meta for layer 0 finalize (residual = feat_in_layer0 = (dt,p))
    std::unique_ptr<deploy::SpscRing<EM_L0>> q_meta_l0;
    // Sum → sum: layer 1 meta (residual = feat_in_layer1 = layer 0 finalised feat).
    // Actually the same thread emits this — we use a queue for simplicity.
    std::unique_ptr<deploy::SpscRing<EM_L1>> q_meta_l1;

    // Sum → stage1+2+3 thread.
    std::unique_ptr<deploy::SpscRing<StageInMsg>> q_to_stages;

    // Per-event measurement
    std::vector<double> lat_drop_s0, lat_drop_s1, lat_drop_s2, lat_emit_s3;

    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done_stages{0};
    alignas(deploy::kCacheLine) std::atomic<bool> done{false};
};

// Compute which shards own at least one of the patches at columns
// {x-1, x, x+1} on a (W) grid given num_shards. Returns mask of shard ids.
inline std::uint16_t affected_shard_mask(int x, int W, int num_shards) {
    std::uint16_t mask = 0;
    auto shard_of_col = [&](int c) {
        if (c < 0) c = 0;
        if (c >= W) c = W - 1;
        const int strip_w = (W + num_shards - 1) / num_shards;
        int s = c / strip_w;
        if (s >= num_shards) s = num_shards - 1;
        return s;
    };
    mask |= (std::uint16_t{1} << shard_of_col(x - 1));
    mask |= (std::uint16_t{1} << shard_of_col(x));
    mask |= (std::uint16_t{1} << shard_of_col(x + 1));
    return mask;
}

// Compute the patch-ownership mask (9 bits) for shard k at event (x, y)
// on a W-wide grid. Patch p is at column dx_offset = (1 - dx). Need to
// figure out the column of each delta patch and check ownership.
inline std::uint16_t shard_patch_mask(int k, int x, int W, int num_shards) {
    const int strip_w = (W + num_shards - 1) / num_shards;
    auto shard_of_col = [&](int c) {
        if (c < 0) c = 0;
        if (c >= W) c = W - 1;
        int s = c / strip_w;
        if (s >= num_shards) s = num_shards - 1;
        return s;
    };
    std::uint16_t m = 0;
    for (int p = 0; p < 9; ++p) {
        // delta = (1 - dy) * 3 + (1 - dx)  →  dx = 1 - (p % 3),  dy = 1 - (p / 3)
        const int dx = 1 - (p % 3);
        const int col = x + dx;
        if (col < 0 || col >= W) continue;
        if (shard_of_col(col) == k) m |= (std::uint16_t{1} << p);
    }
    return m;
}

// =====================================================================
// Threads
// =====================================================================

// Each S0 shard handles BOTH layer 0 and layer 1 work for its owned
// patches. Layer 0 messages come from producer (q_l0_in[k]); layer 1
// messages come from sum thread (q_l1_in[k]) once layer 0 is finalised.
void thread_shard(Pipeline& p, int k, int core, bool pin) {
    if (pin) pin_to_core(core);
    SM_L0 in0;
    SM_L1 in1;
    PM    out{};
    // Thread-local scratch — sized to max layer-0/1 output dim (kC0 = 12).
    std::vector<float> qvg_scratch(3 * deploy::kC0);
    std::vector<float> qh_scratch(deploy::kC0);
    while (true) {
        bool did_work = false;

        // L0 work (priority — producer is the rate-controlling stage)
        if (p.q_l0_in[k]->try_pop(in0)) {
            did_work = true;
            if (in0.type == MsgType::Eof) break;
            if (in0.type == MsgType::Marker) {
                // Markers are absorbed by sum thread; shards don't need to forward.
                continue;
            }
            // Produce layer 0 partial
            std::memset(out.partial, 0, sizeof(out.partial));
            p.pipe.shard_layer_forward(/*layer*/ 0, in0.x, in0.y,
                                       in0.feat_in, in0.owned_mask,
                                       qvg_scratch.data(), qh_scratch.data(),
                                       out.partial);
            out.type  = MsgType::Event;
            out.layer = 0;
            out.seq   = in0.seq;
            p.q_partial_l0[k]->push(out);
        }

        // L1 work
        if (p.q_l1_in[k]->try_pop(in1)) {
            did_work = true;
            if (in1.type == MsgType::Eof) {
                continue;
            }
            if (in1.type == MsgType::Marker) continue;
            std::memset(out.partial, 0, sizeof(out.partial));
            p.pipe.shard_layer_forward(/*layer*/ 1, in1.x, in1.y,
                                       in1.feat_in, in1.owned_mask,
                                       qvg_scratch.data(), qh_scratch.data(),
                                       out.partial);
            out.type  = MsgType::Event;
            out.layer = 1;
            out.seq   = in1.seq;
            p.q_partial_l1[k]->push(out);
        }

        if (!did_work) {
            if (p.done.load(std::memory_order_acquire)) break;
            __builtin_ia32_pause();
        }
    }
}

// Sum thread: aggregates per-shard partials by seq + layer, finalises,
// and dispatches layer-1 work back to shards. Once layer 1 is finalised,
// forwards (seq, x, y, feat_out_l1) to stages 1-3 thread.
void thread_sum(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    const int N = p.num_shards;
    // Sum-thread-local scratch for layer_finalize residual
    std::vector<float> residual_scratch(deploy::kC0);

    // Pending layer-0 events keyed by seq (in seq order from producer).
    // We use a vector of (seq, accumulator_state) and process them in
    // arrival order from q_meta_l0 — that gives FIFO seq processing.
    struct Pending {
        bool is_marker = false;
        std::uint64_t seq = 0;
        std::uint64_t t_arr_tsc = 0;
        int x = 0, y = 0;
        std::uint8_t expected = 0;
        std::uint8_t received = 0;
        float feat_in_residual[deploy::kC0]{};
        float partial_sum[deploy::kC0]{};
    };

    // pending_l0/l1 hold BOTH events and markers in producer-seq order. A
    // marker at the front of pending_l0 cascades to pending_l1 (via the
    // dispatch loop below) only after all preceding L0 events have been
    // finalised and forwarded to L1. Same for pending_l1 → q_to_stages.
    // This is what guarantees oracle snapshot causality.
    // pending_l0 / pending_l1 are FIFO-ordered lists; events finalize from
    // the front. Lookup-by-seq via separate hash maps (iterators into list
    // are stable across push_back / pop_front of OTHER elements).
    std::list<Pending> pending_l0;
    std::list<Pending> pending_l1;
    std::unordered_map<std::uint64_t, std::list<Pending>::iterator> idx_l0;
    std::unordered_map<std::uint64_t, std::list<Pending>::iterator> idx_l1;

    // Out-of-band dispatch buffers — populated by the cascade (pure local
    // work); drained later in the same outer loop with try_push so we
    // never block-spin while holding pending state. This is what breaks
    // the producer→shard→sum→producer livelock cycle.
    struct L1Dispatch {
        std::uint64_t seq;
        std::uint64_t t_arr_tsc;
        int x, y;
        std::uint16_t shard_mask;
        float feat_in[deploy::kC0];
        // Per-shard "already pushed" bits — once a shard's SM_L1 has
        // been delivered to its q_l1_in, set the bit. We retry only the
        // pending bits in subsequent dispatcher passes.
        std::uint16_t delivered_mask;
    };
    std::deque<L1Dispatch> dispatch_l1_pending;
    std::deque<StageInMsg> stage_dispatch_pending;

    // Partials may arrive at sum BEFORE their meta — race between
    // producer (pushes meta then work) and shard (processes work fast)
    // means partial can land in q_partial_l0 before sum has drained the
    // matching meta from q_meta_l0. Defer unmatched partials and retry
    // each outer iter.
    std::deque<PM> deferred_partials_l0;
    std::deque<PM> deferred_partials_l1;

    auto try_match_partial = [&](std::unordered_map<std::uint64_t,
                                     std::list<Pending>::iterator>& idx,
                                  const PM& pm, int OUT_DIM_) -> bool {
        auto it = idx.find(pm.seq);
        if (it == idx.end()) return false;
        Pending& pe = *(it->second);
        for (int i = 0; i < OUT_DIM_; ++i) pe.partial_sum[i] += pm.partial[i];
        ++pe.received;
        return true;
    };
    auto process_partial_l0 = [&](const PM& pm) {
        if (!try_match_partial(idx_l0, pm, deploy::kC0)) {
            deferred_partials_l0.push_back(pm);
        }
    };
    auto process_partial_l1 = [&](const PM& pm) {
        if (!try_match_partial(idx_l1, pm, deploy::kC0)) {
            deferred_partials_l1.push_back(pm);
        }
    };

    while (true) {
        bool did_work = false;

        // Drain meta queues (producer → sum)
        EM_L0 m0;
        while (p.q_meta_l0->try_pop(m0)) {
            did_work = true;
            if (m0.type == MsgType::Eof) {
                // Forward EOF to shards' L1 queues to wind down
                for (int k = 0; k < N; ++k) {
                    SM_L1 e{}; e.type = MsgType::Eof;
                    p.q_l1_in[k]->push(e);
                }
                StageInMsg em{}; em.type = MsgType::Eof;
                p.q_to_stages->push(em);
                return;
            }
            Pending pe{};
            if (m0.type == MsgType::Marker) {
                pe.is_marker = true;
                pe.seq = m0.seq;
            } else {
                pe.seq        = m0.seq;
                pe.t_arr_tsc  = m0.t_arr_tsc;
                pe.x          = m0.x;
                pe.y          = m0.y;
                pe.expected   = m0.expected_count;
                pe.feat_in_residual[0] = m0.feat_in[0];
                pe.feat_in_residual[1] = m0.feat_in[1];
            }
            pending_l0.push_back(pe);
            if (!pe.is_marker) {
                idx_l0[pe.seq] = std::prev(pending_l0.end());
            }
        }
        // q_meta_l1 was redundant — sum is both producer and consumer of
        // L1 metas. We instead push directly into pending_l1 from the
        // pending_l0 cascade below, avoiding a self-queue that could
        // deadlock if it filled mid-cascade.

        // Retry deferred partials now that more metas may be in pending.
        // O(1) lookup via idx maps, so this is O(D) per iter.
        for (std::size_t i = deferred_partials_l0.size(); i > 0; --i) {
            PM pm = deferred_partials_l0.front();
            deferred_partials_l0.pop_front();
            if (try_match_partial(idx_l0, pm, deploy::kC0)) did_work = true;
            else deferred_partials_l0.push_back(pm);
        }
        for (std::size_t i = deferred_partials_l1.size(); i > 0; --i) {
            PM pm = deferred_partials_l1.front();
            deferred_partials_l1.pop_front();
            if (try_match_partial(idx_l1, pm, deploy::kC0)) did_work = true;
            else deferred_partials_l1.push_back(pm);
        }

        // Drain partials
        PM pm;
        for (int k = 0; k < N; ++k) {
            while (p.q_partial_l0[k]->try_pop(pm)) {
                did_work = true;
                process_partial_l0(pm);
            }
            while (p.q_partial_l1[k]->try_pop(pm)) {
                did_work = true;
                process_partial_l1(pm);
            }
        }

        // Drain pending_l0 from front in seq order. A marker at the front
        // cascades to pending_l1 (via q_meta_l1) — preserving causality.
        // Cascade pending_l0 — PURE LOCAL WORK ONLY. No spinning push.
        // Newly-finalized events get appended to dispatch_l1_pending (for
        // SM_L1 push to shards) and pending_l1 (for tracking).
        while (!pending_l0.empty()) {
            Pending& pe = pending_l0.front();
            if (pe.is_marker) {
                Pending pl1m{};
                pl1m.is_marker = true;
                pl1m.seq = pe.seq;
                pending_l1.push_back(pl1m);
                pending_l0.pop_front();
                did_work = true;
                continue;
            }
            if (pe.received < pe.expected) break;  // wait for partials
            // Finalize L0
            p.pipe.layer_finalize(/*layer*/ 0, pe.feat_in_residual,
                                  residual_scratch.data(), pe.partial_sum);
            const int x = pe.x, y = pe.y;
            const int W = p.pipe.W();
            std::uint16_t shard_mask = affected_shard_mask(x, W, N);
            std::uint8_t expected = 0;
            for (int k = 0; k < N; ++k) if ((shard_mask >> k) & 1) ++expected;
            Pending pl1{};
            pl1.seq        = pe.seq;
            pl1.t_arr_tsc  = pe.t_arr_tsc;
            pl1.x          = x;
            pl1.y          = y;
            pl1.expected   = expected;
            std::memcpy(pl1.feat_in_residual, pe.partial_sum, sizeof(pl1.feat_in_residual));
            pending_l1.push_back(pl1);
            idx_l1[pl1.seq] = std::prev(pending_l1.end());

            L1Dispatch d{};
            d.seq = pe.seq;
            d.t_arr_tsc = pe.t_arr_tsc;
            d.x = x; d.y = y;
            d.shard_mask = shard_mask;
            d.delivered_mask = 0;
            std::memcpy(d.feat_in, pe.partial_sum, sizeof(d.feat_in));
            dispatch_l1_pending.push_back(d);

            idx_l0.erase(pe.seq);
            pending_l0.pop_front();
            did_work = true;
        }

        // Drain dispatch_l1_pending — try_push to q_l1_in[k]; on full,
        // mark this shard as not-yet-delivered and bail to outer loop so
        // we can drain partials (which lets shard pop from q_l1_in).
        while (!dispatch_l1_pending.empty()) {
            L1Dispatch& d = dispatch_l1_pending.front();
            bool any_blocked = false;
            for (int k = 0; k < N; ++k) {
                if (((d.shard_mask >> k) & 1) == 0) continue;
                if ((d.delivered_mask >> k) & 1) continue;
                SM_L1 work{};
                work.type = MsgType::Event;
                work.layer = 1;
                work.seq = d.seq;
                work.t_arr_tsc = d.t_arr_tsc;
                work.x = d.x; work.y = d.y;
                work.owned_mask = shard_patch_mask(k, d.x, p.pipe.W(), N);
                std::memcpy(work.feat_in, d.feat_in, sizeof(work.feat_in));
                if (p.q_l1_in[k]->try_push_nb(work)) {
                    d.delivered_mask |= (std::uint16_t{1} << k);
                    did_work = true;
                } else {
                    any_blocked = true;
                    break;  // come back later
                }
            }
            if (any_blocked) break;
            dispatch_l1_pending.pop_front();
        }

        // Cascade pending_l1 — same pattern: pure local finalize, then
        // append to stage_dispatch_pending.
        while (!pending_l1.empty()) {
            Pending& pe = pending_l1.front();
            if (pe.is_marker) {
                StageInMsg sm{};
                sm.type = MsgType::Marker;
                sm.seq = pe.seq;
                stage_dispatch_pending.push_back(sm);
                pending_l1.pop_front();
                did_work = true;
                continue;
            }
            if (pe.received < pe.expected) break;
            p.pipe.layer_finalize(/*layer*/ 1, pe.feat_in_residual,
                                  residual_scratch.data(), pe.partial_sum);
            StageInMsg sm{};
            sm.type = MsgType::Event;
            sm.seq  = pe.seq;
            sm.t_arr_tsc = pe.t_arr_tsc;
            sm.x = pe.x; sm.y = pe.y;
            std::memcpy(sm.feat, pe.partial_sum, sizeof(sm.feat));
            stage_dispatch_pending.push_back(sm);
            idx_l1.erase(pe.seq);
            pending_l1.pop_front();
            did_work = true;
        }

        // Drain stage_dispatch_pending — try_push to q_to_stages.
        while (!stage_dispatch_pending.empty()) {
            const StageInMsg& sm = stage_dispatch_pending.front();
            if (!p.q_to_stages->try_push_nb(sm)) break;
            stage_dispatch_pending.pop_front();
            did_work = true;
        }

        if (!did_work) {
            if (p.done.load(std::memory_order_acquire) &&
                pending_l0.empty() && pending_l1.empty() &&
                dispatch_l1_pending.empty() && stage_dispatch_pending.empty() &&
                deferred_partials_l0.empty() && deferred_partials_l1.empty()) break;
            __builtin_ia32_pause();
        }
    }
}

// Stage 1+2+3+head thread: receives stage 0 output, runs the rest.
void thread_stages(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    StageInMsg in;
    std::vector<float> feat1(deploy::kC1), feat2(deploy::kC2), feat3(deploy::kC3);
    while (true) {
        if (!p.q_to_stages->try_pop(in)) {
            if (p.done.load(std::memory_order_acquire)) break;
            __builtin_ia32_pause();
            continue;
        }
        if (in.type == MsgType::Eof) break;
        if (in.type == MsgType::Marker) {
            p.marker_done_stages.store(in.seq, std::memory_order_release);
            continue;
        }
        // Apply pool/tdrop after stage 0
        int x = in.x, y = in.y;
        if (!p.pipe.tdrop_and_pool(0, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat_drop_s0.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        // Stage 1
        p.pipe.stage_forward(1, x, y, in.feat, feat1.data());
        if (!p.pipe.tdrop_and_pool(1, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat_drop_s1.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        // Stage 2
        p.pipe.stage_forward(2, x, y, feat1.data(), feat2.data());
        if (!p.pipe.tdrop_and_pool(2, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat_drop_s2.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        // Stage 3 + head
        p.pipe.stage_forward(3, x, y, feat2.data(), feat3.data());
        p.pipe.head_decode_cell(3, x, y, feat3.data());
        const auto t1 = deploy::rdtsc_now();
        p.lat_emit_s3.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
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

    const std::size_t qcap = std::size_t{1} << args.q_capacity_pow2;
    pl.q_l0_in.resize(args.num_shards);
    pl.q_l1_in.resize(args.num_shards);
    pl.q_partial_l0.resize(args.num_shards);
    pl.q_partial_l1.resize(args.num_shards);
    for (int k = 0; k < args.num_shards; ++k) {
        pl.q_l0_in[k]       = std::make_unique<deploy::SpscRing<SM_L0>>(qcap);
        pl.q_l1_in[k]       = std::make_unique<deploy::SpscRing<SM_L1>>(qcap);
        pl.q_partial_l0[k]  = std::make_unique<deploy::SpscRing<PM>>(qcap);
        pl.q_partial_l1[k]  = std::make_unique<deploy::SpscRing<PM>>(qcap);
    }
    pl.q_meta_l0    = std::make_unique<deploy::SpscRing<EM_L0>>(qcap);
    pl.q_meta_l1    = std::make_unique<deploy::SpscRing<EM_L1>>(qcap);
    pl.q_to_stages  = std::make_unique<deploy::SpscRing<StageInMsg>>(qcap);

    pl.lat_drop_s0.reserve(events.size());
    pl.lat_drop_s1.reserve(events.size());
    pl.lat_drop_s2.reserve(events.size());
    pl.lat_emit_s3.reserve(events.size());

    std::fprintf(stderr, "[s3] events=%zu shards=%d warmup=%d\n",
                 events.size(), args.num_shards, args.warmup);

    // Spawn threads
    std::vector<std::thread> shards;
    for (int k = 0; k < args.num_shards; ++k) {
        shards.emplace_back(thread_shard, std::ref(pl), k,
                            args.base_core + k, args.pin_cores);
    }
    std::thread t_sum(thread_sum, std::ref(pl),
                      args.base_core + args.num_shards, args.pin_cores);
    std::thread t_stages(thread_stages, std::ref(pl),
                         args.base_core + args.num_shards + 1, args.pin_cores);

    if (args.pin_cores) pin_to_core(args.base_core + args.num_shards + 2);

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);
    const int W = pl.pipe.W();
    const int H = pl.pipe.H();
    const int N = args.num_shards;

    // Push one event through the system. preprocess() runs ON THE PRODUCER
    // (single-threaded with respect to last_t_us_, which is the only
    // pre-stage-0 mutable state). Layer 0 input feat = (dt_norm, polarity).
    auto push_one = [&](std::uint64_t seq, const openeva::Event& e) {
        const int ex = static_cast<int>(e.x);
        const int ey = static_cast<int>(e.y);
        if (ex < 0 || ex >= W || ey < 0 || ey >= H) return;
        float feat[deploy::kInDim];
        pl.pipe.preprocess(e, feat);
        const std::uint64_t t = deploy::rdtsc_now();
        const std::uint16_t shard_mask = affected_shard_mask(ex, W, N);
        std::uint8_t expected = 0;
        for (int k = 0; k < N; ++k) if ((shard_mask >> k) & 1) ++expected;
        // Meta to sum thread
        EM_L0 mm{};
        mm.type = MsgType::Event;
        mm.layer = 0;
        mm.seq = seq;
        mm.t_arr_tsc = t;
        mm.x = ex; mm.y = ey;
        mm.expected_count = expected;
        mm.feat_in[0] = feat[0]; mm.feat_in[1] = feat[1];
        pl.q_meta_l0->push(mm);
        // Work to each affected shard
        for (int k = 0; k < N; ++k) {
            if (((shard_mask >> k) & 1) == 0) continue;
            SM_L0 work{};
            work.type = MsgType::Event;
            work.layer = 0;
            work.seq = seq;
            work.t_arr_tsc = t;
            work.x = ex; work.y = ey;
            work.owned_mask = shard_patch_mask(k, ex, W, N);
            work.feat_in[0] = feat[0]; work.feat_in[1] = feat[1];
            pl.q_l0_in[k]->push(work);
        }
    };
    auto push_marker = [&](std::uint64_t seq) {
        EM_L0 mm{};
        mm.type = MsgType::Marker;
        mm.seq = seq;
        pl.q_meta_l0->push(mm);
    };
    auto wait_marker = [&](std::uint64_t seq) {
        while (pl.marker_done_stages.load(std::memory_order_acquire) < seq) {
            __builtin_ia32_pause();
        }
    };

    // Warmup
    std::fprintf(stderr, "[s3] pushing %zu warmup events...\n", warmup);
    for (std::size_t i = 0; i < warmup; ++i) push_one(static_cast<std::uint64_t>(i), events[i]);
    std::fprintf(stderr, "[s3] pushing warmup marker seq=%zu\n", warmup);
    push_marker(static_cast<std::uint64_t>(warmup));
    std::fprintf(stderr, "[s3] waiting for marker_done_stages>=%zu (currently %lu)...\n",
                 warmup, (unsigned long)pl.marker_done_stages.load());
    wait_marker(static_cast<std::uint64_t>(warmup));
    std::fprintf(stderr, "[s3] warmup drain complete\n");

    std::unique_ptr<deploy::OracleRecorder> oracle;
    if (!args.oracle_dump.empty()) {
        oracle = std::make_unique<deploy::OracleRecorder>(
            pl.pipe.num_anchors_total(), pl.pipe.cols(),
            args.oracle_every, args.oracle_dump);
    }

    // Pacing
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
                std::fprintf(stderr, "[s3] checkpoint mi=%zu seq=%lu push+wait...\n",
                             mi, (unsigned long)mseq);
                push_marker(mseq);
                wait_marker(mseq);
                std::fprintf(stderr, "[s3] checkpoint mi=%zu drained\n", mi);
                oracle->record(static_cast<std::uint64_t>(mi), pl.pipe.last_output());
            }
        }
    }
    push_marker(static_cast<std::uint64_t>(total));
    wait_marker(static_cast<std::uint64_t>(total));
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start).count();

    // Drain
    EM_L0 eof{}; eof.type = MsgType::Eof; pl.q_meta_l0->push(eof);
    for (int k = 0; k < N; ++k) {
        SM_L0 e0{}; e0.type = MsgType::Eof; pl.q_l0_in[k]->push(e0);
    }
    pl.done.store(true, std::memory_order_release);
    for (auto& t : shards) t.join();
    t_sum.join();
    t_stages.join();

    std::vector<double> all;
    all.reserve(pl.lat_drop_s0.size() + pl.lat_drop_s1.size() +
                pl.lat_drop_s2.size() + pl.lat_emit_s3.size());
    for (auto v : pl.lat_drop_s0) all.push_back(v);
    for (auto v : pl.lat_drop_s1) all.push_back(v);
    for (auto v : pl.lat_drop_s2) all.push_back(v);
    for (auto v : pl.lat_emit_s3) all.push_back(v);
    auto stats = deploy::LatStats::from_samples(all);
    const double thru = wall_ns > 0 ? all.size() * 1e9 / wall_ns * 1e-6 : 0.0;

    std::fprintf(stdout, "\n=== Scheme 3: stage-0 sharded (%d shards) ===\n", N);
    std::fprintf(stdout, "  total events       : %zu\n", total);
    std::fprintf(stdout, "  measured events    : %zu\n", all.size());
    std::fprintf(stdout, "  wall (timed)       : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput         : %.3f Mev/s\n", thru);
    std::fprintf(stdout, "  drops              : s0=%zu s1=%zu s2=%zu emit=%zu\n",
                 pl.lat_drop_s0.size(), pl.lat_drop_s1.size(),
                 pl.lat_drop_s2.size(), pl.lat_emit_s3.size());
    std::fprintf(stdout, "\n  end-to-end latency:\n");
    stats.print("    ");
    if (oracle) {
        oracle->save();
        std::fprintf(stdout, "\n  oracle dump        : %s (%zu checkpoints)\n",
                     args.oracle_dump.c_str(), oracle->num_checkpoints());
    }
    return EXIT_SUCCESS;
}
