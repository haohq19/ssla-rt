// s4_owner_shard.cpp — Scheme 4: owner-shard architecture (no central sum thread).
//
// Architectural change vs S3:
//   - Each shard FINALISES its own events (does layer_finalize locally).
//   - When a shard's event needs patches owned by a neighbor (boundary
//     events at strip edges), the OWNER shard sends a "contribute"
//     request to the neighbor and waits for the neighbor's partial
//     to arrive. Owner sums its own + neighbor partials, calls
//     layer_finalize, then forwards to its per-shard outbound queue.
//   - Stage thread (single, downstream) reads from N per-shard outbound
//     queues and merges by seq order before running stages 1-3 + head.
//
// This eliminates the S3 central sum thread that was the throughput
// bottleneck.
//
// Per shard k:
//   inbound queues:
//     q_own[k]              — events owned by this shard (from producer)
//     q_contrib[k]          — contribution requests (from producer)
//                             when this shard owns boundary patches of
//                             a neighbor's owner-event
//     q_partial_in[k]       — partials returning from neighbors for
//                             MY owner-events (single queue: tagged by side)
//   outbound:
//     q_partial_out[neighbor] — when I'm a contributor, push to neighbor's
//                                q_partial_in
//     q_to_stage[k]         — finalised stage-0 events ready for the
//                                stage thread
//
// Stage thread:
//     Reads from q_to_stage[0..N-1] in seq order (a small in-memory
//     min-seq merge), runs stages 1-3 + head as in S2/S3.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
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

// Producer → owner shard. Carries the layer-0 input feature.
struct OwnMsg {
    MsgType        type;
    std::uint8_t   need_left;   // 1 iff left-neighbor contrib expected for L0
    std::uint8_t   need_right;
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x, y;          // event x at stage-0 res
    std::uint16_t  own_mask_l0;   // patch bits this shard owns for L0
    float          feat_in_l0[deploy::kInDim];   // (dt_norm, polarity)
};

// Producer → contributor shard. Says "contribute the patches in
// `contrib_mask` (which are MY columns for an event whose center is at
// `x` in the OWNER's strip). Send back partial to `owner_shard`."
struct ContribMsg {
    MsgType        type;
    std::uint8_t   layer;        // 0 or 1
    std::uint8_t   owner_shard;  // who to send the partial back to
    std::uint64_t  seq;
    int            x, y;
    std::uint16_t  contrib_mask;
    float          feat_in[deploy::kC0]; // L0: only [0..1] used; L1: full
};

// Contributor shard → owner shard. Carries the partial and seq for matching.
struct PartialMsg {
    MsgType        type;
    std::uint8_t   layer;
    std::uint64_t  seq;
    float          partial[deploy::kC0];
};

// Owner shard → stage thread. Final stage-0 output.
struct ToStageMsg {
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
    int W = 0, H = 0;

    // q_own[k]: producer → shard k (owner work)
    std::vector<std::unique_ptr<deploy::SpscRing<OwnMsg>>>     q_own;
    // q_contrib[k]: producer → shard k (contribution requests)
    std::vector<std::unique_ptr<deploy::SpscRing<ContribMsg>>> q_contrib;
    // q_partial_in[k]: incoming partials (from any neighbor) for my owner-events
    std::vector<std::unique_ptr<deploy::SpscRing<PartialMsg>>> q_partial_in;
    // q_to_stage[k]: shard k → stage thread, finalised stage-0 events
    std::vector<std::unique_ptr<deploy::SpscRing<ToStageMsg>>> q_to_stage;

    // next_emit_seq[k] (per-shard atomic) — the seq of the NEXT event/marker
    // shard k will push to q_to_stage[k]. UINT64_MAX if shard has nothing
    // pending and EOF received. Stage thread uses this to know it's safe
    // to emit earlier seqs from other shards without waiting for k.
    std::vector<std::unique_ptr<std::atomic<std::uint64_t>>> next_emit_seq;

    // Per-shard latency samples
    std::vector<std::vector<double>> lat_drop_s0, lat_drop_s1, lat_drop_s2, lat_emit_s3;

    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done_stages{0};
    alignas(deploy::kCacheLine) std::atomic<bool> done{false};
};

// Strip ownership: column c → shard owning it.
inline int shard_of_col(int c, int W, int N) {
    if (c < 0) c = 0;
    if (c >= W) c = W - 1;
    const int strip_w = (W + N - 1) / N;
    int s = c / strip_w;
    if (s >= N) s = N - 1;
    return s;
}

// Compute owned-patch mask (9 bits) for shard k at event center column x.
inline std::uint16_t shard_patch_mask(int k, int x, int W, int N) {
    std::uint16_t m = 0;
    for (int p = 0; p < 9; ++p) {
        const int dx = 1 - (p % 3);
        const int col = x + dx;
        if (col < 0 || col >= W) continue;
        if (shard_of_col(col, W, N) == k) m |= (std::uint16_t{1} << p);
    }
    return m;
}

// =====================================================================
// Shard thread
//
// Each shard does owner-finalize: for events it owns, it keeps a
// pending-events list (FIFO by seq) and accumulates partials from
// neighbors. Once all expected partials for an event are in, it sums
// + finalizes + forwards.
// =====================================================================

struct PendingEvent {
    bool          is_marker = false;
    std::uint64_t seq = 0;
    std::uint64_t t_arr_tsc = 0;
    int           x = 0, y = 0;
    std::uint8_t  expected_l0 = 0;   // own + neighbors that will contribute (0..2)
    std::uint8_t  received_l0 = 0;
    std::uint8_t  expected_l1 = 0;
    std::uint8_t  received_l1 = 0;
    bool          l0_done = false;
    float         feat_in_l0[deploy::kC0]{};   // residual input for L0 finalize ([0..1] used)
    float         partial_l0[deploy::kC0]{};
    float         feat_in_l1[deploy::kC0]{};   // = layer 0 finalised output
    float         partial_l1[deploy::kC0]{};
};

void thread_shard(Pipeline& p, int k, int core, bool pin) {
    if (pin) pin_to_core(core);
    const int W = p.W;
    const int N = p.num_shards;

    std::list<PendingEvent> pending;
    std::unordered_map<std::uint64_t, std::list<PendingEvent>::iterator> idx;
    std::deque<PartialMsg> deferred_partials;

    std::vector<float> qvg_scratch(3 * deploy::kC0);
    std::vector<float> qh_scratch (deploy::kC0);
    std::vector<float> residual_scratch(deploy::kC0);

    auto match_partial = [&](const PartialMsg& pm) -> bool {
        auto it = idx.find(pm.seq);
        if (it == idx.end()) return false;
        PendingEvent& pe = *(it->second);
        if (pm.layer == 0) {
            for (int i = 0; i < deploy::kC0; ++i) pe.partial_l0[i] += pm.partial[i];
            ++pe.received_l0;
        } else {
            for (int i = 0; i < deploy::kC0; ++i) pe.partial_l1[i] += pm.partial[i];
            ++pe.received_l1;
        }
        return true;
    };

    // Outgoing-partial backlog per owner — when q_partial_in[owner_k] is
    // full, queue here and retry in the outer loop. This breaks the
    // producer→contrib→owner cycle when owner is busy.
    std::vector<std::deque<PartialMsg>> outgoing_partials(p.num_shards);
    auto send_partial_to_owner = [&](int owner_k, const PartialMsg& pm) {
        if (!p.q_partial_in[owner_k]->try_push_nb(pm)) {
            outgoing_partials[owner_k].push_back(pm);
        }
    };

    // Outgoing-contrib backlog (when this shard issues L1 contribs to
    // neighbors that are momentarily full).
    std::vector<std::deque<ContribMsg>> outgoing_contribs(p.num_shards);
    auto send_contrib = [&](int target_k, const ContribMsg& cm) {
        if (!p.q_contrib[target_k]->try_push_nb(cm)) {
            outgoing_contribs[target_k].push_back(cm);
        }
    };

    bool got_eof = false;
    constexpr int kPerIterCap = 32;   // per-queue work cap so no queue is starved
    while (true) {
        bool did_work = false;

        // ---- Drain q_own (owner work, capped) ----
        OwnMsg om;
        for (int n = 0; n < kPerIterCap && p.q_own[k]->try_pop(om); ++n) {
            did_work = true;
            if (om.type == MsgType::Eof) {
                got_eof = true;
                continue;   // continue draining; let pending finish
            }
            PendingEvent pe{};
            if (om.type == MsgType::Marker) {
                pe.is_marker = true;
                pe.seq = om.seq;
                pending.push_back(pe);
                continue;
            }
            // Owner event: process own L0 patches now
            pe.seq        = om.seq;
            pe.t_arr_tsc  = om.t_arr_tsc;
            pe.x          = om.x;
            pe.y          = om.y;
            pe.feat_in_l0[0] = om.feat_in_l0[0];
            pe.feat_in_l0[1] = om.feat_in_l0[1];
            // expected_l0 = 1 (own) + (need_left ? 1 : 0) + (need_right ? 1 : 0)
            pe.expected_l0 = 1 + om.need_left + om.need_right;
            // Compute own L0 partial
            std::memset(pe.partial_l0, 0, sizeof(pe.partial_l0));
            p.pipe.shard_layer_forward(0, om.x, om.y,
                                       om.feat_in_l0, om.own_mask_l0,
                                       qvg_scratch.data(), qh_scratch.data(),
                                       pe.partial_l0);
            ++pe.received_l0;
            pending.push_back(pe);
            idx[pe.seq] = std::prev(pending.end());
        }

        // ---- Drain q_contrib (contributor work, capped) ----
        ContribMsg cm;
        for (int n = 0; n < kPerIterCap && p.q_contrib[k]->try_pop(cm); ++n) {
            did_work = true;
            if (cm.type == MsgType::Eof) continue;
            if (cm.type == MsgType::Marker) continue;
            // Compute partial for the contributed patches (L0 or L1)
            PartialMsg outpm{};
            outpm.type  = MsgType::Event;
            outpm.layer = cm.layer;
            outpm.seq   = cm.seq;
            std::memset(outpm.partial, 0, sizeof(outpm.partial));
            p.pipe.shard_layer_forward(cm.layer, cm.x, cm.y,
                                       cm.feat_in, cm.contrib_mask,
                                       qvg_scratch.data(), qh_scratch.data(),
                                       outpm.partial);
            send_partial_to_owner(cm.owner_shard, outpm);
        }

        // ---- Drain q_partial_in (partials returning from neighbors, capped) ----
        PartialMsg pm;
        for (int n = 0; n < kPerIterCap && p.q_partial_in[k]->try_pop(pm); ++n) {
            did_work = true;
            if (pm.type != MsgType::Event) continue;
            if (!match_partial(pm)) deferred_partials.push_back(pm);
        }
        // Retry deferred (incoming partials whose meta wasn't yet present)
        for (std::size_t i = deferred_partials.size(); i > 0; --i) {
            PartialMsg dpm = deferred_partials.front();
            deferred_partials.pop_front();
            if (match_partial(dpm)) did_work = true;
            else deferred_partials.push_back(dpm);
        }

        // Retry outgoing-partial backlog (we couldn't deliver to owner earlier)
        for (int ok = 0; ok < p.num_shards; ++ok) {
            while (!outgoing_partials[ok].empty()) {
                if (p.q_partial_in[ok]->try_push_nb(outgoing_partials[ok].front())) {
                    outgoing_partials[ok].pop_front();
                    did_work = true;
                } else break;
            }
            while (!outgoing_contribs[ok].empty()) {
                if (p.q_contrib[ok]->try_push_nb(outgoing_contribs[ok].front())) {
                    outgoing_contribs[ok].pop_front();
                    did_work = true;
                } else break;
            }
        }

        // ---- Cascade pending ----
        while (!pending.empty()) {
            PendingEvent& pe = pending.front();
            if (pe.is_marker) {
                // Forward marker to stage thread (carries seq)
                ToStageMsg sm{};
                sm.type = MsgType::Marker;
                sm.seq = pe.seq;
                if (!p.q_to_stage[k]->try_push_nb(sm)) break;  // retry later
                pending.pop_front();
                did_work = true;
                continue;
            }

            if (!pe.l0_done) {
                if (pe.received_l0 < pe.expected_l0) break;  // still waiting
                // Finalize L0
                p.pipe.layer_finalize(0, pe.feat_in_l0,
                                      residual_scratch.data(), pe.partial_l0);
                // partial_l0 is now the L0 output. Set up L1.
                std::memcpy(pe.feat_in_l1, pe.partial_l0, sizeof(pe.feat_in_l1));
                std::memset(pe.partial_l1, 0, sizeof(pe.partial_l1));
                pe.l0_done = true;

                // Compute own L1 patches partial
                const std::uint16_t own_l1 = shard_patch_mask(k, pe.x, W, N);
                p.pipe.shard_layer_forward(1, pe.x, pe.y,
                                           pe.feat_in_l1, own_l1,
                                           qvg_scratch.data(), qh_scratch.data(),
                                           pe.partial_l1);
                ++pe.received_l1;

                // Determine L1 contrib needs (same neighbor pattern as L0,
                // since the same 9 patches are touched).
                int need_left  = 0, need_right = 0;
                if (k > 0   && shard_of_col(pe.x - 1, W, N) == k - 1) need_left  = 1;
                if (k < N-1 && shard_of_col(pe.x + 1, W, N) == k + 1) need_right = 1;
                pe.expected_l1 = 1 + need_left + need_right;

                // Send L1 contrib requests to neighbors (pushing feat_in_l1).
                if (need_left) {
                    ContribMsg lm{};
                    lm.type = MsgType::Event;
                    lm.layer = 1;
                    lm.owner_shard = static_cast<std::uint8_t>(k);
                    lm.seq = pe.seq;
                    lm.x = pe.x; lm.y = pe.y;
                    lm.contrib_mask = shard_patch_mask(k - 1, pe.x, W, N);
                    std::memcpy(lm.feat_in, pe.feat_in_l1, sizeof(lm.feat_in));
                    send_contrib(k - 1, lm);
                }
                if (need_right) {
                    ContribMsg lm{};
                    lm.type = MsgType::Event;
                    lm.layer = 1;
                    lm.owner_shard = static_cast<std::uint8_t>(k);
                    lm.seq = pe.seq;
                    lm.x = pe.x; lm.y = pe.y;
                    lm.contrib_mask = shard_patch_mask(k + 1, pe.x, W, N);
                    std::memcpy(lm.feat_in, pe.feat_in_l1, sizeof(lm.feat_in));
                    send_contrib(k + 1, lm);
                }
                did_work = true;
            }

            // L0 done, waiting for L1 partials
            if (pe.received_l1 < pe.expected_l1) break;
            // Finalize L1
            p.pipe.layer_finalize(1, pe.feat_in_l1,
                                  residual_scratch.data(), pe.partial_l1);
            // Forward to stage thread
            ToStageMsg sm{};
            sm.type = MsgType::Event;
            sm.seq  = pe.seq;
            sm.t_arr_tsc = pe.t_arr_tsc;
            sm.x = pe.x; sm.y = pe.y;
            std::memcpy(sm.feat, pe.partial_l1, sizeof(sm.feat));
            if (!p.q_to_stage[k]->try_push_nb(sm)) break;  // retry later
            idx.erase(pe.seq);
            pending.pop_front();
            did_work = true;
        }

        // Publish next-emit-seq for stage thread's merger.
        // Invariant: shard k's NEXT push to q_to_stage[k] will have
        // seq >= next_emit_seq[k]. If pending empty + EOF received,
        // publish UINT64_MAX (we will never push another event/marker).
        if (!pending.empty()) {
            p.next_emit_seq[k]->store(pending.front().seq,
                                       std::memory_order_release);
        } else if (got_eof) {
            p.next_emit_seq[k]->store(UINT64_MAX,
                                       std::memory_order_release);
        }

        if (!did_work) {
            // Exit when our local EOF has arrived AND we have no more
            // pending work in this shard. Other shards may still be
            // running — that's fine, they have their own state.
            bool outgoing_empty = true;
            for (auto& q : outgoing_partials) if (!q.empty()) { outgoing_empty = false; break; }
            for (auto& q : outgoing_contribs) if (!q.empty()) { outgoing_empty = false; break; }
            if (got_eof && pending.empty() && deferred_partials.empty() && outgoing_empty) break;
            __builtin_ia32_pause();
        }
    }
    p.next_emit_seq[k]->store(UINT64_MAX, std::memory_order_release);
}

// =====================================================================
// Stage thread (single)
//
// Reads from N per-shard q_to_stage queues. Merges in seq order and
// runs stages 1, 2, 3 + head — same pattern as S2/S3.
// =====================================================================

void thread_stages(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    const int N = p.num_shards;
    std::vector<float> feat1(deploy::kC1), feat2(deploy::kC2), feat3(deploy::kC3);

    // Min-seq merger: maintain a cached front from each shard's queue
    // that hasn't been consumed yet. Pick the smallest seq each iter.
    // Caching avoids repeated try_pop/try_push calls.
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

        // Compute the smallest seq we could safely emit now.
        // For each shard k:
        //   - if cache[k].have, candidate is cache[k].msg.seq
        //   - else, candidate is shard's next_emit_seq[k] (>= what k will
        //     ever push). If this is UINT64_MAX, shard is done forever.
        // Min across all candidates is the safe-emit seq from the shard
        // whose cache has it. If the smallest candidate IS in cache[min_k],
        // we can safely emit.
        std::uint64_t min_cand = UINT64_MAX;
        int min_k = -1;
        bool min_in_cache = false;
        for (int k = 0; k < N; ++k) {
            std::uint64_t c;
            bool in_cache;
            if (cache[k].have) {
                c = cache[k].msg.seq;
                in_cache = true;
            } else {
                c = p.next_emit_seq[k]->load(std::memory_order_acquire);
                in_cache = false;
            }
            if (c < min_cand) {
                min_cand = c;
                min_k = k;
                min_in_cache = in_cache;
            }
        }

        if (min_cand == UINT64_MAX) {
            // All shards finished, all caches empty.
            break;
        }

        if (!min_in_cache) {
            // Smallest candidate seq is in some shard's pending — we have
            // to wait for that shard to push it.
            __builtin_ia32_pause();
            continue;
        }

        // Emit min_k's msg
        ToStageMsg in = cache[min_k].msg;
        cache[min_k].have = false;

        if (in.type == MsgType::Marker) {
            p.marker_done_stages.store(in.seq, std::memory_order_release);
            continue;
        }

        // Run stages 1, 2, 3 + head
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

    const std::size_t qcap = std::size_t{1} << args.q_capacity_pow2;
    const int N = args.num_shards;
    pl.q_own.resize(N); pl.q_contrib.resize(N);
    pl.q_partial_in.resize(N); pl.q_to_stage.resize(N);
    pl.next_emit_seq.resize(N);
    for (int k = 0; k < N; ++k) {
        pl.q_own[k]         = std::make_unique<deploy::SpscRing<OwnMsg>>(qcap);
        pl.q_contrib[k]     = std::make_unique<deploy::SpscRing<ContribMsg>>(qcap);
        pl.q_partial_in[k]  = std::make_unique<deploy::SpscRing<PartialMsg>>(qcap);
        pl.q_to_stage[k]    = std::make_unique<deploy::SpscRing<ToStageMsg>>(qcap);
        pl.next_emit_seq[k] = std::make_unique<std::atomic<std::uint64_t>>(0);
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

    std::fprintf(stderr, "[s4] events=%zu shards=%d warmup=%d\n",
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

    // Producer-side back-off helper: spin-pause but periodically yield so
    // we don't deadlock if the consumer thread happens to be on the same
    // core or starved.
    auto push_with_yield = [&](auto* q, const auto& msg) {
        int spins = 0;
        while (!q->try_push_nb(msg)) {
            if (++spins > 1024) { std::this_thread::yield(); spins = 0; }
            else __builtin_ia32_pause();
        }
    };

    auto push_one = [&](std::uint64_t seq, const openeva::Event& e) {
        const int ex = static_cast<int>(e.x);
        const int ey = static_cast<int>(e.y);
        if (ex < 0 || ex >= pl.W || ey < 0 || ey >= pl.H) return;
        // Preprocess on producer (single-threaded — last_t_us is pl.pipe's
        // global state, only producer writes it)
        float feat[deploy::kInDim];
        pl.pipe.preprocess(e, feat);
        const std::uint64_t t = deploy::rdtsc_now();
        const int owner_k = shard_of_col(ex, pl.W, N);
        const int need_left  = (owner_k > 0     && shard_of_col(ex - 1, pl.W, N) == owner_k - 1) ? 1 : 0;
        const int need_right = (owner_k < N - 1 && shard_of_col(ex + 1, pl.W, N) == owner_k + 1) ? 1 : 0;

        // OWNER event
        OwnMsg om{};
        om.type = MsgType::Event;
        om.seq = seq;
        om.t_arr_tsc = t;
        om.x = ex; om.y = ey;
        om.need_left  = static_cast<std::uint8_t>(need_left);
        om.need_right = static_cast<std::uint8_t>(need_right);
        om.own_mask_l0 = shard_patch_mask(owner_k, ex, pl.W, N);
        om.feat_in_l0[0] = feat[0]; om.feat_in_l0[1] = feat[1];
        push_with_yield(pl.q_own[owner_k].get(), om);

        // CONTRIB requests (L0 only — L1 contribs are issued by owner shard
        // when it finalizes L0, since L1's feat_in is the L0 output not
        // available at producer time).
        if (need_left) {
            ContribMsg cm{};
            cm.type = MsgType::Event;
            cm.layer = 0;
            cm.owner_shard = static_cast<std::uint8_t>(owner_k);
            cm.seq = seq;
            cm.x = ex; cm.y = ey;
            cm.contrib_mask = shard_patch_mask(owner_k - 1, ex, pl.W, N);
            cm.feat_in[0] = feat[0]; cm.feat_in[1] = feat[1];
            push_with_yield(pl.q_contrib[owner_k - 1].get(), cm);
        }
        if (need_right) {
            ContribMsg cm{};
            cm.type = MsgType::Event;
            cm.layer = 0;
            cm.owner_shard = static_cast<std::uint8_t>(owner_k);
            cm.seq = seq;
            cm.x = ex; cm.y = ey;
            cm.contrib_mask = shard_patch_mask(owner_k + 1, ex, pl.W, N);
            cm.feat_in[0] = feat[0]; cm.feat_in[1] = feat[1];
            push_with_yield(pl.q_contrib[owner_k + 1].get(), cm);
        }
    };

    auto push_marker = [&](std::uint64_t seq) {
        // Push marker to ALL shards' q_own — so every shard's pending
        // list has the marker at the right seq position. Each shard
        // forwards it to q_to_stage[k] in its own seq order. Stage
        // thread merges and waits for marker on all queues before
        // signalling marker_done_stages.
        for (int k = 0; k < N; ++k) {
            OwnMsg om{};
            om.type = MsgType::Marker;
            om.seq = seq;
            push_with_yield(pl.q_own[k].get(), om);
        }
    };

    auto wait_marker = [&](std::uint64_t seq) {
        while (pl.marker_done_stages.load(std::memory_order_acquire) < seq) {
            __builtin_ia32_pause();
        }
    };

    // Warmup
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

    // Tear down
    for (int k = 0; k < N; ++k) {
        OwnMsg eof{}; eof.type = MsgType::Eof;
        push_with_yield(pl.q_own[k].get(), eof);
    }
    pl.done.store(true, std::memory_order_release);
    for (auto& t : shards) t.join();
    t_stages.join();

    std::vector<double> all;
    for (int k = 0; k < N; ++k) {
        for (auto v : pl.lat_drop_s0[k]) all.push_back(v);
        for (auto v : pl.lat_drop_s1[k]) all.push_back(v);
        for (auto v : pl.lat_drop_s2[k]) all.push_back(v);
        for (auto v : pl.lat_emit_s3[k]) all.push_back(v);
    }
    auto stats = deploy::LatStats::from_samples(all);
    const double thru = wall_ns > 0 ? all.size() * 1e9 / wall_ns * 1e-6 : 0.0;

    std::size_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < N; ++k) {
        s0 += pl.lat_drop_s0[k].size();
        s1 += pl.lat_drop_s1[k].size();
        s2 += pl.lat_drop_s2[k].size();
        s3 += pl.lat_emit_s3[k].size();
    }

    std::fprintf(stdout, "\n=== Scheme 4: owner-shard (%d shards, no central sum) ===\n", N);
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
