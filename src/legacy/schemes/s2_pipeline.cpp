// s2_pipeline.cpp — Scheme 2: 4-stage threaded pipeline.
//
// Architecture:
//
//   producer  →  q0  →  S0 thread  →  q1  →  S1 thread  →  q2  →  S2 thread  →  q3  →  S3 thread (+head)
//
// Each stage thread is single-threaded internally (events processed in
// FIFO order from its input queue), so each stage's hidden state is
// updated in the same temporal order as S1. Cross-stage state is
// independent (each stage's hidden grid + tdrop counter is private).
// last_t_us_ is touched only by S0. predictions tensor is touched only
// by S3. Therefore the pipeline produces bit-identical outputs to S1
// at any quiescent point.
//
// Drain markers:
//   To take an oracle snapshot at "after event N", the producer pushes
//   a marker message tagged seq=N which always passes (no tdrop) and
//   carries no feature work. The marker propagates through every stage's
//   queue. When S3 sees it, it sets a public atomic so the producer can
//   safely snapshot the predictions tensor — at that instant, all events
//   ≤ N have completed (either dropped at some stage or emitted to head).
//
// Per-event end-to-end latency:
//   Each input event carries a t_arr_tsc captured at producer push time.
//   When the event is "consumed" (dropped or emitted), the consuming
//   stage records (now - t_arr) into a per-stage latency vector. The
//   merged distribution across all stages is the end-to-end latency.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
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
    int  random_n  = 1000000;
    int  random_h  = 240;
    int  random_w  = 304;
    std::uint32_t random_seed = 1;
    std::string oracle_dump;
    std::size_t oracle_every = 10000;
    int  q_capacity_pow2 = 16;   // 65536 slots per queue
    bool pin_cores = true;
    int  base_core = 0;          // 4 stage threads pinned to base_core..+3
    double target_rate_mev = 0.0;  // 0 = saturation (no pacing)
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir>\n"
        "          [--events <npy> | --random-n N --random-hw H W --random-seed S]\n"
        "          [--warmup K] [--oracle-dump P] [--oracle-every K]\n"
        "          [--q-cap-pow2 N] [--no-pin] [--base-core N]\n", argv0);
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
        if (auto v = eat("--q-cap-pow2"))   { a.q_capacity_pow2 = std::atoi(v); continue; }
        if (auto v = eat("--base-core"))    { a.base_core = std::atoi(v); continue; }
        if (auto v = eat("--target-mev"))   { a.target_rate_mev = std::atof(v); continue; }
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
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(core_id, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

// =====================================================================
// Inter-stage message types. All are POD, fixed size — copies fit one
// cache line where possible.
// =====================================================================

enum class MsgType : std::uint8_t { Event = 0, Marker = 1, Eof = 2 };

// Producer → S0
struct InMsg {
    MsgType         type;
    std::uint64_t   seq;
    std::uint64_t   t_arr_tsc;
    openeva::Event  ev;
};

// Generic stage→stage message templated by feat dim.
template <int FEAT_DIM>
struct StageMsg {
    MsgType        type;
    std::uint64_t  seq;
    std::uint64_t  t_arr_tsc;
    int            x;
    int            y;
    float          feat[FEAT_DIM];
};

using Msg01 = StageMsg<deploy::kC0>;
using Msg12 = StageMsg<deploy::kC1>;
using Msg23 = StageMsg<deploy::kC2>;
using Msg3H = StageMsg<deploy::kC3>;

// =====================================================================
// Pipeline state — shared by all threads.
// =====================================================================

struct Pipeline {
    deploy::SslaSPipeline pipe;
    std::unique_ptr<deploy::SpscRing<InMsg>> q_in;
    std::unique_ptr<deploy::SpscRing<Msg01>> q01;
    std::unique_ptr<deploy::SpscRing<Msg12>> q12;
    std::unique_ptr<deploy::SpscRing<Msg23>> q23;
    std::unique_ptr<deploy::SpscRing<Msg3H>> q3h;  // unused (head folded into S3)

    // Per-stage latency samples (filled by the stage that *consumes* the
    // event — i.e. drops it or emits it from S3). One vector per stage so
    // we don't lock; merged at end.
    std::vector<double> lat0, lat1, lat2, lat3;

    // Drain protocol: each stage publishes the highest seq of any marker
    // it has fully forwarded. Producer waits for marker_done[3] >= N to
    // snapshot.
    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done0{0};
    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done1{0};
    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done2{0};
    alignas(deploy::kCacheLine) std::atomic<std::uint64_t> marker_done3{0};

    // EOF flag — once producer has pushed Eof, all stages drain and exit.
    alignas(deploy::kCacheLine) std::atomic<bool> done{false};
};

// =====================================================================
// Stage threads
// =====================================================================

void thread_s0(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    InMsg in;
    Msg01 out{};
    while (true) {
        if (!p.q_in->try_pop(in)) {
            if (p.done.load(std::memory_order_acquire)) break;
            __builtin_ia32_pause();
            continue;
        }
        if (in.type == MsgType::Eof) {
            // forward EOF downstream
            Msg01 e{}; e.type = MsgType::Eof; e.seq = in.seq; e.t_arr_tsc = in.t_arr_tsc;
            p.q01->push(e);
            break;
        }
        if (in.type == MsgType::Marker) {
            Msg01 m{}; m.type = MsgType::Marker; m.seq = in.seq;
            p.q01->push(m);
            p.marker_done0.store(in.seq, std::memory_order_release);
            continue;
        }
        // Event
        const int ex0 = static_cast<int>(in.ev.x);
        const int ey0 = static_cast<int>(in.ev.y);
        if (ex0 < 0 || ex0 >= p.pipe.W() || ey0 < 0 || ey0 >= p.pipe.H()) {
            // Drop: out-of-bounds
            const auto t1 = deploy::rdtsc_now();
            p.lat0.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        float feat_in[deploy::kInDim];
        p.pipe.preprocess(in.ev, feat_in);
        int x = ex0, y = ey0;
        p.pipe.stage_forward(0, x, y, feat_in, out.feat);
        if (!p.pipe.tdrop_and_pool(0, x, y)) {
            // Dropped at stage 0 boundary
            const auto t1 = deploy::rdtsc_now();
            p.lat0.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        out.type      = MsgType::Event;
        out.seq       = in.seq;
        out.t_arr_tsc = in.t_arr_tsc;
        out.x = x; out.y = y;
        p.q01->push(out);
    }
}

void thread_s1(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    Msg01 in;
    Msg12 out{};
    while (true) {
        if (!p.q01->try_pop(in)) { __builtin_ia32_pause(); continue; }
        if (in.type == MsgType::Eof) {
            Msg12 e{}; e.type = MsgType::Eof; p.q12->push(e); break;
        }
        if (in.type == MsgType::Marker) {
            Msg12 m{}; m.type = MsgType::Marker; m.seq = in.seq;
            p.q12->push(m);
            p.marker_done1.store(in.seq, std::memory_order_release);
            continue;
        }
        int x = in.x, y = in.y;
        p.pipe.stage_forward(1, x, y, in.feat, out.feat);
        if (!p.pipe.tdrop_and_pool(1, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat1.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        out.type      = MsgType::Event;
        out.seq       = in.seq;
        out.t_arr_tsc = in.t_arr_tsc;
        out.x = x; out.y = y;
        p.q12->push(out);
    }
}

void thread_s2(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    Msg12 in;
    Msg23 out{};
    while (true) {
        if (!p.q12->try_pop(in)) { __builtin_ia32_pause(); continue; }
        if (in.type == MsgType::Eof) {
            Msg23 e{}; e.type = MsgType::Eof; p.q23->push(e); break;
        }
        if (in.type == MsgType::Marker) {
            Msg23 m{}; m.type = MsgType::Marker; m.seq = in.seq;
            p.q23->push(m);
            p.marker_done2.store(in.seq, std::memory_order_release);
            continue;
        }
        int x = in.x, y = in.y;
        p.pipe.stage_forward(2, x, y, in.feat, out.feat);
        if (!p.pipe.tdrop_and_pool(2, x, y)) {
            const auto t1 = deploy::rdtsc_now();
            p.lat2.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
            continue;
        }
        out.type      = MsgType::Event;
        out.seq       = in.seq;
        out.t_arr_tsc = in.t_arr_tsc;
        out.x = x; out.y = y;
        p.q23->push(out);
    }
}

// Stage 3 + head folded together (only emitting stage in default config).
void thread_s3(Pipeline& p, int core, bool pin) {
    if (pin) pin_to_core(core);
    Msg23 in;
    std::vector<float> feat3(deploy::kC3);
    while (true) {
        if (!p.q23->try_pop(in)) { __builtin_ia32_pause(); continue; }
        if (in.type == MsgType::Eof) break;
        if (in.type == MsgType::Marker) {
            p.marker_done3.store(in.seq, std::memory_order_release);
            continue;
        }
        const int x = in.x, y = in.y;
        p.pipe.stage_forward(3, x, y, in.feat, feat3.data());
        p.pipe.head_decode_cell(3, x, y, feat3.data());
        const auto t1 = deploy::rdtsc_now();
        p.lat3.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - in.t_arr_tsc));
    }
}

// Wait until S3 has fully processed the marker tagged with `seq`.
void wait_for_marker(Pipeline& p, std::uint64_t seq) {
    while (p.marker_done3.load(std::memory_order_acquire) < seq) {
        __builtin_ia32_pause();
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    deploy::TscClock::instance().tsc_to_ns(0);

    std::vector<openeva::Event> events;
    if (!args.events_path.empty()) {
        events = openeva::load_events_npy(args.events_path);
    } else {
        events = make_random_events(args.random_n, args.random_h, args.random_w,
                                    args.random_seed);
    }
    if (events.empty()) { std::fprintf(stderr, "no events\n"); return EXIT_FAILURE; }
    std::fprintf(stderr, "[s2] events=%zu warmup=%d q_cap=2^%d\n",
                 events.size(), args.warmup, args.q_capacity_pow2);

    Pipeline pl;
    pl.pipe.load(args.weights_dir);
    pl.pipe.reset();
    const std::size_t qcap = std::size_t{1} << args.q_capacity_pow2;
    pl.q_in = std::make_unique<deploy::SpscRing<InMsg>>(qcap);
    pl.q01  = std::make_unique<deploy::SpscRing<Msg01>>(qcap);
    pl.q12  = std::make_unique<deploy::SpscRing<Msg12>>(qcap);
    pl.q23  = std::make_unique<deploy::SpscRing<Msg23>>(qcap);
    // Reserve full size for every per-stage latency vector. Per-stage
    // push_back must NEVER reallocate during the timed loop — a single
    // realloc on lat0 (4 MB → 8 MB grow) stalls S0 thread for ~100 µs,
    // backs up q_in, and shows as a multi-ms tail across all events
    // arriving during the stall. Memory cost: 4 stages × N × 8 B,
    // ~6 MB for N=200k. Cheap.
    pl.lat0.reserve(events.size());
    pl.lat1.reserve(events.size());
    pl.lat2.reserve(events.size());
    pl.lat3.reserve(events.size());

    // Spawn stage threads
    std::thread t0(thread_s0, std::ref(pl), args.base_core + 0, args.pin_cores);
    std::thread t1(thread_s1, std::ref(pl), args.base_core + 1, args.pin_cores);
    std::thread t2(thread_s2, std::ref(pl), args.base_core + 2, args.pin_cores);
    std::thread t3(thread_s3, std::ref(pl), args.base_core + 3, args.pin_cores);

    // Producer (this thread, pinned to base_core + 4)
    if (args.pin_cores) pin_to_core(args.base_core + 4);

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);

    auto push_event = [&](std::uint64_t seq, const openeva::Event& e) {
        InMsg m{};
        m.type      = MsgType::Event;
        m.seq       = seq;
        m.t_arr_tsc = deploy::rdtsc_now();
        m.ev        = e;
        pl.q_in->push(m);
    };
    auto push_marker = [&](std::uint64_t seq) {
        InMsg m{};
        m.type = MsgType::Marker;
        m.seq  = seq;
        pl.q_in->push(m);
    };

    // Warmup
    for (std::size_t i = 0; i < warmup; ++i) push_event(static_cast<std::uint64_t>(i), events[i]);
    push_marker(static_cast<std::uint64_t>(warmup));    // marker N=warmup
    wait_for_marker(pl, static_cast<std::uint64_t>(warmup));

    std::unique_ptr<deploy::OracleRecorder> oracle;
    if (!args.oracle_dump.empty()) {
        oracle = std::make_unique<deploy::OracleRecorder>(
            pl.pipe.num_anchors_total(), pl.pipe.cols(),
            args.oracle_every, args.oracle_dump);
    }

    // Rate pacing: if target rate set, busy-spin between pushes to maintain
    // a steady inter-arrival interval. At 10 Mev/s = 100 ns between events,
    // we can't sleep — must busy-wait via rdtsc.
    const double ns_per_tick = deploy::TscClock::instance().tsc_to_ns(1);
    std::uint64_t pace_ticks = 0;
    if (args.target_rate_mev > 0.0) {
        const double ns_per_event = 1000.0 / args.target_rate_mev;  // 10 Mev/s -> 100 ns
        pace_ticks = static_cast<std::uint64_t>(ns_per_event / ns_per_tick);
        std::fprintf(stderr, "[s2] pacing producer at %.2f Mev/s (%.1f ns/event = %lu TSC ticks)\n",
                     args.target_rate_mev, ns_per_event,
                     static_cast<unsigned long>(pace_ticks));
    }

    // Measured loop
    const auto wall_start = std::chrono::steady_clock::now();
    std::uint64_t next_push_tsc = deploy::rdtsc_now();
    for (std::size_t i = warmup; i < total; ++i) {
        if (pace_ticks > 0) {
            while (deploy::rdtsc_now() < next_push_tsc) __builtin_ia32_pause();
            next_push_tsc += pace_ticks;
        }
        push_event(static_cast<std::uint64_t>(i), events[i]);

        // Oracle: at every K measured events, push marker, drain, snapshot.
        if (oracle) {
            const std::size_t mi = i - warmup;
            if (mi % args.oracle_every == 0) {
                const std::uint64_t mseq = static_cast<std::uint64_t>(i + 1);
                push_marker(mseq);
                wait_for_marker(pl, mseq);
                oracle->record(static_cast<std::uint64_t>(mi),
                               pl.pipe.last_output());
            }
        }
    }
    // Final drain marker
    const std::uint64_t final_seq = static_cast<std::uint64_t>(total);
    push_marker(final_seq);
    wait_for_marker(pl, final_seq);
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start).count();

    // Signal EOF + join
    InMsg eof{}; eof.type = MsgType::Eof; pl.q_in->push(eof);
    pl.done.store(true, std::memory_order_release);
    t0.join(); t1.join(); t2.join(); t3.join();

    // Merge per-stage latencies
    std::vector<double> all_lat;
    all_lat.reserve(pl.lat0.size() + pl.lat1.size() + pl.lat2.size() + pl.lat3.size());
    for (double v : pl.lat0) all_lat.push_back(v);
    for (double v : pl.lat1) all_lat.push_back(v);
    for (double v : pl.lat2) all_lat.push_back(v);
    for (double v : pl.lat3) all_lat.push_back(v);
    auto stats = deploy::LatStats::from_samples(all_lat);
    const double thru_meps = (wall_ns > 0.0)
        ? all_lat.size() * 1e9 / wall_ns * 1e-6 : 0.0;

    std::fprintf(stdout, "\n=== Scheme 2: 4-stage threaded pipeline ===\n");
    std::fprintf(stdout, "  total events       : %zu\n", total);
    std::fprintf(stdout, "  warmup events      : %zu\n", warmup);
    std::fprintf(stdout, "  measured events    : %zu\n", all_lat.size());
    std::fprintf(stdout, "  wall (timed)       : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput         : %.3f Mev/s\n", thru_meps);
    std::fprintf(stdout, "  drop counts        : s0=%zu s1=%zu s2=%zu emitted=%zu\n",
                 pl.lat0.size(), pl.lat1.size(), pl.lat2.size(), pl.lat3.size());
    std::fprintf(stdout, "\n  end-to-end latency (ALL events):\n");
    stats.print("    ");
    std::fprintf(stdout, "\n  per-event distribution:\n");
    deploy::print_histogram(all_lat, "    ");

    // Per-stage tail
    auto print_stage = [&](const char* nm, std::vector<double>& v) {
        if (v.empty()) return;
        auto s = deploy::LatStats::from_samples(v);
        std::fprintf(stdout, "  [%s] n=%zu mean=%.1fns p99.9=%.1fns max=%.1fns\n",
                     nm, s.n, s.mean_ns, s.p99_9_ns, s.max_ns);
    };
    std::fprintf(stdout, "\n  per-termination-stage:\n");
    print_stage("drop@s0", pl.lat0);
    print_stage("drop@s1", pl.lat1);
    print_stage("drop@s2", pl.lat2);
    print_stage("emit@s3", pl.lat3);

    if (oracle) {
        oracle->save();
        std::fprintf(stdout, "\n  oracle dump        : %s (%zu checkpoints)\n",
                     args.oracle_dump.c_str(), oracle->num_checkpoints());
    }
    return EXIT_SUCCESS;
}
