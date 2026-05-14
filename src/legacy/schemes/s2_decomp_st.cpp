// s2_decomp_st.cpp — Phase 1 of S2: single-thread decomposed pipeline.
//
// Validates that the deploy/-only SSLA-S kernel port is bit-equivalent to
// the reference SSLAModel (S1) on the same event stream. This is the
// equivalence gate for Phase 2 (multi-thread S2 / S3 / S4).
//
// Per event:
//   1. preprocess(e) -> (dt_norm, polarity)
//   2. stage_forward(0, ev_x, ev_y, feat_in, feat_out0)
//   3. tdrop+pool(0) -> if pass, ev_x, ev_y halved
//   4. stage_forward(1, ...) -> stage_out1
//   5. tdrop+pool(1)
//   6. stage_forward(2, ...) -> stage_out2
//   7. tdrop+pool(2)
//   8. stage_forward(3, ...) -> stage_out3
//   9. head_decode_cell(emitting stage, gx, gy, feat)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
    int  random_n  = 1000000;
    int  random_h  = 240;
    int  random_w  = 304;
    std::uint32_t random_seed = 1;
    std::string oracle_dump;
    std::size_t oracle_every = 10000;
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir>\n"
        "          [--events <npy> | --random-n N --random-hw H W --random-seed S]\n"
        "          [--warmup K] [--oracle-dump <path>] [--oracle-every K]\n", argv0);
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
    std::fprintf(stderr, "[s2-decomp-st] events=%zu warmup=%d\n",
                 events.size(), args.warmup);

    deploy::SslaSPipeline pipe;
    pipe.load(args.weights_dir);
    pipe.reset();

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);

    // Per-event scratch buffers (allocated once, sized to max channel count).
    float feat_in[deploy::kInDim];
    std::vector<float> stage_feat[deploy::kNumStages] = {
        std::vector<float>(deploy::kC0),
        std::vector<float>(deploy::kC1),
        std::vector<float>(deploy::kC2),
        std::vector<float>(deploy::kC3),
    };

    // Process one event end-to-end through the decomposed pipeline.
    auto process_one = [&](const openeva::Event& e) {
        const int ex0 = static_cast<int>(e.x);
        const int ey0 = static_cast<int>(e.y);
        if (ex0 < 0 || ex0 >= pipe.W() || ey0 < 0 || ey0 >= pipe.H()) {
            return;
        }
        pipe.preprocess(e, feat_in);
        int x = ex0, y = ey0;
        int touched_x[deploy::kNumStages], touched_y[deploy::kNumStages];

        // Stage 0
        touched_x[0] = x; touched_y[0] = y;
        pipe.stage_forward(0, x, y, feat_in, stage_feat[0].data());
        if (!pipe.tdrop_and_pool(0, x, y)) return;

        // Stage 1
        touched_x[1] = x; touched_y[1] = y;
        pipe.stage_forward(1, x, y, stage_feat[0].data(), stage_feat[1].data());
        if (!pipe.tdrop_and_pool(1, x, y)) return;

        // Stage 2
        touched_x[2] = x; touched_y[2] = y;
        pipe.stage_forward(2, x, y, stage_feat[1].data(), stage_feat[2].data());
        if (!pipe.tdrop_and_pool(2, x, y)) return;

        // Stage 3 (no tdrop after)
        touched_x[3] = x; touched_y[3] = y;
        pipe.stage_forward(3, x, y, stage_feat[2].data(), stage_feat[3].data());

        // Head decode for each emitting stage. With default config only
        // stage 3 emits; with multi-scale heads, we'd need stage_feat at
        // earlier stages too — for SSLA-S with one head level, this is
        // just the stage 3 cell.
        // (head_decode_cell is a no-op for stages that don't emit.)
        for (int s = 0; s < deploy::kNumStages; ++s) {
            pipe.head_decode_cell(s, touched_x[s], touched_y[s], stage_feat[s].data());
        }
    };

    // Warmup
    for (std::size_t i = 0; i < warmup; ++i) process_one(events[i]);

    // Oracle setup
    std::unique_ptr<deploy::OracleRecorder> oracle;
    if (!args.oracle_dump.empty()) {
        oracle = std::make_unique<deploy::OracleRecorder>(
            pipe.num_anchors_total(), pipe.cols(),
            args.oracle_every, args.oracle_dump);
        std::fprintf(stderr, "[s2-decomp-st] oracle: shape=%dx%d every=%zu dest=%s\n",
                     pipe.num_anchors_total(), pipe.cols(),
                     args.oracle_every, args.oracle_dump.c_str());
    }

    const std::size_t measured_first = warmup;
    const std::size_t measured_n = total - measured_first;

    std::vector<double> per_event_ns;
    per_event_ns.reserve(measured_n);

    const auto wall_start = std::chrono::steady_clock::now();
    for (std::size_t i = measured_first; i < total; ++i) {
        const std::uint64_t t0 = deploy::rdtsc_now();
        process_one(events[i]);
        const std::uint64_t t1 = deploy::rdtsc_now();
        per_event_ns.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - t0));
        if (oracle) {
            oracle->record(static_cast<std::uint64_t>(i - measured_first),
                           pipe.last_output());
        }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(
        wall_end - wall_start).count();

    const std::size_t lo = measured_n / 20;
    const std::size_t hi = measured_n - (measured_n / 20);
    std::vector<double> ss(per_event_ns.begin() + lo,
                           per_event_ns.begin() + hi);
    auto stats = deploy::LatStats::from_samples(ss);

    const double ss_frac = static_cast<double>(hi - lo) / measured_n;
    const double ss_wall_ns = wall_ns * ss_frac;
    const double thru_meps = (ss_wall_ns > 0.0)
        ? (hi - lo) * 1e9 / ss_wall_ns * 1e-6 : 0.0;

    std::fprintf(stdout, "\n=== Scheme 2 — phase 1: single-thread decomposed ===\n");
    std::fprintf(stdout, "  total events      : %zu\n", total);
    std::fprintf(stdout, "  warmup events     : %zu\n", warmup);
    std::fprintf(stdout, "  measured events   : %zu\n", measured_n);
    std::fprintf(stdout, "  steady-state win  : [%zu, %zu) (5%%-trim each side)\n", lo, hi);
    std::fprintf(stdout, "  wall (timed loop) : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput (ss)   : %.3f Mev/s\n", thru_meps);
    std::fprintf(stdout, "\n  per-event latency:\n");
    stats.print("    ");
    std::fprintf(stdout, "\n  per-event distribution:\n");
    deploy::print_histogram(ss, "    ");

    if (oracle) {
        oracle->save();
        std::fprintf(stdout, "\n  oracle dump       : %s (%zu checkpoints)\n",
                     args.oracle_dump.c_str(), oracle->num_checkpoints());
    }
    return EXIT_SUCCESS;
}
