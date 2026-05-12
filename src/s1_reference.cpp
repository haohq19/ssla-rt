// s1_reference.cpp — Scheme 1: single-threaded reference.
//
// Establishes the latency distribution baseline for SSLA-S, and produces
// the oracle dump that S2/S3/S4 must match for strict semantic equivalence.
//
// Per-event latency is measured tightly around model->step() with rdtsc.
// Throughput is reported over a steady-state window (drops first 5% and
// last 5% of post-warmup events to skip warmup bleed-through and any
// final-batch tail effects).
//
// Usage:
//   s1_reference --weights <dir> [--events <npy>] [--random-n N]
//                [--random-hw H W] [--random-seed S] [--warmup K]
//                [--oracle-dump <path>] [--oracle-every K]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "openeva/event.h"
#include "openeva/runtime_model.h"
#include "src/io.h"

#include "lat_stats.h"
#include "oracle.h"
#include "timed.h"

namespace {

struct Args {
    std::string weights_dir;
    std::string events_path;
    int         warmup       = 10000;
    int         random_n     = 1000000;
    int         random_h     = 240;
    int         random_w     = 304;
    std::uint32_t random_seed = 1;
    std::string oracle_dump;
    std::size_t oracle_every = 10000;
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir>\n"
        "          [--events <npy> | --random-n N --random-hw H W --random-seed S]\n"
        "          [--warmup K]\n"
        "          [--oracle-dump <path>] [--oracle-every K]\n", argv0);
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
        ev[i].t = static_cast<float>(i);   // 1 µs / event nominal
        ev[i].x = static_cast<float>(dx(rng));
        ev[i].y = static_cast<float>(dy(rng));
        ev[i].p = static_cast<float>(dp(rng));
    }
    return ev;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // ---- Calibrate TSC clock before any timed work ----
    // Touch the singleton so the 50 ms calibration sleep happens up front.
    deploy::TscClock::instance().tsc_to_ns(0);

    // ---- Load events ----
    std::vector<openeva::Event> events;
    if (!args.events_path.empty()) {
        events = openeva::load_events_npy(args.events_path);
    } else {
        events = make_random_events(args.random_n, args.random_h, args.random_w,
                                    args.random_seed);
    }
    if (events.empty()) {
        std::fprintf(stderr, "no events\n"); return EXIT_FAILURE;
    }
    std::fprintf(stderr, "[s1] events=%zu warmup=%d\n",
                 events.size(), args.warmup);

    // ---- Construct model ----
    auto model = openeva::create_model("ssla");
    model->load_weights(args.weights_dir);
    model->reset();

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);
    if (total <= warmup) {
        std::fprintf(stderr, "warmup >= total events\n"); return EXIT_FAILURE;
    }

    // ---- Warmup (untimed) ----
    int num_anchors = 0, cols = 0;
    for (std::size_t i = 0; i < warmup; ++i) {
        const auto& out = model->step(events[i]);
        // Peek head shape from any warmup output that has predictions.
        if (num_anchors == 0) {
            if (auto* d = std::get_if<openeva::DetectionOutput>(&out)) {
                if (d->predictions.shape.size() == 2) {
                    num_anchors = static_cast<int>(d->predictions.shape[0]);
                    cols        = static_cast<int>(d->predictions.shape[1]);
                }
            }
        }
    }

    std::unique_ptr<deploy::OracleRecorder> oracle;
    if (!args.oracle_dump.empty() && num_anchors > 0 && cols > 0) {
        oracle = std::make_unique<deploy::OracleRecorder>(
            num_anchors, cols, args.oracle_every, args.oracle_dump);
        std::fprintf(stderr, "[s1] oracle: shape=%dx%d every=%zu dest=%s\n",
                     num_anchors, cols, args.oracle_every, args.oracle_dump.c_str());
    }

    // Measured starts immediately after warmup — same alignment as S2.
    const std::size_t measured_first = warmup;
    const std::size_t measured_n = total - measured_first;

    // ---- Timed loop ----
    std::vector<double> per_event_ns;
    per_event_ns.reserve(measured_n);

    const auto wall_start = std::chrono::steady_clock::now();
    for (std::size_t i = measured_first; i < total; ++i) {
        const std::uint64_t t0 = deploy::rdtsc_now();
        const auto& out = model->step(events[i]);
        const std::uint64_t t1 = deploy::rdtsc_now();

        per_event_ns.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - t0));

        if (oracle) {
            // Record at every K events (relative to the measured-event index)
            oracle->record(static_cast<std::uint64_t>(i - measured_first), out);
        }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ns = std::chrono::duration<double, std::nano>(wall_end - wall_start).count();

    // ---- Steady-state window: drop first/last 5% of measured events ----
    const std::size_t lo = measured_n / 20;
    const std::size_t hi = measured_n - (measured_n / 20);
    std::vector<double> ss(per_event_ns.begin() + lo,
                           per_event_ns.begin() + hi);
    auto stats = deploy::LatStats::from_samples(ss);

    // Throughput from the steady-state window's wall span (proportional).
    const double ss_frac = static_cast<double>(hi - lo) / measured_n;
    const double ss_wall_ns = wall_ns * ss_frac;
    const double thru_meps = (ss_wall_ns > 0.0)
        ? (hi - lo) * 1e9 / ss_wall_ns * 1e-6 : 0.0;

    std::fprintf(stdout, "\n=== Scheme 1 (single-thread reference) ===\n");
    std::fprintf(stdout, "  total events      : %zu\n", total);
    std::fprintf(stdout, "  warmup events     : %zu\n", warmup);
    std::fprintf(stdout, "  measured events   : %zu\n", measured_n);
    std::fprintf(stdout, "  steady-state win  : [%zu, %zu) (5%%-trim each side)\n", lo, hi);
    std::fprintf(stdout, "  wall (timed loop) : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput (ss)   : %.3f Mev/s\n", thru_meps);
    std::fprintf(stdout, "\n  per-event latency:\n");
    stats.print("    ");
    std::fprintf(stdout, "\n  per-event distribution (ns log bins):\n");
    deploy::print_histogram(ss, "    ");

    if (oracle) {
        oracle->save();
        std::fprintf(stdout, "\n  oracle dump       : %s (%zu checkpoints)\n",
                     args.oracle_dump.c_str(), oracle->num_checkpoints());
    }
    return EXIT_SUCCESS;
}
