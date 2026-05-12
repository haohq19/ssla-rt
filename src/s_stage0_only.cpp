// s_stage0_only.cpp — measure stage-0 only on the deploy/-only kernel port.
//
// Purpose: at the 100k ev/s + 1ms-max target, the proposed CPU/GPU split
// puts stage 0 (and possibly stage 1) on the CPU. This bench measures
// the per-event cost of stage 0 IN ISOLATION on this machine so we can
// budget the rest of the pipeline.
//
// Per event:
//   1. preprocess(e) -> (dt_norm, polarity)
//   2. stage_forward(0, ev_x, ev_y, feat_in, feat_out0)
//   3. tdrop_and_pool(0)  — counts pass/drop only; downstream not run.
//
// Optional rate-pacing via TSC busy-wait at --target-mev events/sec, so
// we can measure intrinsic cost AND back-pressure-free latency at the
// real target rate.

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

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "openeva/event.h"
#include "src/io.h"

#include "lat_stats.h"
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
    double target_mev = 0.0;     // 0 = saturate
    int    pin_core   = -1;      // -1 = no pin
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir>\n"
        "          [--events <npy> | --random-n N --random-hw H W --random-seed S]\n"
        "          [--warmup K]\n"
        "          [--target-mev R]   # rate-pace at R Mev/s (default: saturation)\n"
        "          [--pin-core C]\n", argv0);
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
        if (auto v = eat("--target-mev"))   { a.target_mev = std::atof(v); continue; }
        if (auto v = eat("--pin-core"))     { a.pin_core = std::atoi(v); continue; }
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

void pin_to_core(int core) {
#if defined(__linux__)
    if (core < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)core;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    pin_to_core(args.pin_core);

    deploy::TscClock::instance().tsc_to_ns(0);

    std::vector<openeva::Event> events;
    if (!args.events_path.empty()) {
        events = openeva::load_events_npy(args.events_path);
    } else {
        events = make_random_events(args.random_n, args.random_h, args.random_w,
                                    args.random_seed);
    }
    if (events.empty()) { std::fprintf(stderr, "no events\n"); return EXIT_FAILURE; }
    std::fprintf(stderr, "[stage0] events=%zu warmup=%d target=%.3f Mev/s pin=%d\n",
                 events.size(), args.warmup, args.target_mev, args.pin_core);

    deploy::SslaSPipeline pipe;
    pipe.load(args.weights_dir);
    pipe.reset();

    const std::size_t total  = events.size();
    const std::size_t warmup = std::min(static_cast<std::size_t>(std::max(0, args.warmup)), total);

    float feat_in[deploy::kInDim];
    std::vector<float> feat0(deploy::kC0);

    std::size_t pass_stage0 = 0;
    std::size_t oob_count = 0;

    auto process_one = [&](const openeva::Event& e) {
        const int ex0 = static_cast<int>(e.x);
        const int ey0 = static_cast<int>(e.y);
        if (ex0 < 0 || ex0 >= pipe.W() || ey0 < 0 || ey0 >= pipe.H()) {
            ++oob_count;
            return;
        }
        pipe.preprocess(e, feat_in);
        int x = ex0, y = ey0;
        pipe.stage_forward(0, x, y, feat_in, feat0.data());
        if (pipe.tdrop_and_pool(0, x, y)) ++pass_stage0;
    };

    // Warmup
    for (std::size_t i = 0; i < warmup; ++i) process_one(events[i]);
    pass_stage0 = 0;
    oob_count = 0;

    const std::size_t measured_first = warmup;
    const std::size_t measured_n = total - measured_first;

    std::vector<double> per_event_ns;
    per_event_ns.reserve(measured_n);

    // Rate-pacing via TSC busy-wait. ns_per_event = 1e9 / (target_mev * 1e6).
    const double ns_per_tick = deploy::TscClock::instance().tsc_to_ns(1);
    const double ticks_per_event = (args.target_mev > 0.0)
        ? (1e3 / args.target_mev) / ns_per_tick   // ns/event = 1000 / Mev/s
        : 0.0;

    const auto wall_start = std::chrono::steady_clock::now();
    const std::uint64_t base_tsc = deploy::rdtsc_now();
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
        const std::uint64_t t0 = deploy::rdtsc_now();
        process_one(events[i]);
        const std::uint64_t t1 = deploy::rdtsc_now();
        per_event_ns.push_back(deploy::TscClock::instance().tsc_to_ns(t1 - t0));
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

    const std::size_t in_bounds = measured_n - oob_count;
    const double pass_rate = (in_bounds > 0)
        ? 100.0 * static_cast<double>(pass_stage0) / static_cast<double>(in_bounds) : 0.0;

    std::fprintf(stdout, "\n=== Stage-0 only (deploy port) ===\n");
    std::fprintf(stdout, "  total events      : %zu\n", total);
    std::fprintf(stdout, "  warmup events     : %zu\n", warmup);
    std::fprintf(stdout, "  measured events   : %zu\n", measured_n);
    std::fprintf(stdout, "  out-of-bounds     : %zu\n", oob_count);
    std::fprintf(stdout, "  steady-state win  : [%zu, %zu) (5%%-trim each side)\n", lo, hi);
    std::fprintf(stdout, "  wall (timed loop) : %.2f ms\n", wall_ns * 1e-6);
    std::fprintf(stdout, "  throughput (ss)   : %.3f Mev/s\n", thru_meps);
    if (args.target_mev > 0.0) {
        std::fprintf(stdout, "  target rate       : %.3f Mev/s%s\n",
                     args.target_mev,
                     thru_meps < args.target_mev * 0.95 ? "  ** UNDER **" : "");
    }
    std::fprintf(stdout, "  stage-0 pass rate : %.2f%% (%zu / %zu in-bounds)\n",
                 pass_rate, pass_stage0, in_bounds);
    std::fprintf(stdout, "\n  per-event latency (stage-0 only):\n");
    stats.print("    ");
    std::fprintf(stdout, "\n  per-event distribution (ns log bins):\n");
    deploy::print_histogram(ss, "    ");

    return EXIT_SUCCESS;
}
