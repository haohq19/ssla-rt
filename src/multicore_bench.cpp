// multicore_bench.cpp — SSLA multicore throughput PoC.
//
// Goal: answer "can SSLA-S sustain 10 MHz event throughput on this CPU
// when we shard work across cores?". Builds outside the main cpp/ tree;
// links libopeneva_core.a as a black box.
//
// Two sharding modes:
//
//   --mode replicate   Each thread runs the FULL event stream on its own
//                      independent SSLAModel. Pure throughput-scaling
//                      upper bound; tells us how cache and memory
//                      contention scale. NOT a valid sharded inference.
//
//   --mode strip       Vertical-strip partition by x-column. Worker k
//                      handles events with x in [k*W/N, (k+1)*W/N).
//                      Each worker keeps a full-resolution SSLAModel
//                      (so weights / strides are unchanged), but only
//                      gets fed events in its strip. **Approximate**:
//                      events near strip boundaries do not propagate
//                      patch updates into neighbouring strips. Real
//                      sharded inference would need a halo protocol
//                      (TODO). Use this mode for throughput numbers
//                      only.
//
// Reported metric: events/sec on the slowest worker (= overall pipeline
// throughput). Per-event mean latency is computed from wall-time / events
// per worker; we don't time individual events because the steady_clock
// call costs ~40 ns and would distort the numbers we care about.

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
#include "openeva/runtime_model.h"
#include "src/io.h"

namespace {

struct Args {
    std::string method      = "ssla";
    std::string weights_dir;
    std::string events_path;          // if empty -> generate random
    std::string mode        = "strip";
    int         threads     = 1;
    int         warmup      = 10000;
    int         random_n    = 1000000;
    int         random_h    = 240;
    int         random_w    = 304;
    std::uint32_t random_seed = 1;
    bool        pin_cores   = true;
};

void die_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --weights <dir> [--events <npy>] [--threads N]\n"
        "          [--mode strip|replicate] [--warmup K]\n"
        "          [--random-n N --random-hw H W --random-seed S]\n"
        "          [--no-pin]\n", argv0);
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
        if (auto v = eat("--weights"))     { a.weights_dir = v; continue; }
        if (auto v = eat("--events"))      { a.events_path = v; continue; }
        if (auto v = eat("--mode"))        { a.mode        = v; continue; }
        if (auto v = eat("--threads"))     { a.threads     = std::atoi(v); continue; }
        if (auto v = eat("--warmup"))      { a.warmup      = std::atoi(v); continue; }
        if (auto v = eat("--random-n"))    { a.random_n    = std::atoi(v); continue; }
        if (auto v = eat("--random-seed")) { a.random_seed = static_cast<std::uint32_t>(std::atoi(v)); continue; }
        if (std::strcmp(argv[i], "--random-hw") == 0) {
            if (i + 2 >= argc) die_usage(argv[0]);
            a.random_h = std::atoi(argv[++i]);
            a.random_w = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--no-pin") == 0) { a.pin_cores = false; continue; }
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) die_usage(argv[0]);
        std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
        die_usage(argv[0]);
    }
    if (a.weights_dir.empty()) die_usage(argv[0]);
    if (a.threads <= 0)        a.threads = 1;
    if (a.mode != "strip" && a.mode != "replicate") {
        std::fprintf(stderr, "--mode must be 'strip' or 'replicate'\n");
        std::exit(EXIT_FAILURE);
    }
    return a;
}

// Generate random events sorted by t with monotonically nondecreasing
// timestamps and unique-pixel-time pairs (so SSLA's same-pixel
// dt_norm path is exercised).
std::vector<openeva::Event>
make_random_events(int n, int h, int w, std::uint32_t seed) {
    std::vector<openeva::Event> ev(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dx(0, w - 1);
    std::uniform_int_distribution<int> dy(0, h - 1);
    std::uniform_int_distribution<int> dp(0, 1);
    for (int i = 0; i < n; ++i) {
        ev[i].t = static_cast<float>(i);     // 1 µs / event (1 MHz nominal)
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

// Estimate effective sensor bounds from events (used for strip partition).
struct SensorBounds {
    int max_x;
    int max_y;
};

SensorBounds bounds_of(const std::vector<openeva::Event>& ev) {
    int mx = 0, my = 0;
    for (const auto& e : ev) {
        if (static_cast<int>(e.x) > mx) mx = static_cast<int>(e.x);
        if (static_cast<int>(e.y) > my) my = static_cast<int>(e.y);
    }
    return {mx, my};
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // ---- 1. Load events --------------------------------------------------
    std::vector<openeva::Event> events;
    if (!args.events_path.empty()) {
        events = openeva::load_events_npy(args.events_path);
    } else {
        events = make_random_events(args.random_n, args.random_h, args.random_w,
                                    args.random_seed);
    }
    if (events.empty()) {
        std::fprintf(stderr, "no events to process\n");
        return EXIT_FAILURE;
    }
    SensorBounds sb = bounds_of(events);
    std::fprintf(stderr,
        "[bench] events=%zu max_x=%d max_y=%d mode=%s threads=%d warmup=%d\n",
        events.size(), sb.max_x, sb.max_y, args.mode.c_str(),
        args.threads, args.warmup);

    // ---- 2. Per-thread event slices -------------------------------------
    //
    // - replicate: every thread sees every event (full stream replayed).
    // - strip    : strip k owns x in [k*W/N, (k+1)*W/N).
    //              We slice ONCE up front so the timed loop is just a
    //              tight `for (auto& e : my_slice) model->step(e)`.
    const int N = args.threads;
    std::vector<std::vector<openeva::Event>> slices(N);

    if (args.mode == "replicate") {
        for (int k = 0; k < N; ++k) slices[k] = events;
    } else {
        const int sensor_w = sb.max_x + 1;
        const int strip_w  = (sensor_w + N - 1) / N;
        for (auto& s : slices) s.reserve(events.size() / N + 16);
        for (const auto& e : events) {
            const int xi = static_cast<int>(e.x);
            const int k  = std::min(N - 1, xi / strip_w);
            slices[k].push_back(e);
        }
    }

    // Decide warmup count per thread (proportional to slice size).
    std::vector<int> warmup_per_thread(N);
    for (int k = 0; k < N; ++k) {
        const std::size_t s = slices[k].size();
        const std::size_t w = std::min<std::size_t>(args.warmup, s / 4);
        warmup_per_thread[k] = static_cast<int>(w);
    }

    // ---- 3. Spawn workers -----------------------------------------------
    //
    // Each worker:
    //   1. pins itself
    //   2. constructs + loads SSLAModel
    //   3. resets, processes its warmup slice (untimed)
    //   4. waits at barrier_start, processes measured slice, signals done
    //
    // Wall time = max(per-thread measured wall) — the slowest worker
    // gates pipeline throughput.

    std::atomic<int>     ready{0};
    std::atomic<int>     done{0};
    std::atomic<bool>    go{false};
    std::vector<double>  measured_ns(N, 0.0);
    std::vector<std::size_t> measured_evs(N, 0);
    std::mutex           err_mu;
    std::vector<std::string> errors(N);

    auto worker = [&](int k) {
        try {
            if (args.pin_cores) pin_to_core(k);

            auto model = openeva::create_model(args.method);
            model->load_weights(args.weights_dir);
            model->reset();

            const auto& slice = slices[k];
            const int   wn    = warmup_per_thread[k];

            // Warmup
            for (int i = 0; i < wn; ++i) model->step(slice[i]);

            // Sync: signal ready, spin on `go`.
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            // Measured loop — tight, no per-event timing.
            const auto t0 = std::chrono::steady_clock::now();
            for (std::size_t i = wn; i < slice.size(); ++i) {
                model->step(slice[i]);
            }
            const auto t1 = std::chrono::steady_clock::now();

            measured_ns[k]  = std::chrono::duration<double, std::nano>(t1 - t0).count();
            measured_evs[k] = slice.size() - wn;
        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lk(err_mu);
            errors[k] = ex.what();
        }
        done.fetch_add(1, std::memory_order_release);
    };

    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int k = 0; k < N; ++k) threads.emplace_back(worker, k);

    // Wait for all workers to finish warmup.
    while (ready.load(std::memory_order_acquire) < N) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Release all workers simultaneously.
    go.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();

    // Surface any worker errors before reporting.
    for (int k = 0; k < N; ++k) {
        if (!errors[k].empty()) {
            std::fprintf(stderr, "[worker %d] ERROR: %s\n", k, errors[k].c_str());
            return EXIT_FAILURE;
        }
    }

    // ---- 4. Aggregate ----------------------------------------------------
    double max_ns       = 0.0;
    std::size_t total_evs = 0;
    for (int k = 0; k < N; ++k) {
        if (measured_ns[k] > max_ns) max_ns = measured_ns[k];
        total_evs += measured_evs[k];
    }
    // Throughput is governed by the slowest worker. In strip mode that's
    // the partition with the most events; in replicate mode all slices
    // are equal.
    const double max_s     = max_ns * 1e-9;
    const double thru_meps = (max_s > 0.0) ? (total_evs / max_s) * 1e-6 : 0.0;
    const double per_ev_ns_mean = total_evs > 0 ? max_ns / total_evs : 0.0;

    // Per-thread breakdown
    std::fprintf(stdout, "\nper-thread:\n");
    std::fprintf(stdout, "  %-3s %-10s %-12s %-14s\n",
                 "id", "events", "wall (ms)", "ns/event");
    for (int k = 0; k < N; ++k) {
        const double per_ev = measured_evs[k] > 0
            ? measured_ns[k] / measured_evs[k] : 0.0;
        std::fprintf(stdout, "  %-3d %-10zu %-12.2f %-14.1f\n",
                     k, measured_evs[k], measured_ns[k] * 1e-6, per_ev);
    }

    std::fprintf(stdout, "\nsummary:\n");
    std::fprintf(stdout, "  mode              : %s\n",  args.mode.c_str());
    std::fprintf(stdout, "  threads           : %d\n",  N);
    std::fprintf(stdout, "  total events done : %zu\n", total_evs);
    std::fprintf(stdout, "  wall (slowest)    : %.2f ms\n", max_ns * 1e-6);
    std::fprintf(stdout, "  throughput        : %.2f Mev/s\n", thru_meps);
    std::fprintf(stdout, "  ns/event (eff)    : %.1f\n", per_ev_ns_mean);
    std::fprintf(stdout, "  10MHz target      : %s (%.2fx)\n",
                 thru_meps >= 10.0 ? "MET" : "miss",
                 thru_meps / 10.0);

    return EXIT_SUCCESS;
}
