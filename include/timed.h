#pragma once

// Per-event high-resolution timing. Uses rdtsc on x86 for the lowest
// overhead measurement (~5-10 cycles per call) so per-event timing
// itself doesn't perturb the latency distribution we're measuring.
//
// The TSC frequency is calibrated once against std::steady_clock at
// program start; subsequent timestamps convert to nanoseconds by
// multiplication.

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace deploy {

inline std::uint64_t rdtsc_now() {
#if defined(__x86_64__)
    unsigned int aux;
    return __rdtscp(&aux);   // serializing flavour; ~25 cycles
#else
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

class TscClock {
public:
    static TscClock& instance() {
        static TscClock c;
        return c;
    }

    double tsc_to_ns(std::uint64_t delta_tsc) const {
        return static_cast<double>(delta_tsc) * ns_per_tick_;
    }

private:
    TscClock() {
        // 50 ms calibration window.
        const auto wall0 = std::chrono::steady_clock::now();
        const auto t0    = rdtsc_now();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto t1    = rdtsc_now();
        const auto wall1 = std::chrono::steady_clock::now();
        const double ns =
            std::chrono::duration<double, std::nano>(wall1 - wall0).count();
        const double ticks = static_cast<double>(t1 - t0);
        ns_per_tick_ = ns / ticks;
    }
    double ns_per_tick_;
};

}  // namespace deploy
