#pragma once

// Per-event latency statistics with tail-aware percentiles.
//
// Differs from cpp/src/timer.h::LatencyStats by reporting p99.9 / max in
// addition to p99, since SSLA's tail latency (events that survive all
// 4 tdrops and traverse all stages) is the metric we actually care about
// for real-time deployment.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace deploy {

struct LatStats {
    std::size_t n      = 0;
    double      mean_ns = 0.0;
    double      p50_ns  = 0.0;
    double      p90_ns  = 0.0;
    double      p99_ns  = 0.0;
    double      p99_9_ns = 0.0;
    double      max_ns  = 0.0;

    static LatStats from_samples(std::vector<double> ns_samples) {
        LatStats s;
        s.n = ns_samples.size();
        if (s.n == 0) return s;
        std::sort(ns_samples.begin(), ns_samples.end());
        s.mean_ns = std::accumulate(ns_samples.begin(), ns_samples.end(), 0.0) / s.n;
        auto pct = [&](double p) -> double {
            if (s.n == 1) return ns_samples[0];
            const double idx = p * (s.n - 1);
            const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
            const std::size_t hi = std::min(s.n - 1, lo + 1);
            const double frac = idx - lo;
            return ns_samples[lo] * (1.0 - frac) + ns_samples[hi] * frac;
        };
        s.p50_ns   = pct(0.50);
        s.p90_ns   = pct(0.90);
        s.p99_ns   = pct(0.99);
        s.p99_9_ns = pct(0.999);
        s.max_ns   = ns_samples.back();
        return s;
    }

    void print(const char* prefix = "") const {
        auto fmt = [](double ns) -> const char* {
            static thread_local char buf[32];
            if      (ns >= 1e6) std::snprintf(buf, sizeof(buf), "%.3f ms", ns * 1e-6);
            else if (ns >= 1e3) std::snprintf(buf, sizeof(buf), "%.2f us", ns * 1e-3);
            else                std::snprintf(buf, sizeof(buf), "%.1f ns", ns);
            return buf;
        };
        std::fprintf(stdout, "%s n        : %zu\n",     prefix, n);
        std::fprintf(stdout, "%s mean     : %s\n",      prefix, fmt(mean_ns));
        std::fprintf(stdout, "%s p50      : %s\n",      prefix, fmt(p50_ns));
        std::fprintf(stdout, "%s p90      : %s\n",      prefix, fmt(p90_ns));
        std::fprintf(stdout, "%s p99      : %s\n",      prefix, fmt(p99_ns));
        std::fprintf(stdout, "%s p99.9    : %s\n",      prefix, fmt(p99_9_ns));
        std::fprintf(stdout, "%s max      : %s\n",      prefix, fmt(max_ns));
    }
};

// Bin-and-print helper for the full distribution: log-spaced bins from
// 100 ns to 10 ms. Useful to see whether the long tail is bimodal
// (early-drop events vs full-traversal events).
inline void print_histogram(const std::vector<double>& ns_samples,
                            const char* prefix = "  hist  ") {
    if (ns_samples.empty()) return;
    static const double edges[] = {
        100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 1e7
    };
    constexpr int NB = sizeof(edges) / sizeof(edges[0]) - 1;
    std::vector<std::size_t> bins(NB, 0);
    std::size_t under = 0, over = 0;
    for (double v : ns_samples) {
        if (v < edges[0])  { ++under; continue; }
        if (v >= edges[NB]) { ++over;  continue; }
        for (int i = 0; i < NB; ++i) {
            if (v >= edges[i] && v < edges[i+1]) { ++bins[i]; break; }
        }
    }
    auto pct_of = [&](std::size_t c) {
        return 100.0 * c / ns_samples.size();
    };
    auto fmt_edge = [](double ns, char* buf, std::size_t bs) {
        if      (ns >= 1e6) std::snprintf(buf, bs, "%.0fms", ns * 1e-6);
        else if (ns >= 1e3) std::snprintf(buf, bs, "%.0fus", ns * 1e-3);
        else                std::snprintf(buf, bs, "%.0fns", ns);
    };
    char lo_s[16], hi_s[16];
    if (under) {
        fmt_edge(edges[0], lo_s, sizeof(lo_s));
        std::fprintf(stdout, "%s< %-7s   : %zu (%.2f%%)\n",
                     prefix, lo_s, under, pct_of(under));
    }
    for (int i = 0; i < NB; ++i) {
        if (bins[i] == 0) continue;
        fmt_edge(edges[i],   lo_s, sizeof(lo_s));
        fmt_edge(edges[i+1], hi_s, sizeof(hi_s));
        char range[40];
        std::snprintf(range, sizeof(range), "%s..%s", lo_s, hi_s);
        std::fprintf(stdout, "%s%-12s : %zu (%.2f%%)\n",
                     prefix, range, bins[i], pct_of(bins[i]));
    }
    if (over) {
        fmt_edge(edges[NB], hi_s, sizeof(hi_s));
        std::fprintf(stdout, "%s>= %-6s   : %zu (%.2f%%)\n",
                     prefix, hi_s, over, pct_of(over));
    }
}

}  // namespace deploy
