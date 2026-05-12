#pragma once

// Single-producer single-consumer bounded ring buffer.
//
// Power-of-two capacity, lock-free, cache-line-isolated head/tail to avoid
// false sharing. Push spins on full; pop returns false on empty. Holds T
// by value — caller is responsible for T's size (we use small POD payloads).
//
// Memory ordering: producer writes data, then `head` (release); consumer
// reads `head` (acquire), then data, then `tail` (release). Producer
// reads `tail` (acquire) for full-check.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace deploy {

#ifdef __cpp_lib_hardware_interference_size
constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#else
constexpr std::size_t kCacheLine = 64;
#endif

template <typename T>
class SpscRing {
public:
    explicit SpscRing(std::size_t capacity_pow2) : cap_(capacity_pow2), mask_(cap_ - 1) {
        // require power of two
        if (cap_ < 2 || (cap_ & mask_) != 0) {
            throw std::runtime_error("SpscRing: capacity must be power of two ≥ 2");
        }
        buf_.resize(cap_);
    }

    // Try push. Returns false if full.
    bool try_push(const T& v) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        if (h - t >= cap_) return false;
        buf_[h & mask_] = v;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    // Push, spinning on full.
    void push(const T& v) {
        while (!try_push(v)) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
    }

    // try_push variant exposed for callers that must never block on full
    // (avoids the producer/consumer livelock when one thread does both
    // pushes and pops on different queues — see s3_sharded.cpp sum thread).
    bool try_push_nb(const T& v) { return try_push(v); }

    // Pop. Returns false if empty.
    bool try_pop(T& out) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        const std::size_t h = head_.load(std::memory_order_acquire);
        if (h == t) return false;
        out = buf_[t & mask_];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    // Approximate occupancy (snapshot, may race).
    std::size_t size() const {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        return h - t;
    }

    std::size_t capacity() const { return cap_; }

private:
    const std::size_t cap_;
    const std::size_t mask_;
    std::vector<T>    buf_;
    alignas(kCacheLine) std::atomic<std::size_t> head_{0};
    alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
};

}  // namespace deploy
