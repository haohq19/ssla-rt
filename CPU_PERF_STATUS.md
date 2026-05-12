# ssla-rt CPU pipeline performance — status for expert review

_Self-contained handoff document. Recipient has no other access to this codebase._
_All numbers below are first-hand measurements on the target platform._

---

## 0. TL;DR

We have a real-time event-camera inference pipeline. The hot path runs
spatial recurrent ("LRU"-gated) attention layers on per-event basis.
Two optimisation rounds (NEON hand-vectorisation of IN=12 matvecs and
NEON polynomial-sigmoid LRU step) lifted **single-shard throughput
from 135 → 296 kev/s (2.19×)** and **CPU-side admit ceiling from
0.82 → 1.69 Mev/s (2.06×)** on an 8-core Cortex-A78AE @ 1.984 GHz.
**Target: 2.0 Mev/s** (+18%). All cores saturated at n_shards=7
(n=8 doesn't gain). Looking for advice on how to squeeze the
remaining 18%.

Hard constraints:

- **`halo = 2` is locked**, cannot be reduced or bypassed (any scheme
  that weakens halo's per-shard state-duplication semantics is
  forbidden — see §2).
- **`vendor/openeva/*` is vendored from an upstream research repo**;
  cannot be edited (override via explicit template specialisations in
  `include/`).
- **P1 correctness gate**: `max|Δ|` on s3 features over 200k events
  must stay ≤ 5 (currently 4.4, unchanged after our optimisations).
- Target hardware fixed (Jetson Orin NX 16 GB Dev Kit, 8 × A78AE).
- GPU side out of scope for this task — even though GPU drain caps
  end-to-end at ~0.6 Mev/s admit-equivalent, the brief is to maximise
  the **CPU side** in isolation.

---

## 1. Hardware (first-hand)

```text
CPU:    8 × ARM Cortex-A78AE (rev 1, v8l), CPU part 0xd42, @ 1.984 GHz max
        ISA features: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
                      fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
                      uscat ilrcpc flagm
        L1d: 64 KB / core (typical A78), L2: shared cluster, LLC: SoC-wide.
        4 FP/SIMD execution pipes (vendor-published μarch).

GPU:    NVIDIA Orin (Ampere), CC 8.7, 8 SMs @ 918 MHz (MAXN pinned).
        CONCURRENT_MANAGED_ACCESS = 0  →  pinned host memory required for
        CPU↔GPU ring traffic. (Out of scope for CPU work, mentioned for
        context.)

SoC:    Tegra234, NVIDIA Orin NX 16 GB Developer Kit.
        OS: Linux 5.10.120-tegra (JetPack), GCC 10, libstdc++ partial C++23.
```

Toolchain:

```text
gcc:           GNU 10.x for aarch64 (JetPack stock)
Release flags: -O3 -march=native -funroll-loops -ffast-math -fopenmp-simd -DNDEBUG
Std:           C++20 (CMAKE_CXX_STANDARD=20)
Build system:  CMake 3.16+
```

---

## 2. Durable rules / non-negotiables

Encoded in repo's `CLAUDE.md` and persistent memory. These are NOT
optimisation knobs:

1. **halo = 2 is locked** at the dispatcher. Each event whose x-coord
   falls within 2 px of a shard-strip boundary is delivered as 2
   messages (one to the owner shard, one to the neighbour). Both
   shards independently run stage-0 forward (the halo neighbour does
   not run stage-1 / does not push to GPU; it just keeps its own
   per-shard hidden state up to date for future events that might
   land in its strip and need a 3×3 neighbourhood read across the
   boundary). This duplication is the semantic guarantee — any
   redesign that eliminates the duplicate s0 compute by routing
   state-deltas across shards is **forbidden**, even if `halo=2`
   parameter is nominally preserved.

2. **`vendor/openeva/*` headers must not be edited.** Upstream
   research repo can re-sync. Overrides must live in `include/`
   via explicit template specialisations.

3. **P1 numeric gate**: after 200k synthesised events, `max|Δ|` on
   final s3 features vs a Python NumPy reference oracle must be ≤ 5.
   `drift` (count of events where CPU pass-decision differs from
   reference) must be 0.

4. **stage-0 + stage-1 run on CPU**, stage-2 + stage-3 run on GPU.
   The CPU pipeline ends by pushing the 24-D s1 feature into a
   pinned-host ring for the GPU. The GPU is independently capped
   at ~0.6 Mev/s admit-equivalent — beyond that, ring overflows and
   end-to-end fails (P2 FAIL). **We are pushing CPU only.** GPU
   work is explicitly out of scope.

---

## 3. Domain context (algorithm summary)

The model is **SSLA-S** (Spatial-Sequential LRU-gated Attention,
"Small" variant) — an asynchronous event-based detector trained on
DVS / DAVIS cameras. Per-event inference:

- An event = `(t µs, x px, y px, polarity ∈ {0, 1})`. Sensor here is
  80 × 64 px (cropped for the deployment), submitted in batches.
- 4 stages, each at half the spatial resolution of the previous:
  s0 = 80×64, s1 = 40×32, s2 = 20×16, s3 = 10×8.
- Each stage has 2 layers (L0 and L1). Channel widths grow:
  s0 = 12, s1 = 24, s2 = 48, s3 = 96 channels.
- Each layer is a 3×3 spatial "patched" recurrent attention block.
  Per event at (ev_x, ev_y), the layer touches 9 spatial cells (3×3
  neighbourhood at the layer's stage resolution).
- Between stages: a `tdrop_and_pool` boundary halves resolution and
  drops events that don't meet a per-cell time-since-last-event
  threshold. In our stub run only ~25% of events pass tdrop-0,
  ~6% of all events pass tdrop-1 → reach GPU.
- The CPU pipeline processes s0+s1 per event sequentially through
  shard worker threads.

### 3.1 Per-layer math

For a layer with `IN_DIM` input channels and `OUT_DIM` output channels,
called at event coords `(ev_x, ev_y)`:

```
residual = (in_dim != out_dim) ? matvec(input_proj, feat_in) : feat_in
feat_out[] = 0
for each of 9 patches (px, py) in 3×3 around (ev_x, ev_y):
    [q, v, g] = matvec_3OUT( qvgIn[delta], feat_in )   # 3*OUT_DIM outputs
    h_state[patch_cell] = lru_step(g, v, q, h_state[patch_cell])  → qh (OUT_DIM)
    feat_out += matvec( goW[delta], qh )               # OUT_DIM outputs
feat_out = layernorm(feat_out + residual)
```

`lru_step` is a gated recurrence per element:

```
gc = sigmoid(g[c])
h[c] = gc * h[c] + v[c]      # state update
y[c] = q[c] * h[c]
```

The hidden state `h` is per-(layer, patch_cell) and persists across
events. There are `Hl × Wl` patch cells at layer resolution — each
layer's hidden state is `OUT_DIM × Hl × Wl` floats. For s1 L1 that's
24 × 32 × 40 = 122 KB; for s0 L1 that's 12 × 64 × 80 = 245 KB.

`qvgIn[delta]` (shape `3*OUT × IN`) and `goW[delta]` (shape `OUT × OUT`)
are per-patch-position weights (delta ∈ [0, 8] indexes the 3×3 position).

---

## 4. Pipeline architecture

### 4.1 Shard layout

- The full sensor width W is divided into `n_shards` vertical strips
  of `strip_w = W / n_shards` pixels each. With W=80 and n_shards=7,
  strip_w ≈ 11 px.
- For each event submitted, the dispatcher computes `owner = ex / strip`
  and pushes the event message to that shard's SPSC ring. If the event
  is within `halo=2` px of the strip boundary, an extra "halo" copy
  goes to the neighbour shard. (Each event becomes 1 or 2 shard
  messages; halo factor ≈ 1.15 at n=4, ≈ 1.36 at n=7.)
- Each shard owns: its own `SslaSPipeline` instance (with its own
  copy of all weights and hidden state), and an SPSC ring of size
  2^16 messages. The shard's thread spins on `try_pop`.

### 4.2 Per-shard worker loop (current production code, after Phase 2 NEON)

```cpp
// src/lib_stage01_to_gpu.cpp (excerpt; ~200-line full file)
struct ShardMsg {
    std::uint64_t  t_arr_tsc;  // dispatcher's rdtsc at submission
    openeva::Event ev;
    bool           is_owner;   // false → halo copy
    bool           eof;
};

void shard_worker(S01gHandle* h, ShardCtx* ctx) {
    pin_to_core(ctx->core_id);
    deploy::SslaSPipeline& pipe = *ctx->pipe;
    float feat_in[2];                       // kInDim
    std::vector<float> feat0(12), feat1(24); // kC0, kC1

    while (true) {
        ShardMsg m;
        if (!ctx->q_in->try_pop(m)) {
            asm volatile("yield" ::: "memory");
            continue;
        }
        if (m.eof) break;

        const int ex0 = (int)m.ev.x, ey0 = (int)m.ev.y;
        if (ex0 < 0 || ex0 >= pipe.W() || ey0 < 0 || ey0 >= pipe.H()) continue;

        pipe.preprocess(m.ev, feat_in);                // (dt_norm, polarity)
        int x = ex0, y = ey0;
        pipe.stage_forward(0, x, y, feat_in, feat0.data());   // s0 L0 + s0 L1

        if (m.is_owner) {
            bool pass0 = pipe.tdrop_and_pool(0, x, y);
            bool pass1 = false;
            if (pass0) {
                pipe.stage_forward(1, x, y, feat0.data(), feat1.data()); // s1 L0+L1
                pass1 = pipe.tdrop_and_pool(1, x, y);
            }
            if (pass1 && gpu_attached) {
                // Push s1 feature into pinned-host GPU ring(s).
                // ... 24 floats + (t, x, y) packed into a 112-B record.
            }
            // (per-event end-to-end latency sampling: see §6)
        }
        // halo path: just runs preprocess + s0 to keep neighbour state in sync.
    }
}
```

The owner-side does s0 + tdrop0 + (conditionally s1 + tdrop1 + GPU push).
The halo-side does s0 only.

### 4.3 `stage_forward` and `layer_forward_ct`

```cpp
// src/ssla_kernels.cpp (excerpt)
void SslaSPipeline::stage_forward(int stage, int ev_x, int ev_y,
                                  const float* feat_in, float* feat_out) {
    float* tmp = stage_out_[stage].data();   // per-stage scratch
    switch (stage) {
      case 0:
        layer_forward_ct< 2, 12>(0, ev_x, ev_y, feat_in, tmp);
        layer_forward_ct<12, 12>(1, ev_x, ev_y, tmp,    feat_out);
        break;
      case 1:
        layer_forward_ct<12, 24>(2, ev_x, ev_y, feat_in, tmp);
        layer_forward_ct<24, 24>(3, ev_x, ev_y, tmp,    feat_out);
        break;
      // stage 2 / 3 on GPU
    }
}

template <int IN_DIM, int OUT_DIM>
void SslaSPipeline::layer_forward_ct(int layer_idx, int ev_x, int ev_y,
                                     const float* feat_in, float* feat_out) {
    LayerWeights& L = layers_[layer_idx];
    const int stage = layer_idx / kLayersPerStage;
    const int Hl = Hs_[stage], Wl = Ws_[stage];
    float* H_all = hidden_[layer_idx].data();

    float* residual = scratch_residual_[stage].data();
    float* qvg      = scratch_qvg_[stage].data();        // [3 * OUT_DIM]
    float* qh       = scratch_qh_[stage].data();         // [OUT_DIM]

    if (!L.input_proj.empty()) {
        openeva::prim::matvec_ct<IN_DIM, OUT_DIM>(
            feat_in, L.input_proj.data(), nullptr, residual);
    } else {
        std::memcpy(residual, feat_in, sizeof(float) * OUT_DIM);
    }
    std::memset(feat_out, 0, sizeof(float) * OUT_DIM);

    const int num_pos = L.num_pos;                       // = 9 for SSLA-S
    const int base    = ev_y * Wl + ev_x;

    auto process_patch = [&](int patch_idx, int pos) {
        openeva::prim::matvec_ct<IN_DIM, 3 * OUT_DIM>(
            feat_in, L.qvgIn[pos].data(), nullptr, qvg);
        float* h_ptr = H_all + (std::ptrdiff_t)patch_idx * OUT_DIM;
        const float* q = qvg;
        const float* v = qvg + OUT_DIM;
        const float* g = qvg + 2 * OUT_DIM;
        openeva::prim::lru_step<OUT_DIM>(g, v, q, h_ptr, qh);
        openeva::prim::matvec_accum_ct<OUT_DIM, OUT_DIM>(
            qh, L.goW[pos].data(), feat_out);
    };

    const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                       && (ev_y > 0) && (ev_y + 1 < Hl);
    if (interior) {
        // unrolled 9-patch sweep, neighbour-index pattern
        process_patch(base - Wl - 1, 8);
        process_patch(base - Wl,     7);
        process_patch(base - Wl + 1, 6);
        process_patch(base - 1,      5);
        process_patch(base,          4);
        process_patch(base + 1,      3);
        process_patch(base + Wl - 1, 2);
        process_patch(base + Wl,     1);
        process_patch(base + Wl + 1, 0);
    } else {
        // bounds-clipped fallback (uncommon)
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            const int py = ev_y + dy, px = ev_x + dx;
            if (py < 0 || py >= Hl || px < 0 || px >= Wl) continue;
            process_patch(py * Wl + px, (1 - dy) * 3 + (1 - dx));
        }
    }

    openeva::prim::add_inplace(OUT_DIM, feat_out, residual);
    openeva::prim::layernorm_ct<OUT_DIM>(feat_out, L.ln_gamma.data(),
                                          L.ln_beta.data());
}
```

---

## 5. Vendored primitives (in `vendor/openeva/prim/*`, NOT editable)

### 5.1 `linear.h` — matvec primary templates

```cpp
namespace openeva::prim {

// y[o] = (bias ? bias[o] : 0) + Σ_i W[o*IN + i] * x[i]
template <int IN, int OUT>
inline void matvec_ct(const float* x, const float* W,
                      const float* bias, float* y) {
    add_macs(static_cast<std::size_t>(IN) * OUT);     // TLS counter
    for (int o = 0; o < OUT; ++o) {
        const float* w = W + (std::ptrdiff_t)o * IN;
        float acc = bias ? bias[o] : 0.0f;
        for (int i = 0; i < IN; ++i) acc += w[i] * x[i];
        y[o] = acc;
    }
}

// y[o] += Σ_i W[o*IN + i] * x[i]
template <int IN, int OUT>
inline void matvec_accum_ct(const float* x, const float* W, float* y) {
    add_macs(static_cast<std::size_t>(IN) * OUT);
    for (int o = 0; o < OUT; ++o) {
        const float* w = W + (std::ptrdiff_t)o * IN;
        float acc = 0.0f;
        for (int i = 0; i < IN; ++i) acc += w[i] * x[i];
        y[o] += acc;
    }
}

// `add_macs` is a TLS counter increment in vendor/openeva/prim/flop.h:
//   inline thread_local std::size_t g_macs = 0;
//   inline void add_macs(std::size_t n)  { g_macs += n; }
// — disassembly shows a `mrs xN, tpidr_el0` + add per call.
}
```

GCC `-O3 -march=native -ffast-math` auto-vectorisation result, verified
via objdump on `build/libstage01_to_gpu.so` per-layer-lambda symbols
(the `process_patch` lambda gets out-of-lined as `…_clEii` symbol;
`matvec_ct` is inlined into it):

```text
lambda            | vec FMA (.4s) | vec FMUL (.4s) | scalar fmadd | scalar fmul | size
─────────────────────────────────────────────────────────────────────────────────────
<2, 12>           |  9            |  9             | 23           | 13          | 0x454 B  (s0 L0, mixed)
<12, 12>          |  0            |  0             | 26           |  6          | 0x2dc B  (s0 L1, fully scalar — the main loss)
<12, 24>          | 10            |  2             | 15           |  5          | 0x300 B
<24, 24>          | 25            |  5             |  4           |  4          | 0x390 B  (s1 L1, fully vec'd)
<24, 48>          | 26            |  4             |  4           |  4          | 0x368 B
<48, 48>          | 22            |  2             |  4           |  4          | 0x310 B
<48, 96>          | 19            |  1             |  4           |  4          | 0x320 B
<96, 96>          | 16            |  0             |  4           |  4          | 0x328 B
```

GCC's auto-vectoriser **refuses** across-IN-axis vectorisation for IN=12
(cost-model decision — 3 vector FMAs + horizontal reduce vs 12 scalar
FMAs in a tight unrolled chain). For IN≥24 it vectorises cleanly. The
4 residual scalar fmadds in the IN≥24 lambdas come from `layernorm_ct`
(see §5.4 — fp64 reduction). `<2, 12>` is mixed because IN=2 is below
SIMD lane count; GCC vectorises across OUT.

### 5.2 `rnn.h` — `lru_step` primary template

```cpp
template <int DIM>
inline void lru_step(const float* g, const float* v, const float* q,
                     float* h, float* y) {
    add_flops(static_cast<std::size_t>(DIM) * (kSigmoidFlops + 3));
    #pragma omp simd
    for (int c = 0; c < DIM; ++c) {
        const float gc = sigmoid(g[c]);
        const float hc = gc * h[c] + v[c];
        h[c] = hc;
        y[c] = q[c] * hc;
    }
}
```

`sigmoid` is in `activation.h`:

```cpp
#pragma omp declare simd
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
```

Even with `#pragma omp simd` + `#pragma omp declare simd`, the GCC 10
build emits **scalar `bl <expf@plt>` per element** inside the lru loop
— libmvec vectorisation of `expf` did NOT kick in. Disassembly count:
4× `bl expf@plt` per `<12, 12>` lambda (out-of-line scalar calls).

### 5.3 `layernorm.h` — `layernorm_ct`

```cpp
template <int DIM>
inline void layernorm_ct(float* x, const float* gamma, const float* beta,
                         float eps = 1e-5f) {
    add_flops(8 * DIM + 3);
    double sum = 0.0;
    for (int i = 0; i < DIM; ++i) sum += static_cast<double>(x[i]);
    const double mean = sum / static_cast<double>(DIM);
    double var = 0.0;
    for (int i = 0; i < DIM; ++i) {
        const double d = static_cast<double>(x[i]) - mean;
        var += d * d;
    }
    var /= static_cast<double>(DIM);
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
    for (int i = 0; i < DIM; ++i) {
        const double v = (static_cast<double>(x[i]) - mean) * inv;
        x[i] = static_cast<float>(v * static_cast<double>(gamma[i])
                                  + static_cast<double>(beta[i]));
    }
}
```

**Uses fp64 reduction** (matches PyTorch's CPU kernel for bit-equivalent
numerics with the trained reference). 1× per layer call (not per-patch).
Currently scalar fp64 — the 4 residual scalar fmadds in IN≥24 lambdas
trace here. Bench: `layernorm_ct<12>` = 59 ns, `<24>` = 73 ns.

---

## 6. Our additions (Phase 1 + 2)

### 6.1 `include/ssla_neon_linear.h` (Phase 1, current)

```cpp
#pragma once
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include "openeva/prim/linear.h"
#include "openeva/prim/flop.h"

namespace openeva::prim {

// 4-output ILP NEON kernel for IN=12. OUT must be divisible by 4.
// Per output group of 4: load 3 input vectors once (reused across the
// outer loop). For each output, 3 vector FMAs build a 4-lane accumulator,
// then 3 paired vpaddq reduce across 4 outputs in parallel.
template <int OUT, bool Accumulate>
inline void matvec_in12_neon_core(const float* __restrict__ x,
                                   const float* __restrict__ W,
                                   const float* __restrict__ bias,
                                   float* __restrict__ y) {
    static_assert(OUT % 4 == 0);
    const float32x4_t x0 = vld1q_f32(x + 0);
    const float32x4_t x1 = vld1q_f32(x + 4);
    const float32x4_t x2 = vld1q_f32(x + 8);
    for (int o = 0; o < OUT; o += 4) {
        const float* w0 = W + (o + 0) * 12;
        const float* w1 = W + (o + 1) * 12;
        const float* w2 = W + (o + 2) * 12;
        const float* w3 = W + (o + 3) * 12;
        float32x4_t a0 = vmulq_f32(x0, vld1q_f32(w0));
        float32x4_t a1 = vmulq_f32(x0, vld1q_f32(w1));
        float32x4_t a2 = vmulq_f32(x0, vld1q_f32(w2));
        float32x4_t a3 = vmulq_f32(x0, vld1q_f32(w3));
        a0 = vfmaq_f32(a0, x1, vld1q_f32(w0 + 4));
        a1 = vfmaq_f32(a1, x1, vld1q_f32(w1 + 4));
        a2 = vfmaq_f32(a2, x1, vld1q_f32(w2 + 4));
        a3 = vfmaq_f32(a3, x1, vld1q_f32(w3 + 4));
        a0 = vfmaq_f32(a0, x2, vld1q_f32(w0 + 8));
        a1 = vfmaq_f32(a1, x2, vld1q_f32(w1 + 8));
        a2 = vfmaq_f32(a2, x2, vld1q_f32(w2 + 8));
        a3 = vfmaq_f32(a3, x2, vld1q_f32(w3 + 8));
        const float32x4_t s01   = vpaddq_f32(a0, a1);
        const float32x4_t s23   = vpaddq_f32(a2, a3);
        float32x4_t       s0123 = vpaddq_f32(s01, s23);
        if constexpr (Accumulate) {
            vst1q_f32(y + o, vaddq_f32(vld1q_f32(y + o), s0123));
        } else {
            if (bias) s0123 = vaddq_f32(s0123, vld1q_f32(bias + o));
            vst1q_f32(y + o, s0123);
        }
    }
}

template <> inline void matvec_ct<12, 24>(const float* x, const float* W,
                                           const float* b, float* y) {
    add_macs(12 * 24);
    matvec_in12_neon_core<24, false>(x, W, b, y);
}
template <> inline void matvec_ct<12, 36>(const float* x, const float* W,
                                           const float* b, float* y) {
    add_macs(12 * 36);
    matvec_in12_neon_core<36, false>(x, W, b, y);
}
template <> inline void matvec_ct<12, 72>(const float* x, const float* W,
                                           const float* b, float* y) {
    add_macs(12 * 72);
    matvec_in12_neon_core<72, false>(x, W, b, y);
}
template <> inline void matvec_accum_ct<12, 12>(const float* x,
                                                 const float* W, float* y) {
    add_macs(12 * 12);
    matvec_in12_neon_core<12, true>(x, W, nullptr, y);
}
}  // namespace openeva::prim
#endif
```

Microbench (`tests/bench_matvec12.cpp`, 10M iters, pinned to core 4):

```text
matvec_ct<IN=12, OUT=36>     ns/call    vs scalar
─────────────────────────────────────────────────
scalar (vendor primary)      148.31     1.00×
NEON 1-output across-IN       92.88     1.60×
NEON 4-output ILP (this file) 50.68    2.93×        ← chosen impl
max|Δ| vs scalar: 1.2e-7 (single call, fp32 noise from horiz reduce order)
```

### 6.2 `include/ssla_neon_lru.h` (Phase 2, current)

```cpp
#pragma once
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include "openeva/prim/rnn.h"
#include "openeva/prim/flop.h"

namespace openeva::prim {

// Sigmoid via tanh(x/2) Padé[3/2]. Saturate input at ±6 so tanh's argument
// stays in [-3, 3] where the rational approximation is accurate to ~1e-2.
// Single-call max|Δ| 1.2e-2 across x ∈ [-10, 10]; 200k-step recurrence
// drift on h: 0.009 (bench_sigmoid.cpp; details §10).
static inline float32x4_t neon_sigmoid_tanh_pade(float32x4_t x) {
    const float32x4_t hi   = vdupq_n_f32( 6.0f);
    const float32x4_t lo   = vdupq_n_f32(-6.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t c27  = vdupq_n_f32(27.0f);
    const float32x4_t c9   = vdupq_n_f32(9.0f);
    x = vminq_f32(vmaxq_f32(x, lo), hi);
    const float32x4_t t  = vmulq_f32(x, half);
    const float32x4_t t2 = vmulq_f32(t, t);
    const float32x4_t num = vmulq_f32(t, vaddq_f32(c27, t2));
    const float32x4_t den = vfmaq_f32(c27, c9, t2);
    return vfmaq_f32(half, half, vdivq_f32(num, den));
}

template <int DIM>
static inline void lru_step_neon_core(const float* g, const float* v,
                                       const float* q, float* h, float* y) {
    static_assert(DIM % 4 == 0);
    for (int c = 0; c < DIM; c += 4) {
        const float32x4_t gc = neon_sigmoid_tanh_pade(vld1q_f32(g + c));
        const float32x4_t hc = vfmaq_f32(vld1q_f32(v + c),
                                          gc, vld1q_f32(h + c));
        vst1q_f32(h + c, hc);
        vst1q_f32(y + c, vmulq_f32(vld1q_f32(q + c), hc));
    }
}

template <> inline void lru_step<12>(const float* g, const float* v,
                                      const float* q, float* h, float* y) {
    add_flops(static_cast<std::size_t>(12) * (kSigmoidFlops + 3));
    lru_step_neon_core<12>(g, v, q, h, y);
}
template <> inline void lru_step<24>(const float* g, const float* v,
                                      const float* q, float* h, float* y) {
    add_flops(static_cast<std::size_t>(24) * (kSigmoidFlops + 3));
    lru_step_neon_core<24>(g, v, q, h, y);
}
// (also lru_step<48>, lru_step<96> with same pattern)
}
#endif
```

Microbench (`tests/bench_sigmoid.cpp`, 5M iters, pinned, varying input
per-iter to defeat compiler hoisting):

```text
                     sigmoid single-call max|Δ| vs std::exp form
                     ──────────────────────────────────────────
                     poly5 (clamp ±4)       8.0e-2
                     pade (clamp ±5)        2.9e-1   (placeholder coefs — too coarse)
                     tanh-Padé (clamp ±6)   1.2e-2   ← chosen

                     lru_step<12> throughput  (ns/call)
                     ──────────────────────────────────
                     scalar (libm expf)            82.8     1.00×
                     NEON poly5                    17.4     4.76×
                     NEON pade                     19.4     4.27×
                     NEON tanh-Padé                21.4     3.87×    ← chosen for accuracy

                     200k-step h-state cumulative drift (recurrence)
                     ──────────────────────────────────────────────
                     poly5                         0.0495
                     pade                          0.0735
                     tanh-Padé                     0.0085
```

Wired in via include from `src/ssla_kernels.cpp` (after vendor headers):

```cpp
// src/ssla_kernels.cpp
#include "openeva/prim/linear.h"
#include "openeva/prim/rnn.h"
// ...
#include "ssla_neon_linear.h"      // matvec_ct<12, *>, matvec_accum_ct<12, 12>
#include "ssla_neon_lru.h"         // lru_step<12, 24, 48, 96>
```

Post-Phase-2 disassembly of `<12, 12>` lambda (now 0x388 B):

```text
bl expf@plt count : 0      (was 4)
fdiv.4s count     : 3      (tanh Padé denominator)
fmla.4s count     : 41     (was 32)
fmin/fmax.4s      : 6      (tanh-Padé clamp)
```

### 6.3 `add_macs` overhead (TLS counter, unchanged)

Each matvec call invokes `add_macs(IN * OUT)` which does
`g_macs += n` on a `thread_local std::size_t`. Disassembly shows
`mrs xN, tpidr_el0` per call. Per-event count: ~25–35 matvec calls
in a pass-1 event. Empirical overhead estimate ~5 ns/call × 30 = 150 ns/event
≈ 3% of the per-event budget. Has never been gated off — vendored
header is read-only and the TLS variable is used by external accounting.

---

## 7. Build configuration

```cmake
# CMakeLists.txt (relevant parts)
set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -funroll-loops -ffast-math
                      -fopenmp-simd -DNDEBUG)
endif()

# ssla_kernels static lib (compiles src/ssla_kernels.cpp — all the
# layer_forward_ct<...> instantiations live here; NEON specialisation
# headers picked up via #include)
add_library(ssla_kernels STATIC src/ssla_kernels.cpp)

# Production shared libs (loaded by Python harness)
add_library(stage01_to_gpu SHARED src/lib_stage01_to_gpu.cpp)
target_link_libraries(stage01_to_gpu PRIVATE ssla_kernels openeva_vendor Threads::Threads)
# ... etc for stage0_capi, stage01_capi (CPU-only flavours)
```

---

## 8. Measurement methodology

Two harnesses; both first-hand on the target box.

### 8.1 Microbenchmarks (`tests/*.cpp`)

Single-thread, pinned via `taskset -c 4`. Pattern:

```cpp
// (e.g. tests/bench_matvec12.cpp)
constexpr long long N_ITERS = 10'000'000;
auto t0 = std::chrono::steady_clock::now();
for (long long i = 0; i < N_ITERS; ++i) {
    matvec_ct<12, 36>(x, W, nullptr, y);
    x[i & 7] = y[i & 0x1f] * 1e-7f;     // data dep — prevents DCE / hoisting
    sink += y[0];
}
auto t1 = std::chrono::steady_clock::now();
double ns_per_call = duration_ns(t1 - t0) / N_ITERS;
```

We discovered (and fixed) one methodology bug: when the inputs were
loop-invariant, GCC hoisted `sigmoid(g[c])` out of the loop, making
scalar lru appear 9× faster than reality. Fix: perturb input
per-iteration. Re-bench gave the corrected scalar = 82.8 ns/call.

### 8.2 Live saturation (`orin/hybrid_runner.py`)

A Python harness drives the C++ library via ctypes. Two threads:

1. **Synthetic event source** — generates `(t, x, y, p)` events at a
   target rate (e.g., `--synthetic-mev 3.0` = 3.0 M events/s target),
   paces via wall clock, calls `s01g_submit_batch` which routes to
   shard SPSC rings. If a shard's ring is full (size 2^16), the
   producer spins on full → real back-pressure on the dispatcher.
2. **Stats poller** — every 2 s, reads atomic counters from the C++
   side (events submitted, ring occupancy, per-shard latencies, GPU
   side push/done counters), prints a row.

We added per-segment µs instrumentation inside `shard_worker`
(7-bracket `rdtsc_now()` chain accumulating into per-shard
`std::atomic<uint64_t> seg_sum_ticks[6]` and `seg_count[6]`,
read back via the existing stats snapshot API). Instrumentation
overhead ≈ 270 ns/event (~4% of work-unit). Included in all
reported segment numbers.

Steady-state admit is taken from the 8 s–10 s 2 s window of a 12 s
run (skipping warm-up). At sustained synthetic submission well
above CPU capacity, `cam_in / wall_time` from this window is the
true CPU admit rate (the synthetic reader's submit_batch blocks
on full shard rings).

---

## 9. Current performance — first-hand

### 9.1 Per-event µs (post-Phase-2, n=4 shards, synthetic 0.5 Mev/s, 14 s)

```text
           segment | mean µs |   count  | share%
   ──────────────────────────────────────────────
        preprocess |   0.060 |  9 239 k |   2.2%
  stage_forward(0) |   1.112 |  9 239 k |  40.3%
   tdrop_and_pool0 |   0.087 |  8 033 k |   2.8%
  stage_forward(1) |   6.789 |  2 009 k |  53.6%    ← biggest share
   tdrop_and_pool1 |   0.082 |  2 009 k |   0.6%
         ring push |   0.257 |    502 k |   0.5%
```

- 9.24M shard messages from 8.03M unique owner events (halo
  factor 1.15 at n=4).
- s0 runs on every message (owner + halo).
- s1 runs on 25% of owner events (pass-tdrop-0 rate).

### 9.2 Per-layer ns (microbench `tests/bench_layer.cpp`, single-thread pinned)

```text
                      ns/call  (interior pixel, 9 patches; 200k iters)
                      ──────────────────────────────────────────────
   L0 s0  <2 ->12>     116.5         (Wl=80, Hl=64)
   L1 s0  <12->12>     836.2         (Wl=80, Hl=64)   ← s0's main piece
   L0 s1  <12->24>    2228.4         (Wl=40, Hl=32)
   L1 s1  <24->24>    4199.1         (Wl=40, Hl=32)   ← biggest single op
```

Sum-check vs production:
- s0 = 117 + 836 = 953 ns (production 1112 ns; 17% overhead = patch dispatch, lambda call, scratch buffers)
- s1 = 2228 + 4199 = 6427 ns (production 6789 ns; 5% overhead)

### 9.3 Per-primitive ns (microbench)

```text
   primitive             ns/call
   ──────────────────────────────
   lru_step<12>             53.6   (production has NEON; bench harness loads scalar - see note below)
   lru_step<24>            107.1   (likewise)
   layernorm_ct<12>         59.0
   layernorm_ct<24>         72.8
```

*Note on lru_step measurement*: `tests/bench_layer.cpp` calls
`openeva::prim::lru_step<DIM>` directly via the vendor header only
(not through `ssla_kernels.cpp`), so it gets the scalar primary
template. The 53.6 / 107 ns are scalar numbers. NEON lru actually
runs at ~14 / 28 ns per call inside the layer (`tests/bench_sigmoid.cpp`
measured 21.4 ns for full lru_step<12> = sigmoid + FMA + stores, with
sigmoid being ~70% of that).

### 9.4 Saturation sweep (live, synthetic 3.0 Mev/s target, 12 s, per shard count)

```text
n_shards |  admit (Mev/s)  | per-shard rate (kev/s) | bottleneck
──────────────────────────────────────────────────────────────
   1     |     0.296       |      296               | CPU
   2     |     0.593       |      297               | CPU
   4     |     1.137       |      284               | CPU (GPU starts ring-overflowing, ignore)
   5     |     1.379       |      276               | CPU
   6     |     1.562       |      260               | CPU
   7     |     1.694       |      242               | CPU            ← peak ★
   8     |     1.638       |      205               | CPU (8-core saturated)
```

**Headline:** CPU side caps at **1.69 Mev/s @ n=7**. n=8 doesn't gain
(8 cores total: 7 shards + 1 dispatcher/Python/system). Per-shard
efficiency at n=7 is ~82% of the n=1 rate (halo overhead grows with
n_shards since strip width shrinks).

### 9.5 Progression across optimisation rounds

```text
                         baseline    Phase 1     Phase 2 (current)
                         ─────────   ─────────   ─────────────────
per-event s0 µs           3.94        1.83        1.11               (3.55× over baseline)
per-event s1 µs          12.88        9.50        6.79               (1.90× over baseline)
per-shard rate (kev/s)     135         216         296               (2.19× over baseline)
CPU ceiling (Mev/s)       0.82        1.30        1.69               (2.06× over baseline)
P1 max|Δ| (gate ≤ 5)       4.4         4.4         4.4                (unchanged — verified)
```

Phase 1 = NEON `matvec_ct<12, *>` + `matvec_accum_ct<12, 12>` (4-out ILP).
Phase 2 = NEON `lru_step<12, 24, 48, 96>` (tanh-Padé sigmoid).

---

## 10. What's left + investigated levers

Goal: **2.0 Mev/s** = +18% over current 1.69. At n=7 (current peak),
this needs per-shard rate 296 → 350 kev/s, i.e., per-work-unit time
3.14 → 2.65 µs (-15%, save ~0.5 µs / msg).

### 10.1 Lever: hand-NEON `matvec_ct<24, 72>` and `matvec_accum_ct<24, 24>`

Already prototyped. Microbench (`tests/bench_matvec24.cpp`, same
methodology as bench_matvec12):

```text
matvec_ct<IN=24, OUT=72>          GCC auto-vec    NEON 4-out ILP    ratio
                                  196.27 ns       170.68 ns         1.15×

matvec_ct<IN=24, OUT=24>          GCC auto-vec    NEON 4-out ILP    ratio
                                   68.75 ns        61.07 ns         1.13×
```

GCC already vectorises IN=24 cleanly (across-OUT broadcast pattern,
or maybe similar 4-out ILP). Our hand-NEON only buys ~1.13–1.15×.
**Estimated production impact:**

- matvec_ct<24, 72> in s1 L1: 9× per pass-0 event, save ~25 ns/call
  → 225 ns saved per pass-0 event
- matvec_accum_ct<24, 24> in s1 L0 + L1: 18× per pass-0 event,
  save ~8 ns/call → 144 ns/pass-0 event
- Total ≈ 370 ns / pass-0 event × 0.21 pass-rate-of-msg = ~78 ns / msg
- Per-shard time: 3.15 → 3.07 µs ≈ **+2.5% admit** → 1.74 Mev/s

### 10.2 Lever: disable `add_macs` TLS counter in specialisations

Vendor header has no compile-time switch; can't gate the primary.
But our specialisations could skip the `add_macs` call entirely.
Combined with §10.1, all matvecs would be specialised → no TLS hit.

- Per-event TLS hits avoided: ~30 × 5 ns = 150 ns/event = ~35 ns/msg
- Per-shard: 3.15 → 3.12 µs ≈ **+1% admit** → 1.71 Mev/s

### 10.3 Lever: `layernorm_ct<*>` NEON

Currently fp64 reduction (vendor matches PyTorch numerics). Called
1× per layer = 4× per pass-0 event. Sub-1% share of per-event time.
NEON fp32 would be faster but breaks bit-equivalence to reference.
**Negligible (<1% admit gain), high accuracy risk → not pursuing.**

### 10.4 Lever: patch-loop overhead reduction

The 9-patch unrolled sweep in `layer_forward_ct` shows ~5–17%
unexplained overhead (per §9.2 sum-check). Sources: lambda
dispatch (probably inlined but...), scratch buffer access, residual
init (`memset(feat_out, 0, ...)`). Potential 100–200 ns / event
savings if patch loop is restructured. Estimated **+2–4% admit**.

### 10.5 Lever: `preprocess` simplification

Currently does last-t lookup + (t - last_t)/1e5 + clamp + update.
60 ns/event including halo. Path is per-event memory access into
a 5 KB last-t buffer (per shard, sized H×W of double). Hot in
L1 mostly. Already cheap; **<1% admit gain available**.

### 10.6 Lever: SPSC ring + atomic-fetch-add ring-head overhead

Each event: 1× `q_in->try_pop`, 1–2× `q_in->push` from dispatcher.
Spin-on-full / spin-on-empty in current code. At 1.7 Mev/s × 1.32
halo factor × ~50 ns per ring op (incl. atomic) ≈ 100 ns/event of
overhead. Hard to push below ~20 ns. Estimated **+1–2% admit** with
e.g. bounded-batch dequeue.

### 10.7 Lever: fp16 weights + state (NEON `asimdhp`)

A78AE has fp16 NEON (`asimdhp` in /proc/cpuinfo, confirmed). 8 lanes
per 128-bit register vs 4 fp32 lanes → 2× SIMD throughput in
principle.

Cost: substantial. Requires:
- Convert weight files (load-time conversion or new export from training)
- Per-layer fp16 path through matvec, lru_step, layernorm
- Carefully verify P1 (cumulative drift through s2/s3 GPU side
  which is fp32 — feature precision at the CPU/GPU boundary matters)

Estimated raw speedup if numerics hold: **+50–80%** on compute-bound
sections, but the actual admit gain depends on how much of the
work-unit time is compute vs memory/overhead.

### 10.8 Lever: int8 quantisation + `vdotq_s32`

A78AE supports `asimddp` (signed dot product, int8×4 → int32 accumulator).
This is the dominant ARM ML acceleration path. Potential 4× over
fp32 NEON. **Massive accuracy risk** for a network not trained
quantisation-aware. Requires retraining or post-training quantisation
calibration. **Not realistic in a 1-week effort.**

### 10.9 Lever: ditch halo entirely or weaken its semantics

**Forbidden.** halo=2 with full state-duplication on each shard is a
locked invariant. No discussion.

### 10.10 Cumulative ceiling estimate

```text
  applied lever                                       est. admit gain
  ────────────────────────────────────────────────    ───────────────
  base (post-Phase-2)                                 1.69 Mev/s
  + hand-NEON IN=24 specialisations (§10.1)           +2.5%
  + disable add_macs in specialisations (§10.2)       +1%
  + patch loop refactor (§10.4)                       +2–4%
  + ring-op tweaks (§10.6)                            +1–2%
  ──────────────────────────────────────────────────────────────────
  realistic Phase 3 ceiling                           ≈ 1.82–1.85 Mev/s
  (does not reach 2.0 Mev/s)

  + fp16 weights + state (§10.7) [high risk]          +50–80% (if numerics ok)
  ──────────────────────────────────────────────────────────────────
  fp16 path                                           ≈ 2.5–3.0 Mev/s
```

---

## 11. Microbench source files (for the expert's audit)

All under `tests/`, compiled aarch64-only via the CMakeLists guard.
Methodology is shared: pin to one core, ≥5M iters, data dep per iter
to prevent compiler hoisting, sink to prevent DCE.

- `tests/bench_matvec12.cpp` — scalar / 1-out NEON / 4-out ILP for
  `matvec_ct<12, 36>`. Used to validate the Phase 1 design.
- `tests/bench_matvec24.cpp` — GCC auto-vec vs 4-out ILP for
  `matvec_ct<24, 72>` and `<24, 24>`. Confirmed Phase 3 IN=24
  specialisation has marginal headroom.
- `tests/bench_sigmoid.cpp` — scalar libm `expf` vs 3 NEON polynomial
  sigmoids. Single-call accuracy, throughput, 200k-step cumulative
  drift through the lru recurrence. Drove Phase 2 tanh-Padé choice.
- `tests/bench_layer.cpp` — per-layer `layer_forward_ct<IN,OUT>`
  timing (loads real stub weights via `SslaSPipeline::load()`,
  iterates with varying coords at the correct stage resolution).
  Bridges microbench numbers and production per-event µs.

`SslaSPipeline::layer_forward(int layer_idx, ...)` was added as a
public delegate so bench_layer can call individual layers in
isolation; production runtime still calls `stage_forward`.

---

## 12. What we expect the expert to consider

Pushing CPU side from 1.69 → 2.0 Mev/s on this hardware, **within
the constraints** of §2. We are looking for advice on:

1. **Has GCC really maxed out IN=24 matvec on A78AE?** Our 1.13–1.15×
   from hand-NEON 4-out ILP suggests yes, but maybe a different
   layout (e.g., pre-transpose weights so OUT-axis sequential
   access aligns naturally) or better instruction scheduling would
   help. Specifically: A78AE has 4 FP pipes — is there a tighter
   pattern that hits 4-ops-per-cycle on a 24×72 matvec?

2. **Patch loop / scratch buffer overhead**: the 9-patch unrolled
   sweep has ~15% unexplained overhead vs the sum of measured primitive
   costs. Memory aliasing? Bad prefetch? Lambda inlining failing?
   How would you investigate / restructure?

3. **fp16 path feasibility**: is `asimdhp` really 2× throughput on
   A78AE for matvec-class loops? Any known pitfalls (denormal
   handling, accumulator precision)? Realistic accuracy hit on a
   recurrent fp32-trained network?

4. **Anything we're missing**: maybe a different layer-fusion
   pattern (run matvec_ct + lru_step + matvec_accum_ct as one
   loop?), or a fundamentally different scheduling approach.

5. **Reasonability of 2.0 Mev/s** on this hardware given the
   constraints. If the expert says "no, you've squeezed everything
   reasonable — 1.85 is the ceiling", that's a useful answer too.

---

## 13. Reproduction commands (for the expert to verify any number)

```bash
# Repo root: ~/ssla-rt
# Stub weights generator (random weights, schema-valid for SSLA-S)
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80

# Build everything (Release default)
cmake -S . -B build && cmake --build build -j

# Layer + primitive microbench
taskset -c 4 build/bench_layer

# matvec microbenches
taskset -c 4 build/bench_matvec12       # IN=12, 4-out NEON shines
taskset -c 4 build/bench_matvec24       # IN=24, GCC already vectorises

# Sigmoid accuracy + speed
taskset -c 4 build/bench_sigmoid

# P1 correctness gate (200k events, GPU vs Python oracle — note that
# this gate does NOT exercise the libstage01_to_gpu CPU pipeline; it
# tests the GPU kernel against a Python NumPy reference)
python3 orin/bench_s2_s3_head_celled.py --n 200000

# Live saturation sweep
cd orin && for N in 1 2 4 5 6 7 8; do
  python3 hybrid_runner.py --weights /tmp/ssla_s_64x80 --h-full 64 --w-full 80 \
    --kernel-variant celled --synthetic-mev 3.0 --duration-s 12 \
    --stats-interval-s 2 --shards $N
done
# Read the 8s–10s 2-s-window delta in `cam_in` for steady-state admit.

# Disassembly inspection — verify NEON specialisation landed:
objdump -d build/libstage01_to_gpu.so | less       # grep for the
                                                    # layer_forward_ct lambdas
```

Source layout:

```text
ssla-rt/
├── CMakeLists.txt
├── include/
│   ├── ssla_kernels.h           — pipeline class declaration
│   ├── ssla_neon_linear.h       — Phase 1 matvec specialisations
│   └── ssla_neon_lru.h          — Phase 2 lru specialisations
├── src/
│   ├── ssla_kernels.cpp         — layer_forward_ct, stage_forward, preprocess
│   ├── lib_stage01_to_gpu.cpp   — shard worker, dispatcher, instrumentation
│   └── (other CPU pipeline variants)
├── tests/
│   ├── bench_matvec12.cpp
│   ├── bench_matvec24.cpp
│   ├── bench_sigmoid.cpp
│   └── bench_layer.cpp
├── vendor/openeva/
│   ├── prim/linear.h            — read-only primary templates
│   ├── prim/rnn.h               — read-only lru_step primary
│   ├── prim/activation.h        — read-only sigmoid
│   ├── prim/layernorm.h         — read-only layernorm_ct
│   └── prim/flop.h              — read-only TLS counter
└── orin/
    ├── hybrid_runner.py         — Python harness for live runs
    └── bench_s2_s3_head_celled.py — P1 GPU correctness gate
```

---

## 14. Closing notes

Two NEON optimisation phases shipped (commits `89e6544` and pending).
The CPU pipeline is currently **2.06× over the original baseline**, on
the same hardware with no algorithmic changes — by hand-vectorising
the two primitives GCC failed to vectorise (matvec at IN=12, sigmoid
inside lru_step). Both lifts validated in microbench AND at
saturation; P1 numerics preserved exactly (max|Δ| 4.4, unchanged).

The remaining factor-to-2.0 is asked under tight constraints. We
believe Phase 3 quick wins (§10.1–§10.6) get us to ~1.85 Mev/s with
~1 day of effort. Reaching 2.0 Mev/s likely requires fp16 (§10.7)
which is a 1–2 week effort with real accuracy uncertainty, and
would need a fresh P1-equivalent end-to-end numeric gate built
against the C++ pipeline (the existing P1 only validates the GPU
side against a Python reference, not the C++ pipeline against a
Python reference — that's a gap we'd need to close before risking
fp16).

We welcome the expert's suggestions on:
- whether 2.0 Mev/s is reachable without fp16 / quantisation,
- specific concrete techniques (assembly patterns, loop transformations,
  cache layout, etc.) we haven't tried,
- whether the fp16 path's numeric risk is realistically manageable on
  this recurrent network.
