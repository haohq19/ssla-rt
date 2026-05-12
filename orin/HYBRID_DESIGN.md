# SSLA Hybrid Deploy — Design

_Draft. Not yet approved. Code lands only after this doc is signed off._

End-to-end SSLA-S real-time inference on Jetson Orin NX with the
iniVation DVXplorer Micro. Splits the pipeline across CPU and GPU:

- **CPU**: stages 0 + 1 — per-event, multi-shard (4 shards × halo = 2),
  already shipped via `deploy/build/libstage01_capi.so` and verified at
  ~450 kev/s admit rate at 60 × 80 effective resolution.
- **GPU**: stages 2 + 3 + head — per-event, **2 blocks × halo = 2**
  spatial sharding, S9-spatial style (per-block private hidden state,
  no atomics, halo broadcast at producer). Each event flows event-by-event
  through the GPU, **no batching**.

This is **not** an extension of `cpp/methods/ssla/` and does **not**
modify any existing python/ or cpp/ files.

---

## 1. Why this split

| Stage | Per-event cost | Surviving rate (steady state) | Best fit |
|---|---:|---:|---|
| s0 + s1 | ~2 µs | 100 % → 6.25 % | CPU shards (small per-event work, big rate; spatial halo = 2 already proven) |
| s2 + s3 + head | ~30 µs (single-thread) → ~5–10 µs (cooperate-per-event, 256 threads) | 6.25 % → 1.56 % → write | GPU blocks (heavier per-event compute, small grid, shared mem absorbs full hidden state) |

At 1 Mev/s admit rate, ~62 kev/s reaches s2 input. With 2 GPU blocks
running cooperate-per-event internally, each block sustains
~100–200 kev/s; halo overhead is ~40 % redundant work, so effective
throughput cap is ~215 kev/s — comfortable margin over the 60 kev/s
input rate.

**Hard rule from user** (recorded for future readers):
> NO BATCH OPERATIONS FOR GPU. GPU IS ALSO PER-EVENT INFERENCE.

This means events flow through the GPU one at a time per block.
Within a block, all 8 warps cooperate on a single event's matvec / LRU
/ LN / pool / head work — no concurrent events in flight inside a
block. Across the 2 blocks, each block runs FIFO from its own input
ring. State on the GPU side is recurrent, exactly like the CPU side:
no batched matmuls, no fixed event windows, no implicit concurrency at
the event granularity.

**Hard rule from user** (also recorded):
> halo 1 is not acceptable in any time

So halo = 2 strict at every stage. That's what shapes the GPU sharding
math below.

---

## 2. Data flow

```
DVXplorer ──► dv_processing thread ──► ctypes ──► libstage01_to_gpu.so
                                                      │
                                       4 CPU shards (s0+s1, halo=2)
                                                      │
                                            owner-pass at s1 → s2-x = full-x >> 2
                                                      │
                                          ┌───────────┴───────────┐
                                          │ route by s2-x to       │
                                          │ GPU block(s) covering  │
                                          │ s2-x in proc range     │
                                          ▼                        ▼
                                  [ pinned ring 0 ]        [ pinned ring 1 ]
                                  (MPSC: 4 CPU pro-       (MPSC)
                                   ducers, GPU consumer)
                                          │                        │
                                          ▼                        ▼
                                  GPU block 0                GPU block 1
                                  s2-owned [0, 10)            s2-owned [10, 20)
                                  s2-proc  [0, 12)            s2-proc  [8, 20)
                                  s3-owned [0, 5)             s3-owned [5, 10)
                                  ┌─────────────────┐        ┌─────────────────┐
                                  │ 8 warps coop-   │        │ 8 warps coop-   │
                                  │ per-event:      │        │ per-event:      │
                                  │   s2 forward    │        │   s2 forward    │
                                  │   if owner: s2  │        │   if owner: s2  │
                                  │     tdrop+pool  │        │     tdrop+pool  │
                                  │   s3 forward    │        │   s3 forward    │
                                  │   if owner: s3  │        │   if owner: s3  │
                                  │     tdrop+pool  │        │     tdrop+pool  │
                                  │   if owner+pass:│        │   if owner+pass:│
                                  │     head decode │        │     head decode │
                                  │   no atomics    │        │   no atomics    │
                                  └─────────────────┘        └─────────────────┘
                                          │                        │
                                          └───────────┬────────────┘
                                                      ▼
                                       predictions tensor (host-visible)
                                       per-cell version counter
```

Pinned host memory required: Orin NX reports
`CONCURRENT_MANAGED_ACCESS = 0` (see [README.md](README.md) §Tegra
coherence caveat). Managed memory is unsafe for live concurrent CPU
writes + GPU reads on this device.

---

## 3. CPU → GPU ring contract

### 3.1 Strip layout (W2 = 20 at 60 × 80 base)

| Block | s2 owned | s2 processed (halo = 2 at s2) | s3 owned | s3 processed |
|---|---|---|---|---|
| 0 | [0, 10) | [0, 12) | [0, 5) | [0, 6) |
| 1 | [10, 20) | [8, 20) | [5, 10) | [4, 10) |

**Halo math.** Halo = 2 at s2 res automatically covers both s2 and s3
patch-touch ranges:
- s2 forward (K = 3) needs ±1 cell of state at s2 res — covered.
- s3 forward (K = 3) needs ±1 cell at s3 res = ±2 cells at s2 res —
  exactly the halo width. Block 0's s2 processing range [0, 12) gives
  s3 cells reachable through pool from [0, 6), so s3 patches at any
  cell in [0, 5] (block 0's s3 owned + 1 patch halo) are correctly
  driven by events block 0 sees. ✓

### 3.2 Producer side (CPU shard worker, after `tdrop_and_pool(1, ...)`)

Existing CPU code in `lib_stage01_capi.cpp`:

```cpp
pipe.preprocess(m.ev, feat_in);
int x = ex0, y = ey0;
pipe.stage_forward(0, x, y, feat_in, feat0.data());
if (m.is_owner) {
    if (pipe.tdrop_and_pool(0, x, y)) {
        pipe.stage_forward(1, x, y, feat0.data(), feat1.data());
        if (pipe.tdrop_and_pool(1, x, y)) {
            // <-- HYBRID HOOK starts here
            const int s2x = x;             // already s2 res after two pools
            const int s2y = y;
            for (int b = 0; b < N_GPU_BLOCKS; ++b) {
                const auto& s = gpu_strip[b];
                if (s2x < s.proc_lo || s2x >= s.proc_hi) continue;
                ring[b].push({ .t = ev.t, .x = s2x, .y = s2y,
                               .feat1 = feat1 });
            }
        }
    }
}
```

Each event lands in 1–2 GPU rings (always exactly one block has it as
owner; 0–1 as halo). The CPU shard's existing owner-only forwarding
rule is preserved upstream, so the per-event broadcast at the hook is
independent of the CPU sharding.

**Owner re-derived GPU-side, not carried in the record.** Each block
knows its own owned range from its `strip[blockIdx.x]` config, so
`is_owner = (rec.x >= owned_lo && rec.x < owned_hi)` is a trivial
check after the record is popped. Saves 4 B per record and removes
one source of CPU/GPU disagreement.

Channel sizes verified against `deploy/include/ssla_kernels.h`:
| | input dim | output dim |
|---|---:|---:|
| CPU stage 0 | `kInDim=2` | `kC0=12` |
| CPU stage 1 | `kC0=12` | **`kC1=24`** ← passed to GPU as `feat1` |
| GPU stage 2 | `kC1=24` | `kC2=48` |
| GPU stage 3 | `kC2=48` | `kC3=96` |
| GPU head | `kC3=96` | `5 + N_classes` |

### 3.3 Ring record format

```c
struct GpuInputRec {
    float    t;            // dt_norm (already host-normalized)
    uint16_t x;            // s2-res x
    uint16_t y;            // s2-res y
    float    feat1[24];    // C1 = 24 features after CPU stage 1
};                          // 4 + 2 + 2 + 96 = 104 B
```

Per-block ring capacity 65 536 records → 7 MB pinned alloc per ring,
14 MB total. At 60 kev/s s2-input the buffer holds ~1 s of headroom.

### 3.4 Synchronization

- **Producer side (CPU)**: each ring is **multi-producer** (4 CPU shard
  workers may push concurrently to the same ring). Use
  `__atomic_fetch_add(&ring_head[b], 1, __ATOMIC_RELAXED)` to claim a
  slot. Write the record. Then `__atomic_thread_fence(release)` so the
  GPU sees the record before observing the head bump (we publish a
  separate `committed_head[b]` after the record write — single
  producer's own bump is local-ordered).
- **Consumer side (GPU)**: single-consumer per ring. Block reads
  `committed_head[b] - tail[b]` for empty-check, no atomic needed for
  tail (block-local).
- **Backpressure**: producer spins on full ring (preserves order; ring
  is 1 s deep so this only fires under sustained overload).

The MPSC ring head increment is the **only atomic in the entire
pipeline**. There are no atomics on the GPU side at all.

---

## 4. GPU kernel design

### 4.1 Two persistent blocks, S9-spatial style

```
gridDim  = (2, 1, 1)
blockDim = (256, 1, 1)         // 8 warps × 32 lanes
launch once, persistent
```

Each block reads its `strip[blockIdx.x]` config (owned range, proc
range, ring pointer, version-counter base) and runs its private
pipeline. **No inter-block writes.** **No atomics on the device.**
Cross-block correctness comes from the producer broadcasting halo
events to both blocks; each block independently keeps state in sync
for cells in its processing range.

### 4.2 Memory layout — full-grid hidden state per block

Each block keeps the **full** s2 and s3 hidden state, not strip-only.
Reasons: (a) ~88 KB / block fits in opt-in shared (168 KB on sm_87),
(b) avoids strip-boundary handoff complexity, (c) updates outside the
block's owned range are fine — they're driven by halo events the
producer sent.

| state | per block |
|---|---:|
| s2 hidden (15 × 20 × 48 fp32) — full grid | 56.3 KB |
| s3 hidden (7 × 10 × 96 fp32) — full grid | 26.3 KB |
| s2 tdrop counters (15 × 20 × uint8) | 0.3 KB |
| s3 tdrop counters (7 × 10 × uint8) | 0.07 KB |
| head predictions (7 × 5_owned × OUT fp32) | < 5 KB |
| version counters (per owned cell) | 0.2 KB |
| **total per block** | **≈ 88 KB** |

Shared-memory is the natural home — no global-memory state round-trips
during steady state.

### 4.3 Per-event flow inside a block (cooperate-per-event)

All 256 threads work on **one event at a time**. Matvec / LRU / LN / pool
math is lane-striped exactly as `kernels/ssla_step.cuh` already does for
S9-base; reductions use warp shuffles + `__syncthreads` (only inside
the per-event critical section, not as the per-warp pattern). No
atomics, no spinlocks.

```c
__shared__ float    s2_hidden [H2 * W2 * C2];   // 56 KB
__shared__ float    s3_hidden [H3 * W3 * C3];   // 26 KB
__shared__ uint8_t  tdrop_s2  [H2 * W2];
__shared__ uint8_t  tdrop_s3  [H3 * W3];
__shared__ float    preds     [H3 * W3_OWNED * OUT];

while (true) {
    // Single-warp lookahead at the ring head; broadcast to all warps.
    if (warp_id == 0 && lane == 0) {
        if (*stop_flag) want_stop = 1;
        ev_avail = (committed_head[block_id] > tail[block_id]);
    }
    __syncthreads();
    if (want_stop) return;
    if (!ev_avail) {
        if (warp_id == 0 && lane == 0) __nanosleep(200);
        __syncthreads();
        continue;
    }

    // Pop one record; warp 0 reads, broadcasts via shared mem.
    GpuInputRec rec;
    if (warp_id == 0) cooperative_load_record(rec, ring[block_id], tail[block_id]);
    __syncthreads();
    if (warp_id == 0 && lane == 0) tail[block_id]++;

    // Re-derive owner from rec.x and the block's owned range (no flag in record).
    const bool is_owner = (rec.x >= strip.owned_lo) && (rec.x < strip.owned_hi);

    // ----- s2 forward (always; halo + owner both run state update) -----
    s2_forward_block(rec, s2_hidden);   // 9-patch matvec/LRU/LN, 256 threads coop

    if (is_owner) {
        bool pass2 = s2_tdrop_pool_block(rec.x, rec.y, tdrop_s2);
        if (pass2) {
            int s3x = rec.x >> 1, s3y = rec.y >> 1;
            s3_forward_block(s3x, s3y, rec.feat1, s2_hidden, s3_hidden);
            bool pass3 = s3_tdrop_pool_block(s3x, s3y, tdrop_s3);
            if (pass3) {
                int hx = s3x >> 1, hy = s3y >> 1;
                head_decode_block(hx, hy, s3_hidden, preds);
                bump_version_counter(hx, hy);
            }
        }
    } else {
        // Halo block: state forward only, no tdrop, no head write.
        // Compute s3 coords directly from the s2 coords (pool is positional).
        int s3x = rec.x >> 1, s3y = rec.y >> 1;
        s3_forward_block(s3x, s3y, rec.feat1, s2_hidden, s3_hidden);
    }
}
```

Halo block runs `stage_forward` at s2 + s3 to keep its private hidden
state consistent. It does NOT call `tdrop_and_pool` (that's an owner
operation that mutates a counter; per-shard counter drift is the only
documented S8 drift source, and we accept the same drift here).

### 4.4 Why cooperate-per-event, not warp-per-event

Two reasons within a block:
1. **No atomics.** Warp-per-event with shared hidden state inside a
   block needs locks (S6 style) — explicitly rejected by user. The
   only lock-free alternative within a block is one-event-at-a-time.
2. **Better matvec parallelism.** A 96 × 96 matvec parallelizes
   beautifully across 256 lanes; a 96 × 96 matvec on a single 32-lane
   warp is 3 × slower per event but only 1/8 as many lanes. Net
   per-block throughput is similar; cooperate-per-event keeps the
   no-atomics invariant for free.

Across blocks: 2 blocks × ~150 kev/s × halo factor 1.4 ≈ 215 kev/s
effective input ceiling. Headroom is ~3.5 × the expected 60 kev/s
input rate.

### 4.5 Predictions assembly

Each block writes only to its **owned** s3 cells in pinned host
memory. Cell ranges:
- Block 0 → cells s3-x ∈ [0, 5) × s3-y ∈ [0, 7)
- Block 1 → cells s3-x ∈ [5, 10) × s3-y ∈ [0, 7)

The two ranges are disjoint, so no conflicting writes — no atomics
needed. A per-cell `version` counter (uint32) is bumped after every
head write so the host viewer / consumer can detect freshness.

### 4.6 What is NOT done in the kernel

- **No atomics anywhere on device.** Producer-side ring head bump is
  the only atomic in the system.
- **No `__syncthreads()` outside the per-event critical section.** The
  per-event work itself uses `__syncthreads` between the matvec and
  reduction steps inside one block (this is fine — events are FIFO so
  the barrier cadence is per-event).
- No prediction post-processing on GPU (sigmoid + NMS stay on host).
- No oracle-dump on first pass — equivalence checked offline by replay.

---

## 5. Equivalence target

### 5.1 Acceptance (Phase 1)

Replay 200 k synthetic post-s1 records (dumped from a CPU run) through:
- the CPU s2+s3+head reference (single-thread, no sharding — gives the
  ground-truth oracle),
- the new GPU kernel (2 blocks, halo broadcast).

| Quantity | Target |
|---|---|
| s2 drop count | within ±0.5 % of CPU reference (per-block tdrop drift, S8-like) |
| s3 drop count | within ±0.5 % of CPU reference |
| head predictions max\|Δ\| | ≤ 5.0 (matches S8 baseline drift) |
| ring deadlock at 30 s saturated input | none |

Drift sources (all bounded, all documented in [README.md](README.md)
§S8):
- per-block tdrop counters drift because halo events don't increment
  them — small (~0.5 %).
- inter-block ordering is implicit via the per-block ring (FIFO within
  a ring), but cross-block scheduling skew adds order variation
  identical to S8 on CPU.

### 5.2 Out of scope for Phase 1

- bit-equivalence with deploy/S1 (not required; bounded drift accepted)
- camera glitch tolerance (drift only widens, never diverges)
- 4-block sharding (halo math doesn't fit at this grid; revisit if
  throughput needs more)

---

## 6. Phasing

| Phase | What lands | Acceptance gate |
|---|---|---|
| **P1** | `kernels/ssla_s2_s3_head.cuh` + Python harness `bench_s2_s3_head.py` that feeds synthetic post-s1 records and diffs against CPU oracle | drops within ±0.5 %; max\|Δ\| ≤ 5.0; 200 k events without crash |
| **P2** | `lib_stage01_to_gpu.cpp` (extends `lib_stage01_capi.cpp` with per-block ring push hook); shared header for ring record dtype + strip config | end-to-end Python driver runs 30 s on live camera with no deadlock; non-zero predictions written by both blocks |
| **P3** | latency + throughput measurement, written up as a new STATUS.md section | numbers reported; user signs off |

Each phase has a **single deliverable** and a **single gate**. We do
not start P2 until P1's gate passes; we do not start P3 until P2's
passes. Lesson taken from the S3-deadlock and S9-base-2.1-kev/s
post-mortems.

---

## 7. File layout (new, all under `deploy/orin/`)

```
deploy/orin/
├── HYBRID_DESIGN.md                    this doc
├── kernels/
│   └── ssla_s2_s3_head.cuh             new — 2-block coop-per-event s2+s3+head
└── orin/                               existing python package
    ├── ring_gpu_input.py               new — pinned MPSC ring (one instance per
    │                                    GPU block; CPU push, GPU pull)
    ├── strip_config.py                 new — block layout, owned/proc ranges
    ├── hybrid_runner.py                new — entry point: open camera, init CPU
    │                                    lib, allocate 2 rings, launch 2-block
    │                                    persistent kernel, monitor stats, exit
    └── ...                             (rest unchanged)
```

In `deploy/`:

```
deploy/src/
└── lib_stage01_to_gpu.cpp              new — fork of lib_stage01_capi.cpp
                                              with per-block ring push hook
                                              (P2 only; P1 uses synthetic ring
                                              from Python)
```

Phase 1 explicitly does **not** touch `lib_stage01_capi.cpp`. We don't
want to drag the existing CPU-only verified library into a phase whose
goal is GPU-side correctness only.

---

## 8. Open questions for user

1. **Block thread count** — 256 (8 warps) or 128 (4 warps)? The Day-1
   measurement of regs/thread on the s2+s3+head-only kernel decides.
   Goal: at least 1 block per SM with full register budget; 2 blocks
   total occupies 2 SMs out of 8 (rest idle but available for camera
   USB / dv_processing thread).
2. **Ring depth** — 65 536 records (~7 MB / ring) is a lot. Acceptable
   on Orin's 16 GB? Yes if no other GPU consumer exists in the session.
3. **Per-cell version counter** — uint32 cumulative, or "frame index"
   resettable from host? Cumulative is simpler; default to that.
4. **Stop-flag flush semantics** — graceful drain (each block finishes
   in-flight events before exiting) is the default.

---

## 9. What this doc commits us to

If approved:
- I write `kernels/ssla_s2_s3_head.cuh` and the P1 harness.
- I do NOT push to camera until P1 passes.
- I do NOT change `lib_stage01_capi.cpp` until P2.
- The architecture (2 blocks, halo = 2 strict at s2 res, full-grid
  hidden state per block, cooperate-per-event, no GPU atomics) is the
  design — not adjustable mid-implementation without revisiting this
  doc.
