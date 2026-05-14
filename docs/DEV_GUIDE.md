# SSLA-RT Development Guide

A working playbook distilled from real-world optimization of SSLA-S
on Jetson Orin NX with DVXplorer Micro. Use this when porting to:
- **a different device** (Orin AGX, future Jetsons, other Ampere/Ada/Hopper hardware)
- **a different event camera** (other DVS sensors, neuromorphic sensors)
- **a different SSLA variant** (SSLA-M, SSLA-L, alternative recurrent backbones)

Everything in here is **verified by experiment** unless explicitly marked
otherwise. Dead-end attempts are documented so you don't repeat them.

---

## Table of contents

- [0. TL;DR — the production recipe](#0-tldr)
- [1. Methodology — the rules](#1-methodology)
- [2. Architecture invariants — DO NOT VIOLATE](#2-architecture-invariants)
- [3. Profiling & instrumentation](#3-profiling)
- [4. Optimization techniques that work](#4-what-works)
- [5. Optimization techniques that DON'T work](#5-what-doesnt-work)
- [6. Camera tuning](#6-camera-tuning)
- [7. New hardware adaptation checklist](#7-new-hardware)
- [8. New camera adaptation checklist](#8-new-camera)
- [9. New SSLA variant adaptation](#9-new-variant)
- [10. Co-design opportunities (not yet shipped)](#10-codesign)
- [11. Testing & validation protocols](#11-testing)
- [12. Common pitfalls & footguns](#12-pitfalls)
- [13. Tools & reference](#13-tools)

---

<a id="0-tldr"></a>
## 0. TL;DR — the production recipe

Current production-shipped state on Jetson Orin NX (2026-05-14):

| component | what / config |
|---|---|
| **CPU side** | 6 shards on cores 1–6, halo=2, NEON fused interior matvecs, **2.80 Mev/s admit ceiling** |
| **CPU→GPU bridge** | per-block SPSC pinned-host rings, 8 blocks |
| **GPU topology** | 2W × 4H = 8 blocks on 8 SMs, **balanced Y splits** (5/3/3/5) |
| **GPU kernel** | `ssla_s2_s3_head_celled.cuh`, cell-owner warp partition (9 warps/block) |
| **GPU smem cache** | L4 qvgIn (121 KB) lives in smem permanently |
| **GPU L2 pinning** | L7 qvgIn (972 KB) via `CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW` (perf bench only) |
| **State-only path** | VG-only matvec for tdrop-dropped events at L5/L7 (no Q matvec, no qh) |
| **A1 opt** | events failing tdrop_s2/s3 skip goW + gather + LN |
| **GPU drain** | **~230 kev/s live, ~247 kev/s offline** |
| **e2e latency** | **~330 µs p50** (output-producing events) |

Build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Run smoke test:
```bash
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80
cd orin
python3 bench_s2_s3_head_celled.py --n 50000   # P1 oracle (must pass first)
python3 perf_celled_multi.py --n-blocks 8       # offline throughput
python3 hybrid_runner_multi.py --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --cpp-synth \
    --synthetic-mev 2.0 --duration-s 8 --pin-python-main --base-core 1
```

---

<a id="1-methodology"></a>
## 1. Methodology — the rules

### 1.1 P1 first, perf second

**Every kernel change MUST pass the P1 oracle before any perf measurement
is meaningful.** The oracle (`bench_s2_s3_head_celled.py`) compares the
GPU `drain_n` output against a single-thread CPU reference, allowing
`max|Δ| ≤ 5.0` on `s3_feat` and `drift = 0` on drop counts.

Workflow:
1. Change kernel
2. `python3 bench_s2_s3_head_celled.py --n 50000` — wait for `P1 PASS`
3. If P1 fails: STOP, fix correctness, do not waste time benchmarking
   broken output
4. If P1 passes: proceed to perf

If your model has bit-identical recurrence requirements, the tolerance
budget (5.0 here) may need adjusting. See [§9](#9-new-variant).

### 1.2 Trust experiments, not projections

**Every optimization estimate in this session was off by 2–10×**, almost
always overestimating gain. Specifically:

| optimization | projection | actual |
|---|---:|---:|
| L4 qvgIn smem cache | +20–30% | +4% |
| L7 L2 persistence | +5–10% | +1.7% offline, **0% live** |
| Pipeline split / 2 blocks/SM | +50–100% | +9% with halo waste |
| Balanced topology | +20–30% | +4% |
| VG-only state-only | +6–13% | +3% |
| **TF32 TC L4 precompute** | **uncertain (expert promised gain)** | **−5% offline** |
| **Batch-local hidden coalescing** | **uncertain (expert promised gain)** | **−6% offline** |

Lesson: **propose an optimization → estimate, then implement minimally
and measure**. If the measurement disagrees with the estimate by more
than 2×, debug before scaling up. Save dead-end results (e.g., this
guide's [§5](#5-what-doesnt-work)) so they aren't repeated.

### 1.3 One change at a time

Bundle no more than one optimization per commit. Each commit message
should include:
- What changed (1–2 sentences)
- Why it should help (analysis)
- Measured P1 result (drift, max|Δ|)
- Measured perf result (throughput, latency)
- Reverted? If yes, why.

This makes `git bisect` viable when something regresses 3 commits later.

### 1.4 Critical thinking under user pressure

Users will ask for optimizations they've read about (Tensor Cores,
sparsity, ...). Do not implement an expert's suggestion just because
they sound authoritative. **Walk through the problem-specific
constraints first** (see [§2](#2-architecture-invariants)) and identify
which constraints the suggestion violates. If all constraints survive,
implement minimally and measure. **Negative experimental results are
data — record them in memory files, not as failures.**

### 1.5 Distinguish output latency from all-event latency

Most events get tdrop-dropped. **Only output-producing events
(`is_owner && pass_tdrop_s2 && pass_tdrop_s3`) matter for downstream
latency.** Inside a batch, output and dropped events share the same
`t_pop_clk` and `t_done_clk`, so kernel-internal latency is the same,
but the report should filter to output-producers (we do — see
`GpuTimingSlot.owner` field).

---

<a id="2-architecture-invariants"></a>
## 2. Architecture invariants — DO NOT VIOLATE

These are HARD constraints. Optimizations that violate them are
either incorrect or destroy properties (race-free design, accuracy)
that the rest of the system depends on.

### 2.1 `halo = 2` is locked

CPU shards process events at boundaries (within ±2 cells of own strip
edge) to correctly compute s2 conv (K=3) + s2→s3 pool + s3 conv (K=3).
Reducing halo below 2 produces wrong predictions at strip boundaries.
**Never reduce halo to 1, never bypass via state-sync schemes, never
even discuss it as a "hypothetical lever".**

### 2.2 Cell-owner warp partition (race-free invariant)

The GPU production kernel uses 9 warps per block with:
```c
warp_id = (cell_y % 3) * 3 + (cell_x % 3)
```

For any event at (evx, evy), the 3×3 patch around it touches 9 cells
with 9 distinct `(cy%3, cx%3)` pairs → exactly one warp per cell.
**No two warps ever write to the same hidden-state cell.** This is the
foundation of the no-atomics, no-`__syncthreads`-in-per-event-step
design.

If you re-partition for a different patch size K, derive an equivalent
guarantee. For K=5 (5×5 patch, 25 cells, but only 9 distinct mod-3
partitions still — same partition works). For K=2 or K=4, the mod-3
hash doesn't fit cleanly; **redesign required**.

### 2.3 Pinned host memory for ring traffic (Tegra coherence)

Jetson Orin NX reports `CONCURRENT_MANAGED_ACCESS = 0`. **Managed
memory is NOT safe for concurrent CPU writes + GPU reads on this
device.** All cross-CPU/GPU traffic must use pinned host memory
(`cuMemHostAlloc`):
- per-block ring buffers
- ring head/tail atomics
- stop flag
- predictions output

Managed memory is fine only for GPU-only state (hidden state, weights
written once at init, tdrop counters).

On other devices (e.g., Orin AGX, dedicated GPUs with `CONCURRENT_MANAGED
_ACCESS = 1`), this constraint may relax. Test with `cuDeviceGetAttribute
(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev)`. **Don't assume.**

### 2.4 No atomics on hidden state

The cell-owner partition guarantees no within-block hidden-state race.
**Never introduce GPU atomic ops on hidden state cells** — it would
mask the race-freedom guarantee and add measurable cost. (Smem
atomics on counters are OK; we use them in `parallel_tdrop`.)

### 2.5 fp32 throughout (no fp16 / int8 unless approved)

`BATCH=8` events through the kernel, each at m=1 per matvec, fp32
inputs and outputs. fp16/int8 are forbidden per project policy. **TF32
is allowed** if it doesn't break P1 (it doesn't — verified, see §5.6).

### 2.6 `BATCH = 8` events per kernel dispatch

Smem layout, contrib buffer, event_slots, task_event/delta are all
sized for BATCH=8. Changing BATCH requires:
- updating smem alloc formulas in 4 kernel entries
- updating per-batch loop bounds
- updating ALL Python harness smem calculators
- re-validating P1 + perf

Currently BATCH=8 is treated as a project constant.

### 2.7 Output-producing events drive latency design

Latency reporting filters by `is_owner && pass2 && pass3` (the `owner`
field in `GpuTimingSlot`). Optimizations targeting per-event latency
should specifically reduce latency for events that pass both tdrops.
Dropped events share the same per-batch timing anyway.

---

<a id="3-profiling"></a>
## 3. Profiling & instrumentation

### 3.1 Three layers of measurement

| layer | tool | what it measures |
|---|---|---|
| **Offline kernel-only** | `perf_celled_multi.py` / `perf_celled.py` | best-case GPU drain with pre-filled rings; isolates GPU compute from ring/CPU overhead |
| **Per-phase µs breakdown** | `perf_celled_profile.py` (uses `_profile.cuh`) | how many µs each kernel phase takes (LOAD, TDROP, L4_COMPUTE, ..., GATHER, OUT) |
| **Live system** | `hybrid_runner_multi.py` | full pipeline (camera/synth → CPU shards → GPU drain → predictions); reports e2e latency for output-producing events |

Always run **all three** when characterizing a change. Differences:
- offline >>> live means CPU/ring is the bottleneck
- live ≈ offline means GPU is the bottleneck
- per-phase tells you WHICH part of GPU compute is heaviest

### 3.2 GPU clock for kernel timing

GPU side uses `clock64()` (SM clock cycles). To convert to ns:
- `cfg.kernel_start_clk[blk]` written at kernel entry
- `cfg.kernel_end_clk[blk]` written at kernel exit
- Host records CPU `CLOCK_MONOTONIC_RAW` at both anchor points
- `ns_per_cycle = ns_span / cyc_span` per block
- Then `t_done_ns_cpu = t_anchor_cpu_ns + (t_done_clk - kernel_start_clk) * ns_per_cycle`

This calibration is biased by ~5 ms due to SM-clock DVFS without
`jetson_clocks` / root. **Treat absolute `emit→done` numbers as
upper-bounded by ~5 ms residual error.** For latency reporting, prefer
the additive form `emit→push + (kernel push→done)` (both
domain-internal, no cross-clock bias).

### 3.3 Per-event timing record (`GpuTimingSlot`)

Each batch writes 8 records into a per-block timing ring (capacity
configurable via `--timing-cap`):
```cpp
struct GpuTimingSlot {
    unsigned long long t_pop_clk;   // batch entered kernel (GPU clock)
    unsigned long long t_done_clk;  // batch exited (GPU clock)
    unsigned int       seq;          // event index in stream
    unsigned int       owner;        // 1 if is_owner && pass2 && pass3
    unsigned long long t_push_ns;   // CPU ring publish (CPU clock)
    unsigned long long t_emit_ns;   // synth/camera emit (CPU clock)
};
```

**Latency analysis filters by `owner == 1`** — only events that produce
a final prediction. See `hybrid_runner_multi.py:319`.

### 3.4 Per-phase profiling

`perf_celled_profile.py` runs a special kernel variant
(`ssla_s2_s3_head_celled_profile.cuh`) with `STAMP()` markers between
phases. Outputs per-phase µs (p50/mean/share%):

```
phase                          p50 µs  share %
POP → LOAD                       8     4 %
DISPATCH_S2 → L4_RES             2     1 %
L4_RES → L4_ZERO                 0.5   0.2 %
L4_ZERO → L4_COMPUTE           40     18 %   ← compute phase
L4_COMPUTE → L4_GATHER         9      4 %
...
total p50                     217    100 %
```

**Use this whenever you suspect a specific layer/phase is the bottleneck.**
If you add a new optimization to L5, this profile tells you immediately
whether L5_COMPUTE got faster.

The profile variant adds ~10% overhead; **don't use for headline
throughput numbers**.

### 3.5 Saturation curve

Always sweep synth rates to find the saturation point:

```bash
for mev in 0.5 1.0 1.5 2.0 2.5 3.0 4.0; do
  python3 hybrid_runner_multi.py --synthetic-mev $mev ...
done
```

Look for:
- **linear region**: synth < saturation, lag = 0, drain ≈ synth × (1/16)
- **saturation point**: lag starts non-zero
- **plateau**: drain stops growing, lag grows with synth

The drain at the plateau is the **system ceiling**. Currently 227–230
kev/s drain at synth ≥ 2.0 Mev/s.

### 3.6 Live test should run ≥ 6 s

OS daemons / IRQs cause p99 spikes that don't show in 1–2 s windows.
Minimum 6 s for reasonable p99/max stats. For p99.9 / max tail, 30+ s
is needed.

---

<a id="4-what-works"></a>
## 4. Optimization techniques that work

Each entry: what it is → why it helps → measured gain → reference.

### 4.1 A1: skip output for tdrop-dropped events

**What:** Events that fail `tdrop_s2` or `tdrop_s3` skip `goW + gather +
LN` since their output is never consumed. Hidden state still updates
(qvg + lru must run for recurrence correctness).

**Why:** With `tdrop_window=4`, 75% of events at each tdrop are dropped.
The "last layer of each stage" (L5 for s2, L7 for s3) produces output
only consumed by downstream stages. Dropped events: output unused →
skip computing it.

**Gain:** GPU 184 → 226 kev/s (+23%), commit `6f287d3`.

**Pattern:** any event-driven recurrent pipeline with stage-gated
output. Identify which path is "state update only" vs "state + output".

### 4.2 VG-only matvec for state-only path

**What:** In state-only patch_cell (skip goW), the Q half of the qvg
matvec is also unused (since `qh = q · h_new` is the only consumer of
q, and qh feeds goW which is also skipped). Skip Q reads + FMAs.

**Why:** qvg matvec is 3 LDG + 3 FMA per inner iter. VG-only is 2 LDG
+ 2 FMA — saves 1/3 of the inner-loop work for state-only events.

**Gain:** Offline 240 → 247 kev/s (+3%), live 225 → 230 kev/s (+2.5%),
commit `f0dc6cf`.

**Pattern:** in any recurrent matmul where the "drop" path is decided
before matmul completes, check whether all matmul outputs are
consumed. Often the gate signal (g, v) and the recurrent feedback (q)
have different downstream paths.

### 4.3 L4 qvgIn smem cache

**What:** Load L4 qvgIn (small layer's weights, 121 KB) into shared
memory once at kernel entry; reuse for every batch.

**Why:** Without it, L4 weights re-read from L2/DRAM every batch.
L4 is the smallest layer's weights so they fit smem.

**Gain:** Offline +4% (227 → 236 kev/s), commit `a76f419`.

**Pattern:** if smem budget allows, cache the layer with smallest
weights AND highest reuse rate (i.e., all-events layers like L4 here).
**Don't cache later-stage weights** — they're too big and the smem
trade-off isn't worth it (L5 = 243 KB, L6 = 486 KB, L7 = 972 KB).

### 4.4 L7 qvgIn L2 persistence (offline only)

**What:** Use `CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW` with
`CU_ACCESS_PROPERTY_PERSISTING` to keep L7 qvgIn (972 KB) "sticky" in
L2 cache, preventing eviction by L4/L5/L6 weight reads between
batches.

**Why:** Total weights = 2.6 MB, L2 = 2 MB. Without pinning, L7
weights get evicted by L4 reads of the next batch, causing DRAM
refetch.

**Gain:** Offline +1.7% (236 → 240 kev/s), p99 tail −25%. **Live: 0%**
(offline gain didn't transfer to live mode).

**Pattern:** L2 persistence is highest-impact when weight set ≫ L2 and
the largest weight is read last per iteration. Use `cuCtxSetLimit
(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, ...)` + per-stream attribute. Limit
your window to **one contiguous buffer** (the largest weight).
Pinning multiple buffers requires allocating them contiguous (which
needs Python weight prep changes).

Status: kept in `perf_celled_multi.py` for reference; not propagated
to live runners because live perf saw no improvement.

### 4.5 Balanced multi-block topology

**What:** For 2W × 4H = 8 blocks, give corner H-strips wider owned-Y
ranges (5/3/3/5 instead of 4/4/4/4) so that **proc widths are equal**
after adding halo.

**Why:** With uniform splits, middle H-strips inherit halo on BOTH
sides → 8 cells proc width vs corner blocks' 6 → middles do 1.33× more
work → middles saturate, corners idle.

**Gain:** Live +3% (218 → 225 kev/s), commit `232f324`. Bonus: lag
distributes evenly across all 8 blocks instead of all on middles.

**Pattern:** any 2D block topology with halo. The interior blocks
always do more work. Tilt the spatial partition to equalize **proc
size**, not owned size.

### 4.6 NEON-vectorized matvec (CPU side)

**What:** Hand-written NEON matvec specializations for IN=12, 24, 48
with fused interior layer kernels (Phase 6 tile-streaming).

**Why:** Compiler-generated AArch64 from generic templates uses scalar
fmadd; hand-NEON exploits 4-way vectorization.

**Gain:** CPU admit went from 0.82 → 2.30 → 2.80 Mev/s (3.4× cumulative,
multiple commits Phase 1–6). See `docs/archive/STATUS.md` §§9–11.

**Pattern:** profile per-segment µs (CPU side: `[cpu seg]` block in
`hybrid_runner.py`). Identify the slowest per-event segment.
Hand-vectorize via NEON intrinsics with explicit float32x4_t register
allocation. **Don't put 18-vector register arrays as `float32x4_t
qvg[18]` — compiler spills to stack. Name the vars explicitly
(`qvg_q0`, `qvg_q1`, ... `qvg_g5`).** See
`include/ssla_fused_layers.h`.

### 4.7 Implicit dispatch (no per-batch task queue)

**What:** Each warp computes its own `(dy, dx)` delta from the event's
`(evx, evy)` directly, rather than reading a per-batch task queue
filled by thread 0.

**Why:** Task queue building was serial (thread 0 only) and required
a `__syncthreads` for the rest of the block to consume. Implicit
dispatch is per-warp parallel and eliminates the sync.

**Gain:** GPU 179 → 184 kev/s (+3%), commit `bdb3c44`.

**Pattern:** if a "scheduler" thread fills a queue and others wait,
check whether each consumer can compute its own slot independently.

### 4.8 Parallel tdrop (9-warp partition)

**What:** Each warp's lane 0 walks the batch and increments the
counter for cells in its `(evy%3, evx%3)` partition.

**Why:** 9 warps process 9 distinct cell partitions in parallel,
each with serial counter access only WITHIN the partition (cell-owner
race-free for counters too).

**Gain:** Versus single-thread (which we tested and it was −2%),
9-warp parallelism wins for BATCH=8 even though each warp does ~1
real increment.

**Pattern:** sequential-looking per-element work can often parallelize
when the elements are partitioned by an independent dimension (here:
cell mod 3).

---

<a id="5-what-doesnt-work"></a>
## 5. Optimization techniques that DON'T work — DO NOT REPEAT

Each entry: what it is, **measured negative result**, why it failed,
when (if ever) it might work elsewhere.

### 5.1 `__launch_bounds__(288, 2)` for 2 blocks/SM on the same kernel

**Tried 2026-05-13.** Reducing register count from 168 to 96 via launch
bounds, then running 8 blocks on 8 SMs in hopes of 2 blocks/SM.

**Result:** Per-block batch time grew **1.7×** when 2 blocks share an
SM. Theoretical 2× scaling collapsed to **1.09×** (memory/compute
contention, not latency-bound).

**Why:** Kernel is **compute/memory-bound**, not latency-bound.
Occupancy hides memory stalls — when there aren't stalls to hide
(plenty of FMA work + cacheable weights), an extra resident block
just splits the same SM resources.

**When might it work elsewhere:** if the kernel is latency-bound (lots
of L2 misses, low FMA density), 2 blocks/SM could help. Profile-first.

See `project_occupancy_gives_little.md`.

### 5.2 Pipeline-split kernels (S2/S3, or 4-layer, in separate blocks)

**Analyzed during 2026-05-13.** Split layers into separate block types
to enable 2 blocks/SM via pipelining.

**Conclusion (without full implementation):** Same SM-contention math
as 5.1. Compute-bound kernels don't benefit from co-resident blocks.
**Expected ~1.1×, not the 1.9× pipeline theory promises.**

Abandoned before implementation. Confirmed by the 16-block test in
5.1 (same effect).

### 5.3 Single-thread `parallel_tdrop`

**Tried 2026-05-14.** Replace 9-warp partition tdrop with single-thread
serial walk.

**Result:** **−2%** (240 → 235 kev/s).

**Why:** BATCH=8 with 9-warp partition is **already parallel**
(different warps handle different cells). The "savings" of avoiding 9
warps' dispatch overhead are smaller than the loss of parallelism.

See section above for the kept `parallel_tdrop`.

### 5.4 TF32 Tensor Core L4 qvg precompute

**Tried 2026-05-14.** Implement m16n8k4 TF32 MMA for L4 qvg matmul,
batching 8 events as n=8. Pre-compute all 9 deltas' Q/V/G into
`qvg_buf` smem; cell-owner loop reads pre-computed values.

**Result:** **−5% offline, −3% live**. P1 unchanged (max|Δ| = 4.202).

**Why TF32 didn't break accuracy:** 4-layer LRU+LN cascade tolerates
TF32's reduced mantissa precision. This is **useful info for future TC
attempts**: TF32 is precision-safe on this network.

**Why net negative:** qvg_buf requires 41 KB smem; only way to fit was
removing the 121 KB L4 qvgIn smem cache (4.3). Smem-cache loss > TC
gain. **TF32 itself is roughly compute-neutral** vs CUDA-core matvec
at our shapes (k=24, n=8 too small for TC throughput to dominate).

**When might it work elsewhere:** if your model has k ≥ 64 (typical
attention/MLP layers in transformers) OR you can keep both smem caches
AND qvg_buf (e.g., larger smem budget on Hopper's 228 KB/block).

See `project_tf32_tc_l4_prototype.md`.

### 5.5 Batch-local hidden-cell coalescing

**Tried 2026-05-14.** Group events targeting the same hidden cell;
load h_cell once per group, apply LRU updates in registers, store once.

**Result:** **−6% offline**. P1 unchanged.

**Why:** Per-batch reuse ratio measured at **1.097 on uniform synth**
(only ~9% of cell-touches are repeats; 91% are unique cells). The
saved load/store is ~30 cycles × 0.7 saves/batch = 21 cycles ≈
0.02 µs. The grouping bookkeeping (smem dynamic indexing, bitmask
control flow, `__ffs` loop) added 16 µs/batch overhead — 700× the
savings.

**When might it work elsewhere:** if your event source has hot pixels
giving k=4+ cell collisions frequently. Run `analyze_coalescing.py`
on your actual data first; if reuse > 1.5 or k=3+ cells > 20%,
consider it. Otherwise no.

See `project_coalescing_net_negative.md`.

### 5.6 cp.async for weight prefetch into smem

**Analyzed 2026-05-13.** Use cp.async to overlap L5/L6/L7 weight loads
with previous-layer compute.

**Blocked:** smem already at 166/167 KB (L4 cache + contrib + ...).
No room for prefetch buffers. Freeing contrib smem (27 KB) requires
algorithm change to gather phase that costs more time than cp.async
would save.

**When might it work elsewhere:** larger smem budget (Hopper 228 KB)
or different gather algorithm.

### 5.7 L6+L7 contiguous L2 persistence (1458 KB pinned)

**Tried 2026-05-13.** Allocate L6 and L7 qvgIn in one contiguous buffer,
pin 1458 KB in L2 persistence (within 1.5 MB budget).

**Result:** **Same throughput as L7-only pinning** (240 kev/s). p99
similar.

**Why:** Pinning 1458/2048 KB leaves only 590 KB for non-persistent
data (L5 qvgIn 243 KB, all goW ~810 KB, hidden state, etc.) → those
get evicted more often. Net wash.

**Lesson:** L2 persistence isn't free; pinning more displaces what's
not pinned. Sweet spot is ONE buffer that's both largest AND read last
per batch.

### 5.8 Removing L4 qvgIn smem cache to fit other smem optimizations

Multiple attempts hit this trade-off. The L4 smem cache is 121 KB =
72% of the 167 KB budget. Almost any new smem-resident structure
displaces it. **The displacement cost (−4%) is usually larger than
the new structure's gain.**

Lesson: when proposing a new smem optimization, **first** check that
it can coexist with the L4 cache. If not, find a way to make it
smaller, or accept that you're trading 4% baseline for the new
optimization.

---

<a id="6-camera-tuning"></a>
## 6. Camera tuning (DVXplorer / DVXplorer Micro)

### 6.1 Open the camera

```python
import dv_processing as dv
cam = dv.io.camera.open()    # picks first connected DVXplorer/Micro
print(cam.getCameraName(), cam.getEventResolution())
```

Tested with **`dv-processing` 2.0.3** (apt: `dv-processing-python`).
**libcaer 3.3.17 does NOT support the Micro variant** — use
dv-processing.

### 6.2 Contrast threshold valid range is 0–17

**This is the #1 footgun.** The chip silently clamps any value above
17, but `getContrastThresholdOn()` returns whatever you `set()`.
Readback alone won't tell you the clamp happened.

Confirmed by event-rate sweep on DXUS0026 (DVXplorer Micro): rates
plateau identically at thresholds 18, 50, 200.

**Always cap your slider at 17** (we do — see `event_viewer.py:CONTRAST_MAX`).

**Useful working points** (measured under uniform random-noise motion
proxy):
| threshold | rate kev/s | comment |
|---:|---:|---|
| 2 | 14 000+ | extreme sensitive, noise-dominated |
| 4 | 1500 | very sensitive |
| 6 | 300 | balanced |
| 8 | 130 | balanced |
| 10 | 115 | balanced |
| 12 | 70 | low |
| 17 | 40 | chip floor / minimum response |

See `project_dvxplorer_contrast_range.md`.

### 6.3 Tuning under controlled motion

Hand-wave motion isn't reproducible. Use `--motion-probe`:

```bash
cd orin/viz
python3 event_viewer.py --motion-probe   # 8 thresholds × 3 s each, ~28s total
```

The user provides **continuous motion** for the duration; the tool
cycles thresholds automatically and reports rate per threshold. **Same
motion pattern → fair comparison.**

For more granular sweep:
```bash
python3 event_viewer.py --motion-probe \
    --motion-probe-thresholds 4,6,8,10,12 \
    --motion-probe-dwell-s 4.0
```

### 6.4 Other tuning knobs

| knob | API | range / note |
|---|---|---|
| ContrastThresholdOn | `setContrastThresholdOn(v)` | 0–17 (clamps) |
| ContrastThresholdOff | `setContrastThresholdOff(v)` | 0–17 (clamps) |
| Subsample (2x2, 4x4, ...) | `setSubSampleHorizontal/Vertical` | reduces total events 4x / 16x |
| ReadoutFPS | `setReadoutFPS(v)` | chip-internal readout rate |
| Hot-pixel filter | not a chip setting — applied in software in `event_viewer.py` (see `--hot-pixel-filter`) | |
| Crop / ROI | `setCropArea` | reduces field of view |
| Flip H/V | `setFlipHorizontal/Vertical` | physical orientation correction |

For full method list on a connected camera:
```bash
python3 event_viewer.py --list-methods
```

---

<a id="7-new-hardware"></a>
## 7. New hardware adaptation checklist

When porting to a different GPU / device, work through this in order.

### 7.1 Capability discovery

```python
from cuda import cuda
cuda.cuInit(0)
err, dev = cuda.cuDeviceGet(0)
name = cuda.cuDeviceGetName(128, dev)[1].decode().strip('\x00')

attrs = {
    "SM count": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    "L2 size": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    "Persistent L2 max": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
    "Smem per block max (optin)": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    "Regs per SM": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
    "Concurrent managed access": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
    "Compute capability major": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    "Compute capability minor": cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
}
for label, attr in attrs.items():
    _, val = cuda.cuDeviceGetAttribute(attr, dev)
    print(f"{label}: {val}")
```

**Reference values for Orin NX (sm_87):**
- SM count: 8
- L2: 2 MB
- Persistent L2 max: 1.5 MB
- Smem per block optin: 167 KB
- Regs per SM: 65 536
- CONCURRENT_MANAGED_ACCESS: 0 (Tegra-specific)

### 7.2 Topology adjustment

Number of blocks should typically equal SM count for max occupancy at
1 block/SM. With 8 SMs we use 8 blocks. On Orin AGX (12 SMs) you'd
likely want 12 blocks. On a discrete GPU with 80+ SMs, you'd want
many more blocks per CPU shard.

Topology grid: for n_blocks divisible into 2W × nH, see
`orin/orin/multi_block.py::grid_topology`. For 12 blocks, 2W × 6H
(unconfirmed) or 4W × 3H — pick what aligns best with W2/H2 grid.

### 7.3 Smem budget recalculation

Run `cuFuncGetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` on the compiled
kernel. The L4 qvgIn smem cache (121 KB) is the biggest consumer; if
the new device has < 167 KB optin, you may have to drop the cache.

### 7.4 Coherence model

Test `CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`. If `1`, you can
use managed memory for ring traffic (simpler, no pinned alloc
required). If `0` (Tegra), MUST use pinned host memory for any buffer
with concurrent CPU writes + GPU reads.

### 7.5 L2 persistence support

`CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE`. If 0, the feature
isn't supported on this device (older arch). Skip the L7 L2
persistence path.

### 7.6 SM clock / DVFS

`nvpmodel -q` + `jetson_clocks --show` to see current frequency. For
reproducible benchmarks, **pin frequencies** (requires root). Without
root, the ~5 ms calibration bias for cross-clock latency stands.

### 7.7 Run the full test suite

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cd orin
python3 bench_s2_s3_head_celled.py --n 50000    # P1
python3 perf_celled_multi.py --n-blocks 8        # offline perf
python3 perf_celled_profile.py --n 30000         # per-phase
python3 hybrid_runner_multi.py --weights ... --gpu-blocks 8 ...   # live
```

Compare numbers to `docs/PROFILE.md`. Any deviation > 20% → investigate.

---

<a id="8-new-camera"></a>
## 8. New camera adaptation checklist

### 8.1 Drivers

- iniVation cameras (DVXplorer, DVXplorer Micro): `dv-processing` 2.0+
- Prophesee Metavision: `metavision-sdk`, completely different API
- Samsung Gen4: vendor-specific SDK
- Generic AER cameras: may need libcaer / custom

Each requires its own event-fetching code in `orin/viz/event_viewer.py`
and `src/lib_stage01_to_gpu.cpp`.

### 8.2 Resolution

Current default: 80×64 post-subsample. Source: 640×480 DVXplorer Micro
with 8×8 software subsample (via `setSubSampleHorizontal/Vertical=3`).

For higher-resolution cameras, **decide subsample at the camera level
if supported** (cheaper than software-subsample after ingest). Update
`H_FULL` / `W_FULL` and the grid dimensions.

### 8.3 Threshold / sensitivity equivalent

Each camera has a different sensitivity API. Find the equivalent of
`setContrastThresholdOn/Off` and:
1. Test the valid range (sweep, check rate change)
2. Document chip floor and dynamic range
3. Cap your UI trackbar at the chip's actual valid max

See `analyze_coalescing.py` and `motion_probe()` for the pattern.

### 8.4 Hot-pixel filtering

Some cameras have a fraction of "always firing" pixels at any
threshold. Use software hot-pixel filter (existing in
`event_viewer.py`) or chip-level filter if available.

### 8.5 Event format

DVXplorer format: `{t, x, y, polarity}` as raw uint16/32 packed. Check
the new camera's batch format and adjust `INPUT_DTYPE` in
`orin/orin/hybrid_common.py`.

### 8.6 Verify event flow

Before any modeling, smoke-test event delivery:
```bash
cd orin/viz
python3 event_viewer.py    # GUI should show events with motion
```

If events don't flow, debug at the SDK level before touching the
inference pipeline.

---

<a id="9-new-variant"></a>
## 9. New SSLA variant adaptation

### 9.1 What changes when channel dims change

If a variant has different `C1, C2, C3`:

| dim | depends on |
|---|---|
| `C1` (s2 input) | model architecture |
| `C2` (s2 output, s3 input) | model architecture |
| `C3` (s3 output, head input) | model architecture |

Files to update:
- `include/ssla_kernels.h` (constants, struct layouts)
- `orin/orin/hybrid_common.py` (`C1`, `C2`, `C3` constants)
- `orin/kernels/ssla_s2_s3_head_celled.cuh` (`constexpr int C1`, `C2`, `C3`)
- All smem-size calculators in `orin/perf_*.py` / `bench_*.py` /
  `hybrid_runner*.py` (search for `* C2 * C1` etc.)
- L4 smem cache size = `9 * 3 * C2 * C1 * 4`. Check it still fits in
  167 KB.

### 9.2 What changes when patch size K changes

The current kernel hard-codes K=3 (3×3 patch → 9 cells per event → 9
warps). Changing K requires:

- **K=5**: 25 cells per event, but the cell-owner partition (mod 3)
  still has 9 distinct partitions. Each warp now owns more cells. Per-
  event work in each warp scales by 25/9 ≈ 2.8×.
- **K=2 or K=4**: mod-3 partition doesn't fit cleanly. Need to
  re-derive cell-owner partition (probably mod 2 → 4 warps for K=4,
  or mod 1 → 1 warp for K=2). Major kernel restructure.

P1 oracle (`bench_s2_s3_head_celled.py`) hard-codes K=3 in
`SSLA_S2S3_KERNELS`. Update.

### 9.3 What changes when stage count changes

The current design has 4 GPU layers (L4..L7) split across 2 GPU stages
(s2, s3). For different stage counts:

- **3 GPU stages** (e.g., s2, s3, s4): add tdrop_s4, L8, L9. Re-derive
  smem budget. Add another set of weights / hidden state / counters
  to `HybridS2S3Config`.
- **1 GPU stage**: drop s3 entirely. Simplification.

CPU side (s0+s1) and GPU side are decoupled by the CPU→GPU ring; you
can vary stage counts on each side independently.

### 9.4 What changes when tdrop_window changes

Currently `tdrop_window = 4`. Changing affects:
- pass rate (1 / tdrop_window)
- VG-only path activation rate
- balance between full-work and state-only events

**Higher tdrop_window** (e.g., 8) → fewer events through pipeline →
lower GPU load → likely higher throughput but **lower temporal
resolution** in output. Verify task-specific accuracy.

**Lower tdrop_window** (e.g., 2) → more events through pipeline →
higher GPU load → may bottleneck more.

### 9.5 What changes when recurrence type changes

Current SSLA uses LRU-style gated recurrent: `h_new = sigmoid(g) · h +
v`. Alternatives:

- **Simple EMA**: `h = α · h + (1-α) · x`. No qh, simpler. VG-only
  becomes trivial (no Q exists).
- **GRU / LSTM**: more gates, more matmuls. Likely doesn't fit smem.
- **No state**: feed-forward only. Trivially batchable, Tensor Cores
  become useful (no per-event m=1 constraint).

A **non-recurrent variant** would unlock Tensor Cores at full
efficiency. Major architectural decision.

### 9.6 Re-validation after variant change

1. Update CPU oracle (`orin/orin/ssla_ref.py::LayerRef`, `layer_step`)
   to match new variant math.
2. Update `bench_s2_s3_head_celled.py` (P1) to use new oracle.
3. Run P1 — note new `max|Δ|` (will differ from current 4.2).
4. Update P1 tolerance budget if the new variant has different
   numerical sensitivity.
5. Run perf benches to see how the changes shifted bottlenecks.

---

<a id="10-codesign"></a>
## 10. Co-design opportunities (not yet shipped)

These haven't been implemented in this session but have **plausible
analysis showing positive ROI**. They each require model retraining
or accuracy validation; pursue when budget allows.

### 10.1 Channel reduction via knowledge distillation

**Idea:** Train a smaller SSLA variant (e.g., C1=16, C2=32, C3=64)
distilled from current model. Compute scales with C².

**Expected gain:** Throughput +50–80%, latency −30–40%. Hidden state
fits L2 more comfortably.

**Risk:** mAP loss. KD is well-studied; <1% loss typical.

**Effort:** Training-side (days–weeks).

### 10.2 Hidden state storage in bf16

**Idea:** Store hidden state buffers as bf16 (compute remains fp32 with
cast on read / write).

**Expected gain:** +5–15% throughput (less L2/DRAM bandwidth for the
~600 KB hidden state).

**Risk:** Numerical drift. bf16 has fp32 exponent range; mantissa is
the only loss. P1 must re-pass.

**Effort:** Small kernel + harness changes (~50–100 lines).

### 10.3 Structured 2:4 sparsity in weights

**Idea:** Train with 2:4 sparsity regularization; inference uses
Ampere/Hopper sparse TF32 MMA (2× throughput at same shape).

**Expected gain:** Uncertain. Could finally make TC profitable if
combined with smem cache.

**Risk:** Accuracy depends on whether the model can be sparsified.

**Effort:** Training + custom CUDA path for sparse MMA.

### 10.4 Streaming output (per-event prediction write)

**Idea:** Currently, predictions for all output-producing events in a
batch are written at the end of the batch (OUT phase). Stream them:
write event N's prediction as soon as warp N finishes its L7 gather.

**Expected gain:** −3 µs latency for output-producing events. Zero
throughput impact.

**Risk:** None — pure code change.

**Effort:** ~50 lines kernel.

### 10.5 Reduce CPU→GPU event multiplication (halo replication)

**Idea:** Halo replication in CPU shards roughly doubles GPU input
events (each event in 1–2 shards' proc range). Re-architect so each
event pushes to GPU exactly once.

**Expected gain:** +30–50% live throughput.

**Risk:** Need to preserve halo correctness (each block's hidden
state must reflect events at its boundary).

**Effort:** Significant CPU + ring restructure.

---

<a id="11-testing"></a>
## 11. Testing & validation protocols

### 11.1 The mandatory test trinity

For every kernel change, run **all three** in this order:

1. **P1 correctness** (`bench_s2_s3_head_celled.py --n 50000`):
   - PASS criteria: `drift = 0`, `max|Δ| ≤ 5.0` (current budget)
   - FAIL → revert; do not measure perf on broken kernel
2. **Offline throughput** (`perf_celled_multi.py --n-blocks 8 --n 200000 --runs 3`):
   - Stable values: 240–250 kev/s
   - First run usually lower (PTX cache miss); take run 2 / 3
3. **Live test** (`hybrid_runner_multi.py @ synth 2.0 Mev/s`):
   - Drain ~227 kev/s, latency p50 ~330 µs
   - 8 s duration minimum for stable percentiles

### 11.2 Unit tests

```bash
cd orin
python3 tests/test_proto_layer_pair.py    # GPU primitive vs numpy
python3 tests/test_ring_smoke.py          # SPSC ring stress
python3 tests/test_ring_unit.py           # SPSC ring correctness
python3 tests/test_weights_ssla.py        # weight reshape correctness
python3 tests/test_weights_unit.py        # weight loader
```

All must pass after any change to `orin/orin/` or the active kernels.

### 11.3 Adding a new test

If you implement a new optimization, add a new test in `orin/tests/`
that exercises the specific code path. Pattern:

```python
"""Tests that <new opt> produces identical output to reference."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ...
```

Tests must be self-contained Python scripts (not pytest-only) so they
can run on systems without pytest installed.

### 11.4 Regression tests

After each commit that touches a kernel:
1. P1 oracle: check `max|Δ|` value. If it changed materially (> 0.1),
   investigate.
2. Perf bench: check kev/s. If it dropped > 3% across 3 runs, investigate.
3. Profile: check per-phase µs. Identify which phase changed.

Save these numbers in commit messages so the bisect-trail is readable.

---

<a id="12-pitfalls"></a>
## 12. Common pitfalls & footguns

### 12.1 `/tmp/` is volatile on Orin

Reboot wipes `/tmp/ssla_s_64x80/`. Always regenerate stub weights
after reboot:
```bash
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80
```

### 12.2 NVRTC PTX cache miss = ~80 s rebuild

First run of any new kernel hash compiles from scratch (slow).
Subsequent runs hit `~/.cache/openeva/orin_ptx/` and load in < 1 s.

To force cache miss (e.g., to verify cache correctness):
```bash
OPENEVA_ORIN_NO_PTX_CACHE=1 python3 perf_celled_multi.py ...
```

### 12.3 `__launch_bounds__(288, 1)` required for smem opt-in

Without this, NVRTC defaults to `MAX_THREADS_PER_BLOCK=256`, and
`cuLaunchKernel` returns error 701 (LAUNCH_OUT_OF_RESOURCES) on the
9 warp × 32 lane = 288-thread launch.

### 12.4 Cross-clock CPU/GPU latency has ~5 ms residual bias

Without `jetson_clocks` (requires root), the SM clock DVFS makes the
linear-fit ns-per-cycle calibration error up to 5 ms. The
`emit→done` cross-clock latency can come out NEGATIVE in extreme
cases (the bias exceeds the actual latency on short batches).

Workaround: report **`emit→push` + `kernel`** (both domain-internal,
unbiased) instead of one cross-clock `emit→done` number.

### 12.5 Hybrid stats "kept" column includes pad duplicates

When P1 bench pads block ring sizes to a multiple of BATCH=8 (so
kernel doesn't need a partial-batch path), the padding events pass
through the kernel. Filter via global event index when comparing
oracle vs GPU.

### 12.6 Managed memory page faulting on first GPU touch

Weights stored via `cuMemAllocManaged` are NOT in GPU memory until
first GPU access. The first kernel batch suffers a paging stall.
**Always warm up** with 1–2 batches before measuring throughput.

`perf_celled_multi.py` does this implicitly (the first `run` is
typically 5–10% lower than runs 2 and 3 — discard run 0 for headline
numbers).

### 12.7 `cv2.namedWindow` failure in viz tools

If running headless (no `$DISPLAY`), `cv2.namedWindow` throws. The
`event_viewer.py` falls back to print-only mode if creation fails.
If you need plots without GUI, use `--no-window`.

### 12.8 dv-processing getter doesn't enforce chip clamp

`cam.setContrastThresholdOn(200)` followed by
`cam.getContrastThresholdOn()` returns 200, but the chip is running at
17 (the clamp). The getter just returns whatever was passed in.
**Use rate measurements** to detect clamps, not readbacks. See §6.2.

### 12.9 Per-batch smem layout must match across files

Smem size is computed in **multiple files** (Python harnesses + kernel
ctype mirror). When you change smem layout in `.cuh`, update **all**:
- `perf_celled.py::_smem_size_bytes`
- `perf_celled_multi.py::_smem_size_bytes`
- `perf_celled_profile.py::_smem_size_bytes`
- `bench_s2_s3_head_celled.py::_smem_size_bytes`
- `hybrid_runner.py` inline SMEM calc
- `hybrid_runner_multi.py` inline SMEM calc

Forgetting one → `cuLaunchKernel` error 701 OR silent buffer overrun.

### 12.10 Profile kernel adds overhead

`ssla_s2_s3_head_celled_profile.cuh` inserts `STAMP()` markers between
phases (extra `clock64` calls + smem store). Profile mode is ~10%
slower than production. **Don't quote profile-mode kev/s as headline
numbers.**

---

<a id="13-tools"></a>
## 13. Tools & reference

### 13.1 Active scripts (orin/ root)

| script | purpose | typical args |
|---|---|---|
| `hybrid_runner.py` | 2-block live runner | `--cpp-synth --synthetic-mev 2.0 --duration-s 10` |
| `hybrid_runner_multi.py` | N-block (2/4/8) live runner | `--gpu-blocks 8 --shards 6 --cpp-synth --synthetic-mev 2.0` |
| `bench_s2_s3_head_celled.py` | P1 oracle | `--n 50000` |
| `perf_celled.py` | 2-block offline throughput | `--n 100000 --runs 3` |
| `perf_celled_multi.py` | N-block offline throughput | `--n-blocks 8 --n 200000 --runs 3` |
| `perf_celled_profile.py` | per-phase µs profile | `--n 30000` |
| `analyze_coalescing.py` | hidden-cell reuse statistics | `--n 200000 --tdrop 4` |
| `viz/event_viewer.py` | live event viewer + tuning | (GUI); see §6 |

### 13.2 Memory files (Claude project memory)

These live in
`/home/nanod/.claude/projects/-home-nanod-ssla-rt/memory/` and capture
verified findings:

- `project_deploy_target.md` — Jetson Orin NX target, Tegra coherence
- `project_dvxplorer_micro_libcaer.md` — libcaer 3.3.17 doesn't open
  Micro; use dv-processing
- `project_cell_owner_design.md` — race-free 9-warp partition
- `project_skip_output_for_dropped.md` — A1 optimization rationale
- `project_occupancy_gives_little.md` — 2 blocks/SM only +1.1×, not 2×
- `project_cpu_gpu_event_ratio.md` — GPU sees 1/16 of CPU admit (in
  theory; live overhead changes the actual ratio)
- `project_tf32_tc_l4_prototype.md` — TC prototype was net-negative
- `project_coalescing_net_negative.md` — coalescing net-negative
- `project_dvxplorer_contrast_range.md` — contrast threshold clamps at 17
- `feedback_measure_before_promise.md` — don't quote speedup without
  measuring
- `feedback_no_vendor_edits.md` — override via specialization headers
- `feedback_halo_2_locked.md` — halo=2 not negotiable
- `feedback_no_gpu_mention.md` — don't compare GPU when working on CPU
- `feedback_no_substitute_approach.md` — execute user's choice, don't
  substitute
- `feedback_do_the_division.md` — convert rates between stages
- `feedback_no_theoretical_for_live.md` — bottleneck claims need live
  numbers
- `feedback_no_substitute_approach.md` — when user picks X, do X

### 13.3 Key commits (chronological highlights)

| commit | what |
|---|---|
| `89e6544` | CPU NEON specialization for IN=12 (Phase 1) |
| `e9ecf4d` | CPU Phase 3 fused interior layers |
| `7b2bdd2` | CPU Phase 4: fused s0 L0, memcpy elim |
| `830e601` | CPU Phase 5: C++ synth dispatcher |
| `2f13555` | CPU Phase 6: tile-streaming s1 L0+L1 |
| `b91e989` | GPU step 2: multi-block + merged qvg (40→179 kev/s) |
| `bdb3c44` | GPU C1: implicit dispatch + parallel tdrop |
| `6f287d3` | GPU A1: skip output for tdrop-dropped events |
| `d876030` | CPU A1: same idea on CPU side |
| `4e2d2e9` | Multi-block live mode (N=4/8) |
| `f0cc228` | Whole-pipeline latency instrumentation |
| `a76f419` | L4 qvgIn smem cache (+4%) |
| `288fbf3` | Profile kernel mirror of A1 + split |
| `4060a51` | L7 qvgIn L2 persistence |
| `f0dc6cf` | VG-only state-only matvec |
| `232f324` | Balanced multi-block H-strips |
| `93a843b` | Cap contrast threshold trackbar at 17 (chip range) |
| `a3f9392` | Repo reorganization |

### 13.4 Build / run reference card

```bash
# === build ===
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# === weights ===
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80

# === P1 correctness ===
cd orin && python3 bench_s2_s3_head_celled.py --n 50000

# === offline throughput ===
python3 perf_celled_multi.py --n-blocks 8 --n 200000 --runs 3

# === per-phase profile ===
python3 perf_celled_profile.py --n 30000

# === live, synth ===
python3 hybrid_runner_multi.py --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --cpp-synth --synthetic-mev 2.0 \
    --duration-s 8 --pin-python-main --base-core 1

# === live, camera ===
python3 hybrid_runner_multi.py --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --duration-s 30 \
    --pin-python-main --base-core 1

# === camera tuning ===
cd viz
python3 event_viewer.py                        # GUI
python3 event_viewer.py --motion-probe         # threshold sweep under motion

# === cleanup PTX cache (force recompile) ===
rm -rf ~/.cache/openeva/orin_ptx/

# === throughput vs synth-rate sweep ===
for mev in 0.5 1.0 1.5 2.0 2.5 3.0 4.0; do
  python3 hybrid_runner_multi.py --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --cpp-synth --synthetic-mev $mev \
    --duration-s 6 --pin-python-main --base-core 1
done
```

---

## Appendix A: optimization decision tree

When you have a kernel and want to make it faster:

```
1. Run P1 — ensure baseline correct.
2. Run perf_celled_multi → headline kev/s.
3. Run perf_celled_profile → identify dominant phase.
4. Is dominant phase COMPUTE? (≥ 60 % share)
   ├── YES → memory-bound or instruction-bound?
   │   ├── Check arithmetic intensity (FMA / byte). If low → memory
   │   │   ├── Try smem caching of the most-reused weight subset
   │   │   ├── Try L2 persistence on largest weight buffer
   │   │   └── Try reducing memory reads (e.g., VG-only for state-only events)
   │   └── Check FMA utilization. If low → instruction-bound
   │       ├── Profile with nsys / ncu (if available)
   │       └── Look for stalls (sync, mem latency, divergence)
   └── NO → which phase dominates?
       ├── GATHER → try reducing cross-warp smem traffic
       ├── LOAD/POP → try cp.async or larger batches (if smem allows)
       ├── DISPATCH → can each warp compute its own task (implicit dispatch)?
       └── SYNC overhead → reduce __syncthreads count, use __syncwarp instead
5. After implementing: P1 → perf → profile. Compare each phase to baseline.
   ├── Phase you targeted: did it shrink?
   ├── Other phases: did any regress?
   └── Overall throughput: net +/-?
6. Decision:
   ├── If P1 fails → revert
   ├── If target phase shrunk and net positive → commit
   ├── If target phase shrunk but other phases regressed → analyze trade-off
   └── If target phase didn't shrink → analysis was wrong, debug
```

## Appendix B: what to do when a new "expert suggestion" arrives

1. **Read it carefully.** Identify which constraints it touches.
2. **Cross-check against [§2](#2-architecture-invariants).** If it
   violates an invariant, ask the suggester to revise.
3. **Cross-check against [§5](#5-what-doesnt-work).** If it's been
   tried and failed in this session, share the prior result.
4. **Estimate gain conservatively.** Use a cycle-count or memory-
   traffic model, not a "should be 2× because TC".
5. **Implement minimally.** Don't propagate to all 4 kernels until
   it's proven on one.
6. **Run P1 first.** Then perf. Then live.
7. **Compare against the conservative estimate.** If net positive AND
   ≥ 50% of the estimated gain → commit. If net negative or marginal
   → revert and record the negative result in memory.

The goal isn't to implement every suggestion; the goal is to
**convert suggestions into experimental data, positive or negative**,
that the next developer can rely on.
