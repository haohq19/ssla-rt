# SSLA-S Real-Time Inference Pipeline — Current Design

**Status as of 2026-05-13.** Self-contained reference for expert consultation
on remaining optimization paths.

---

## 1. System target

- **Application:** Real-time inference of an SSLA-S event-camera model
  (4-stage recurrent backbone + YOLOX head). Streams events from a DVXplorer
  / DVXplorer Micro through the full network end-to-end **with no batching
  across events** — each event updates per-cell hidden state and may produce
  a prediction.
- **Hardware:** NVIDIA Jetson Orin NX (ARM aarch64 + Ampere sm_87 GPU).
  Key device parameters:
  - 8 SMs at 918 MHz (pinned in MAXN power mode).
  - 128 KB combined L1/smem per SM; **167 KB max optin smem per block**.
  - 65,536 registers per SM.
  - 2 MB L2 cache, **1.5 MB max persistent L2 reservation** (Ampere feature).
  - 58.8 GB/s DRAM bandwidth (LPDDR5).
  - `CONCURRENT_MANAGED_ACCESS = 0` → **managed memory is NOT safe for
    concurrent CPU writes + GPU reads on this device**; all cross-CPU/GPU
    traffic must use pinned host memory.

## 2. SSLA-S architecture

Four stages, each with 2 layers (L0..L7), 2× spatial downsample between
stages, hidden state at every cell of every stage:

| stage | layers | grid (full) | grid (typical) | input dim | output dim |
|-------|--------|------|------|-----------|------------|
| s0    | L0, L1 | 80×64 | 80×64 | 4  | 12  |
| s1    | L2, L3 | 40×32 | 40×32 | 12 | 24  |
| s2    | L4, L5 | 20×16 | 20×16 | 24 | 48  |
| s3    | L6, L7 | 10×8  | 10×8  | 48 | 96  |

Each layer has 9 spatial "delta" weight tensors (K=3 spatial conv) plus
per-position projection matrices: `qvgIn (9, in, 3·out)` (combined Q/V/G
matvec inputs), `goW (9, out, out)` (output projection), optional
`input_proj (in, out)` for channel-change residual, and per-stage
`ln_gamma / ln_beta`. Recurrence is gated linear (LRU-style) per cell:

```
qh = lru_step(g·sigmoid + v, q, h)   # h is per-cell global memory
contrib = goW · qh                    # output of one patch position
event_output = LN(input_proj·in + Σ_9_positions contrib)
```

Each event also gates per-stage **tdrop** counters: a per-cell counter
increments; the event "passes" only when `count % tdrop_window == 0`
(default `tdrop_window=4`). An event that fails tdrop_s2 still updates
L4/L5 hidden state but is dropped before reaching s3.

The pipeline performs 2 sequential tdrops between CPU admit and GPU work
(after CPU s0 and CPU s1), and 2 more inside GPU (after s2 pool and after
s3 forward). At `tdrop_window=4` each tdrop nominally passes 1/4 events,
so GPU input rate ≈ CPU admit rate / 16, with halo replication scaling
this back up by ~2×.

## 3. Pipeline overview

```
camera/synth  →  CPU shards (s0+s1)  →  per-block ring  →  GPU blocks (s2+s3+head)
   (events)        N shards, halo=2     pinned host mem    2W×4H = 8 blocks,
                                        SPSC per block      9 warps/block
```

- **CPU side runs s0+s1** (low-channel, high-resolution stages).
  Implemented in C++ with NEON intrinsics, 6–8 worker shards on dedicated
  cores. Each shard owns a horizontal s2 strip with halo=2 (input space)
  and reproduces the work for boundary events.
- **GPU side runs s2+s3+head** (high-channel, low-resolution stages,
  recurrent + YOLOX head).
- **Boundary:** each shard owns one GPU block; CPU pushes records into a
  per-block pinned-memory SPSC ring. GPU blocks pop and process events
  one batch at a time (BATCH=8 events per dispatch).

## 4. GPU multi-block topology

For N=8 GPU blocks at s2 grid 20×16:

```
        Y →   [0, 5)    [5, 8)    [8, 11)   [11, 16)    ←  owned Y ranges (balanced)
              proc [0,7) [3,10)   [6,13)    [9,16)      ←  all proc Y widths = 7
   X
   [0, 10)    blk0      blk1      blk2      blk3
   [10, 20)   blk4      blk5      blk6      blk7
   proc X
   [0, 12)   [8, 20)
```

- **Halo = 2 (LOCKED).** Each block's proc range extends ±2 cells into
  neighbouring blocks' owned ranges, so the 3×3 patch dependencies for
  s2 conv and the post-pool 3×3 dependencies for s3 conv are all satisfied
  by the block's own hidden state copy.
- **Balanced Y splits** (commit 232f324): with the default uniform 4×4×4×4
  split, middle H-strips had halo on BOTH sides, giving them 1.33× more
  proc cells than corners. Live measurements at synth ≥2 Mev/s showed
  middles saturating (52k-event backlog) while corners sat idle at 0 lag.
  Switching to 5/3/3/5 owned Y sizes equalizes proc widths to 7 cells
  everywhere. Measured gain: +4% drain, lag distributed 21–30k across all
  8 blocks instead of concentrated.
- Each block has its own hidden state buffer (per layer) sized to its
  proc range, its own tdrop counter array, and its own prediction
  destination cells (determined by `s3_owned_y = owned_y >> 1`).
- Weights are **shared across blocks** — all 8 blocks read the same
  qvgIn / goW / input_proj / ln_* pointers per layer.

## 5. GPU kernel design — cell-owner warp partition

The production kernel is `k_ssla_s2s3_celled_persistent_multi` in
`orin/kernels/ssla_s2_s3_head_celled.cuh`. Key design choices:

### 5.1 Cell-owner warp partition (race-free invariant)

- 9 warps per block (288 threads = 9·32 lanes). Each warp owns the
  hidden-state cells with `(cell_y % 3, cell_x % 3) == (warp_id / 3, warp_id % 3)`.
- For any event at position `(evx, evy)`, the 3×3 patch around it touches
  9 cells with 9 *distinct* `(cy%3, cx%3)` pairs → exactly one warp per
  cell. **No two warps ever write to the same hidden-state cell.**
- Consequence: **no GPU atomics needed.** `__syncwarp()` only inside the
  per-warp inner loops, `__syncthreads()` only between batched phases.

### 5.2 Implicit dispatch (no per-batch task queue)

Earlier revisions built a `task[warp][slot]` queue mapping events to
warps. Current kernel skips this — each warp computes its own delta
directly from event coordinates:

```c
const int dy = ((own_y - evy) % 3 + 3) % 3 - 1;   // ∈ {-1, 0, 1}
const int dx = ((own_x - evx) % 3 + 3) % 3 - 1;
// Bounds-check (py, px); if in proc range, do the patch.
```

### 5.3 Per-event flow (one batch)

```
for batch_base in [0..n_events) step BATCH:
    LOAD     : 9 warps cooperatively pull BATCH event records from ring
               into smem (event_slots[BATCH]).
    TDROP_S2 : 9 warp-leaders run parallel_tdrop — each warp handles
               events at cells with its (evy%3, evx%3); counter increments
               within a warp stay in-order, across warps disjoint.
               Output: pass2[BATCH] mask.
    L4       : 9 warps × BATCH events; each warp computes its owned-cell
               patch (qvg matvec + lru_step + goW matvec) and writes
               its contrib slot in smem. 1 warp per event runs the
               residual matvec.
    L4_GATHER: 1 warp per event sums 9 contribs + residual, applies LN,
               writes back to event_slot.in_feat (= input to L5).
    L5       : Same as L4 but with input_mask = none, output_mask = pass2
               (so events failing tdrop_s2 only run qvg + lru — they still
               update hidden state but skip goW + gather + LN).
    POOL     : Right-shift evx, evy by 1 → s3 coordinates.
    TDROP_S3 : parallel_tdrop on s3 grid with mask=pass2 (events failed
               at s2 don't even enter s3 tdrop). Output: pass3[BATCH].
    L6       : input_mask = pass2 (s2-dropped events skip L6 entirely);
               output is full for events that pass.
    L7       : input_mask = pass2, output_mask = pass3 (similar A1 logic).
    HEAD     : YOLOX classification + box regression for owner-passed
               events at their s3 cell. Each block writes to its s3_owned
               prediction cells; readers double-buffer via a `version[]`
               array.
```

Phase boundaries are `__syncthreads()`. Per-batch median compute time
≈ 217 µs in offline benchmark.

### 5.4 Smem layout (per block, ~166 KB, opt-in)

```
event_slots[BATCH]                808 B × 8     = 6,464 B
contrib[BATCH][N_WARPS][OUT_MAX]  6912 floats   = 27,648 B   ← largest
per_warp_in_feat_sm               864 floats    = 3,456 B
per_warp_qh_sm                    864 floats    = 3,456 B
task_event[N_WARPS][BATCH]                      = 288 B
task_delta[N_WARPS][BATCH]                      = 288 B
task_count[N_WARPS]                             = 36 B
pass2[BATCH]                                    = 32 B
pass3[BATCH]                                    = 32 B
qvg_l4_sm                         9·3·48·24 fl  = 124,416 B  ← L4 qvgIn cache
-----------------------------------------------------------------
total (16-byte aligned)                         ≈ 166,128 B   (limit 167 KB)
```

### 5.5 Hidden state, weights, ring placement

| buffer | type | size (per block / total) | access pattern |
|--------|------|-----------|----------------|
| hidden state (4 layers) | managed | per block: L4 30+ KB, L5 38 KB, L6 30 KB, L7 30 KB | GPU read+write only |
| qvgIn (4 layers) | managed | shared: L4 121 KB, L5 243 KB, L6 486 KB, L7 972 KB | CPU writes once, GPU reads forever |
| goW (4 layers) | managed | shared: L4 81 KB, L5 81 KB, L6 324 KB, L7 324 KB | GPU read |
| input_proj, ln_*  | managed | small | GPU read |
| input ring | **pinned host** | 8.4 MB × 8 blocks | CPU writes, GPU reads |
| ring head/tail | **pinned host** | u64 each | producer/consumer atomics |
| stop_flag | **pinned host** | u32 | CPU writes |
| predictions | **pinned host** | per block | GPU writes, CPU reads |
| tdrop_s2/s3 counters | managed | per block, H·W bytes | GPU read+write |

Total weight footprint = 2.6 MB > L2 (2 MB) → weights cannot all coexist
in L2. **L7 qvgIn (972 KB, largest single buffer) is pinned in L2 via
`CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`** with persistent budget
reserved through `cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE)`.
(Note: pinning was measured as +1.7% drain offline but **0% in live**;
kept in perf_celled_multi.py only, not propagated to production runner.)

### 5.6 Persistent-kernel control flow

Each block runs as a persistent kernel that polls its ring tail:

```c
while (!stop_flag):
    seq = ring[tail & mask].seq_done;
    if (seq != tail + 1):
        __nanosleep(spin_ns);
        continue;
    // count how many consecutive slots are ready (up to BATCH)
    process_batch();
    tail += batch_size;
    events_done[block] = tail;
```

Producer side (CPU shard, `lib_stage01_to_gpu.cpp`) publishes each record
with `__threadfence_system()` and writes `seq_done = tail + 1` last;
consumer fences after reading the seq mark.

### 5.7 Optimizations applied (with measured deltas, all offline unless noted)

| opt | commit | offline gain | notes |
|-----|--------|-------|-------|
| `__launch_bounds__(288, 1)` | base | — | 168 regs/thread, 0 spills |
| multi-block sub-strips + merged qvg matvec | b91e989 | 40 → 179 kev/s | first big lift |
| implicit dispatch + parallel_tdrop | bdb3c44 | 179 → 184 kev/s | dropped explicit task queue |
| A1: skip goW + gather + LN for tdrop-dropped events | 6f287d3 | 184 → 226 kev/s | masks split into enter_mask / output_mask |
| **L4 qvgIn smem cache** (121 KB) | a76f419 | 227 → 236 kev/s (+4%) | only layer that fits in smem |
| **L7 qvgIn L2 persistence** (972 KB) | 4060a51 | 236 → 240 kev/s (+1.7%); **0% in live** | offline gain didn't transfer |
| **Balanced multi-block H splits** | 232f324 | live 218 → 227 kev/s (+4%) | corners and middles now share lag |

## 6. CPU side (summary)

CPU side runs s0 + s1 with NEON-specialized matvec kernels and is **not**
the current optimization focus. Top-line numbers (post commit `d876030`
"CPU A1"):

| cap (Mev/s) | admit (Mev/s) |
|------------:|---------------:|
| 16          | 2.58          |
| 32          | 2.71          |
| 64          | **2.80**       |

Halo replication on the CPU shard side roughly doubles per-shard event
counts versus admit (each event in 1–2 shards), and the two intermediate
tdrop stages (s0→s1, s1→GPU s2) each cut by ~4× in expectation. The
combined effect of halo replication and tdrop is that GPU sees roughly
`CPU_admit · (2.x / 16)` events per second, with significant per-block
asymmetry depending on event spatial distribution.

## 7. Measured live behaviour (synth sweep, balanced topology)

`hybrid_runner_multi.py --gpu-blocks 8 --shards 6 --cpp-synth`,
6-second windows, balanced topology (commit 232f324):

| synth Mev/s | GPU drain (kev/s) | per-block lag spread | regime |
|------------:|------------------:|---------------------:|--------|
| 1.0  | 132 | 0–10        | catching up |
| 1.5  | 198 | 30–70       | near saturation |
| **2.0** | **227** | **21–28k** | **saturated** |
| 2.5  | (not re-measured) | — | saturated |
| 3.0  | 224 | 24–30k | saturated |
| 4.0  | 208 | 23–28k | saturated, slight regression (likely contention) |

- **GPU drain saturates around 220–227 kev/s in live mode** (vs ~240
  kev/s offline pre-filled-ring benchmark — live overhead ~10%).
- Per-batch median kernel time at saturation:
  - corner blocks: ~253 µs
  - middle blocks: ~309 µs *before* balancing; comparable to corners *after*.
- Profile breakdown of one batch (from `perf_celled_profile.py`,
  representative of saturated steady state):
  | phase | µs/batch | %  |
  |-------|---------:|---:|
  | L4 compute   | 41 | 19 |
  | L5 compute   | 49 | 23 |
  | L6 compute   | 34 | 15 |
  | L7 compute   | 36 | 16 |
  | L4–L7 gather | 25 | 12 |
  | L4–L7 residual | 7 | 3 |
  | tdrops + pool + dispatch + load + out | ~25 | 12 |
  | total p50    | ~217 | 100 |

  Compute phases dominate at ~73%.

## 8. Where the system saturates

- **In live mode the GPU is the binding stage.** CPU at 2.80 Mev/s admit,
  with halo replication and per-tdrop fan-out, produces more GPU-input
  events than the GPU can drain at ~227 kev/s.
- Saturation point in synth-Mev/s is roughly 1.5–2.0 Mev/s above which
  per-block backlogs accumulate. Beyond 2.0 Mev/s, GPU drain plateaus
  and the difference between admit and drain piles into ring lag.
- Halo asymmetry has been mitigated (commit 232f324) but boundary blocks
  still see slightly less per-event work than interior blocks, so a small
  residual asymmetry remains (boundary events have OOB 3×3 patches that
  the kernel skips).
- The kernel is **not** FMA-throughput bound: 240 kev/s ≈ 40 GFMA/s vs the
  theoretical 470 GFMA/s for 8 SMs × 64 FP32 FMA/cycle × 918 MHz (~8.5%
  of peak). The bottleneck appears to be memory latency / smem-and-register
  pressure / sync overhead, not raw arithmetic.

## 9. Things tried and rejected

| attempt | what | result | why didn't work |
|---------|------|--------|-----------------|
| `__launch_bounds__(288, 2)` only | force ≤96 regs/thread, expecting 2 blocks/SM speedup | 8-block offline: 227 → 218 kev/s (-4%) | with only 8 blocks total, second SM slot stays empty; pure register-pressure tax |
| 16 blocks at (288, 2) | actually fill 2 blocks/SM via doubled topology | offline 248 kev/s (+9%); but per-block batch time 1.7× longer | compute/memory-bound, not latency-bound — kernels share L2/SFU/MIO, occupancy doesn't hide what is not stalled |
| pipeline split (S2/S3 or 4-layer separate kernels) | distribute layers across block types to enable 2 blocks/SM | abandoned during planning | same contention bound — splitting layers doesn't increase total compute resources |
| single-thread `parallel_tdrop` | replace 9-warp tdrop with one thread (BATCH=8 work is small) | 240 → 235 kev/s (-2%) | 9-warp tdrop is genuinely parallel (disjoint cells), and serializing it leaves 8 warps stalled |
| L6+L7 contiguous + L2 persistent (1458 KB) | extend L2 persistence beyond just L7 | same as L7-only (240 kev/s) | pinning more in L2 squeezes the budget for non-persistent data (L5, goW, hidden); net-zero |
| L7 L2 persistence in production runner | propagate offline gain to live | 212 → 211 kev/s | offline +1.7% gain did not transfer to live |
| float4 vectorized hidden-state loads | reduce LDG instruction count | not attempted | hidden state is already coalesced and L1-hot; layout change would need lane-partition redesign |
| L4+L5 layer fusion | save the L4→L5 smem round-trip | not attempted | L5 strictly depends on L4 gathered+LN'd output; matvec needs full input before starting; no overlap possible at our granularity |
| GATHER + RES + LN fusion into COMPUTE | hide gather phase | not attempted | gather is already mostly parallel (1 warp per event); freeing the contrib smem (needed for cross-warp reduction) requires algorithm changes that increase total time |

## 10. Open optimization paths (none cleanly viable)

### 10.1 cp.async weight prefetch into smem

- **What:** while L4 compute runs (40 µs), prefetch L5 qvgIn (243 KB) via
  `cp.async.commit_group` into a smem ring buffer, double-buffered with L4
  reads. Similarly L5→L6, L6→L7.
- **Block:** the contrib buffer takes 27 KB of smem and is *needed* for the
  current parallel-batch gather. Smem currently 166 KB / 167 KB optin —
  no room for an L5 weight tile.
- **Free contrib smem?** Options analysed and rejected:
  1. **Per-event accumulator via smem atomicAdd.** 9 warps competing on
     same accumulator address → 9× serialized atomic. 6912 atomics/layer ×
     4 layers ≈ 2 ms/batch overhead at ~270 cycles each. Kills perf.
  2. **Time-interleaved gather** (gather event-by-event instead of all 8
     in parallel). Adds 7× the per-layer gather time per batch ≈ +170 µs,
     well above current 217 µs total.
  3. **Partial-sum tree reduction in registers** (e.g., 5 partials of 2
     warps each, ~7.5 KB smem freed). 200–400 LOC restructure with
     uncertain payoff (5–10% estimated).
- **L2 prefetch alternative.** `prefetch.global.L2` PTX hint during L4
  compute could fill L2 with L5 weights without smem changes. Estimated
  +2–5% if L5 currently misses L2. Cheap to try (10 LOC) but unmeasured.

### 10.2 Layer fusion (L4+L5)

- L5 input = LayerNorm(Σ contribs + residual) of L4. Cannot start L5 qvg
  matvec until that scalar is finalized for each output channel. No way
  to overlap matvec compute with input availability at our granularity.

### 10.3 Increase BATCH

- BATCH=8 is **fixed** by project constraint. Larger batches would amortize
  per-batch overhead but increase per-batch latency (less desirable for
  real-time inference).

### 10.4 Mixed precision (fp16 / bf16 / int8)

- **Forbidden** by project constraint (CLAUDE.md §0).

### 10.5 Reduce CPU→GPU event multiplication

- The CPU shard halo replication ~doubles GPU input. Re-architecting the
  CPU side to push only owner events (with downstream state sync) would
  cut GPU load by ~2× and effectively raise system throughput. This is
  a CPU-side change, not a GPU optimization, and may conflict with the
  halo=2 correctness rule.

## 11. Hard constraints (DO NOT VIOLATE without explicit project sign-off)

1. **`halo = 2` is locked.** Not reducible, not bypass-able via state-sync
   schemes, not even discussable as a hypothetical lever.
2. **Cell-owner warp partition.** Any kernel restructure must preserve the
   property that `(cell_y % 3, cell_x % 3) → warp_id` and only the owning
   warp writes to a given hidden-state cell. This is what enables the
   no-atomic / no-`__syncthreads`-in-inner-loop design.
3. **Pinned host memory for ring traffic.** Tegra coherence
   (`CONCURRENT_MANAGED_ACCESS = 0`) makes managed memory unsafe for
   concurrent CPU writes + GPU reads.
4. **No fp16, no int8.** fp32 throughout.
5. **BATCH = 8.**
6. **No GPU atomics, no `__syncthreads` inside per-event step.**

## 12. Key files for the GPU side

```
orin/kernels/
    ssla_s2_s3_head_celled.cuh        production kernel (4 entry points)
    ssla_s2_s3_head_celled_profile.cuh  profile variant (per-phase clock64 stamps)
    proto_layer_pair.cuh              matvec_w, lru_step_w, layernorm_w primitives

orin/orin/
    multi_block.py        topology + per-block resource allocation
    hybrid_common.py      ctypes struct mirrors of HybridS2S3Config etc.
    cuda_util.py          alloc_pinned, alloc_managed wrappers
    nvrtc_util.py         NVRTC compile, PTX cache by hash, CudaModule

orin/
    hybrid_runner_multi.py     production live driver (--gpu-blocks 8)
    perf_celled_multi.py       offline N-block throughput harness
    bench_s2_s3_head_celled.py P1 oracle (CPU reference vs GPU drain_n)
    perf_celled_profile.py     phase-by-phase µs breakdown
```

## 13. Question for the expert

Given the constraints above and the data in §7–10:

- The GPU is the binding live bottleneck at ~220 kev/s drain.
- The kernel is at ~8.5% of FMA peak — clearly not arithmetic-bound.
- Smem is at 166/167 KB; the only way to free meaningful space is to
  restructure the cross-warp gather, which appears to lose more time
  than it saves.
- 16-block / 2-blocks-per-SM occupancy experiments showed compute/memory
  contention prevents 2× scaling (best observed: 1.09× over 8-block).
- L2 is 2 MB and ~2.6 MB of weights compete for it; the largest single
  buffer (L7 qvgIn = 972 KB) is pinned via L2 persistence, but pinning
  more buffers crowds out hidden-state traffic.

**Are there standard Ampere-era techniques we're overlooking that could
plausibly unlock another 10–30% throughput on this kind of pipeline?**
Specifically: any way to make cp.async profitable given the smem
constraint, any non-obvious occupancy/scheduling trick on sm_87, or any
restructure of the cross-warp gather that doesn't cost more than it
saves?
