# SSLA-S Full-Pipeline Profile (refreshed 2026-05-14)

**Setup:** Jetson Orin NX, MAXN. Topology: 8 GPU blocks (2W × 4H, balanced
Y splits). 6 CPU shards on cores 1–6, C++ synth dispatcher on core 7,
Python main pinned to core 0. tdrop_window=4. Stub weights (random).
Measurement windows: 6 s each unless noted.

**Session-end state** (after all 2026-05-13 / 2026-05-14 optimizations
applied: L4 qvgIn smem cache, L7 qvgIn L2 persistence, balanced Y-strip
topology, VG-only state-only patch_cell). Numbers below are
re-measured 2026-05-14 with all optimizations active.

End-to-end path:
```
synth/camera emit (t_emit_ns)
  → CPU shards: s0 (L0+L1) → tdrop → s1 (L2+L3) → tdrop
  → ring push (t_push_ns)
  → GPU block: tdrop_s2 → L4+L5 → pool → tdrop_s3 → L6+L7 → head → predictions
```

`t_emit_ns` and `t_push_ns` are both CLOCK_MONOTONIC_RAW on CPU side;
`emit→push` is reliable. GPU kernel duration uses `clock64()` at SM clock,
also reliable. Cross-clock (CPU emit ns → GPU done clock) calibration is
biased by ~5 ms due to SM-clock DVFS, so absolute `emit→done` is reported
as `(emit→push) + (kernel)` rather than direct measurement.

---

## 1. Throughput sweep (full pipeline)

Live `hybrid_runner_multi --gpu-blocks 8 --shards 6 --cpp-synth`, 6 s,
balanced topology (5/3/3/5 owned Y).

| synth (Mev/s) | drain (kev/s) | max per-block lag | regime |
|---:|---:|---:|---|
| 0.5 | **66.0** | 0 | well below saturation, no backlog |
| 1.0 | **130.2** | 0–1 | well below saturation |
| 1.5 | **197.9** | 0 | just below saturation |
| 2.0 | **227.9** | **21–27k** | **saturated** |
| 2.5 | **230.0** | 22–28k | saturated (peak) |
| 3.0 | 227.3 | 22–29k | saturated |
| 4.0 | 226.9 | 21–27k | saturated |

**Saturation throughput ≈ 227–230 kev/s** (GPU drain, all 8 blocks combined).
This is what falls out the bottom of the pipeline as completed events.

Observations:
- Linear scaling holds up to ~1.5 Mev/s synth.
- Above ~1.7 Mev/s synth, GPU is the binding stage and total drain plateaus.
- Peak at synth 2.5 (230 kev/s); above 3 Mev/s slight regression (~227)
  attributable to L2 / DRAM contention as ring back-pressure grows.
- Drain split evenly across 8 blocks after balanced topology: corner
  blocks ~28 kev/s, middle blocks ~28 kev/s. Pre-balancing (2026-05-13
  morning state), middles plateaued at 25.9 kev/s while corners idled.

## 2. End-to-end latency at saturation (synth 2.0 Mev/s)

Per-block, **output-producing events only** (`is_owner AND pass_tdrop_s2
AND pass_tdrop_s3`, i.e. events that actually wrote a final prediction —
not just spatial owners). The `owner` field in `GpuTimingSlot` is the
AND of all three flags; counts shown as `n_pred` ≈ 311–561 over 6 s window.

Dropped events (failing either tdrop) have the same per-batch `t_pop_clk`
and `t_done_clk` as output-producing events in the same batch, so the
kernel-internal latency would be similar; we filter them out because
their predictions are never read by downstream consumers — the metric is
only meaningful for events that actually deliver an output.

Latency quoted in **µs**:

```
blk | n_pred|  emit→push (CPU)   kernel (GPU)        e2e (sum)
    |       |  p50 / p99 / max   p50 / p99 / max     p50 / p99 / max
----|-------|-------------------|-------------------|-------------------
  0 |  561  |   23 /  70 /  97  | 303 / 1934 / 2249 | 326 / 2004 / 2346
  1 |  314  |   23 /  70 /  79  | 331 / 1882 / 2191 | 354 / 1952 / 2270
  2 |  358  |   26 /  87 /  98  | 322 / 1702 / 2108 | 348 / 1789 / 2206
  3 |  545  |   25 /  73 / 117  | 301 / 1922 / 2186 | 326 / 1995 / 2303
  4 |  530  |   17 /  55 /  83  | 304 / 1113 / 2091 | 321 / 1168 / 2174
  5 |  338  |   17 /  54 /  71  | 315 / 1434 / 2197 | 332 / 1488 / 2268
  6 |  333  |   19 /  75 / 103  | 303 /  999 / 2098 | 322 / 1074 / 2201
  7 |  536  |   17 /  58 /  80  | 304 / 1046 / 1883 | 321 / 1104 / 1963
```

### Summary across all blocks (saturated, synth 2.0 Mev/s, 8 s window)

| metric | p50 | p99 | max |
|--------|----:|----:|----:|
| **emit → ring push (CPU pipeline)** | ~21 µs | ~68 µs | ~120 µs |
| **GPU kernel** | ~310 µs | ~1500 µs | ~2200 µs |
| **whole pipeline (emit → GPU done)** | **~330 µs** | **~1600 µs** | **~2300 µs** |

### Latency vs offered load (re-measured 2026-05-14)

```
synth 0.5 Mev/s — kernel p50:  130 µs        — emit→push p50:   8 µs   — e2e p50: ~140 µs
synth 1.0 Mev/s — kernel p50:  135 µs        — emit→push p50:   8 µs   — e2e p50: ~143 µs
synth 1.5 Mev/s — kernel p50:  220 µs (mix)  — emit→push p50:   8 µs   — e2e p50: ~230 µs
synth 2.0 Mev/s — kernel p50:  310 µs        — emit→push p50:  21 µs   — e2e p50: ~330 µs
synth 2.5 Mev/s — kernel p50:  310 µs        — emit→push p50:  20 µs   — e2e p50: ~330 µs
synth 3.0 Mev/s — kernel p50:  312 µs        — emit→push p50:  20 µs   — e2e p50: ~332 µs
synth 4.0 Mev/s — kernel p50:  310 µs        — emit→push p50:  20 µs   — e2e p50: ~330 µs
```

- At low load (≤1 Mev/s), kernel p50 is ~131 µs — partial-batch effect
  (rings often have fewer than 8 events; per-batch fixed overhead dominates).
- Above saturation, kernel p50 stabilizes at ~305 µs because the ring is
  always full and BATCH=8 is consistently formed.
- emit→push jumps from 8 µs to 20 µs at saturation: CPU shards spend more
  time waiting for ring slots (back-pressure from saturated GPU).
- **p99 kernel tail** is the dominant concern: 1.5–1.9 ms at saturation,
  ~7× the p50. Driven by (a) OS daemons / IRQs on non-RT cores, (b) DRAM
  contention spikes, (c) ring head/tail polling spin during back-pressure.

## 3. CPU per-segment profile (synth 2.0 Mev/s, 2-block runner)

`hybrid_runner.py --kernel-variant celled --cpp-synth --synthetic-mev 2.0`.

This is the 2-block (legacy) live runner — emits the per-segment shard
timing; the 8-block runner doesn't. Numbers are mean µs across all
shards over 6 s. Note that `count` reflects events that passed each
preceding tdrop:

| segment            | mean µs | events  | share % | notes |
|--------------------|--------:|--------:|--------:|-------|
| preprocess         | 0.079   | 6.07 M  | 3.3 %   | input record build |
| stage_forward(0)   | 0.912   | 6.07 M  | 38.6 %  | CPU s0 = L0+L1 (24→48 channels, 80×64 grid) |
| tdrop_and_pool(0)  | 0.095   | 6.07 M  | 4.0 %   | s0→s1 tdrop counter + pool |
| stage_forward(1)   | 5.711   | 1.32 M  | 52.5 %  | CPU s1 = L2+L3 (12→24 channels, 40×32 grid) |
| tdrop_and_pool(1)  | 0.078   | 1.32 M  | 0.7 %   | s1→GPU s2 tdrop |
| ring push          | 0.345   | 330 k   | 0.8 %   | publish to GPU ring |
| **total / event-eq** | ~8.0 µs (averaged including post-tdrop fan-in) | | | |

Observations:
- s1 = stage_forward(1) is by far the most expensive CPU stage at 5.7 µs
  per event, despite handling 1/4 the event volume (post-tdrop).
- s1 dominates because it has bigger channel (12→24) and the bigger
  per-event matvec.
- Ring push is fast (~0.35 µs) so CPU/GPU coupling is not the bottleneck.

## 4. GPU per-phase profile (offline, perf_celled_profile.py)

50k random events, 1 of 2 representative blocks. Each row is the duration
between adjacent STAMP markers — segments named "PREV→NEXT" measure the
NEXT phase's duration:

| phase                          | p50 µs | mean µs | share % | category |
|--------------------------------|-------:|--------:|--------:|----------|
| POP → LOAD                      | 8.47   | 9.03    | 3.8 %   | ring/IO |
| LOAD → DISPATCH_S2              | 3.28   | 4.57    | 1.9 %   | setup |
| DISPATCH_S2 → L4_RES            | 2.64   | 2.87    | 1.2 %   | L4 residual matvec |
| L4_RES → L4_ZERO                | 0.54   | 0.54    | 0.2 %   | L4 contrib zero |
| **L4_ZERO → L4_COMPUTE**        | **40.80** | 46.02 | **19.2 %** | L4 patch compute |
| L4_COMPUTE → L4_GATHER          | 9.29   | 9.69    | 4.0 %   | L4 gather+LN |
| L4_GATHER → L5_RES              | 0.61   | 0.64    | 0.3 %   | L5 residual |
| L5_RES → L5_ZERO                | 0.53   | 0.53    | 0.2 %   | L5 zero |
| **L5_ZERO → L5_COMPUTE**        | **51.72** | 54.61 | **22.7 %** | L5 patch compute (LARGEST) |
| L5_COMPUTE → L5_GATHER          | 4.32   | 4.13    | 1.7 %   | L5 gather (skipped for tdrop-dropped) |
| L5_GATHER → TDROP_S2            | 0.08   | 0.08    | 0.0 %   |  |
| TDROP_S2 → POOL                 | 0.49   | 0.82    | 0.3 %   | (now done EARLIER, before L5; this is residual) |
| POOL → DISPATCH_S3              | 2.56   | 3.26    | 1.4 %   | s2→s3 coord pool |
| DISPATCH_S3 → L6_RES            | 3.98   | 4.09    | 1.7 %   |  |
| L6_RES → L6_ZERO                | 0.95   | 0.95    | 0.4 %   |  |
| **L6_ZERO → L6_COMPUTE**        | **36.11** | 38.43 | **16.0 %** | L6 patch compute |
| L6_COMPUTE → L6_GATHER          | 10.82  | 11.62   | 4.8 %   | L6 gather |
| L6_GATHER → L7_RES              | 0.82   | 0.82    | 0.3 %   |  |
| L7_RES → L7_ZERO                | 0.91   | 0.92    | 0.4 %   |  |
| **L7_ZERO → L7_COMPUTE**        | **35.61** | 40.95 | **17.1 %** | L7 patch compute |
| L7_COMPUTE → L7_GATHER          | 0.80   | 2.03    | 0.8 %   | L7 gather (skipped for tdrop-dropped) |
| L7_GATHER → TDROP_S3            | 0.08   | 0.08    | 0.0 %   |  |
| TDROP_S3 → OUT                  | 2.86   | 3.43    | 1.4 %   | head + prediction write |
| **total**                       | **224.2** | **240.1** | **100 %** | per-batch p50 |

### Rolled up by category:

| category | µs p50 | share % |
|----------|-------:|--------:|
| **COMPUTE** (L4–L7 patch matvec) | **164** | **73 %** |
| **GATHER** (L4–L7 cross-warp reduction + LN) | 25 | 12 % |
| **RESIDUAL** (input_proj matvec per layer) | 7 | 3 % |
| **TDROP / POOL / DISPATCH** | 8 | 4 % |
| **LOAD / OUT (ring IO)** | 17 | 8 % |
| **ZERO** (contrib reset) | 3 | 1 % |

- L5 is the slowest compute layer (52 µs) — `IN=48, OUT=48` matvec is twice
  L4's per-event work and L5 weights (243 KB) don't fit in smem (L4 weights
  do, 121 KB).
- L4 compute (41 µs) benefits from L4 qvgIn smem cache (committed
  a76f419, +4% drain).
- GATHER for L4 (9 µs) and L6 (11 µs) are surprisingly large — gather is
  "1 warp per event, sum 9 warps' contribs + LN" and seems memory-latency
  bound on the cross-warp smem reads.
- TDROP_S2 is now zero-cost in the profiled position because it was moved
  EARLIER in the kernel (before L5 compute) to enable A1 skip-output.
- BATCH=8 fixed, so per-batch overhead amortizes across 8 events:
  per-event compute = 224.2 / 8 = **28 µs/event**.
  At 8 SMs running 1 block each: **224 kev/s aggregate**, matches measured
  offline drain.

## 5. Per-block kernel time, corner vs middle (at saturation)

After balanced topology (commit 232f324), proc widths are uniform (7 cells Y).
Residual asymmetry: corner blocks have events at boundary (some 3×3 patches
OOB → skipped), middle blocks have all events interior (full 9 patches each).

Synth 2.0 Mev/s, average over 6 s:

| block class | blocks | kernel p50 | drain/block | events pushed (CPU) |
|-------------|--------|-----------:|------------:|--------------------:|
| Corner (0, 3, 4, 7) | 4 | 296–300 µs | ~28 kev/s | 32 kev/s |
| Middle (1, 2, 5, 6) | 4 | 307–321 µs | ~27 kev/s | 32 kev/s |

- Corner blocks ~3–6% faster per batch (boundary OOB cells skipped).
- Both classes see backlog ~22–28k events at end of 6 s window — neither
  is idle; lag is roughly distributed.
- Pre-balancing (uniform 4/4/4/4 Y): corners 0 lag, middles 52k lag.
  Balanced version moves backlog off middles onto a uniform distribution.

## 6. Backlog dynamics

Per-second per-block done counts at synth 2.0 Mev/s:

```
    t |  kev/s GPU | block-wise events_done deltas per 1 s window
  1.0s |      228 | corner ~28.9k, middle ~28.0k
  2.0s |      223 | corner ~28.4k, middle ~27.5k
  3.0s |      218 | corner ~27.8k, middle ~26.9k
  4.0s |      220 | corner ~27.9k, middle ~27.0k
  5.0s |      221 | corner ~28.0k, middle ~27.0k
```

Pattern: starts at ~228 kev/s, dips to ~218 around 3 s as backlog fills the
rings, stabilizes at 220–222 kev/s. The dip likely reflects the rings
filling and the kernel spending more cycles on ring-tail polling vs
compute.

## 7. System bottleneck summary

- **Live drain ceiling (2026-05-14):** ~230 kev/s peak (synth 2.5),
  ~227 kev/s steady state.
- **Per-block ceiling:** ~28 kev/s. With 8 blocks, total ceiling ≈ 230 kev/s.
- **CPU side:** stable up to ~2.8 Mev/s admit (post commit d876030 CPU A1),
  produces ~280 kev/s of GPU-ring traffic at this admit rate after halo
  replication and 2× tdrops. So CPU produces ~20% more than GPU drains
  (the backlog at saturation).
- **GPU side:** 8.5% of theoretical FMA peak. NOT arithmetic-bound. Likely
  memory-latency / sync-overhead bound. Cannot directly add occupancy
  (167 KB smem near limit, 168 reg/thread fits 1 block/SM exactly).
- **End-to-end latency at saturation:** p50 ~330 µs, p99 ~1.6 ms,
  max ~2.3 ms. The tail (p99/p50 ≈ 5×) is the worst aspect.

## 8. What's known to NOT help (already tested)

- Single-thread tdrop: −2 % (parallel 9-warp tdrop is real)
- `__launch_bounds__(288, 2)` on 8 blocks: −4 %
- 16-block topology to fill 2 blocks/SM: +9 % (with cost: 1.7× per-block
  batch time due to L2/SFU contention)
- L7 qvgIn L2 persistence on live runner: 0 %
- L6+L7 combined L2 persistence (1458 KB contiguous): 0 %
- **TF32 Tensor Core L4 qvg precompute** (2026-05-14): −5 % offline,
  −3 % live. P1 unchanged (max\|Δ\| = 4.202). Root cause: qvg_buf needed
  41 KB smem; only way to fit was removing L4 qvgIn smem cache (121 KB,
  +4% gain in commit a76f419). The smem cache loss exceeded TC compute
  savings. See `project_tf32_tc_l4_prototype.md`.
- **Batch-local hidden-cell coalescing** (2026-05-14): −6 % offline.
  Synth reuse only 1.097 (k=2 collisions in 9% of cells, k=3+ rare);
  overhead 700× the savings due to smem dynamic indexing, bitmask
  control flow, and lost compiler scheduling. See
  `project_coalescing_net_negative.md`.

## 9. Reproduction

```bash
# Throughput sweep
for mev in 0.5 1.0 1.5 2.0 2.5 3.0 4.0; do
  python3 hybrid_runner_multi.py --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --cpp-synth --synthetic-mev $mev \
    --duration-s 6 --pin-python-main --base-core 1
done

# CPU per-segment profile (2-block runner, emits cpu seg block)
python3 hybrid_runner.py --weights /tmp/ssla_s_64x80/ \
  --kernel-variant celled --cpp-synth --synthetic-mev 2.0 \
  --duration-s 6 --pin-python-main

# GPU per-phase profile (offline)
python3 perf_celled_profile.py --n 50000
```
