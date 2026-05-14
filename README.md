# ssla-rt

**Real-time SSLA-S inference runtime for edge devices.**

Streams events from a DVS camera (iniVation DVXplorer / DVXplorer Micro)
through a 4-stage SSLA-S backbone + YOLOX detection head, end-to-end,
without batching. Reference platform: **NVIDIA Jetson Orin NX**.

Originally a subtree of [OpenEVA](https://github.com/openeva-dev) —
split out because the deploy concerns (latency, hardware coherence,
persistent CUDA kernels) move on a different cadence from the research
benchmark.

---

## Headline numbers (2026-05-14, session-end)

Measured on Jetson Orin NX (MAXN, 8 SMs Ampere @ 918 MHz) at 80×64
post-subsample resolution. 8 GPU blocks (2W × 4H balanced topology) with
the cell-owner warp kernel; 6 CPU shards (halo=2).

| metric | value | notes |
|---|---:|---|
| **GPU drain (live, saturated)** | **~230 kev/s** | output-producing events |
| GPU drain (offline benchmark) | ~247 kev/s | pre-filled rings |
| **End-to-end latency p50** | **~330 µs** | emit → GPU prediction write |
| End-to-end latency p99 | ~1.6 ms | OS/IRQ-dominated tail |
| End-to-end latency max | ~2.3 ms | |
| CPU admit ceiling | ~2.80 Mev/s | post commit `d876030` |
| P1 oracle (drop drift) | 0 | bit-identical hidden state |
| P1 oracle (max\|Δ\| s3_feat) | 4.2 (≤ 5.0 budget) | passing |

See [`docs/PROFILE.md`](docs/PROFILE.md) for the full per-phase
breakdown, throughput sweep, and saturation analysis. See
[`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md) for the
architecture, constraints, and what's been tried (including dead
ends — kept for context).

---

## Architecture (one screen)

```
DVXplorer ── dv-processing ── libstage01_to_gpu.so ── pinned ring × 8 ── persistent CUDA kernel
                              │ 6 CPU shards          │ 1 ring per         │ 8 blocks × 9 cell-
                              │ halo=2 spatial        │   GPU block        │ owner warps
                              │ stages s0 + s1        │ SPSC, pinned host  │ stages s2 + s3 + head
                              │ (NEON-optimised)      │   memory           │ race-free by design
```

**Key properties:**

- **CPU shards** run s0 + s1 with halo=2 spatial sharding (each shard
  owns a horizontal s2 strip with 2-cell halo overlap). NEON
  fused-interior kernels carry the matvec work.
- **GPU side** runs 8 blocks × 9 warps each on the 8 SMs. Hidden state
  cells are partitioned by `warp_id = (cy % 3) * 3 + (cx % 3)`. Any
  event's 3×3 patch touches 9 cells with 9 distinct owners → no
  within-block race, no atomics needed.
- **Per-batch gather** sums 9 per-warp contributions for each event via
  warp-shuffle reduction in shared memory.
- **No GPU atomics, no `__syncthreads` inside per-event step** — only
  `__syncwarp` and per-batch barriers between phases.
- **Tegra-coherence-safe**: rings, head/tail atomics, and stop flags
  use **pinned host memory** (`cuMemHostAlloc`). Hidden state and
  weights use managed memory but are GPU-only-write at runtime
  (`CONCURRENT_MANAGED_ACCESS = 0` on this device).

---

## Quickstart

### Requirements

- NVIDIA Jetson Orin NX (or compatible Ampere `sm_87`) on JetPack 5+
- CUDA toolkit ≥ 11.4, `pip install cuda-python`
- GCC 9.4+, CMake 3.16+
- numpy
- (optional, live camera) `dv-processing` 2.0+ with the DVXplorer
  Micro driver (libcaer 3.3.17 does NOT work for Micro — use
  dv-processing)

### Build

```bash
git clone <repo>
cd ssla-rt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Produces three shared libs under `build/`:
- `libstage01_to_gpu.so` — production hybrid pipeline (used by
  `orin/hybrid_runner*.py`)
- `libstage01_capi.so` — CPU-only baseline (stages s0 + s1)
- `libstage0_capi.so` — stage-0-only baseline

Plus `oracle_diff` static helper.

> To also build the 11 historical x86 CPU-sharding scheme binaries
> (s_stage0_*, s1..s8), configure with `-DSSLA_RT_BUILD_SCHEMES=ON`.
> These won't compile on aarch64 without patches — they were used in
> the early CPU optimization log and are kept for reference.

### Generate stub weights (no training needed)

For benchmarks / smoke testing, generate schema-valid random weights:

```bash
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80
```

`/tmp` is volatile on Orin — regenerate after reboot.

### Run

The Python runners live in `orin/`. **Run from inside `orin/`** so the
`orin/orin/` Python package is on `sys.path`:

```bash
cd orin

# === P1 correctness oracle (offline) ===
# Compares GPU drain_n output to a CPU oracle, prints drop drift +
# max|Δ| of s3_feat. Must pass before any other change is shipped.
python3 bench_s2_s3_head_celled.py --n 50000
# expect: drop drift = 0, max|Δ| ≤ 5.0

# === Offline throughput (N-block) ===
# Sweep multi-block GPU drain — no live ring, no CPU side.
python3 perf_celled_multi.py --n-blocks 8 --n 200000 --runs 3

# === Offline per-phase µs profile ===
python3 perf_celled_profile.py --n 50000

# === Live throughput + latency (8-block production) ===
python3 hybrid_runner_multi.py \
    --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --cpp-synth \
    --synthetic-mev 2.0 --duration-s 8 \
    --pin-python-main --base-core 1
# Reports drain rate per block + end-to-end latency for
# output-producing events.

# === Live 2-block (legacy runner, also production-tested) ===
python3 hybrid_runner.py \
    --weights /tmp/ssla_s_64x80/ \
    --cpp-synth --synthetic-mev 2.0 \
    --duration-s 10 --pin-python-main

# === Live camera (real DVXplorer Micro) ===
# Drop --synthetic-mev to use camera events.
python3 hybrid_runner_multi.py \
    --weights /tmp/ssla_s_64x80/ \
    --gpu-blocks 8 --shards 6 --duration-s 30 \
    --pin-python-main --base-core 1
```

### Live event viewer + DVXplorer tuner

```bash
cd orin/viz
python3 event_viewer.py                           # GUI with trackbars
python3 event_viewer.py --motion-probe            # sweep thresholds under
                                                  # continuous motion
python3 event_viewer.py --probe-thresholds        # static threshold sweep
python3 event_viewer.py --list-methods            # dump camera API
```

Note: DVXplorer contrast threshold valid range is **0–17** (chip clamps
silently above that). The trackbar caps at 17 since commit `93a843b`.

### Tests

```bash
# Python unit tests (proto_layer_pair primitive, weights, ring SPSC):
cd orin
python3 -m pytest tests/ -v

# C++ benches:
./build/bench_fused
./build/bench_layer
./build/bench_matvec24
./build/bench_sigmoid
```

---

## Repo layout

```
ssla-rt/
├── CLAUDE.md / README.md / LICENSE / CMakeLists.txt
├── include/                  public C/C++ headers
├── src/                      CPU pipeline (production)
│   ├── lib_stage01_to_gpu.cpp   hybrid CPU s0+s1 → ring → GPU
│   ├── lib_stage{0,01}_capi.cpp CPU-only baselines
│   ├── ssla_kernels.cpp         NEON-optimised forward kernels
│   ├── oracle_diff.cpp          P1 oracle diff helper
│   ├── multicore_bench.cpp      CPU multicore bench
│   └── legacy/schemes/          11 historical x86 schemes (opt-in build)
├── tests/                    C++ benches (bench_fused, bench_layer, ...)
├── tools/make_ssla_stub.py   schema-valid random-weight generator
├── scripts/run_scaling.sh    helper for throughput scan
├── vendor/                   vendored openeva (event, prim, heads, weights_loader)
├── docs/
│   ├── PIPELINE_DESIGN.md       current architecture + constraints
│   ├── PROFILE.md               current measurement snapshot
│   ├── DEV_LOG.md               historical dev log
│   └── archive/                 superseded docs (STATUS, HYBRID_DESIGN, ...)
└── orin/                     Python + GPU side
    ├── orin/                    Python utility package (cuda_util, nvrtc_util,
    │                            hybrid_common, multi_block, weights_ssla, ssla_ref)
    ├── kernels/
    │   ├── ssla_s2_s3_head_celled.cuh           PRODUCTION kernel
    │   ├── ssla_s2_s3_head_celled_profile.cuh   profile variant
    │   ├── proto_layer_pair.cuh                 warp-cooperative primitives
    │   └── legacy/                              6 superseded kernels
    ├── tests/                                   unit tests
    ├── viz/event_viewer.py                      live event viewer + tuner
    ├── hybrid_runner.py                         2-block live runner
    ├── hybrid_runner_multi.py                   N-block live runner (production)
    ├── bench_s2_s3_head_celled.py               P1 oracle
    ├── perf_celled.py / perf_celled_multi.py    offline throughput
    ├── perf_celled_profile.py                   per-phase µs profile
    ├── analyze_coalescing.py                    offline reuse-statistics analyzer
    └── legacy/                                  superseded scripts + tests
```

---

## Development guide

Before modifying anything, read **[`docs/DEV_GUIDE.md`](docs/DEV_GUIDE.md)** —
it captures the methodology, architecture invariants, working
optimizations, dead-end experiments, hardware/camera/variant
adaptation checklists, common pitfalls, and a decision tree for new
optimizations. Read this **first** when porting to new hardware /
camera / SSLA variant.

## What's been tried and the bottom line

See [`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md) §§ 9–10 for
the full list. Briefly:

**Shipped optimizations (cumulative on this session's branch):**

- L4 qvgIn smem cache (121 KB persistent in smem, avoids per-batch L2 reload)
- L7 qvgIn L2 persistence (`CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`,
  perf benches only)
- Balanced H-strip topology (5/3/3/5 owned Y instead of 4/4/4/4 —
  equalizes proc-width across corner and middle blocks)
- VG-only state-only patch (events failing tdrop skip the Q half of the
  qvg matvec — 75% of L5/L7 events benefit)

**Experiments that didn't pay off (reverted):**

- Pipeline split / 2 blocks/SM via `__launch_bounds__(_, 2)`: compute
  contention swallowed the occupancy win (1.1×, not 2×)
- 16-block topology: same contention
- TF32 Tensor Core L4 qvg precompute: m=1 / m=8 padding kills TC
  efficiency on our problem size (-5 % net)
- Batch-local hidden-cell coalescing: reuse only 1.097 on uniform synth,
  overhead 700× the savings (-6 %)

**Hard constraints (do not violate without project sign-off):**

- `halo = 2` is locked
- Cell-owner warp partition (race-free invariant)
- Pinned host memory for ring traffic (Tegra coherence)
- No fp16, no int8
- `BATCH = 8`

---

## Known limitations / open work

1. **GPU is the live bottleneck.** CPU at 2.80 Mev/s admit produces
   more GPU events than the GPU's 230 kev/s drain can absorb at full
   halo replication. Backlog accumulates at saturation.
2. **p99 tail latency (~1.6 ms) is OS / DRAM-burst dominated.** Going
   below ~500 µs would need root + SCHED_FIFO + memory locking — out
   of scope under no-root.
3. **DVXplorer contrast threshold caps at 17** chip-side. See memory
   `project_dvxplorer_contrast_range.md` and commit `93a843b`.

---

## License

[Apache License 2.0](LICENSE) © 2026 Haoqiang Hao.

The vendored code under `vendor/` originates from the OpenEVA research
repository (same author) and is re-licensed under Apache-2.0 here for
consistency with this redistribution.

---

## Citing

```
@software{ssla-rt,
  title  = {ssla-rt: Real-time SSLA-S inference runtime for edge devices},
  author = {Hao, Haoqiang},
  year   = {2026},
  url    = {https://github.com/haohq19/ssla-rt},
}
```
