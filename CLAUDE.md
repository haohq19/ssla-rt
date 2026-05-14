# ssla-rt — Development Guide

_Last updated: 2026-05-12 (initial split from openeva-dev/deploy)_

## 0. What this repo is

Real-time SSLA-S inference runtime for edge devices. Streams events from
an iniVation DVXplorer / DVXplorer Micro through a 4-stage SSLA-S
backbone + YOLOX head, end-to-end, **no batching**.

Reference platform: **NVIDIA Jetson Orin NX**.

## 0.1 What this repo is NOT

- Not a training codebase. SSLA-S training, evaluation, and weight
  export live in the upstream [openeva](https://github.com/...) research
  repo (under `~/openeva-dev/` on this dev machine).
- Not a multi-method runtime. Only SSLA-S is implemented here. Other
  event-based methods (DAGr, FARSE-CNN, AsyNet, AEGNN, NVS) remain in
  openeva.
- Not a deployment target for x86 EPYC. CPU sharding schemes S1..S8
  under `src/s*.cpp` are historical artifacts from the openeva dev log;
  they use x86 intrinsics (`__builtin_ia32_pause`) and **do not compile
  on aarch64 without patches**. The production path is the three SHARED
  libraries built by default (`libstage0_capi.so`, `libstage01_capi.so`,
  `libstage01_to_gpu.so`) + the Python runtime under `orin/`.

## 0.2 Development guide (READ THIS FIRST when porting / optimizing)

**[`docs/DEV_GUIDE.md`](docs/DEV_GUIDE.md)** — comprehensive playbook
distilled from real optimization work. Covers methodology, architecture
invariants, what works, what doesn't (with measured negative results),
profiling protocols, camera tuning, hardware adaptation checklists,
SSLA-variant adaptation, common pitfalls, optimization decision tree.

Read this **before** proposing or implementing any kernel/topology
change. Failure modes are extensively documented to prevent re-running
known dead ends.

## 1. Repo layout

```
ssla-rt/
├── CLAUDE.md / README.md / LICENSE / CMakeLists.txt / .gitignore
├── include/                    public C/C++ headers (ssla_kernels.h, spsc.h, ...)
├── src/                        production CPU pipeline sources
│   ├── lib_stage01_to_gpu.cpp     production hybrid pipeline (CPU s0+s1 → ring → GPU)
│   ├── lib_stage01_capi.cpp       CPU-only stages s0+s1
│   ├── lib_stage0_capi.cpp        CPU-only stage 0
│   ├── ssla_kernels.cpp           per-stage SSLA-S forward kernels (NEON-optimised)
│   ├── oracle_diff.cpp / multicore_bench.cpp   benches
│   └── legacy/schemes/             11 historical x86 CPU sharding scheme experiments
│                                   (opt-in build via -DSSLA_RT_BUILD_SCHEMES=ON;
│                                    x86-only, won't compile on aarch64)
├── tests/                      C++ benches (bench_fused, bench_layer, bench_matvec*, bench_sigmoid)
├── tools/
│   └── make_ssla_stub.py          schema-valid random-weight export generator
├── scripts/
│   └── run_scaling.sh             throughput scan helper
├── vendor/                     vendored openeva pieces (event, prim/, heads/, weights_loader)
├── docs/
│   ├── PIPELINE_DESIGN.md         CURRENT canonical architecture + constraints
│   ├── PROFILE.md                 CURRENT measured throughput + latency snapshot
│   ├── DEV_LOG.md                 historical PoC dev log
│   └── archive/                   superseded design docs (STATUS, HYBRID_DESIGN,
│                                   RISK_RETIREMENT, CPU_PERF_STATUS) — kept for
│                                   context but not maintained
└── orin/                       Python + GPU side
    ├── README.md                  orin-specific quickstart
    ├── orin/                      Python utility package
    │   ├── cuda_util.py             cuMemHostAlloc, cuMemAllocManaged wrappers
    │   ├── nvrtc_util.py            NVRTC compile + PTX cache + CudaModule
    │   ├── hybrid_common.py         ctypes mirrors of HybridS2S3Config etc.
    │   ├── multi_block.py           topology + per-block resource allocation
    │   ├── weights_ssla.py          reshape qvgIn / goW for matvec-friendly layout
    │   ├── weights.py               npz-to-managed-memory loader
    │   └── ssla_ref.py              single-thread CPU oracle (LayerRef, layer_step)
    ├── kernels/
    │   ├── ssla_s2_s3_head_celled.cuh         PRODUCTION GPU kernel
    │   ├── ssla_s2_s3_head_celled_profile.cuh per-phase profile variant
    │   ├── proto_layer_pair.cuh               warp-cooperative primitives
    │   └── legacy/                            6 superseded kernels (block-coop,
    │                                          warp-per-event with races, etc.)
    ├── tests/                      unit tests (proto_layer_pair, weights, ring)
    ├── viz/event_viewer.py         live event viewer + DVXplorer tuning tool
    ├── hybrid_runner.py            2-block production live runner
    ├── hybrid_runner_multi.py      N-block (2/4/8) production live runner
    ├── bench_s2_s3_head_celled.py  P1 oracle (CPU reference vs GPU drain_n)
    ├── perf_celled.py              offline throughput (2 block)
    ├── perf_celled_multi.py        offline throughput (N block)
    ├── perf_celled_profile.py      per-phase µs profile
    ├── analyze_coalescing.py       offline batch-local hidden-cell reuse analyzer
    └── legacy/                     superseded benches / demos / one-off scripts
        └── tests/                  tests for legacy kernels (not maintained)
```

## 2. Critical invariants

These are durable rules, not preferences. Violating them silently breaks
correctness or causes runtime hangs.

### 2.1 Tegra coherence: pinned host memory for ring traffic

Orin NX reports `CONCURRENT_MANAGED_ACCESS = 0`. **Managed memory is
unsafe for live concurrent CPU writes + GPU reads on this device.**

All cross-CPU/GPU traffic must go through pinned host memory
(`cuMemHostAlloc`):
- per-block ring buffers (CPU producer → GPU consumer)
- ring head/tail atomics
- stop flag
- predictions output

Managed memory is only safe for GPU-only state:
- hidden state (read/write by GPU only)
- weights (CPU writes once at init, GPU reads forever)
- tdrop counters
- timing slots

See `orin/orin/hybrid_common.py::alloc_pinned` vs `alloc_managed`.

### 2.2 Halo = 2 strict at every CPU sharding stage

User-enforced rule: **halo = 1 is not acceptable in any time**.

The CPU pipeline (`lib_stage01_to_gpu.cpp`) uses halo = 2 spatial
sharding at the strip boundaries. This covers both the K=3 spatial conv
patches at the current resolution AND, after the s1 pool, the K=3
patches at the next resolution (which would need ±2 cells at the
current res = halo 2).

Do not reduce halo below 2 anywhere — not at block level, not at warp
level, not at any sub-level.

### 2.3 Cell-owner warp partition (race-free invariant)

The production GPU kernel (`ssla_s2_s3_head_celled.cuh`) uses 9 warps
per block. Hidden state cells are owned by exactly one warp:

```
warp_id = (cell_y % 3) * 3 + (cell_x % 3)
```

Key property: any event's 3×3 patch update writes to 9 cells with 9
distinct (cy%3, cx%3) pairs → 9 distinct warp owners → **only the
owning warp ever writes to any given cell**. No GPU atomics needed.

If you change the kernel structure, preserve this invariant or the
P1 drift gate will start failing.

### 2.4 No GPU atomics, no __syncthreads inside per-event step

- `ssla_s2_s3_head_celled.cuh` only uses `__syncthreads()` between
  batched phases (DISPATCH / COMPUTE / GATHER / TDROP), not inside
  per-event step.
- Within a warp's queue, sequential processing of (event, delta) tasks
  uses only `__syncwarp()` and warp shuffles.
- tdrop counters are serialized by thread 0 of the block, not via
  atomicAdd.

## 3. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Builds three SHARED libs (default config):
- `libstage01_to_gpu.so` — production hybrid pipeline
- `libstage01_capi.so` — CPU-only baseline
- `libstage0_capi.so` — stage-0-only baseline

Plus `oracle_diff` static helper.

Scheme experiments opt-in: `-DSSLA_RT_BUILD_SCHEMES=ON` (x86-only;
won't compile on aarch64 without porting).

## 4. Run

### Stub weights (no training needed, for benches)
```bash
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80
```

### Hybrid pipeline (production)
```bash
cd orin
python3 hybrid_runner.py --weights /tmp/ssla_s_64x80/ \
        --kernel-variant celled \
        --synthetic-mev 0.5 --duration-s 10
```

Drop `--synthetic-mev` for live camera input (DVXplorer Micro on USB2).

### P1 correctness gate
```bash
cd orin
python3 bench_s2_s3_head_celled.py --n 200000
# expect: drift = 0, max|Δ| ≤ 5
```

## 5. Test gates

| Phase | What | Pass criterion |
|---|---|---|
| P1 | Offline drain_n vs CPU oracle | drop drift = 0, max\|Δ\| ≤ 5 |
| P2 | Live 30 s synthetic at 0.05 Mev/s | no deadlock, both blocks write 40/40 cells |
| P3 | Saturation sweep 0.05/0.5/1.0/2.0 Mev/s | ring lag bounded; document drain ceiling |

See `docs/PROFILE.md` for current passing numbers (throughput + latency
sweep). Historical measurement log: `docs/archive/STATUS.md` §§6–7.

## 6. Headline measurements (current snapshot lives in `docs/PROFILE.md`; full historical log in `docs/archive/STATUS.md`)

| metric | legacy | Phase 1+2 (matvec+lru NEON) | **Phase 3 (fused layers)** | total vs legacy |
|---|---:|---:|---:|---:|
| GPU drain saturation (live persistent kernel) | 5 kev/s | ~47 kev/s | ~47 kev/s | 9.4× |
| owner-pass-3 p50 | 530 µs | 127 µs | 127 µs | 4.2× |
| Per-event stage_forward(0) on CPU | 3.94 µs | 1.83 µs | **1.05 µs** | **3.75×** |
| Per-event stage_forward(1) on CPU | 12.88 µs | 9.50 µs | **5.34 µs** | **2.41×** |
| CPU per-shard rate | 135 kev/s | 216 kev/s | **327 kev/s @ n=4** | **2.42×** |
| **CPU ceiling stable** | 0.82 Mev/s @ n=8 | 1.30 Mev/s @ n=7 | **1.85 Mev/s @ n=6** | **2.26×** |
| **CPU ceiling burst** | — | — | **~2.0 Mev/s @ n=8** | **2.44×** |

CPU side now exceeds the 1 Mev/s design target by 1.85–2.0×. GPU drain ceiling
(~0.6 Mev/s admit-equivalent) is the binding system-level limit; the CPU can
produce more than 3× what the GPU can drain. Path past 0.6 Mev/s end-to-end
requires GPU-side work (e.g. BATCH 8→4 → 2 blocks/SM).

## 7. Cross-references

### Upstream (openeva research repo, ~/openeva-dev/)
- Trained SSLA-S checkpoints + weight export → `python/exporter.py`
  produces the `weights.npz + meta.json` schema this repo expects.
- SSLA reference paper: `doc/bib/2603.06228v1.pdf`
- CPU SSLA forward reference: `cpp/methods/ssla/ssla_detection_yolox.cpp`
  (vendored under `vendor/openeva/heads/async_yolox.h` here).

### iniVation DVXplorer
- DVXplorer Micro (USB 2.0): `dv-processing` 2.0.3+ (vendored under
  `orin/third_party/dv-processing/include/` if present, else apt/pip).
- libcaer 3.3.17 does NOT support the Micro variant — see auto-memory
  `project_dvxplorer_micro_libcaer.md` for details.

### NVIDIA Jetson Orin NX
- SM clock pinned at **918 MHz** in MAXN power mode (warmup ramps
  306 → 714 → 918 over ~1.2 s). Used for `clock64()` → ns conversion.
- `nvpmodel -q` / `jetson_clocks --show` to confirm.

## 8. Design principles

- **Measure first, then optimize.** §3 in `docs/archive/STATUS.md` projected 50–100×
  speedup for a "warp-per-event" rewrite; the prototype's actual gain
  was 2–3×, and only after switching to the cell-owner partition design
  did we realize the projected gains (4–9×). Don't trust projections
  — run the bench.
- **No unrequested features.** No extra error handling, fallback paths,
  comments, or design improvements beyond the task.
- **Keep `docs/PROFILE.md` authoritative for current measurements.** New measurement
  data lands in a new §-section (currently §7 is latest). Don't edit
  historical sections.
- **Public README stays terse.** Headline numbers + quickstart only.
  Detailed measurement logs live in `docs/archive/STATUS.md`.

## 9. Common pitfalls

- **`/tmp/ssla_s_64x80/` is volatile** — Orin clears /tmp on reboot. If
  hybrid_runner fails with "cannot open /tmp/ssla_s_64x80/meta.json",
  regenerate via `tools/make_ssla_stub.py`.
- **NVRTC PTX cache miss = ~80 s rebuild.** First run of a kernel
  variant compiles from scratch. Subsequent runs hit the cache (`~0 s`).
  Cache lives in `~/.cache/openeva/orin_ptx/`. Setting
  `OPENEVA_ORIN_NO_PTX_CACHE=1` disables it (useful for verifying
  cache correctness).
- **`__launch_bounds__(288, 1)` required for celled kernel.** Default
  NVRTC caps `MAX_THREADS_PER_BLOCK=256`; without the explicit hint,
  cuLaunchKernel returns error 701 (LAUNCH_OUT_OF_RESOURCES).
- **Hybrid stats column `kept` includes pad duplicates.** When P1
  bench pads block ring sizes to a multiple of BATCH=8, the duplicates
  pass through the kernel. Dedupe via global event index.
