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

## 1. Repo layout

```
ssla-rt/
├── CMakeLists.txt         standalone build, no parent repo dependency
├── include/               public C/C++ headers (ssla_kernels.h, spsc.h, lat_stats.h, ...)
├── src/                   CPU pipeline sources
│   ├── lib_stage01_to_gpu.cpp   the production hybrid pipeline (CPU s0+s1 → pinned ring → GPU)
│   ├── lib_stage01_capi.cpp     CPU-only stages s0+s1
│   ├── lib_stage0_capi.cpp      CPU-only stage 0
│   ├── ssla_kernels.cpp         per-stage SSLA-S forward kernels
│   └── s*.cpp                   historical scheme experiments (opt-in build)
├── orin/
│   ├── hybrid_runner.py         production end-to-end driver (--kernel-variant celled)
│   ├── bench_s2_s3_head_celled.py     P1 oracle harness for celled GPU kernel
│   ├── bench_s2_s3_head.py            P1 oracle for legacy block-coop kernel
│   ├── bench_s2_s3_head_w.py          P1 oracle for warp-per-event (deprecated, races)
│   ├── kernels/
│   │   ├── ssla_s2_s3_head_celled.cuh   PRODUCTION GPU kernel (cell-owner warps)
│   │   ├── ssla_s2_s3_head.cuh          legacy block-coop kernel
│   │   ├── ssla_s2_s3_head_w.cuh        warp-per-event kernel (deprecated; has races)
│   │   ├── proto_layer_pair.cuh         warp-cooperative layer primitives
│   │   ├── ssla_primitives.cuh, ssla_layer.cuh   block-coop primitives
│   │   └── ssla_step.cuh                full-pipeline persistent kernel proto
│   ├── orin/                     Python utility package
│   │   ├── cuda_util.py          cuMemHostAlloc, cuMemAllocManaged wrappers
│   │   ├── nvrtc_util.py         NVRTC compile + PTX cache by hash + CudaModule
│   │   ├── hybrid_common.py      ctypes mirrors of HybridS2S3Config etc.
│   │   ├── weights_ssla.py       reshape qvgIn / goW for matvec-friendly layout
│   │   └── ssla_ref.py           single-thread CPU oracle (LayerRef, layer_step)
│   ├── tests/                    unit tests (test_proto_layer_pair.py, ...)
│   ├── viz/                      live event-viewer (oscilloscope + DVXplorer controls)
│   ├── STATUS.md                 full measurement log §§1–7 (READ THIS FIRST)
│   └── HYBRID_DESIGN.md          design doc for the CPU/GPU split
├── tools/
│   └── make_ssla_stub.py         schema-valid random-weight export generator
├── vendor/                       vendored openeva pieces (event, prim/, heads/, weights_loader)
└── docs/
    └── DEV_LOG.md                historical PoC dev log (predates the celled rewrite)
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

See `orin/STATUS.md` §§6–7 for current passing numbers.

## 6. Headline measurements (as of 2026-05-12; see STATUS.md §9 for the CPU NEON layer that supersedes §8's CPU numbers)

| metric | legacy | **current (post-NEON)** | speedup vs legacy |
|---|---:|---:|---:|
| GPU drain saturation (live persistent kernel) | 5 kev/s | **~47 kev/s agg** | **9.4×** |
| owner-pass-3 p50 | 530 µs | **127 µs** | **4.2×** |
| owner-pass-3 p99 | 1.8 ms | **310 µs** | **5.8×** |
| Per-event stage_forward(0) on CPU | 3.94 µs | **1.83 µs** | **2.15×** (§9) |
| Per-event stage_forward(1) on CPU | 12.88 µs | **9.50 µs** | **1.36×** (§9) |
| CPU admit per shard | 135 kev/s | **216 kev/s** | **1.60×** (§9) |
| **CPU side ceiling** (max sustained admit) | 0.82 Mev/s @ n=8 | **1.30 Mev/s @ n=7** | **1.60×** (§9) |

CPU side now exceeds the 1 Mev/s design target. GPU drain ceiling (~0.6 Mev/s
admit-equivalent) is the binding system-level limit; the CPU can produce more
than the GPU can drain. Path past 1 Mev/s end-to-end requires GPU-side work
(e.g. BATCH 8→4 → 2 blocks/SM).

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

- **Measure first, then optimize.** §3 in STATUS.md projected 50–100×
  speedup for a "warp-per-event" rewrite; the prototype's actual gain
  was 2–3×, and only after switching to the cell-owner partition design
  did we realize the projected gains (4–9×). Don't trust projections
  — run the bench.
- **No unrequested features.** No extra error handling, fallback paths,
  comments, or design improvements beyond the task.
- **Keep STATUS.md authoritative for measurements.** New measurement
  data lands in a new §-section (currently §7 is latest). Don't edit
  historical sections.
- **Public README stays terse.** Headline numbers + quickstart only.
  Detailed measurement logs live in `orin/STATUS.md`.

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
