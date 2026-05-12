# ssla-rt

**Real-time SSLA-S inference runtime for edge devices.**

Streams events from a DVS camera (iniVation DVXplorer / DVXplorer Micro)
through a 4-stage SSLA-S backbone + YOLOX detection head, end-to-end,
without batching. Targets the NVIDIA Jetson Orin NX as the reference
platform.

Originally a subtree of [OpenEVA](https://github.com/...) — split out
because the deploy concerns (latency, hardware coherence, persistent
CUDA kernels) move on a different cadence from the research benchmark.

---

## Headline numbers

Measured on Jetson Orin NX (MAXN, 8 SM Ampere @ 918 MHz) at 64 × 80
post-subsample resolution. 2-block GPU sharding with the cell-owner warp
kernel; CPU side at 4 shards halo=2 (lib_stage01_to_gpu).

| metric | block-coop (legacy) | **ssla-rt celled** | speedup |
|---|---:|---:|---:|
| GPU drain (saturation) | 5 kev/s | **45 kev/s** | **9.0×** |
| GPU owner-pass-3 p50 | 530 µs | **127 µs** | **4.2×** |
| GPU owner-pass-3 p99 | 1.8 ms | **310 µs** | **5.8×** |
| End-to-end admit ceiling | 0.07 Mev/s | **~0.6 Mev/s** | **8.6×** |
| Drift vs CPU oracle (drop count) | 0 (with risk) | **0** (race-free by design) | — |
| Drift vs CPU oracle (max\|Δ\|) | 4.40 | 4.40 | matches |

The system bottleneck moved from GPU (`block-coop` was drain-limited) to
CPU side admit. See [docs/architecture.md](docs/architecture.md) for
the design rationale and [orin/STATUS.md](orin/STATUS.md) §§6–7 for the
full measurement log.

---

## Architecture

```
DVXplorer ──► dv_processing ──► libstage01_to_gpu.so ──► pinned MPSC ring ──► persistent CUDA kernel
                                │ 4 CPU shards          │ ×2 (one per          │ 2 blocks × 9 cell-
                                │ halo = 2              │   GPU block)         │ owner warps each,
                                │ stages s0 + s1        │                      │ stages s2 + s3 + head
                                │                       │                      │ race-free by partition
```

Key design properties:

* **CPU shards** run s0 + s1 with halo-=-2 spatial sharding. Owner-pass
  events get pushed to per-GPU-block pinned rings via lock-free MPSC.
* **GPU side** runs 2 blocks × 9 warps each. Hidden state cells are
  partitioned by `warp_id = (cy % 3) * 3 + (cx % 3)`. Any event's 3×3
  patch update writes to 9 cells, each owned by a different warp →
  no within-block race on hidden state OR tdrop counters, by construction.
* **Per-batch gather** sums 9 per-warp contributions into each event's
  out_feat (lane-striped warp-shuffle reduction).
* **No GPU atomics, no `__syncthreads` inside the per-event step** —
  only `__syncwarp` and per-batch barriers.
* **Tegra-coherence-safe**: rings + control flags use pinned host memory
  (`cuMemHostAlloc`); hidden state + weights use managed memory with no
  concurrent CPU/GPU writes.

---

## Quickstart

### Requirements

* NVIDIA Jetson Orin NX (or compatible Ampere) running JetPack 5+
* CUDA toolkit ≥ 11.4, `pip install cuda-python`
* GCC 9.4+, CMake 3.16+
* numpy
* (optional, for live camera) `dv-processing` 1.7+ with the
  DVXplorer Micro driver

### Build

```bash
git clone <repo>
cd ssla-rt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces three shared libraries under `build/`:

* `libstage01_to_gpu.so` — the production hybrid pipeline (used by `orin/hybrid_runner.py`)
* `libstage01_capi.so` — CPU-only baseline (stages s0 + s1)
* `libstage0_capi.so` — stage-0-only baseline

To also build the historical CPU-sharding scheme binaries (S2..S8),
configure with `-DSSLA_RT_BUILD_SCHEMES=ON`. Note: these expect x86-64
intrinsics and may not compile on aarch64 without patches.

### Run with stub weights (no training needed)

For benchmarking / smoke testing, generate random-init SSLA-S weights
that satisfy the C++ loader's schema:

```bash
python3 tools/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80
```

Then run the hybrid pipeline with the cell-owner GPU kernel:

```bash
cd orin
python3 hybrid_runner.py \
    --weights /tmp/ssla_s_64x80/ \
    --kernel-variant celled \
    --synthetic-mev 0.5 \
    --duration-s 10
```

Expected output: `P2 PASS — both blocks advanced and wrote non-zero
predictions; no ring deadlock.` plus a GPU latency table.

### Run with trained weights (real predictions)

Export weights via OpenEVA's `python/exporter.py` from a trained SSLA-S
checkpoint. The export schema is documented in
[orin/STATUS.md](orin/STATUS.md). Replace the `--weights` path above.

For live camera input (drop `--synthetic-mev`), connect a DVXplorer USB
camera and run `hybrid_runner.py` with `--duration-s 30`.

---

## Repo layout

```
ssla-rt/
├── CMakeLists.txt           standalone build, no parent dependency
├── include/                 public C/C++ headers (ssla_kernels.h, lat_stats.h, ...)
├── src/                     CPU pipeline sources (lib_stage01_to_gpu.cpp, ssla_kernels.cpp, ...)
├── orin/
│   ├── hybrid_runner.py     production end-to-end driver
│   ├── bench_*.py           offline drain_n harnesses + P1 oracles
│   ├── kernels/             CUDA source (ssla_s2_s3_head_celled.cuh + primitives)
│   ├── orin/                Python utility package (cuda_util, nvrtc_util, hybrid_common)
│   ├── tests/               unit tests
│   ├── viz/                 live event-viewer (oscilloscope + controls)
│   ├── STATUS.md            full measurement log (§§1–7)
│   └── HYBRID_DESIGN.md     design doc for the CPU/GPU split
├── tools/
│   └── make_ssla_stub.py    schema-valid random-weight export generator
├── vendor/                  vendored OpenEVA pieces (event, prim/, heads/, weights_loader)
└── docs/
    └── DEV_LOG.md           historical PoC dev log (predates the celled rewrite)
```

---

## Status

| Phase | Result |
|---|---|
| P1 — kernel correctness vs CPU oracle (200 k events offline) | ✅ PASS (drift 0, max\|Δ\| 4.40) |
| P2 — live synthetic 10 s | ✅ PASS (ring lag 0, all 80/80 cells written) |
| P3 — saturation sweep (0.05/0.5/1.0/2.0 Mev/s synthetic) | ✅ PASS (CPU caps at 0.6 Mev/s admit) |
| Live camera (DVXplorer Micro) | ✅ PASS at low admit; high-rate scene TBD |

See [orin/STATUS.md](orin/STATUS.md) §§6–7 for the full measurement
tables and tail-latency distributions.

### Known limitations / next steps

1. **CPU s0+s1 is now the bottleneck.** End-to-end admit hard-caps at
   ~0.6 Mev/s (4 shards × halo=2 throughput). Going past that requires
   CPU-side work: more shards, SIMD-vectorized matvecs, or halo-overhead
   reduction.
2. **Single block per SM** on the GPU side. SMEM (~41 KB) is the limit;
   halving BATCH from 8 to 4 would enable 2 blocks/SM → another
   ~1.5–2× GPU drain ceiling.
3. **Camera knobs not auto-tuned.** The DVXplorer's `contrastThreshold`
   and `MIPITimeoutValue` control admit rate; `orin/viz/event_viewer.py`
   is the manual exploration UI.

---

## License

[Apache License 2.0](LICENSE) © 2026 Haoqiang Hao.

The vendored code under `vendor/` originates from the OpenEVA research
repository (same author) and is re-licensed under Apache-2.0 here for
consistency with this redistribution.

---

## Citing

If this work is useful, please cite:

```
@software{ssla-rt,
  title  = {ssla-rt: Real-time SSLA-S inference runtime for edge devices},
  author = {Hao, Haoqiang},
  year   = {2026},
  url    = {https://github.com/haohq19/ssla-rt},
}
```

And the underlying SSLA paper:

```
@article{ssla2024,
  title  = {SSLA: ...},
  author = {Hao et al.},
  year   = {2024},
  url    = {https://arxiv.org/abs/2603.06228},
}
```
