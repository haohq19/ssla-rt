# OpenEVA SSLA Real-Time Deployment on Jetson Orin

_Last updated: 2026-05-07_

Live-camera deployment of SSLA-S on a Jetson Orin board with an
iniVation DVXplorer (Micro) event camera. Companion to the parent
`deploy/` PoC; nothing in this subtree is bound by `../CLAUDE.md`'s
custom-op rules.

## Hardware target (verified)

```
Board     : NVIDIA Orin NX 16 GB Devkit (R35.4.1, JetPack 5.1.2)
CPU       : 8 × Cortex-A78AE
GPU       : Ampere GA10B, sm_87, 8 SMs, 1024 CUDA cores, 32 tensor cores
Memory    : 16 GB LPDDR5, unified between CPU and GPU (no PCIe)
CUDA      : 11.4 (libnvrtc.so.11.4.300, /usr/local/cuda)
Camera    : DVXplorer Micro (USB 2.0, vendor 152a:8419, 640 × 480)
```

Live ingest measured at **6.6–7.5 Mev/s** in casual desk-scene motion —
already above the parent `deploy/`'s ~3 Mev/s CPU peak on EPYC. GPU
offload is required for SSLA-S inference at this rate on this Jetson.

## Toolchain (verified)

```
camera I/O : dv_processing 2.0.3 Python bindings  (apt: dv-processing-python)
GPU glue   : cuda-python 11.7.1                   (pip: cuda-python==11.7.1)
JIT        : nvrtc 11.4.300                       (in /usr/local/cuda)
viz        : OpenCV 4.2 Python                    (libopencv 4.5 system)
```

**Why Python orchestration, not C++.** We attempted a `libcaer` /
`dv-processing` C++ reader first; both blocked:
- `libcaer 3.3.17` (latest) doesn't recognise the DVXplorer Micro
  chip ID — `caerDeviceOpen` returns errno=-4. iniVation moved camera
  I/O for newer products into `dv-processing` 2.x and stopped
  maintaining libcaer for new device variants.
- The `dv-processing` C++ headers are header-only but require
  C++20 with `<source_location>` and `libfmt 10` — neither shipped
  by the Jetson's GCC 9 / 10 toolchain or focal libfmt 9.

Python `dv_processing` works out of the box because its precompiled
.so links against the newer libstdc++ already present at runtime.
For an event-stream PoC, the Python producer at ~110 MB/s (7 Mev/s ×
16 B) is well within numpy's reach; the persistent CUDA kernel still
runs in raw CUDA C++. Memory between them is shared via CUDA managed
allocations on Tegra (no PCIe copy).

## Why a persistent kernel

Per-launch kernels cost ~5–15 µs each on Jetson — at 1 Mev/s with a
launch per event that's 5–15 s of overhead per second of real time.
Even batched at 10k events / launch the launch tax is 0.5–1.5 µs/event.

A **persistent kernel** is launched once and runs for the entire
inference session. Thread blocks pull events from a unified-memory
ring buffer in a tight loop, process them through the SSLA pipeline,
and write predictions back to host-visible memory. Launch tax is paid
once.

**Tegra coherence caveat.** Orin NX 16 GB reports
`CONCURRENT_MANAGED_ACCESS = 0`: the host cannot safely write to a
managed-memory page while a kernel is reading the same page — it
segfaults. For the persistent loop the SPSC ring and the cross-side
control scalars (`ring_head`, `stop_flag`, `events_done`) must live in
**pinned host memory** (`cuMemHostAlloc`) instead, which IS coherent
for concurrent CPU/GPU access on Tegra. Weights and hidden state stay
in managed memory because each is touched by only one side at runtime.

## Proposed scheme — S9: persistent-kernel SSLA-S on Orin GPU

Three milestones, each a verifiable, runnable artifact:

### S9-base — single-block correctness baseline

```
[Python producer]                     [GPU persistent kernel — 1 block]
  dv_processing → managed ring   ───►   block 0: stage 0→1→2→3→head (FIFO)
                                                  hidden state in global mem
                                                  no contention (single block)
  predictions  ◄───  unified mem
```

- 1 thread block, 256 threads, persistent for session lifetime.
- Threads parallelise each event's matvec / elu / sigmoid; events
  stay serial across the block (FIFO order, deterministic).
- Goal: **bit-identical to deploy/S1** at oracle-dump checkpoints.
- Use: prove the kernel skeleton + ring buffer + weight loading.
- Throughput estimate: ~1–1.5 Mev/s (one SM busy, others idle).

### S9-spatial — port of deploy/S6 to GPU

```
managed ring  ─►  Python dispatcher  ─┬─►  block 0 (cols   0..79  + halo)
                                      ├─►  block 1 (cols  80..159 + halo)
                                      ├─►  ...
                                      └─►  block 7 (cols 560..639 + halo)
                                           each block runs FULL 4-stage + head
                                           per-patch atomicCAS spinlocks
```

- 8 blocks × full pipeline, column-strip sharded.
- Per-patch atomic spinlocks for boundary events touching adjacent
  shards.
- Goal: **drops match S1**, bounded fp32 drift, **5–8 Mev/s**.
- Use: production deployment baseline.

### S9-stage — workload-matched block specialization (stretch)

- Block count per stage matches cost ratio in deploy/README §1.
- Inter-stage handoff via global-memory SPSC queues.
- Goal: better p50 latency than S9-spatial at similar throughput.
- Only attempt after S9-spatial is stable.

## Planned file layout

```
deploy/orin/
├── README.md                     this file
├── viz/
│   └── event_viewer.py           live event viewer (done)
├── kernels/
│   └── ssla_s_kernel.cu          CUDA C++ source string, JIT-compiled
├── orin/                         Python package
│   ├── __init__.py
│   ├── ring.py                   managed-memory SPSC ring + producer thread
│   ├── weights.py                load exported SSLA-S weights to device
│   ├── kernel.py                 NVRTC compile + launch helpers
│   ├── runner.py                 main loop: open camera, launch, read preds
│   └── viz_overlay.py            overlay predictions on event frame
├── run_s9_base.py                entry point — single-block runner
└── run_s9_spatial.py             entry point — 8-block sharded runner
```

No CMake, no C++ static libs. The .cu file is read at startup and
compiled to PTX by NVRTC; cuda-python loads the resulting module and
launches the kernel.

## Equivalence protocol

Mirror the parent deploy's oracle-dump approach:
- runner takes `--oracle-dump path.bin --oracle-every N`,
- producer pushes a `seq=N` marker every N events,
- GPU snapshots predictions tensor when all blocks have processed up
  to that seq,
- compare against `cpp/methods/ssla/` S1 dump via the parent's
  `oracle_diff` binary.

Acceptance:
- S9-base: `max|Δ| ≤ fp32 ULP scale` (≈ 1e-5) — strictly bit-equivalent.
- S9-spatial: drops match S1 exactly; `max|Δ|` bounded (S6 measured
  4.4 on EPYC; expected similar on GPU).

## Setup (one-time)

```sh
# iniVation Python bindings + headless OpenCV
sudo apt install -y dv-processing-python libzstd-dev

# CUDA-Python (NVRTC + driver API bindings)
pip3 install --user 'cuda-python==11.7.1'

# Optional: libcaer dev headers (for future C++ helpers, not the
# camera reader — see "Why Python orchestration" above).
sudo apt install -y libcaer-dev
```

## Run (planned, not yet implemented)

```sh
# Live camera → S9-base persistent kernel → predictions
python3 deploy/orin/run_s9_base.py \
    --weights outputs/export/ssla_s_det_Gen1/

# Same but 8-block sharded (after S9-spatial lands)
python3 deploy/orin/run_s9_spatial.py \
    --weights outputs/export/ssla_s_det_Gen1/

# Equivalence vs deploy/S1 (synthetic events, no camera)
python3 deploy/orin/run_s9_base.py \
    --weights outputs/export/ssla_s_det_Gen1/ \
    --random-events 200000 --seed 1 \
    --oracle-dump /tmp/oracle_s9.bin --oracle-every 10000
./cpp/build/oracle_diff /tmp/oracle_s1.bin /tmp/oracle_s9.bin --tol 1e-5
```
