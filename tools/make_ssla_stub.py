"""Generate a stub SSLA-S export (random weights) so the CPU lib can load.

The deploy CPU pipeline reads weights.npz + meta.json via
deploy/src/ssla_kernels.cpp::SslaSPipeline::load. This script writes a
schema-valid export with random weights — no training, just enough so the
loader doesn't throw.

Used by deploy/orin/hybrid_runner.py when the real export is unavailable
(e.g. dev machine without torch_geometric for python/export.py).

Usage:
    python3 /tmp/make_ssla_stub.py /tmp/ssla_s_64x80/ --h 64 --w 80
"""
import argparse
import json
import os

import numpy as np


# SSLA-S structural constants (mirror deploy/include/ssla_kernels.h).
IN_DIM = 2
CHANS  = (12, 24, 48, 96)
N_STAGES = 4
LAYERS_PER_STAGE = 2
N_LAYERS = N_STAGES * LAYERS_PER_STAGE   # 8


def build_layer(in_d: int, out_d: int, K: int, rng: np.random.Generator,
                 prefix: str, weights: dict):
    """Schema mirrors cpp/methods/ssla/ssla_detection_yolox.cpp::Impl::load_layer.

    Shapes:
      W_in_cat   (in_d, n_pos * out_d)
      W_out_cat  (n_pos * out_d, out_d)
      q/v/g/o_proj/weight  (out_d, out_d)
      input_proj/weight    (out_d, in_d)  optional, present iff in_d != out_d
      norm/weight, norm/bias  (out_d,)
    """
    A = K * K
    n_pos = A
    weights[f"{prefix}/W_in_cat"]  = rng.normal(scale=0.3, size=(in_d, n_pos * out_d)).astype(np.float32)
    weights[f"{prefix}/W_out_cat"] = rng.normal(scale=0.3, size=(n_pos * out_d, out_d)).astype(np.float32)
    for proj in ("q_proj", "v_proj", "g_proj", "o_proj"):
        weights[f"{prefix}/{proj}/weight"] = rng.normal(scale=0.3, size=(out_d, out_d)).astype(np.float32)
    if in_d != out_d:
        weights[f"{prefix}/input_proj/weight"] = rng.normal(scale=0.3, size=(out_d, in_d)).astype(np.float32)
    weights[f"{prefix}/norm/weight"] = (rng.normal(scale=0.1, size=(out_d,)) + 1.0).astype(np.float32)
    weights[f"{prefix}/norm/bias"]   = rng.normal(scale=0.1, size=(out_d,)).astype(np.float32)


def build_yolox_head_level(C_in: int, C_hid: int, n_cls: int,
                            level_idx: int, rng: np.random.Generator,
                            weights: dict):
    """Schema mirrors cpp/include/openeva/heads/async_yolox.h::load_yolox_level.

    Per-level keys:
      head/cls_convs/<k>/0/{weight, bias?}   conv 1x1 → C_hid
      head/cls_convs/<k>/1/{weight, bias, running_mean, running_var}  BN
      head/reg_convs/<k>/0,1   same pattern
      head/cls_preds/<k>/{weight, bias}      (n_cls, C_hid) → cls logits
      head/reg_preds/<k>/{weight, bias}      (4, C_hid)     → bbox
      head/obj_preds/<k>/{weight, bias}      (1, C_hid)     → objectness
    """
    k = str(level_idx)
    def conv_bn(branch: str):
        # conv 1x1: (C_hid, C_in) (cpp uses shape[0], shape[1] only)
        weights[f"head/{branch}/{k}/0/weight"] = rng.normal(scale=0.1, size=(C_hid, C_in)).astype(np.float32)
        # BN: gamma, beta, running_mean, running_var — all (C_hid,)
        weights[f"head/{branch}/{k}/1/weight"]       = (rng.normal(scale=0.05, size=(C_hid,)) + 1.0).astype(np.float32)
        weights[f"head/{branch}/{k}/1/bias"]         = rng.normal(scale=0.05, size=(C_hid,)).astype(np.float32)
        weights[f"head/{branch}/{k}/1/running_mean"] = np.zeros((C_hid,), dtype=np.float32)
        weights[f"head/{branch}/{k}/1/running_var"]  = np.ones((C_hid,), dtype=np.float32)
    conv_bn("cls_convs")
    conv_bn("reg_convs")
    weights[f"head/cls_preds/{k}/weight"] = rng.normal(scale=0.05, size=(n_cls, C_hid)).astype(np.float32)
    weights[f"head/cls_preds/{k}/bias"]   = rng.normal(scale=0.05, size=(n_cls,)).astype(np.float32)
    weights[f"head/reg_preds/{k}/weight"] = rng.normal(scale=0.05, size=(4, C_hid)).astype(np.float32)
    weights[f"head/reg_preds/{k}/bias"]   = rng.normal(scale=0.05, size=(4,)).astype(np.float32)
    weights[f"head/obj_preds/{k}/weight"] = rng.normal(scale=0.05, size=(1, C_hid)).astype(np.float32)
    weights[f"head/obj_preds/{k}/bias"]   = rng.normal(scale=0.05, size=(1,)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=80)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--tdrop", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-head-levels", type=int, default=1,
                    help="Number of YOLOX head pyramid levels. The C++ loader "
                          "auto-detects by checking head/cls_convs/<k>/0/weight "
                          "for k=0..3 in order. SSLA-S typically uses 1 level "
                          "at the smallest stage (H/8 × W/8).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    weights = {}
    # L0: K=1 (2→12), L1: K=3 (12→12), L2: K=3 (12→24), L3: K=3 (24→24),
    # L4: K=3 (24→48), L5: K=3 (48→48), L6: K=3 (48→96), L7: K=3 (96→96)
    layer_specs = [(IN_DIM, CHANS[0], 1)]   # L0
    layer_specs.append((CHANS[0], CHANS[0], 3))  # L1
    layer_specs.append((CHANS[0], CHANS[1], 3))  # L2
    layer_specs.append((CHANS[1], CHANS[1], 3))  # L3
    layer_specs.append((CHANS[1], CHANS[2], 3))  # L4
    layer_specs.append((CHANS[2], CHANS[2], 3))  # L5
    layer_specs.append((CHANS[2], CHANS[3], 3))  # L6
    layer_specs.append((CHANS[3], CHANS[3], 3))  # L7
    for li, (in_d, out_d, K) in enumerate(layer_specs):
        build_layer(in_d, out_d, K, rng, f"layers/{li}", weights)

    # YOLOX head — `--num-head-levels` levels starting from the smallest stage.
    # head_stage_idx_[k] = kNumStages - num_head_levels + k
    # For 1 level: stage 3 only (H/8, W/8 grid).
    # For 3 levels: stages 1, 2, 3 (H/2, H/4, H/8).
    c_last = CHANS[N_STAGES - 1]   # 96
    for k in range(args.num_head_levels):
        s = N_STAGES - args.num_head_levels + k
        build_yolox_head_level(C_in=CHANS[s], C_hid=c_last, n_cls=args.num_classes,
                                level_idx=k, rng=rng, weights=weights)

    npz_path = os.path.join(args.out_dir, "weights.npz")
    np.savez(npz_path, **weights)   # uncompressed (C++ loader requires this)
    print(f"Wrote {npz_path} with {len(weights)} tensors")

    meta = {
        "schema_version": 1,
        "model":   "ssla_s_yolox_det",
        "method":  "ssla",
        "height":  args.h,
        "width":   args.w,
        "num_classes": args.num_classes,
        "num_anchors": 1,
        "ssla_tdrop_window": args.tdrop,
        "dtype":  "float32",
        "layout": "C_order_row_major",
        "key_separator": "/",
        "checkpoint": "stub_random",
        "total_params": sum(int(np.prod(w.shape)) for w in weights.values()),
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(f"Total params: {meta['total_params']:,}")


if __name__ == "__main__":
    main()
