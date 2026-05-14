"""Convert a DSEC-trained SSLA-S Lightning checkpoint to the
weights.npz + meta.json schema this deploy expects.

The upstream checkpoint stores SSLA layers at indices 0,1,4,5,8,9,12,13
(slots 2,3 / 6,7 / 10,11 hold stateless spatial_downsample +
temporal_dropout ops with no parameters). Our deploy schema is 8
consecutive layers 0..7 — so we remap:
    ckpt 0,1   → deploy 0,1   (stage 0)
    ckpt 4,5   → deploy 2,3   (stage 1)
    ckpt 8,9   → deploy 4,5   (stage 2)
    ckpt 12,13 → deploy 6,7   (stage 3)

We use ema_state_dict if present (EMA-of-weights from the trainer, decay
0.9999 — almost always better than the live optimizer state at eval).
pixel2patch_lut buffers are skipped (C++ recomputes them from H/W).

Schema details mirror tools/make_ssla_stub.py.
"""
import argparse
import json
import os
import numpy as np
import torch


CKPT_TO_DEPLOY_IDX = {0: 0, 1: 1, 4: 2, 5: 3, 8: 4, 9: 5, 12: 6, 13: 7}

# Keys to drop (recomputed in C++ or unused at inference).
_SKIP_PARAM_SUFFIXES = ("pixel2patch_lut", "num_batches_tracked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Lightning .ckpt file")
    ap.add_argument("out_dir")
    ap.add_argument("--h", type=int, default=120,
                    help="Runtime input height (must match --h-full at run time).")
    ap.add_argument("--w", type=int, default=160,
                    help="Runtime input width  (must match --w-full at run time).")
    ap.add_argument("--use-ema", action="store_true", default=True,
                    help="Use ema_state_dict if present (default).")
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema_state_dict") if args.use_ema else None
    src = "ema_state_dict"
    if sd is None:
        sd = ck["state_dict"]
        src = "state_dict"
    print(f"[convert] using {src} ({len(sd)} tensors) from {args.ckpt}")

    hparams = ck.get("hyper_parameters", {})
    num_classes = int(hparams.get("num_classes", 2))
    tdrop_windows = []
    for stage in hparams.get("stages", []):
        for layer in stage.get("layers", []):
            if layer.get("type") == "temporal_dropout":
                tdrop_windows.append(int(layer["window"]))
    tdrop = tdrop_windows[0] if tdrop_windows else 4
    print(f"[convert] num_classes={num_classes}  tdrop={tdrop}")

    weights = {}
    n_skipped = 0
    n_layer_kept = 0
    n_head_kept = 0
    for name, tensor in sd.items():
        if any(name.endswith(s) for s in _SKIP_PARAM_SUFFIXES):
            n_skipped += 1
            continue
        arr = tensor.detach().cpu().to(torch.float32).numpy()

        if name.startswith("layers."):
            parts = name.split(".")
            ckpt_idx = int(parts[1])
            if ckpt_idx not in CKPT_TO_DEPLOY_IDX:
                # Unexpected layer index — skip with a warning.
                print(f"[convert] WARN: skipping {name} (ckpt idx {ckpt_idx} "
                      f"not in remap)")
                continue
            new_idx = CKPT_TO_DEPLOY_IDX[ckpt_idx]
            new_key = f"layers/{new_idx}/" + "/".join(parts[2:])
            weights[new_key] = arr
            n_layer_kept += 1
        elif name.startswith("head."):
            new_key = name.replace(".", "/")
            weights[new_key] = arr
            n_head_kept += 1
        else:
            print(f"[convert] WARN: unrecognized key {name} — skipping")
            n_skipped += 1

    print(f"[convert] kept {n_layer_kept} layer + {n_head_kept} head tensors, "
          f"skipped {n_skipped}")

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, "weights.npz")
    np.savez(npz_path, **weights)   # uncompressed (C++ loader requires this)
    print(f"[convert] wrote {npz_path} ({len(weights)} tensors)")

    meta = {
        "schema_version":     1,
        "model":              hparams.get("model", "ssla_s_yolox_det"),
        "method":             "ssla",
        "height":             args.h,
        "width":              args.w,
        "num_classes":        num_classes,
        "num_anchors":        1,
        "ssla_tdrop_window":  tdrop,
        "dtype":              "float32",
        "layout":             "C_order_row_major",
        "key_separator":      "/",
        "checkpoint":         os.path.abspath(args.ckpt),
        "ckpt_epoch":         int(ck.get("epoch", -1)),
        "ckpt_global_step":   int(ck.get("global_step", -1)),
        "source_state_dict":  src,
        "ckpt_native_h":      int(hparams.get("height", 0)),
        "ckpt_native_w":      int(hparams.get("width", 0)),
        "total_params":       int(sum(int(np.prod(w.shape)) for w in weights.values())),
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[convert] wrote {meta_path}  total_params={meta['total_params']:,}")


if __name__ == "__main__":
    main()
