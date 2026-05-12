// async_yolox.h — task-level shared YOLOX detection head for async event runtimes.
//
// Header-only / inline. Mirrors python/modules/yolo/yolox.py but only the
// per-cell decode the streaming cpp runtimes need.
//
// Per-cell decode (`decode_yolox_cell` + `decode_yolox_cell_ct`):
//   feat_cell: (C_in,) — one cell's worth of backbone features
//   out_row:   (5 + n_cls,) [cx, cy, w, h, obj_logit, cls_logit_0...]
//
//   cls_branch = silu(conv(feat_cell)) → cls_pred → cls_logits
//   reg_branch = silu(conv(feat_cell)) → reg_pred → 4-d reg
//                                      → obj_pred → 1-d obj
//   cx = (tx + gx) · stride
//   cy = (ty + gy) · stride
//   w  = exp(tw) · stride
//   h  = exp(th) · stride
//
// obj/cls are RAW logits (Python postprocess does sigmoid; cpp leaves it
// to the caller for parity with python/models/<m>_detection.py:postprocess).
//
// Each method's cpp runtime owns the YoloxLevel(s), scratch buffers, and
// drives `decode_yolox_cell{,_ct}` for the touched cell once per event.
// SSLA / DAGr's per-method head implementations historically lived in their
// own *_model.cpp; SSLA now uses this shared header. DAGr keeps its private
// FPN-stem variant (different stem topology) — see
// `cpp/methods/dagr/dagr_detection_yolox.cpp`.

#pragma once

#include "openeva/event.h"
#include "openeva/heads/detail.h"
#include "openeva/prim/activation.h"
#include "openeva/prim/bn_fusion.h"
#include "openeva/prim/flop.h"
#include "openeva/prim/linear.h"
#include "openeva/prim/matmul.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace openeva::heads {

// One head level. Names match python/modules/yolo/yolox.py:
//   cls_convs/<idx>/0  (Conv2d 1×1)
//   cls_convs/<idx>/1  (BatchNorm2d, fused into Conv at load)
//   reg_convs/<idx>/0,1
//   cls_preds/<idx>    (Conv2d 1×1, raw logits — no BN)
//   reg_preds/<idx>    (Conv2d 1×1)
//   obj_preds/<idx>    (Conv2d 1×1)
struct YoloxLevel {
    int C_in     = 0;
    int C_hid    = 0;
    int n_cls    = 0;
    int stride_h = 0;        // input_h / H_grid (anisotropic; stride_h == stride_w
    int stride_w = 0;        // when the backbone happens to give a square stride)
    int H_grid   = 0;
    int W_grid   = 0;
    std::vector<float> W_cls_conv, b_cls_conv;
    std::vector<float> W_reg_conv, b_reg_conv;
    std::vector<float> W_cls_pred, b_cls_pred;
    std::vector<float> W_reg_pred, b_reg_pred;
    std::vector<float> W_obj_pred, b_obj_pred;
};

// Load one head level's BN-fused conv + raw pred weights.
//
// Folds each BN into the preceding 1×1 conv:
//   W' = W · scale,  b' = (b_conv or 0) · scale + shift
// so the streaming decode is just two matvecs per branch (no per-channel
// BN loop in the hot path).
//
// `key_root` is "head" by default (matches the Python module hierarchy).
// `err_prefix` is a short tag for error messages ("SSLA", "AEGNN", ...).
//
// `stride_h` / `stride_w` need NOT be equal: the head's per-cell decode
// uses anisotropic stride. Pass `stride_h == stride_w` for SSLA-style
// backbones where every halving stage gives a square reduction.
inline void load_yolox_level(
    const std::unordered_map<std::string, Tensor>& W,
    YoloxLevel& level,
    int idx,
    int C_in, int C_hid, int n_cls,
    int stride_h, int stride_w, int H_grid, int W_grid,
    const std::string& err_prefix,
    const std::string& key_root = "head") {

    level.C_in     = C_in;
    level.C_hid    = C_hid;
    level.n_cls    = n_cls;
    level.stride_h = stride_h;
    level.stride_w = stride_w;
    level.H_grid   = H_grid;
    level.W_grid   = W_grid;

    auto load_conv_bn = [&](const std::string& cp, const std::string& bp,
                            std::vector<float>& Wo, std::vector<float>& bo) {
        const auto& cw = detail::must_get_t(W, cp + "/weight", err_prefix);
        if (cw.shape.size() < 2) {
            throw std::runtime_error(err_prefix + ": " + cp + " conv rank");
        }
        const int out_c = static_cast<int>(cw.shape[0]);
        const int in_c  = static_cast<int>(cw.shape[1]);
        Wo.assign(cw.data.begin(), cw.data.end());

        const auto& bn_g  = detail::must_get_t(W, bp + "/weight",       err_prefix);
        const auto& bn_b  = detail::must_get_t(W, bp + "/bias",         err_prefix);
        const auto& bn_mu = detail::must_get_t(W, bp + "/running_mean", err_prefix);
        const auto& bn_va = detail::must_get_t(W, bp + "/running_var",  err_prefix);
        const auto bn = openeva::prim::fuse_bn(
            bn_g.data.data(), bn_b.data.data(),
            bn_mu.data.data(), bn_va.data.data(),
            static_cast<int>(bn_g.numel()), 1e-5f);
        const Tensor* cb = detail::maybe_get_t(W, cp + "/bias");
        bo.assign(static_cast<std::size_t>(out_c), 0.0f);
        const float* cb_ptr = cb ? cb->data.data() : nullptr;
        openeva::prim::fold_bn_into_weight_2d(
            Wo.data(), out_c, in_c,
            bn.scale.data(), bn.shift.data(),
            cb_ptr, bo.data());
    };

    auto load_pred = [&](const std::string& p,
                         std::vector<float>& Wo, std::vector<float>& bo) {
        const auto& cw = detail::must_get_t(W, p + "/weight", err_prefix);
        const auto& cb = detail::must_get_t(W, p + "/bias",   err_prefix);
        Wo.assign(cw.data.begin(), cw.data.end());
        bo.assign(cb.data.begin(), cb.data.end());
    };

    const std::string ls = std::to_string(idx);
    load_conv_bn(key_root + "/cls_convs/" + ls + "/0",
                 key_root + "/cls_convs/" + ls + "/1",
                 level.W_cls_conv, level.b_cls_conv);
    load_conv_bn(key_root + "/reg_convs/" + ls + "/0",
                 key_root + "/reg_convs/" + ls + "/1",
                 level.W_reg_conv, level.b_reg_conv);
    load_pred(key_root + "/cls_preds/" + ls, level.W_cls_pred, level.b_cls_pred);
    load_pred(key_root + "/reg_preds/" + ls, level.W_reg_pred, level.b_reg_pred);
    load_pred(key_root + "/obj_preds/" + ls, level.W_obj_pred, level.b_obj_pred);
}

// Per-cell decode: writes (5 + n_cls) floats to `out_row`.
// `feat_cell` is one cell's `level.C_in` floats; `cls_f`/`reg_f`/`cls_logits`
// are caller-owned scratch buffers (resized to ≥ C_hid / ≥ n_cls).
inline void decode_yolox_cell(
    const YoloxLevel& level,
    int gx, int gy,
    const float* feat_cell,
    float* out_row,
    std::vector<float>& cls_f,
    std::vector<float>& reg_f,
    std::vector<float>& cls_logits) {

    const int C_in   = level.C_in;
    const int C_hid  = level.C_hid;
    const int n_cls  = level.n_cls;
    const int stride_h = level.stride_h;
    const int stride_w = level.stride_w;

    if (cls_f.size() < static_cast<std::size_t>(C_hid)) {
        cls_f.resize(C_hid);
        reg_f.resize(C_hid);
    }
    if (cls_logits.size() < static_cast<std::size_t>(n_cls)) {
        cls_logits.resize(n_cls);
    }

    openeva::prim::matvec(feat_cell, C_in, level.W_cls_conv.data(), C_hid,
                          level.b_cls_conv.data(), cls_f.data());
    openeva::prim::silu_inplace(cls_f.data(), C_hid);
    openeva::prim::matvec(cls_f.data(), C_hid, level.W_cls_pred.data(), n_cls,
                          level.b_cls_pred.data(), cls_logits.data());

    openeva::prim::matvec(feat_cell, C_in, level.W_reg_conv.data(), C_hid,
                          level.b_reg_conv.data(), reg_f.data());
    openeva::prim::silu_inplace(reg_f.data(), C_hid);
    float reg_raw[4];
    float obj_logit;
    openeva::prim::matvec(reg_f.data(), C_hid, level.W_reg_pred.data(), 4,
                          level.b_reg_pred.data(), reg_raw);
    openeva::prim::matvec(reg_f.data(), C_hid, level.W_obj_pred.data(), 1,
                          level.b_obj_pred.data(), &obj_logit);

    // 2 exps + 2 adds + 4 muls
    openeva::prim::add_flops(2 * openeva::prim::kExpFlops + 6);
    out_row[0] = (reg_raw[0] + float(gx)) * float(stride_w);
    out_row[1] = (reg_raw[1] + float(gy)) * float(stride_h);
    out_row[2] = std::exp(reg_raw[2])     * float(stride_w);
    out_row[3] = std::exp(reg_raw[3])     * float(stride_h);
    out_row[4] = obj_logit;
    for (int c = 0; c < n_cls; ++c) {
        out_row[5 + c] = cls_logits[c];
    }
}

// Compile-time-dim variant for the hot path. C_IN / C_HID are folded into
// the inner matvec so GCC can fully schedule the small (typically 96-256
// channel) MVs. n_cls stays runtime — sub-1% of total head work, and
// templating it would double instantiations.
template <int C_IN, int C_HID>
inline void decode_yolox_cell_ct(
    const YoloxLevel& level,
    int gx, int gy,
    const float* feat_cell,
    float* out_row,
    std::vector<float>& cls_f,
    std::vector<float>& reg_f,
    std::vector<float>& cls_logits) {

    const int n_cls    = level.n_cls;
    const int stride_h = level.stride_h;
    const int stride_w = level.stride_w;

    if (cls_f.size() < static_cast<std::size_t>(C_HID)) {
        cls_f.resize(C_HID);
        reg_f.resize(C_HID);
    }
    if (cls_logits.size() < static_cast<std::size_t>(n_cls)) {
        cls_logits.resize(n_cls);
    }

    openeva::prim::matvec_ct<C_IN, C_HID>(
        feat_cell, level.W_cls_conv.data(),
        level.b_cls_conv.data(), cls_f.data());
    openeva::prim::silu_inplace_ct<C_HID>(cls_f.data());
    openeva::prim::matvec(cls_f.data(), C_HID, level.W_cls_pred.data(), n_cls,
                          level.b_cls_pred.data(), cls_logits.data());

    openeva::prim::matvec_ct<C_IN, C_HID>(
        feat_cell, level.W_reg_conv.data(),
        level.b_reg_conv.data(), reg_f.data());
    openeva::prim::silu_inplace_ct<C_HID>(reg_f.data());
    float reg_raw[4];
    float obj_logit;
    openeva::prim::matvec_ct<C_HID, 4>(
        reg_f.data(), level.W_reg_pred.data(),
        level.b_reg_pred.data(), reg_raw);
    openeva::prim::matvec_ct<C_HID, 1>(
        reg_f.data(), level.W_obj_pred.data(),
        level.b_obj_pred.data(), &obj_logit);

    openeva::prim::add_flops(2 * openeva::prim::kExpFlops + 6);
    out_row[0] = (reg_raw[0] + float(gx)) * float(stride_w);
    out_row[1] = (reg_raw[1] + float(gy)) * float(stride_h);
    out_row[2] = std::exp(reg_raw[2])     * float(stride_w);
    out_row[3] = std::exp(reg_raw[3])     * float(stride_h);
    out_row[4] = obj_logit;
    for (int c = 0; c < n_cls; ++c) {
        out_row[5 + c] = cls_logits[c];
    }
}

}  // namespace openeva::heads
