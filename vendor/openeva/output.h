#pragma once

#include <cstddef>
#include <variant>
#include <vector>

namespace openeva {

struct Tensor {
    std::vector<std::size_t> shape;
    std::vector<float>       data;

    std::size_t numel() const {
        std::size_t n = 1;
        for (std::size_t d : shape) n *= d;
        return n;
    }
};

/// Detection head output. Decoded to pixel space, one row per anchor.
///
/// `predictions` has shape `{N_anchors_total, 5 + num_classes}`:
///   columns `[cx, cy, w, h, obj, cls_0, cls_1, ...]`
///
/// Contract (matches each method's Python head output exactly, so Python's
/// postprocess can consume it unchanged):
///   - cx, cy, w, h are decoded to pixel space
///   - obj / cls follow the method's convention:
///       * SSLA / DAGr (YOLOXHead): obj and cls are RAW LOGITS
///         (postprocess applies sigmoid)
///       * FARSE-CNN (FARSEYOLOHead, YOLO v1 style): obj has sigmoid
///         applied, cls are RAW logits
///   - no confidence threshold, no argmax, no NMS — Python owns postprocessing.
struct DetectionOutput {
    Tensor predictions;
};

/// Classification head output. shape = `{num_classes}`.
struct ClassificationOutput {
    Tensor logits;
};

/// Segmentation head output. shape = `{num_classes, H, W}` or `{H, W}`.
struct SegmentationOutput {
    Tensor mask;
};

/// Task-agnostic per-step return type. `std::monostate` lets async models
/// signal that an event caused no head emission.
using ModelOutput = std::variant<
    std::monostate,
    DetectionOutput,
    ClassificationOutput,
    SegmentationOutput>;

}  // namespace openeva
