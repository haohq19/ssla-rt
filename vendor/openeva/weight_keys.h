#pragma once

// weight_keys.h — canonical tensor-name conventions for OpenEVA exports.
//
// The contract between python/exporter.py and the C++ loaders is:
//   - every parameter / buffer is saved to weights.npz with key =
//     `named_parameters()` or `named_buffers()` name, with `.` → `/`;
//   - all tensors are float32, C-order row-major (no transposes at export);
//   - meta.json carries `schema_version`, `model`, `method`, `height`,
//     `width`, `num_classes` — everything else architectural is either
//     implied by the variant in `model` (channel tables baked into C++
//     variant structs) or derived from it by the loader.
//
// This header centralizes the well-known *suffixes* so that a Python rename
// is a single-point edit. If you rename a PyTorch module, bump
// `kSupportedSchemaVersion` in weights_loader.h and update the relevant
// suffix below — old exports fail fast at load_meta() with a clear error.
//
// Methods may still construct full keys by concatenating `prefix +
// kSomeSuffix`. This header does NOT try to enumerate every full key;
// method-specific load_weights() functions know their own prefix tree.

namespace openeva::weight_keys {

// ---- PyTorch canonical submodule suffixes ---------------------------------
// (Used by multiple methods; kept here to avoid stringly-typed drift.)
inline constexpr const char* kWeight = "weight";
inline constexpr const char* kBias   = "bias";

// BatchNorm running stats (exporter emits these as buffers).
inline constexpr const char* kBnRunningMean = "running_mean";
inline constexpr const char* kBnRunningVar  = "running_var";
inline constexpr const char* kBnEps         = "eps";

// ---- SSLA / SSLA2 ---------------------------------------------------------
namespace ssla {
inline constexpr const char* kScatterProj = "scatter_proj_weights";
inline constexpr const char* kGatherProj  = "gather_proj_weights";
inline constexpr const char* kQProj = "q_proj/weight";
inline constexpr const char* kVProj = "v_proj/weight";
inline constexpr const char* kGProj = "g_proj/weight";
inline constexpr const char* kOProj = "o_proj/weight";
}  // namespace ssla

// ---- FARSE-CNN ------------------------------------------------------------
// NOTE: head output layout is `H_grid · W_grid · (num_classes + 5·num_anchors)`
// produced by FARSEYOLOHead (YOLO v1 style); see python/modules/farse_cnn/
// layers/yolo_head.py. §3 work (num_anchors != 1) depends on this.
namespace farse_cnn {
inline constexpr const char* kLstmIh = "shared_lstm_stack/weight_ih";
inline constexpr const char* kLstmHh = "shared_lstm_stack/weight_hh";
inline constexpr const char* kLstmBh = "shared_lstm_stack/bias_hh";
inline constexpr const char* kLstmBi = "shared_lstm_stack/bias_ih";
inline constexpr const char* kConvWeights = "conv_weights";
inline constexpr const char* kHeadClassifier = "head/classifier";
}  // namespace farse_cnn

// ---- DAGr -----------------------------------------------------------------
// DAGr v0.1 will populate this once the spline-conv + pooling key tree is
// stabilized. For now the DAGr loader accepts any-or-no weights.
namespace dagr {
}  // namespace dagr

}  // namespace openeva::weight_keys
