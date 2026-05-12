#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "openeva/output.h"

namespace openeva {

// Tensor is defined in openeva/output.h (one shape + float32 row-major payload,
// with a .numel() helper). Weights loaded from weights.npz use the same type.

/// Schema version that this C++ build accepts for exported checkpoints.
/// Bumped by python/exporter.py when the export format or key layout changes.
/// load_meta() throws if meta.json["schema_version"] is unknown.
inline constexpr int kSupportedSchemaVersion = 1;

/// Parsed contents of meta.json. See cpp/DEVELOPMENT.md §3.
struct ModelMeta {
    int         schema_version = 0; // required ≥ 1; legacy exports (missing key) default to 0
    std::string model;          // e.g. "ssla_s_det", "dagr_n_det"
    std::string method;         // e.g. "ssla", "farse_cnn", "dagr"
    int         height      = 0;
    int         width       = 0;
    int         num_classes = 0;
    int         num_anchors = 0;     // FARSE-CNN: anchors per grid cell (Python default 2)
    // Optional TemporalDropout window overrides for the whole backbone.
    // >0 replaces the hardcoded default (4); 1 means keep every event
    // (used for Python↔C++ parity testing); 0 = leave defaults.
    int         farse_cnn_tdrop_window = 0;
    int         farse_cnn_head_width    = 0;   // YOLOX hidden conv width; 0 = use C_in
    int         ssla_tdrop_window      = 0;
    // DAGr-specific runtime flags. `dagr_keep_temporal_ordering` is 0/1
    // mirroring config/model/dagr_*_det.yaml's boolean; `dagr_pooling_dim_at_output`
    // is the "HxW" string (e.g. "5x7") that drives the 4-pool voxel grid
    // calculation in dagr.py:compute_pooling_at_each_layer. Both default
    // to the values checked into the yamls; a non-DAGr checkpoint leaves
    // them at defaults.
    int         dagr_keep_temporal_ordering = 0;
    std::string dagr_pooling_dim_at_output;
    // AsyNet-specific runtime fields. The Python pipeline runs
    // `F.interpolate(events, size=(asynet_input_h, asynet_input_w))` before
    // the backbone, then the YOLOv1 head emits a (asynet_grid_h, asynet_grid_w)
    // detection grid. Both pairs are dataset-derived (Gen1 vs NCaltech101)
    // and are written by `python/exporter.py`. 0 means "use the dev-time
    // default for this sensor shape".
    int         asynet_input_h = 0;
    int         asynet_input_w = 0;
    int         asynet_grid_h  = 0;
    int         asynet_grid_w  = 0;
    // AEGNN-specific runtime fields. `aegnn_grid_{h,w}` is the YOLOv1
    // detection grid (mirrors `cfg["cell_map_shape"]`). Network input
    // H/W comes from `meta.height/width` directly — AEGNN runs at native
    // sensor resolution (no F.interpolate, vs AsyNet which does). The
    // graph hyperparameters (radius, max_neighbors, beta) drive the
    // online radius_graph builder + time normalization. All written by
    // `python/exporter.py` from the detection yaml. 0 / 0.0 means
    // "missing from meta.json".
    int         aegnn_grid_h         = 0;
    int         aegnn_grid_w         = 0;
    float       aegnn_radius         = 0.0f;
    int         aegnn_max_neighbors  = 0;
    float       aegnn_beta           = 0.0f;
    int         aegnn_head_width     = 0;   // YOLOX hidden conv width (1×1 stem); 0 = use C_in
    int         asynet_head_width    = 0;   // (yolox path) — same semantic as aegnn_head_width
    // pool5 voxel size (pixels). Drives the geometric Cartesian
    // max_value bound; default upstream values mirror
    // `python/modules/aegnn/layers.py:GraphResBackbone.pool5_voxel_size`.
    int         aegnn_pool5_voxel_w  = 0;
    int         aegnn_pool5_voxel_h  = 0;
    // NVS-specific runtime fields. Mirrors AEGNN with extra knobs for
    // NVS's per-block channels + voxel sizes (the GraphWen backbone is
    // configurable in both, vs AEGNN's fixed `[64,128,128,256,128,128,128]`).
    // `nvs_input_h/w` is the network input shape (NVS runs at native
    // sensor resolution by default → equals `meta.height/width`, but kept
    // explicit so a future `input_resize` config can diverge without a
    // schema bump). `nvs_channels` / `nvs_voxel_sizes` are comma-
    // separated 4-element lists (e.g. "64,128,256,512" / "4,8,16,32").
    int         nvs_input_h              = 0;
    int         nvs_input_w              = 0;
    int         nvs_grid_h               = 0;
    int         nvs_grid_w               = 0;
    float       nvs_radius               = 0.0f;
    int         nvs_max_neighbors        = 0;
    float       nvs_beta                 = 0.0f;
    int         nvs_voxel_pool_dim       = 0;
    int         nvs_final_pool_voxel_div = 0;
    int         nvs_final_pool_size      = 0;
    int         nvs_head_width           = 0;   // YOLOX hidden conv width; 0 = use C_in
    std::string nvs_channels;     // CSV of NUM_CONV_BLOCKS ints
    std::string nvs_voxel_sizes;  // CSV of NUM_CONV_BLOCKS ints
    std::size_t total_params = 0;
    std::string dtype;          // always "float32" in v0
    std::string layout;         // "C_order_row_major"
    std::string key_separator;  // "/"
    std::string checkpoint;
};

/// Parse meta.json from `<dir>/meta.json`.
ModelMeta load_meta(const std::string& dir);

/// Load every tensor from `<dir>/weights.npz` into a name → Tensor map.
/// Keys are the raw NPZ entry names with the trailing ".npy" stripped.
std::unordered_map<std::string, Tensor>
load_weights_npz(const std::string& dir);

}  // namespace openeva
