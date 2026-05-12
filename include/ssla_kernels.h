#pragma once

// SSLA-S per-stage forward kernels — deploy/-only port.
//
// Reuses the shared primitive headers from cpp/include/openeva/prim/* and
// the YOLOX head decoder from cpp/include/openeva/heads/async_yolox.h.
// The weight loader is reused from cpp/src/weights_loader.h.
//
// What this header EXPOSES that cpp/methods/ssla/ does NOT:
//   - per-stage forward (stage_forward_ct<...>) callable independently
//   - explicit boundary (tdrop_and_pool) so the caller controls flow
//   - exposed preprocess + head decode + prepopulate
// The numerics are bit-equivalent to cpp/methods/ssla/ssla_detection_yolox.cpp
// at SSLA-S (channels {12, 24, 48, 96}). The only thing this header does
// NOT support is non-S variants — keeping it focused.

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "openeva/event.h"
#include "openeva/output.h"            // must precede heads/* (provides Tensor)
#include "openeva/heads/async_yolox.h"
#include "openeva/prim/elementwise.h"
#include "openeva/prim/layernorm.h"
#include "openeva/prim/linear.h"
#include "openeva/prim/rnn.h"

namespace deploy {

// ============================================================================
// Compile-time SSLA-S constants
// ============================================================================
constexpr int kInDim  = 2;     // (dt_norm, polarity)
constexpr int kC0     = 12;    // SSLA-S stage 0 channels
constexpr int kC1     = 24;
constexpr int kC2     = 48;
constexpr int kC3     = 96;
constexpr int kPatchArea = 9;  // 3×3 sliding-window
constexpr int kNumStages = 4;
constexpr int kLayersPerStage = 2;
constexpr int kNumLayers = kNumStages * kLayersPerStage;  // 8

// Per-layer weights, packed into the same form the cpp runtime uses:
//   qvgIn[p]: (3*OUT, IN) row-major = vstack(Q,V,G) @ scatter[p]
//   goW[p]  : (OUT, OUT) row-major  = gather[p].T @ oW (transpose baked in)
struct LayerWeights {
    int in_dim   = 0;
    int out_dim  = 0;
    int num_pos  = 1;          // 1 (no-PAP) or 9 (PAP)
    std::array<std::vector<float>, kPatchArea> qvgIn{};
    std::array<std::vector<float>, kPatchArea> goW{};
    std::vector<float> input_proj;   // (OUT, IN) if in_dim != out_dim, else empty
    std::vector<float> ln_gamma;     // (OUT,)
    std::vector<float> ln_beta;      // (OUT,)
};

// ============================================================================
// SslaSPipeline — owns weights + per-stage state. Stateful; not thread-safe.
//
// One instance models the full SSLA-S pipeline. For S2 multi-thread, each
// stage thread owns its own STATE for its assigned layers but shares the
// loaded weights through a shared pointer. We keep that distinction
// outside this class — load weights once, copy into per-thread state.
// ============================================================================
class SslaSPipeline {
public:
    SslaSPipeline() = default;

    // Reads <dir>/meta.json and <dir>/weights.npz. Throws on schema mismatch.
    void load(const std::string& dir);

    // Reset hidden states / tdrop counters / last-t buffer (one cold start).
    void reset();

    // Stage 0 input preprocessing — also updates the per-pixel last-t buffer.
    // Returns (dt_norm, polarity) into feat_out[2]. Must be called serially.
    void preprocess(const openeva::Event& e, float feat_out[kInDim]);

    // Per-stage forward.
    //
    // ev_x / ev_y are coordinates AT THIS STAGE'S RESOLUTION (= ev_x0 / 2^stage).
    // The two layers of the stage are run back-to-back; feat_in is at the
    // stage's input dim, feat_out at the stage's output dim.
    //
    // dispatched by stage index 0..3.
    void stage_forward(int stage, int ev_x, int ev_y,
                       const float* feat_in, float* feat_out);

    // Bench-only public accessor for per-layer weight buffers. Used by
    // tests/bench_fused.cpp to compare the generic layer_forward against a
    // hand-fused interior kernel. Not part of the runtime contract.
    const LayerWeights& bench_layer_weights(int layer_idx) const {
        return layers_[layer_idx];
    }
    int bench_Wl(int stage) const { return Ws_[stage]; }
    int bench_Hl(int stage) const { return Hs_[stage]; }
    float* bench_hidden(int layer_idx) { return hidden_[layer_idx].data(); }

    // Single-layer forward. Like stage_forward, but evaluates exactly one
    // SSLA layer specified by global layer index (0..7).
    //   layer_idx | (IN,  OUT) | stage / role
    //   --------- + ---------- + --------------
    //         0   | ( 2, 12)   | stage 0 L0
    //         1   | (12, 12)   | stage 0 L1
    //         2   | (12, 24)   | stage 1 L0
    //         3   | (24, 24)   | stage 1 L1
    //         4   | (24, 48)   | stage 2 L0
    //         5   | (48, 48)   | stage 2 L1
    //         6   | (48, 96)   | stage 3 L0
    //         7   | (96, 96)   | stage 3 L1
    // Used by tests/bench_layer.cpp to time individual layers in isolation.
    // Production runtime path uses stage_forward instead.
    void layer_forward(int layer_idx, int ev_x, int ev_y,
                       const float* feat_in, float* feat_out);

    // -------- Sharded layer-forward primitives (S3) --------
    //
    // SHARDED STAGE 0 ONLY (other stages run single-threaded in S3 since
    // they are not the bottleneck). For shard-aware spatial sharding, the
    // 9 patches around an event's pixel may span multiple shards; each
    // shard processes only its owned patches via shard_layer_forward()
    // and produces a partial feat_out. A central sum thread accumulates
    // partials by seq and calls layer_finalize() to apply residual + LN.
    //
    // owned_patch_mask: bit p is 1 iff shard owns patch at delta-pos p
    // (delta = (1 - dy) * 3 + (1 - dx), p ∈ [0, 8]). For num_pos == 1
    // (kernel_size 1 layers), only bit 0 is meaningful.
    //
    // SCRATCH: callers MUST provide thread-local scratch buffers because
    // S3 runs shard_layer_forward in N parallel threads that would race on
    // any internal scratch. qvg_scratch >= 3 * OUT_DIM_layer, qh_scratch
    // >= OUT_DIM_layer floats. partial_feat_out >= OUT_DIM_layer floats,
    // ZERO-initialised by the caller (sum thread re-zeros per seq).
    void shard_layer_forward(int layer_idx, int ev_x, int ev_y,
                             const float* feat_in,
                             std::uint16_t owned_patch_mask,
                             float* qvg_scratch,
                             float* qh_scratch,
                             float* partial_feat_out);

    // Finalise a layer's feat_out: apply input_proj on feat_in_for_residual
    // (or identity copy if dims match), add residual, layernorm in place.
    // partial_feat_inout is the accumulated sum of partials from all shards.
    // residual_scratch must be sized >= OUT_DIM_layer floats (caller-owned).
    void layer_finalize(int layer_idx,
                        const float* feat_in_for_residual,
                        float* residual_scratch,
                        float* partial_feat_inout);

    // -------- S5: Shared-memory + per-patch lock ---------
    //
    // Run one full layer forward (L0 or L1) for an event, doing ALL 9
    // patches locally on the shared hidden state. Boundary patches are
    // protected by the caller-supplied per-patch spinlock array (one
    // std::atomic_flag per (y * Ws[stage] + x) position; size Hs[stage]
    // * Ws[stage] for the layer's stage). Works for stage 0 only here
    // (layer 0 / 1).
    //
    // SCRATCH: caller provides qvg / qh / residual scratch (per-thread).
    //
    // Semantics: lock acquisition is mutex-only — does NOT enforce
    // producer-seq order across shards. Two events at adjacent shards
    // touching the same boundary patch may apply in lock-acquisition
    // order, which roughly tracks processing-time order. The user has
    // accepted this bounded reordering as long as no events are lost.
    void layer_forward_locked(int layer_idx, int ev_x, int ev_y,
                              const float* feat_in,
                              std::atomic_flag* patch_locks,
                              float* qvg_scratch,
                              float* qh_scratch,
                              float* residual_scratch,
                              float* feat_out);

    // Stage forward (all 4 stages) under per-layer locks. patch_locks_a /
    // patch_locks_b are the lock arrays for the stage's two layers (each
    // sized to Hs[stage] * Ws[stage]). l0_out_scratch is the per-event
    // intermediate between the two layers.
    //
    // Caller provides scratch and lock arrays so multiple shards can run
    // this concurrently without sharing any internal scratch.
    void stage_forward_locked(int stage, int ev_x, int ev_y,
                              const float* feat_in,
                              std::atomic_flag* patch_locks_a,
                              std::atomic_flag* patch_locks_b,
                              float* qvg_scratch,
                              float* qh_scratch,
                              float* residual_scratch,
                              float* l0_out_scratch,
                              float* feat_out);

    // Lock-grid size for a given stage (= Hs[stage] * Ws[stage]). Each
    // stage's two layers can share the same lock array OR have separate
    // arrays — within a single event the two layers don't run in
    // parallel so contention is OK either way.
    int stage_lock_grid_size(int stage) const {
        return (stage >= 0 && stage < kNumStages) ? Hs_[stage] * Ws_[stage] : 0;
    }
    int stage0_lock_grid_size() const { return stage_lock_grid_size(0); }

    // Atomic tdrop+pool. Uses fetch_add on a uint8_t atomic counter
    // array (sized stage_lock_grid_size(stage+1)) instead of the
    // internal non-atomic counter. Returns true if event passes the
    // tdrop window. ev_x and ev_y are halved on pass.
    bool tdrop_and_pool_atomic(int stage, int& ev_x, int& ev_y,
                                std::atomic<std::uint8_t>* atomic_counters);

    // Locked head decode for one cell. Multiple shards may try to write
    // the same cell concurrently; per-cell spinlock serializes. Like
    // head_decode_cell, but takes the lock array + per-thread scratch.
    //
    // head_cell_locks: array of size num_anchors_total, one lock per
    // (level, gy, gx) cell flattened in the same order as predictions().
    // cls_scratch / reg_scratch / cls_logits_scratch are per-thread
    // scratch (sized to head_[*].C_hid / num_classes respectively).
    void head_decode_cell_locked(int stage, int gx, int gy, const float* feat,
                                 std::atomic_flag* head_cell_locks,
                                 std::vector<float>& cls_scratch,
                                 std::vector<float>& reg_scratch,
                                 std::vector<float>& cls_logits_scratch);

    // Tdrop + pool boundary (between stages). Updates the post-stage tdrop
    // counter for (ev_x/2, ev_y/2). Returns true if event passes, false if
    // dropped. On pass, ev_x and ev_y are updated in place to next-stage
    // resolution (halved).
    //
    // stage in [0, 2]: there is no pool/dropout after stage 3.
    bool tdrop_and_pool(int stage, int& ev_x, int& ev_y);

    // Head decode for one cell at the given stage's resolution. Writes one
    // row of last_predictions_. Stage must be in head_stage_idx_ (i.e. an
    // emitting stage); otherwise no-op.
    void head_decode_cell(int stage, int gx, int gy, const float* feat);

    // Pre-populate last_predictions_ with head(0_feat) for every cell of
    // every emitting level. Mirrors ssla_detection_yolox.cpp:prepopulate_head_zero.
    // Call once after reset() (we call it inside reset()).
    void prepopulate_head_zero();

    // Const view of the current decoded predictions tensor (rows × cols).
    const std::vector<float>& predictions() const { return last_predictions_; }
    const openeva::ModelOutput& last_output() const { return last_output_; }

    // Manually copy last_predictions_ into last_output_ (the variant
    // returned by last_output()). For S5/S6 where shards write to
    // last_predictions_ directly without calling head_decode_cell, this
    // is needed before each oracle->record() call.
    void publish_output_now() { publish_output(); }

    // Geometry
    int H() const { return H_; }
    int W() const { return W_; }
    int num_classes() const { return num_classes_; }
    int num_anchors_total() const { return total_anchors_; }
    int cols() const { return 5 + num_classes_; }
    int tdrop_window() const { return tdrop_window_; }

private:
    // ---- config / weights ----------------------------------------------
    int H_ = 0, W_ = 0, num_classes_ = 0;
    int tdrop_window_ = 4;
    std::array<int, kNumStages> Hs_{}, Ws_{};
    std::array<LayerWeights, kNumLayers> layers_{};

    // Head levels (only stage 3 emits by default; up to 4 levels supported).
    static constexpr int kMaxHeadLevels = 4;
    std::array<openeva::heads::YoloxLevel, kMaxHeadLevels> head_{};
    int num_head_levels_ = 0;
    std::array<int, kMaxHeadLevels> head_stage_idx_{};
    int total_anchors_ = 0;

    // ---- persistent state (mutated per event) ---------------------------
    std::array<std::vector<float>, kNumLayers>      hidden_{};
    std::array<std::vector<std::uint8_t>, 3>       tdrop_counter_{};
    std::vector<double>                             last_t_us_;

    // ---- scratch — one slot per stage so the threaded pipeline (S2/S3/S4)
    // can run all 4 stage_forward() concurrently without sharing scratch.
    // Single-thread drivers just use scratch_*_[stage] of whichever stage
    // they're calling.
    std::array<std::vector<float>, kNumStages> scratch_residual_{};
    std::array<std::vector<float>, kNumStages> scratch_qvg_{};
    std::array<std::vector<float>, kNumStages> scratch_qh_{};
    std::vector<float> zero_feat_cell_;
    // Head scratch reused per call (decode_yolox_cell_ct expects mutable
    // vectors; sized to max head hidden in load()).
    std::vector<float> scratch_cls_f_;
    std::vector<float> scratch_reg_f_;
    std::vector<float> scratch_cls_logits_;
    // Per-stage final feature (one cell's worth, sized max channels).
    std::array<std::vector<float>, kNumStages> stage_out_{};

    // ---- output ---------------------------------------------------------
    std::vector<float>     last_predictions_;       // (anchors_total * cols,)
    openeva::DetectionOutput last_det_{};
    openeva::ModelOutput   last_output_{std::monostate{}};

    // ---- helpers --------------------------------------------------------
    template <int IN_DIM, int OUT_DIM>
    void layer_forward_ct(int layer_idx, int ev_x, int ev_y,
                          const float* feat_in, float* feat_out);

    void publish_output();   // write last_predictions_ → last_det_ → last_output_
};

}  // namespace deploy
