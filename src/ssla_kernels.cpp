// ssla_kernels.cpp — SSLA-S per-stage forward implementation.
//
// Mirrors cpp/methods/ssla/ssla_detection_yolox.cpp's Impl::ssla_layer_forward_ct
// and load_layer / load_head_level / prepopulate_head_zero, but exposed as
// independent per-stage functions so a multi-thread driver can run them
// in a pipeline.

#include "ssla_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "openeva/output.h"            // must precede heads/*
#include "openeva/heads/async_yolox.h"
#include "openeva/prim/elementwise.h"
#include "openeva/prim/layernorm.h"
#include "openeva/prim/linear.h"
#include "openeva/prim/matmul.h"
#include "openeva/prim/rnn.h"
#include "src/weights_loader.h"

// NEON specializations for matvec_ct / matvec_accum_ct at IN=12. Must come
// after openeva/prim/linear.h (primary templates must be declared first) and
// before any layer_forward_ct<…> instantiation in this TU. The instantiations
// happen below in stage_forward() — keeping this include here covers all of
// them. See include/ssla_neon_linear.h for rationale and microbench data.
#include "ssla_neon_linear.h"

// NEON specializations for lru_step<DIM> at DIM ∈ {12, 24, 48, 96}. Same
// rule as ssla_neon_linear: include AFTER vendor/openeva/prim/rnn.h. See
// include/ssla_neon_lru.h for the tanh-Padé sigmoid + drift verification.
#include "ssla_neon_lru.h"

// Fused interior-only kernels for s0 L1, s1 L0, s1 L1 (Phase 3). Used by
// stage_forward below when ev_x/ev_y are interior; falls back to the
// generic layer_forward_ct on boundary pixels.
#include "ssla_fused_layers.h"

namespace deploy {

namespace {

const openeva::Tensor& must_get(
    const std::unordered_map<std::string, openeva::Tensor>& m,
    const std::string& k) {
    auto it = m.find(k);
    if (it == m.end()) throw std::runtime_error("ssla: missing tensor " + k);
    return it->second;
}
const openeva::Tensor* maybe_get(
    const std::unordered_map<std::string, openeva::Tensor>& m,
    const std::string& k) {
    auto it = m.find(k);
    return it == m.end() ? nullptr : &it->second;
}

// vstack(Q, V, G) — 3 (out, in) blocks → one (3*out, in) row-major buffer.
std::vector<float> pack_qvg(const std::vector<float>& Q,
                            const std::vector<float>& V,
                            const std::vector<float>& G,
                            int out_dim, int in_dim) {
    std::vector<float> P(static_cast<std::size_t>(3) * out_dim * in_dim);
    const std::size_t block = static_cast<std::size_t>(out_dim) * in_dim;
    std::memcpy(P.data(),                 Q.data(), block * sizeof(float));
    std::memcpy(P.data() + block,         V.data(), block * sizeof(float));
    std::memcpy(P.data() + 2 * block,     G.data(), block * sizeof(float));
    return P;
}

void load_layer(const std::unordered_map<std::string, openeva::Tensor>& W,
                const std::string& prefix,
                LayerWeights& L, int in_dim, int out_dim) {
    L.in_dim  = in_dim;
    L.out_dim = out_dim;

    const auto& Win  = must_get(W, prefix + "/W_in_cat");
    const auto& Wout = must_get(W, prefix + "/W_out_cat");
    if (Win.shape.size() != 2 || Wout.shape.size() != 2) {
        throw std::runtime_error("ssla: " + prefix + " W_*_cat rank");
    }
    const int Win_cols  = static_cast<int>(Win.shape[1]);
    const int Wout_rows = static_cast<int>(Wout.shape[0]);
    if (Win_cols % out_dim != 0 || Wout_rows % out_dim != 0) {
        throw std::runtime_error(
            "ssla: " + prefix + " W_*_cat shape not multiple of out_dim");
    }
    const int n_pos_in  = Win_cols  / out_dim;
    const int n_pos_out = Wout_rows / out_dim;
    const int num_pos   = std::max(n_pos_in, n_pos_out);
    L.num_pos = num_pos;

    const auto& qW = must_get(W, prefix + "/q_proj/weight");
    const auto& vW = must_get(W, prefix + "/v_proj/weight");
    const auto& gW = must_get(W, prefix + "/g_proj/weight");
    const auto& oW = must_get(W, prefix + "/o_proj/weight");
    const std::vector<float> qV(qW.data.begin(), qW.data.end());
    const std::vector<float> vV(vW.data.begin(), vW.data.end());
    const std::vector<float> gV(gW.data.begin(), gW.data.end());
    const std::vector<float> oV(oW.data.begin(), oW.data.end());

    std::vector<float> scatter(static_cast<std::size_t>(out_dim) * in_dim);
    std::vector<float> gather (static_cast<std::size_t>(out_dim) * out_dim);
    for (int p = 0; p < num_pos; ++p) {
        const int pin  = (n_pos_in  == 1) ? 0 : p;
        const int pout = (n_pos_out == 1) ? 0 : p;
        // scatter[d, c] = W_in_cat[c, pin*D + d]
        for (int d = 0; d < out_dim; ++d) {
            for (int c = 0; c < in_dim; ++c) {
                scatter[static_cast<std::size_t>(d) * in_dim + c] =
                    Win.data[static_cast<std::ptrdiff_t>(c) * Win_cols
                             + pin * out_dim + d];
            }
        }
        // gather[i, j] = W_out_cat[pout*D + j, i]  (transpose baked in)
        for (int i = 0; i < out_dim; ++i) {
            for (int j = 0; j < out_dim; ++j) {
                gather[static_cast<std::size_t>(i) * out_dim + j] =
                    Wout.data[static_cast<std::ptrdiff_t>(pout * out_dim + j)
                              * out_dim + i];
            }
        }

        auto q_in = openeva::prim::mm(qV, out_dim, out_dim, scatter, out_dim, in_dim);
        auto v_in = openeva::prim::mm(vV, out_dim, out_dim, scatter, out_dim, in_dim);
        auto g_in = openeva::prim::mm(gV, out_dim, out_dim, scatter, out_dim, in_dim);
        L.qvgIn[p] = pack_qvg(q_in, v_in, g_in, out_dim, in_dim);
        L.goW[p]   = openeva::prim::mm(gather, out_dim, out_dim, oV, out_dim, out_dim);
        // Pre-compute goW transpose for tile-streaming fused kernels:
        //   goW_T[p][i*OUT + j] = goW[p][j*OUT + i]
        // One-time cost (out_dim² floats per patch position) traded for
        // contiguous column access in the per-tile inner loop.
        L.goW_T[p].assign(static_cast<std::size_t>(out_dim) * out_dim, 0.0f);
        for (int j = 0; j < out_dim; ++j) {
            for (int i = 0; i < out_dim; ++i) {
                L.goW_T[p][static_cast<std::size_t>(i) * out_dim + j]
                    = L.goW[p][static_cast<std::size_t>(j) * out_dim + i];
            }
        }
    }

    if (const openeva::Tensor* ip = maybe_get(W, prefix + "/input_proj/weight")) {
        L.input_proj.assign(ip->data.begin(), ip->data.end());
    } else {
        L.input_proj.clear();
    }
    const auto& gn = must_get(W, prefix + "/norm/weight");
    const auto& bb = must_get(W, prefix + "/norm/bias");
    L.ln_gamma.assign(gn.data.begin(), gn.data.end());
    L.ln_beta.assign(bb.data.begin(),  bb.data.end());
}

}  // namespace

// ============================================================================
// load() / reset() / preprocess()
// ============================================================================

void SslaSPipeline::load(const std::string& dir) {
    const openeva::ModelMeta meta = openeva::load_meta(dir);
    if (meta.model != "ssla_s_yolox_det") {
        throw std::runtime_error(
            "deploy/ssla_kernels: only ssla_s_yolox_det supported, got " + meta.model);
    }
    H_ = meta.height > 0 ? meta.height : 240;
    W_ = meta.width  > 0 ? meta.width  : 304;
    num_classes_ = meta.num_classes > 0 ? meta.num_classes : 2;
    if (meta.ssla_tdrop_window > 0) tdrop_window_ = meta.ssla_tdrop_window;

    Hs_[0] = H_;     Ws_[0] = W_;
    Hs_[1] = H_ / 2; Ws_[1] = W_ / 2;
    Hs_[2] = H_ / 4; Ws_[2] = W_ / 4;
    Hs_[3] = H_ / 8; Ws_[3] = W_ / 8;

    auto W = openeva::load_weights_npz(dir);

    // Discover SSLA layer indices in the export — Python's stages.layers
    // schema means SSLA blocks may not sit at indices 0..7.
    std::vector<int> ssla_indices;
    for (const auto& [k, _] : W) {
        const std::string prefix = "layers/";
        if (k.size() <= prefix.size() ||
            k.compare(0, prefix.size(), prefix) != 0) continue;
        const std::string rest = k.substr(prefix.size());
        const auto slash = rest.find('/');
        if (slash == std::string::npos) continue;
        if (rest.substr(slash + 1) != "W_in_cat") continue;
        try { ssla_indices.push_back(std::stoi(rest.substr(0, slash))); }
        catch (...) {}
    }
    std::sort(ssla_indices.begin(), ssla_indices.end());
    if (ssla_indices.size() != static_cast<std::size_t>(kNumLayers)) {
        throw std::runtime_error(
            "deploy/ssla_kernels: expected " + std::to_string(kNumLayers) +
            " ssla layers, found " + std::to_string(ssla_indices.size()));
    }

    const std::array<int, kNumStages> chans = {kC0, kC1, kC2, kC3};
    int prev = kInDim;
    for (int l = 0; l < kNumLayers; ++l) {
        const int s = l / kLayersPerStage;
        const int c = chans[s];
        const int in_d  = (l % kLayersPerStage == 0) ? prev : c;
        const int out_d = c;
        if (l % kLayersPerStage == kLayersPerStage - 1) prev = c;
        load_layer(W, "layers/" + std::to_string(ssla_indices[l]),
                   layers_[l], in_d, out_d);
    }

    // Detect head levels
    num_head_levels_ = 0;
    for (int k = 0; k < kMaxHeadLevels; ++k) {
        if (maybe_get(W, "head/cls_convs/" + std::to_string(k) + "/0/weight")) {
            ++num_head_levels_;
        } else break;
    }
    if (num_head_levels_ == 0) {
        throw std::runtime_error("deploy/ssla_kernels: no head levels found");
    }
    for (int k = 0; k < num_head_levels_; ++k) {
        head_stage_idx_[k] = kNumStages - num_head_levels_ + k;
    }
    const int c_last = chans[kNumStages - 1];  // head hidden width = last stage chs
    for (int k = 0; k < num_head_levels_; ++k) {
        const int si = head_stage_idx_[k];
        openeva::heads::load_yolox_level(
            W, head_[k], k,
            /*C_in=*/chans[si], /*C_hid=*/c_last, num_classes_,
            /*stride_h=*/H_ / Hs_[si], /*stride_w=*/W_ / Ws_[si],
            /*H_grid=*/Hs_[si], /*W_grid=*/Ws_[si],
            /*err_prefix=*/"deploy/ssla_kernels");
    }
    total_anchors_ = 0;
    for (int k = 0; k < num_head_levels_; ++k) {
        total_anchors_ += head_[k].H_grid * head_[k].W_grid;
    }

    // Allocate state + scratch
    int max_c = 0;
    for (int c : chans) max_c = std::max(max_c, c);
    for (int s = 0; s < kNumStages; ++s) {
        scratch_residual_[s].assign(max_c, 0.0f);
        scratch_qvg_[s].assign(3 * max_c, 0.0f);
        scratch_qh_[s].assign(max_c, 0.0f);
    }
    zero_feat_cell_.assign(max_c, 0.0f);
    for (int s = 0; s < kNumStages; ++s) stage_out_[s].assign(max_c, 0.0f);
    scratch_cls_f_.assign(c_last, 0.0f);
    scratch_reg_f_.assign(c_last, 0.0f);
    scratch_cls_logits_.assign(num_classes_, 0.0f);

    for (int l = 0; l < kNumLayers; ++l) {
        const int stage = l / kLayersPerStage;
        const int out   = chans[stage];
        hidden_[l].assign(static_cast<std::size_t>(Hs_[stage]) * Ws_[stage] * out, 0.0f);
    }
    for (int s = 0; s < 3; ++s) {
        const std::size_t n_pix = static_cast<std::size_t>(Hs_[s + 1]) * Ws_[s + 1];
        tdrop_counter_[s].assign(n_pix, 0u);
    }
    last_t_us_.assign(static_cast<std::size_t>(H_) * W_, -1.0);

    last_predictions_.assign(
        static_cast<std::size_t>(total_anchors_) * (5 + num_classes_), 0.0f);
    last_det_.predictions.shape = {
        static_cast<std::size_t>(total_anchors_),
        static_cast<std::size_t>(5 + num_classes_)};
    last_det_.predictions.data.assign(last_predictions_.size(), 0.0f);
    last_output_ = std::monostate{};

    prepopulate_head_zero();
}

void SslaSPipeline::reset() {
    for (auto& h : hidden_) std::fill(h.begin(), h.end(), 0.0f);
    for (auto& s : stage_out_) std::fill(s.begin(), s.end(), 0.0f);
    for (int s = 0; s < 3; ++s) {
        std::fill(tdrop_counter_[s].begin(), tdrop_counter_[s].end(), 0u);
    }
    std::fill(last_t_us_.begin(), last_t_us_.end(), -1.0);
    std::fill(last_predictions_.begin(), last_predictions_.end(), 0.0f);
    last_output_ = std::monostate{};
    prepopulate_head_zero();
}

void SslaSPipeline::preprocess(const openeva::Event& e, float feat_out[kInDim]) {
    const int ex = static_cast<int>(e.x);
    const int ey = static_cast<int>(e.y);
    if (ex < 0 || ex >= W_ || ey < 0 || ey >= H_) {
        feat_out[0] = 0.0f; feat_out[1] = 0.0f;
        return;
    }
    const int pidx = ey * W_ + ex;
    const double prev = last_t_us_[pidx];
    float dt_norm;
    if (prev < 0.0) {
        dt_norm = 1.0f;
    } else {
        double v = (double(e.t) - prev) / 1e5;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        dt_norm = static_cast<float>(v);
    }
    last_t_us_[pidx] = double(e.t);
    feat_out[0] = dt_norm;
    feat_out[1] = e.p;   // SSLA uses {0,1} polarity, matches Python
}

// ============================================================================
// stage_forward (template-dispatched on the stage's IN/OUT dims)
// ============================================================================

template <int IN_DIM, int OUT_DIM>
void SslaSPipeline::layer_forward_ct(int layer_idx, int ev_x, int ev_y,
                                     const float* feat_in, float* feat_out) {
    LayerWeights& L = layers_[layer_idx];
    const int stage = layer_idx / kLayersPerStage;
    const int Hl = Hs_[stage];
    const int Wl = Ws_[stage];
    float* H_all = hidden_[layer_idx].data();

    float* residual = scratch_residual_[stage].data();
    float* qvg      = scratch_qvg_[stage].data();
    float* qh       = scratch_qh_[stage].data();

    if (!L.input_proj.empty()) {
        openeva::prim::matvec_ct<IN_DIM, OUT_DIM>(
            feat_in, L.input_proj.data(), nullptr, residual);
    } else {
        std::memcpy(residual, feat_in, sizeof(float) * OUT_DIM);
    }

    std::memset(feat_out, 0, sizeof(float) * OUT_DIM);

    const int num_pos = L.num_pos;
    const int base    = ev_y * Wl + ev_x;

    auto process_patch = [&](int patch_idx, int pos) {
        openeva::prim::matvec_ct<IN_DIM, 3 * OUT_DIM>(
            feat_in, L.qvgIn[pos].data(), nullptr, qvg);
        float* h_ptr = H_all +
                       static_cast<std::ptrdiff_t>(patch_idx) * OUT_DIM;
        const float* q = qvg;
        const float* v = qvg + OUT_DIM;
        const float* g = qvg + 2 * OUT_DIM;
        openeva::prim::lru_step<OUT_DIM>(g, v, q, h_ptr, qh);
        openeva::prim::matvec_accum_ct<OUT_DIM, OUT_DIM>(
            qh, L.goW[pos].data(), feat_out);
    };

    if (num_pos == 1) {
        process_patch(base, 0);
    } else {
        const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                           && (ev_y > 0) && (ev_y + 1 < Hl);
        if (interior) {
            process_patch(base - Wl - 1, 8);
            process_patch(base - Wl,     7);
            process_patch(base - Wl + 1, 6);
            process_patch(base - 1,      5);
            process_patch(base,          4);
            process_patch(base + 1,      3);
            process_patch(base + Wl - 1, 2);
            process_patch(base + Wl,     1);
            process_patch(base + Wl + 1, 0);
        } else {
            for (int dy = -1; dy <= 1; ++dy) {
                const int py = ev_y + dy;
                if (py < 0 || py >= Hl) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int px = ev_x + dx;
                    if (px < 0 || px >= Wl) continue;
                    const int delta = (1 - dy) * 3 + (1 - dx);
                    process_patch(py * Wl + px, delta);
                }
            }
        }
    }

    openeva::prim::add_inplace(OUT_DIM, feat_out, residual);
    openeva::prim::layernorm_ct<OUT_DIM>(
        feat_out, L.ln_gamma.data(), L.ln_beta.data());
}

// ============================================================================
// shard_layer_forward / layer_finalize  (S3 sharded primitives)
// ============================================================================

namespace {
// Compute the 9 absolute patch indices around (ev_x, ev_y) on a (Hl, Wl)
// grid, with -1 for out-of-bounds. delta indexing follows
// (1 - dy) * 3 + (1 - dx) — same as the existing layer_forward_ct.
inline void compute_patch_indices(int ev_x, int ev_y, int Hl, int Wl,
                                   int patches[9]) {
    const int base = ev_y * Wl + ev_x;
    const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                       && (ev_y > 0) && (ev_y + 1 < Hl);
    if (interior) {
        patches[8] = base - Wl - 1;
        patches[7] = base - Wl;
        patches[6] = base - Wl + 1;
        patches[5] = base - 1;
        patches[4] = base;
        patches[3] = base + 1;
        patches[2] = base + Wl - 1;
        patches[1] = base + Wl;
        patches[0] = base + Wl + 1;
    } else {
        for (int p = 0; p < 9; ++p) patches[p] = -1;
        for (int dy = -1; dy <= 1; ++dy) {
            const int py = ev_y + dy;
            if (py < 0 || py >= Hl) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int px = ev_x + dx;
                if (px < 0 || px >= Wl) continue;
                const int delta = (1 - dy) * 3 + (1 - dx);
                patches[delta] = py * Wl + px;
            }
        }
    }
}

template <int IN_DIM, int OUT_DIM>
inline void shard_layer_forward_ct(
    LayerWeights& L,
    int ev_x, int ev_y, int Hl, int Wl,
    const float* feat_in,
    std::uint16_t owned_patch_mask,
    float* H_all, float* qvg, float* qh,
    float* partial_feat_out) {
    if (L.num_pos == 1) {
        // SSLA num_pos==1: only patch at delta=0 (center, here patch_idx=base)
        if (owned_patch_mask & 1u) {
            openeva::prim::matvec_ct<IN_DIM, 3 * OUT_DIM>(
                feat_in, L.qvgIn[0].data(), nullptr, qvg);
            float* h_ptr = H_all + static_cast<std::ptrdiff_t>(ev_y * Wl + ev_x) * OUT_DIM;
            const float* q = qvg;
            const float* v = qvg + OUT_DIM;
            const float* g = qvg + 2 * OUT_DIM;
            openeva::prim::lru_step<OUT_DIM>(g, v, q, h_ptr, qh);
            openeva::prim::matvec_accum_ct<OUT_DIM, OUT_DIM>(
                qh, L.goW[0].data(), partial_feat_out);
        }
        return;
    }
    int patches[9];
    compute_patch_indices(ev_x, ev_y, Hl, Wl, patches);
    for (int p = 0; p < 9; ++p) {
        if (((owned_patch_mask >> p) & 1u) == 0) continue;
        if (patches[p] < 0) continue;
        openeva::prim::matvec_ct<IN_DIM, 3 * OUT_DIM>(
            feat_in, L.qvgIn[p].data(), nullptr, qvg);
        float* h_ptr = H_all + static_cast<std::ptrdiff_t>(patches[p]) * OUT_DIM;
        const float* q = qvg;
        const float* v = qvg + OUT_DIM;
        const float* g = qvg + 2 * OUT_DIM;
        openeva::prim::lru_step<OUT_DIM>(g, v, q, h_ptr, qh);
        openeva::prim::matvec_accum_ct<OUT_DIM, OUT_DIM>(
            qh, L.goW[p].data(), partial_feat_out);
    }
}
}  // namespace

void SslaSPipeline::shard_layer_forward(int layer_idx, int ev_x, int ev_y,
                                        const float* feat_in,
                                        std::uint16_t owned_patch_mask,
                                        float* qvg_scratch,
                                        float* qh_scratch,
                                        float* partial_feat_out) {
    LayerWeights& L = layers_[layer_idx];
    const int stage = layer_idx / kLayersPerStage;
    const int Hl = Hs_[stage];
    const int Wl = Ws_[stage];
    float* H_all = hidden_[layer_idx].data();
    switch (layer_idx) {
        case 0:
            shard_layer_forward_ct<kInDim, kC0>(
                L, ev_x, ev_y, Hl, Wl, feat_in, owned_patch_mask,
                H_all, qvg_scratch, qh_scratch, partial_feat_out);
            break;
        case 1:
            shard_layer_forward_ct<kC0, kC0>(
                L, ev_x, ev_y, Hl, Wl, feat_in, owned_patch_mask,
                H_all, qvg_scratch, qh_scratch, partial_feat_out);
            break;
        default:
            throw std::runtime_error(
                "shard_layer_forward only supports layers 0/1 (stage 0)");
    }
}

void SslaSPipeline::layer_finalize(int layer_idx,
                                    const float* feat_in_for_residual,
                                    float* residual_scratch,
                                    float* partial_feat_inout) {
    LayerWeights& L = layers_[layer_idx];
    float* residual = residual_scratch;
    auto run = [&](auto IN_T, auto OUT_T) {
        constexpr int IN  = decltype(IN_T)::value;
        constexpr int OUT = decltype(OUT_T)::value;
        if (!L.input_proj.empty()) {
            openeva::prim::matvec_ct<IN, OUT>(
                feat_in_for_residual, L.input_proj.data(),
                nullptr, residual);
        } else {
            std::memcpy(residual, feat_in_for_residual, sizeof(float) * OUT);
        }
        openeva::prim::add_inplace(OUT, partial_feat_inout, residual);
        openeva::prim::layernorm_ct<OUT>(
            partial_feat_inout, L.ln_gamma.data(), L.ln_beta.data());
    };
    using IC0 = std::integral_constant<int, kC0>;
    using IIN = std::integral_constant<int, kInDim>;
    switch (layer_idx) {
        case 0: run(IIN{}, IC0{}); break;
        case 1: run(IC0{}, IC0{}); break;
        default: throw std::runtime_error(
            "layer_finalize only supports layers 0/1 (stage 0)");
    }
}

// ============================================================================
// S5: Shared-memory + per-patch-lock layer forward
// ============================================================================

namespace {

template <int IN_DIM, int OUT_DIM>
inline void layer_forward_locked_ct(
    LayerWeights& L,
    int ev_x, int ev_y, int Hl, int Wl,
    const float* feat_in,
    std::atomic_flag* patch_locks,
    float* H_all, float* qvg, float* qh,
    float* residual_scratch,
    float* feat_out) {
    // Residual (no shared state)
    if (!L.input_proj.empty()) {
        openeva::prim::matvec_ct<IN_DIM, OUT_DIM>(
            feat_in, L.input_proj.data(), nullptr, residual_scratch);
    } else {
        std::memcpy(residual_scratch, feat_in, sizeof(float) * OUT_DIM);
    }
    std::memset(feat_out, 0, sizeof(float) * OUT_DIM);

    auto process_patch = [&](int patch_idx, int pos) {
        // Local compute (no shared state)
        openeva::prim::matvec_ct<IN_DIM, 3 * OUT_DIM>(
            feat_in, L.qvgIn[pos].data(), nullptr, qvg);
        const float* q = qvg;
        const float* v = qvg + OUT_DIM;
        const float* g = qvg + 2 * OUT_DIM;
        // ACQUIRE lock for this patch — guards the RMW on H_all[patch_idx].
        // Uncontended: ~5 cycles. Boundary patches between shards may
        // contend briefly.
        auto& lk = patch_locks[patch_idx];
        while (lk.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
        float* h_ptr = H_all + static_cast<std::ptrdiff_t>(patch_idx) * OUT_DIM;
        openeva::prim::lru_step<OUT_DIM>(g, v, q, h_ptr, qh);
        lk.clear(std::memory_order_release);
        // Local accumulate (no shared state — qh is per-thread scratch)
        openeva::prim::matvec_accum_ct<OUT_DIM, OUT_DIM>(
            qh, L.goW[pos].data(), feat_out);
    };

    if (L.num_pos == 1) {
        process_patch(ev_y * Wl + ev_x, 0);
    } else {
        const int base = ev_y * Wl + ev_x;
        const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                           && (ev_y > 0) && (ev_y + 1 < Hl);
        if (interior) {
            process_patch(base - Wl - 1, 8);
            process_patch(base - Wl,     7);
            process_patch(base - Wl + 1, 6);
            process_patch(base - 1,      5);
            process_patch(base,          4);
            process_patch(base + 1,      3);
            process_patch(base + Wl - 1, 2);
            process_patch(base + Wl,     1);
            process_patch(base + Wl + 1, 0);
        } else {
            for (int dy = -1; dy <= 1; ++dy) {
                const int py = ev_y + dy;
                if (py < 0 || py >= Hl) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int px = ev_x + dx;
                    if (px < 0 || px >= Wl) continue;
                    const int delta = (1 - dy) * 3 + (1 - dx);
                    process_patch(py * Wl + px, delta);
                }
            }
        }
    }

    openeva::prim::add_inplace(OUT_DIM, feat_out, residual_scratch);
    openeva::prim::layernorm_ct<OUT_DIM>(
        feat_out, L.ln_gamma.data(), L.ln_beta.data());
}

}  // namespace

void SslaSPipeline::layer_forward_locked(int layer_idx, int ev_x, int ev_y,
                                         const float* feat_in,
                                         std::atomic_flag* patch_locks,
                                         float* qvg_scratch,
                                         float* qh_scratch,
                                         float* residual_scratch,
                                         float* feat_out) {
    LayerWeights& L = layers_[layer_idx];
    const int stage = layer_idx / kLayersPerStage;
    const int Hl = Hs_[stage];
    const int Wl = Ws_[stage];
    float* H_all = hidden_[layer_idx].data();
    // Compile-time dim dispatch for SSLA-S (channels 12, 24, 48, 96).
    switch (layer_idx) {
        case 0: layer_forward_locked_ct<kInDim, kC0>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 1: layer_forward_locked_ct<kC0, kC0>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 2: layer_forward_locked_ct<kC0, kC1>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 3: layer_forward_locked_ct<kC1, kC1>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 4: layer_forward_locked_ct<kC1, kC2>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 5: layer_forward_locked_ct<kC2, kC2>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 6: layer_forward_locked_ct<kC2, kC3>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        case 7: layer_forward_locked_ct<kC3, kC3>(
            L, ev_x, ev_y, Hl, Wl, feat_in, patch_locks,
            H_all, qvg_scratch, qh_scratch, residual_scratch, feat_out); break;
        default:
            throw std::runtime_error("layer_forward_locked: bad layer idx");
    }
}

void SslaSPipeline::stage_forward_locked(int stage, int ev_x, int ev_y,
                                          const float* feat_in,
                                          std::atomic_flag* patch_locks_a,
                                          std::atomic_flag* patch_locks_b,
                                          float* qvg_scratch,
                                          float* qh_scratch,
                                          float* residual_scratch,
                                          float* l0_out_scratch,
                                          float* feat_out) {
    if (stage < 0 || stage >= kNumStages) {
        throw std::runtime_error("stage_forward_locked: bad stage idx");
    }
    const int la = stage * kLayersPerStage;       // first layer of stage
    const int lb = la + 1;                         // second layer of stage
    layer_forward_locked(la, ev_x, ev_y, feat_in, patch_locks_a,
                         qvg_scratch, qh_scratch, residual_scratch,
                         l0_out_scratch);
    layer_forward_locked(lb, ev_x, ev_y, l0_out_scratch, patch_locks_b,
                         qvg_scratch, qh_scratch, residual_scratch,
                         feat_out);
}

bool SslaSPipeline::tdrop_and_pool_atomic(int stage, int& ev_x, int& ev_y,
                                           std::atomic<std::uint8_t>* atomic_counters) {
    if (stage < 0 || stage > 2) return true;
    ev_x /= 2;
    ev_y /= 2;
    const int Wp  = Ws_[stage + 1];
    const int idx = ev_y * Wp + ev_x;
    const std::uint8_t pre = atomic_counters[idx].fetch_add(
        1, std::memory_order_relaxed);
    if (tdrop_window_ <= 1) return true;
    return (pre % tdrop_window_) == 0u;
}

void SslaSPipeline::head_decode_cell_locked(int stage, int gx, int gy,
                                             const float* feat,
                                             std::atomic_flag* head_cell_locks,
                                             std::vector<float>& cls_scratch,
                                             std::vector<float>& reg_scratch,
                                             std::vector<float>& cls_logits_scratch) {
    int level = -1;
    for (int k = 0; k < num_head_levels_; ++k) {
        if (head_stage_idx_[k] == stage) { level = k; break; }
    }
    if (level < 0) return;
    const int W_gr = head_[level].W_grid;
    const int H_gr = head_[level].H_grid;
    if (gx < 0 || gx >= W_gr || gy < 0 || gy >= H_gr) return;

    int row_offset = 0;
    for (int k = 0; k < level; ++k) row_offset += head_[k].H_grid * head_[k].W_grid;
    const int cell_idx = row_offset + gy * W_gr + gx;
    const int row_stride = 5 + num_classes_;

    auto& lk = head_cell_locks[cell_idx];
    while (lk.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#endif
    }

    float* out_row = last_predictions_.data() +
                     static_cast<std::ptrdiff_t>(cell_idx) * row_stride;
    switch (stage) {
        case 0: openeva::heads::decode_yolox_cell_ct<kC0, kC3>(
            head_[level], gx, gy, feat, out_row,
            cls_scratch, reg_scratch, cls_logits_scratch); break;
        case 1: openeva::heads::decode_yolox_cell_ct<kC1, kC3>(
            head_[level], gx, gy, feat, out_row,
            cls_scratch, reg_scratch, cls_logits_scratch); break;
        case 2: openeva::heads::decode_yolox_cell_ct<kC2, kC3>(
            head_[level], gx, gy, feat, out_row,
            cls_scratch, reg_scratch, cls_logits_scratch); break;
        case 3: openeva::heads::decode_yolox_cell_ct<kC3, kC3>(
            head_[level], gx, gy, feat, out_row,
            cls_scratch, reg_scratch, cls_logits_scratch); break;
        default: break;
    }
    lk.clear(std::memory_order_release);
}

void SslaSPipeline::stage_forward(int stage, int ev_x, int ev_y,
                                  const float* feat_in, float* feat_out) {
    // Stages have fixed (in, out) dims for SSLA-S. Each stage runs both
    // layers (L0+L1, L2+L3, ...) so the caller only deals with stage-level
    // boundaries. The intermediate buffer between the two layers is the
    // per-stage scratch (stage_out_[stage] reused as a temp).
    float* tmp = stage_out_[stage].data();
    const int Wl = Ws_[stage];
    const int Hl = Hs_[stage];
    const bool interior = (ev_x > 0) && (ev_x + 1 < Wl)
                       && (ev_y > 0) && (ev_y + 1 < Hl);
    switch (stage) {
        case 0:
            if (interior) {
                // s0 L0: fused interior path. num_pos = 1 (K=1, single-cell
                // update — only qvgIn[0] / goW[0] are populated). Skips
                // add_macs TLS counter — the dominant TLS-write source.
                const auto& L0 = layers_[0];
                fused::s0_l0_interior(
                    ev_x, ev_y, Wl, feat_in,
                    L0.input_proj.empty() ? nullptr : L0.input_proj.data(),
                    L0.qvgIn[0].data(), L0.goW[0].data(),
                    L0.ln_gamma.data(), L0.ln_beta.data(),
                    hidden_[0].data(), tmp);
                // s0 L1: fused interior path (1.09× over generic in microbench).
                const auto& L1 = layers_[1];
                fused::s0_l1_interior(
                    ev_x, ev_y, Wl, tmp,
                    L1.qvgIn, L1.goW,
                    L1.ln_gamma.data(), L1.ln_beta.data(),
                    hidden_[1].data(), feat_out);
            } else {
                layer_forward_ct<kInDim, kC0>(0, ev_x, ev_y, feat_in, tmp);
                layer_forward_ct<kC0, kC0>(1, ev_x, ev_y, tmp, feat_out);
            }
            break;
        case 1:
            // s1 L0: tile-streaming fused interior path (Phase 6, goW_T layout).
            if (interior) {
                const auto& L0 = layers_[2];
                fused::s1_l0_interior_tiled(
                    ev_x, ev_y, Wl, feat_in,
                    L0.input_proj.empty() ? nullptr : L0.input_proj.data(),
                    L0.qvgIn, L0.goW_T,
                    L0.ln_gamma.data(), L0.ln_beta.data(),
                    hidden_[2].data(), tmp);
            } else {
                layer_forward_ct<kC0, kC1>(2, ev_x, ev_y, feat_in, tmp);
            }
            // s1 L1: tile-streaming fused interior path (Phase 6, goW_T layout).
            if (interior) {
                const auto& L1 = layers_[3];
                fused::s1_l1_interior_tiled(
                    ev_x, ev_y, Wl, tmp,
                    L1.qvgIn, L1.goW_T,
                    L1.ln_gamma.data(), L1.ln_beta.data(),
                    hidden_[3].data(), feat_out);
            } else {
                layer_forward_ct<kC1, kC1>(3, ev_x, ev_y, tmp, feat_out);
            }
            break;
        case 2:
            layer_forward_ct<kC1, kC2>(4, ev_x, ev_y, feat_in, tmp);
            layer_forward_ct<kC2, kC2>(5, ev_x, ev_y, tmp,    feat_out);
            break;
        case 3:
            layer_forward_ct<kC2, kC3>(6, ev_x, ev_y, feat_in, tmp);
            layer_forward_ct<kC3, kC3>(7, ev_x, ev_y, tmp,    feat_out);
            break;
        default:
            throw std::out_of_range("ssla_kernels: stage out of range");
    }
}

void SslaSPipeline::layer_forward(int layer_idx, int ev_x, int ev_y,
                                   const float* feat_in, float* feat_out) {
    switch (layer_idx) {
        case 0: layer_forward_ct<kInDim, kC0>(0, ev_x, ev_y, feat_in, feat_out); break;
        case 1: layer_forward_ct<kC0,    kC0>(1, ev_x, ev_y, feat_in, feat_out); break;
        case 2: layer_forward_ct<kC0,    kC1>(2, ev_x, ev_y, feat_in, feat_out); break;
        case 3: layer_forward_ct<kC1,    kC1>(3, ev_x, ev_y, feat_in, feat_out); break;
        case 4: layer_forward_ct<kC1,    kC2>(4, ev_x, ev_y, feat_in, feat_out); break;
        case 5: layer_forward_ct<kC2,    kC2>(5, ev_x, ev_y, feat_in, feat_out); break;
        case 6: layer_forward_ct<kC2,    kC3>(6, ev_x, ev_y, feat_in, feat_out); break;
        case 7: layer_forward_ct<kC3,    kC3>(7, ev_x, ev_y, feat_in, feat_out); break;
        default:
            throw std::out_of_range("ssla_kernels: layer_idx out of range");
    }
}

// ============================================================================
// tdrop_and_pool
// ============================================================================

bool SslaSPipeline::tdrop_and_pool(int stage, int& ev_x, int& ev_y) {
    if (stage < 0 || stage > 2) return true;  // no boundary after stage 3
    ev_x /= 2;
    ev_y /= 2;
    const int Wp  = Ws_[stage + 1];
    const int idx = ev_y * Wp + ev_x;
    std::uint8_t& c = tdrop_counter_[stage][idx];
    const std::uint8_t pre = c;
    c = static_cast<std::uint8_t>(c + 1u);
    if (tdrop_window_ <= 1) return true;
    return (pre % tdrop_window_) == 0u;
}

// ============================================================================
// head_decode_cell + prepopulate_head_zero
// ============================================================================

void SslaSPipeline::head_decode_cell(int stage, int gx, int gy, const float* feat) {
    int level = -1;
    for (int k = 0; k < num_head_levels_; ++k) {
        if (head_stage_idx_[k] == stage) { level = k; break; }
    }
    if (level < 0) return;

    const int W_gr = head_[level].W_grid;
    const int H_gr = head_[level].H_grid;
    if (gx < 0 || gx >= W_gr || gy < 0 || gy >= H_gr) return;

    int row_offset = 0;
    for (int k = 0; k < level; ++k) {
        row_offset += head_[k].H_grid * head_[k].W_grid;
    }
    const int row_stride = 5 + num_classes_;
    float* out_row = last_predictions_.data() + static_cast<std::ptrdiff_t>(
        row_offset + gy * W_gr + gx) * row_stride;

    // Compile-time dispatch on the stage's C_in (matches stage_forward).
    switch (stage) {
        case 0:
            openeva::heads::decode_yolox_cell_ct<kC0, kC3>(
                head_[level], gx, gy, feat, out_row,
                scratch_cls_f_, scratch_reg_f_, scratch_cls_logits_); break;
        case 1:
            openeva::heads::decode_yolox_cell_ct<kC1, kC3>(
                head_[level], gx, gy, feat, out_row,
                scratch_cls_f_, scratch_reg_f_, scratch_cls_logits_); break;
        case 2:
            openeva::heads::decode_yolox_cell_ct<kC2, kC3>(
                head_[level], gx, gy, feat, out_row,
                scratch_cls_f_, scratch_reg_f_, scratch_cls_logits_); break;
        case 3:
            openeva::heads::decode_yolox_cell_ct<kC3, kC3>(
                head_[level], gx, gy, feat, out_row,
                scratch_cls_f_, scratch_reg_f_, scratch_cls_logits_); break;
        default: break;
    }
    publish_output();
}

void SslaSPipeline::prepopulate_head_zero() {
    if (num_head_levels_ == 0 || total_anchors_ == 0) return;
    const int row_stride = 5 + num_classes_;
    const float* zero = zero_feat_cell_.data();
    int row_offset = 0;
    for (int k = 0; k < num_head_levels_; ++k) {
        const int W_gr = head_[k].W_grid;
        const int H_gr = head_[k].H_grid;
        for (int gy = 0; gy < H_gr; ++gy) {
            for (int gx = 0; gx < W_gr; ++gx) {
                float* out_row = last_predictions_.data() +
                    static_cast<std::ptrdiff_t>(row_offset + gy * W_gr + gx) * row_stride;
                openeva::heads::decode_yolox_cell(
                    head_[k], gx, gy, zero, out_row,
                    scratch_cls_f_, scratch_reg_f_, scratch_cls_logits_);
            }
        }
        row_offset += H_gr * W_gr;
    }
    publish_output();
}

void SslaSPipeline::publish_output() {
    last_det_.predictions.data = last_predictions_;  // single memcpy
    last_output_ = last_det_;
}

}  // namespace deploy
