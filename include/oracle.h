#pragma once

// Oracle equivalence harness.
//
// Each scheme dumps a sequence of (event_idx, prediction_tensor) snapshots
// at fixed checkpoint indices (e.g. every K events). The reference (S1)
// dump is the ground truth. Other schemes pass equivalence iff their
// dumps match within a fp32 ULP-scale tolerance at every checkpoint.
//
// Binary file layout (little-endian, host = x86):
//   uint32 magic        = 0x4F45 5641 ("OEVA")
//   uint32 version      = 1
//   uint32 num_anchors
//   uint32 cols         (= 5 + num_classes)
//   uint32 num_checkpoints
//   For each checkpoint:
//     uint64 event_idx
//     float[num_anchors * cols] predictions
//
// The recorder is callback-driven so each scheme can plug it in at the
// natural quiescence point (single-thread: after every step(); pipeline:
// after pipeline drain).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "openeva/output.h"

namespace deploy {

class OracleRecorder {
public:
    OracleRecorder(int num_anchors, int cols, std::size_t every_n_events,
                   const std::string& path)
        : num_anchors_(num_anchors), cols_(cols),
          every_(every_n_events), path_(path) {
        if (num_anchors_ <= 0 || cols_ <= 0)
            throw std::invalid_argument("OracleRecorder: invalid shape");
    }

    void record(std::uint64_t event_idx, const openeva::ModelOutput& out) {
        if (every_ == 0) return;
        if ((event_idx % every_) != 0 && event_idx != 0) return;
        const auto* det = std::get_if<openeva::DetectionOutput>(&out);
        if (!det) return;
        const auto& d = det->predictions.data;
        const std::size_t expected =
            static_cast<std::size_t>(num_anchors_) * cols_;
        if (d.size() != expected) return;  // head not ready yet (init phase)
        checkpoints_.push_back({event_idx, d});
    }

    void save() const {
        std::FILE* f = std::fopen(path_.c_str(), "wb");
        if (!f) throw std::runtime_error("oracle: cannot open " + path_);
        const std::uint32_t magic   = 0x4F455641;  // 'OEVA'
        const std::uint32_t version = 1;
        const std::uint32_t na      = static_cast<std::uint32_t>(num_anchors_);
        const std::uint32_t co      = static_cast<std::uint32_t>(cols_);
        const std::uint32_t nc      = static_cast<std::uint32_t>(checkpoints_.size());
        std::fwrite(&magic,   4, 1, f);
        std::fwrite(&version, 4, 1, f);
        std::fwrite(&na,      4, 1, f);
        std::fwrite(&co,      4, 1, f);
        std::fwrite(&nc,      4, 1, f);
        for (const auto& cp : checkpoints_) {
            std::fwrite(&cp.event_idx, 8, 1, f);
            std::fwrite(cp.preds.data(), sizeof(float), cp.preds.size(), f);
        }
        std::fclose(f);
    }

    std::size_t num_checkpoints() const { return checkpoints_.size(); }

private:
    struct Checkpoint {
        std::uint64_t event_idx;
        std::vector<float> preds;
    };
    int num_anchors_;
    int cols_;
    std::size_t every_;
    std::string path_;
    std::vector<Checkpoint> checkpoints_;
};

// Standalone diff: returns max abs delta and number of mismatched
// checkpoints across two saved dumps. Used by the equivalence CLI tool.
struct DiffReport {
    bool        ok          = false;   // shape/checkpoint-count match
    std::size_t n_cp        = 0;
    double      max_abs_delta = 0.0;
    std::size_t n_mismatched_cp = 0;   // checkpoints with any |Δ| > tol
};

inline DiffReport diff_dumps(const std::string& path_a,
                             const std::string& path_b,
                             double tol_abs = 1e-4) {
    auto load = [](const std::string& p,
                   std::uint32_t& na, std::uint32_t& co,
                   std::uint32_t& nc,
                   std::vector<std::uint64_t>& idx,
                   std::vector<std::vector<float>>& preds) {
        std::FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) throw std::runtime_error("oracle: cannot open " + p);
        std::uint32_t magic, version;
        if (std::fread(&magic, 4, 1, f) != 1 ||
            std::fread(&version, 4, 1, f) != 1 ||
            std::fread(&na, 4, 1, f) != 1 ||
            std::fread(&co, 4, 1, f) != 1 ||
            std::fread(&nc, 4, 1, f) != 1) {
            std::fclose(f);
            throw std::runtime_error("oracle: header read failed: " + p);
        }
        if (magic != 0x4F455641u || version != 1)
            throw std::runtime_error("oracle: bad magic/version: " + p);
        const std::size_t per_cp = static_cast<std::size_t>(na) * co;
        idx.resize(nc);
        preds.assign(nc, std::vector<float>(per_cp));
        for (std::uint32_t i = 0; i < nc; ++i) {
            if (std::fread(&idx[i], 8, 1, f) != 1) break;
            if (std::fread(preds[i].data(), sizeof(float), per_cp, f) != per_cp) break;
        }
        std::fclose(f);
    };
    std::uint32_t na_a, co_a, nc_a, na_b, co_b, nc_b;
    std::vector<std::uint64_t> idx_a, idx_b;
    std::vector<std::vector<float>> preds_a, preds_b;
    load(path_a, na_a, co_a, nc_a, idx_a, preds_a);
    load(path_b, na_b, co_b, nc_b, idx_b, preds_b);
    DiffReport r;
    r.ok = (na_a == na_b) && (co_a == co_b) && (nc_a == nc_b);
    r.n_cp = nc_a;
    if (!r.ok) return r;
    for (std::uint32_t i = 0; i < nc_a; ++i) {
        if (idx_a[i] != idx_b[i]) { r.ok = false; break; }
        double cp_max = 0.0;
        const std::size_t per = preds_a[i].size();
        for (std::size_t j = 0; j < per; ++j) {
            const double d = std::abs(double(preds_a[i][j]) - double(preds_b[i][j]));
            if (d > cp_max) cp_max = d;
        }
        if (cp_max > r.max_abs_delta) r.max_abs_delta = cp_max;
        if (cp_max > tol_abs) ++r.n_mismatched_cp;
    }
    return r;
}

}  // namespace deploy
