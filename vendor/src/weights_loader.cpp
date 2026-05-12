// weights_loader.cpp — minimal meta.json + weights.npz reader shared by all
// OpenEVA C++ runtimes. Supports uncompressed (stored) zip entries only, which
// is what np.savez produces. See cpp/DEVELOPMENT.md §3.

#include "src/weights_loader.h"

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace openeva {
namespace {

// ---------- tiny JSON helpers (flat objects only — no nesting) -----------

std::string trim(std::string s) {
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!s.empty() && is_ws(s.front())) s.erase(s.begin());
    while (!s.empty() && is_ws(s.back()))  s.pop_back();
    return s;
}

std::string json_get_string(const std::string& blob, const std::string& key) {
    auto k = "\"" + key + "\"";
    auto p = blob.find(k);
    if (p == std::string::npos) return "";
    p = blob.find(':', p + k.size());
    if (p == std::string::npos) return "";
    ++p;
    // find the first '"' after the colon
    auto q = blob.find('"', p);
    if (q == std::string::npos) return "";
    auto r = blob.find('"', q + 1);
    if (r == std::string::npos) return "";
    return blob.substr(q + 1, r - q - 1);
}

long long json_get_int(const std::string& blob, const std::string& key) {
    auto k = "\"" + key + "\"";
    auto p = blob.find(k);
    if (p == std::string::npos) return 0;
    p = blob.find(':', p + k.size());
    if (p == std::string::npos) return 0;
    ++p;
    while (p < blob.size() && (blob[p] == ' ' || blob[p] == '\t')) ++p;
    auto r = p;
    while (r < blob.size() && blob[r] != ',' && blob[r] != '}' && blob[r] != '\n') ++r;
    std::string num = trim(blob.substr(p, r - p));
    if (num.empty()) return 0;
    long long v = 0;
    auto [_, ec] = std::from_chars(num.data(), num.data() + num.size(), v);
    if (ec != std::errc{}) return 0;
    return v;
}

double json_get_float(const std::string& blob, const std::string& key) {
    auto k = "\"" + key + "\"";
    auto p = blob.find(k);
    if (p == std::string::npos) return 0.0;
    p = blob.find(':', p + k.size());
    if (p == std::string::npos) return 0.0;
    ++p;
    while (p < blob.size() && (blob[p] == ' ' || blob[p] == '\t')) ++p;
    auto r = p;
    while (r < blob.size() && blob[r] != ',' && blob[r] != '}' && blob[r] != '\n') ++r;
    std::string num = trim(blob.substr(p, r - p));
    if (num.empty()) return 0.0;
    // std::from_chars doesn't accept floats with locale handling reliably
    // pre-C++20; use std::stod which is locale-aware but suffices for
    // simple unsigned floats with optional exponent (the meta.json values
    // are written by Python's json.dump which always uses "C" locale).
    try { return std::stod(num); }
    catch (...) { return 0.0; }
}

// ---------- uncompressed zip reader --------------------------------------

#pragma pack(push, 1)
struct LocalFileHeader {
    std::uint32_t signature;             // 0x04034b50
    std::uint16_t version;
    std::uint16_t flags;
    std::uint16_t method;                // must be 0 (stored)
    std::uint16_t mod_time;
    std::uint16_t mod_date;
    std::uint32_t crc32;
    std::uint32_t compressed_size;
    std::uint32_t uncompressed_size;
    std::uint16_t filename_length;
    std::uint16_t extra_length;
};
#pragma pack(pop)
static_assert(sizeof(LocalFileHeader) == 30, "zip local file header size");

// ---------- .npy parser --------------------------------------------------

struct NpyInfo {
    std::vector<std::size_t> shape;
    std::string              dtype;      // e.g. "<f4"
    bool                     fortran;    // C-order if false
    std::size_t              header_len; // absolute offset in bytes
};

NpyInfo parse_npy_header(const char* data, std::size_t len) {
    if (len < 10 || std::memcmp(data, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("weights.npz: entry is not a .npy file");
    }
    std::uint8_t major = static_cast<std::uint8_t>(data[6]);
    std::size_t header_len = 0;
    std::size_t header_start = 0;
    if (major == 1) {
        std::uint16_t hlen = 0;
        std::memcpy(&hlen, data + 8, 2);
        header_len = hlen;
        header_start = 10;
    } else {
        std::uint32_t hlen = 0;
        std::memcpy(&hlen, data + 8, 4);
        header_len = hlen;
        header_start = 12;
    }
    if (header_start + header_len > len) {
        throw std::runtime_error("weights.npz: .npy header overruns entry");
    }
    std::string header(data + header_start, header_len);

    NpyInfo info;
    info.header_len = header_start + header_len;
    info.fortran    = header.find("'fortran_order': True") != std::string::npos;

    // descr
    auto d1 = header.find("'descr':");
    if (d1 != std::string::npos) {
        auto q1 = header.find('\'', d1 + 8);
        auto q2 = header.find('\'', q1 + 1);
        info.dtype = header.substr(q1 + 1, q2 - q1 - 1);
    }
    // shape
    auto s1 = header.find("'shape':");
    if (s1 != std::string::npos) {
        auto lp = header.find('(', s1);
        auto rp = header.find(')', s1);
        std::string shape_str = header.substr(lp + 1, rp - lp - 1);
        std::stringstream ss(shape_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            tok = trim(tok);
            if (!tok.empty()) info.shape.push_back(std::stoull(tok));
        }
        if (info.shape.empty()) info.shape.push_back(1);  // scalar
    }
    return info;
}

}  // namespace

ModelMeta load_meta(const std::string& dir) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += "meta.json";

    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string blob = ss.str();

    ModelMeta m;
    m.schema_version = static_cast<int>(json_get_int(blob, "schema_version"));
    m.model         = json_get_string(blob, "model");
    m.method        = json_get_string(blob, "method");
    m.dtype         = json_get_string(blob, "dtype");
    m.layout        = json_get_string(blob, "layout");
    m.key_separator = json_get_string(blob, "key_separator");
    m.checkpoint    = json_get_string(blob, "checkpoint");
    m.height        = static_cast<int>(json_get_int(blob, "height"));
    m.width         = static_cast<int>(json_get_int(blob, "width"));
    m.num_classes   = static_cast<int>(json_get_int(blob, "num_classes"));
    m.num_anchors   = static_cast<int>(json_get_int(blob, "num_anchors"));
    m.farse_cnn_tdrop_window =
        static_cast<int>(json_get_int(blob, "farse_cnn_tdrop_window"));
    m.farse_cnn_head_width =
        static_cast<int>(json_get_int(blob, "farse_cnn_head_width"));
    m.ssla_tdrop_window =
        static_cast<int>(json_get_int(blob, "ssla_tdrop_window"));
    m.dagr_keep_temporal_ordering =
        static_cast<int>(json_get_int(blob, "dagr_keep_temporal_ordering"));
    m.dagr_pooling_dim_at_output =
        json_get_string(blob, "dagr_pooling_dim_at_output");
    m.asynet_input_h = static_cast<int>(json_get_int(blob, "asynet_input_h"));
    m.asynet_input_w = static_cast<int>(json_get_int(blob, "asynet_input_w"));
    m.asynet_grid_h  = static_cast<int>(json_get_int(blob, "asynet_grid_h"));
    m.asynet_grid_w  = static_cast<int>(json_get_int(blob, "asynet_grid_w"));
    m.asynet_head_width = static_cast<int>(json_get_int(blob, "asynet_head_width"));
    m.aegnn_grid_h         = static_cast<int>(json_get_int(blob, "aegnn_grid_h"));
    m.aegnn_grid_w         = static_cast<int>(json_get_int(blob, "aegnn_grid_w"));
    m.aegnn_radius         = static_cast<float>(json_get_float(blob, "aegnn_radius"));
    m.aegnn_max_neighbors  = static_cast<int>(json_get_int(blob, "aegnn_max_neighbors"));
    m.aegnn_beta           = static_cast<float>(json_get_float(blob, "aegnn_beta"));
    m.aegnn_pool5_voxel_w  = static_cast<int>(json_get_int(blob, "aegnn_pool5_voxel_w"));
    m.aegnn_pool5_voxel_h  = static_cast<int>(json_get_int(blob, "aegnn_pool5_voxel_h"));
    m.aegnn_head_width     = static_cast<int>(json_get_int(blob, "aegnn_head_width"));
    m.nvs_input_h              = static_cast<int>(json_get_int(blob, "nvs_input_h"));
    m.nvs_input_w              = static_cast<int>(json_get_int(blob, "nvs_input_w"));
    m.nvs_grid_h               = static_cast<int>(json_get_int(blob, "nvs_grid_h"));
    m.nvs_grid_w               = static_cast<int>(json_get_int(blob, "nvs_grid_w"));
    m.nvs_radius               = static_cast<float>(json_get_float(blob, "nvs_radius"));
    m.nvs_max_neighbors        = static_cast<int>(json_get_int(blob, "nvs_max_neighbors"));
    m.nvs_beta                 = static_cast<float>(json_get_float(blob, "nvs_beta"));
    m.nvs_voxel_pool_dim       = static_cast<int>(json_get_int(blob, "nvs_voxel_pool_dim"));
    m.nvs_final_pool_voxel_div = static_cast<int>(json_get_int(blob, "nvs_final_pool_voxel_div"));
    m.nvs_final_pool_size      = static_cast<int>(json_get_int(blob, "nvs_final_pool_size"));
    m.nvs_head_width           = static_cast<int>(json_get_int(blob, "nvs_head_width"));
    m.nvs_channels             = json_get_string(blob, "nvs_channels");
    m.nvs_voxel_sizes          = json_get_string(blob, "nvs_voxel_sizes");
    m.total_params  = static_cast<std::size_t>(json_get_int(blob, "total_params"));

    // Back-compat: if `method` is missing, derive it from `model` by
    // stripping the variant/task suffix.
    if (m.method.empty() && !m.model.empty()) {
        if      (m.model.find("ssla2") == 0) m.method = "ssla2";
        else if (m.model.find("ssla")  == 0) m.method = "ssla";
        else if (m.model.find("farse_cnn") == 0) m.method = "farse_cnn";
        else if (m.model.find("dagr")   == 0) m.method = "dagr";
        else if (m.model.find("asynet") == 0) m.method = "asynet";
        else if (m.model.find("aegnn")  == 0) m.method = "aegnn";
    }

    // Validate schema_version. Legacy exports (missing key → 0) are accepted
    // with a one-line warning; unknown future versions are a hard error.
    if (m.schema_version == 0) {
        std::fprintf(stderr,
            "[weights_loader] WARNING: %s has no schema_version "
            "(pre-v1 export); accepting for back-compat. "
            "Re-export with the current python/exporter.py to silence.\n",
            path.c_str());
    } else if (m.schema_version > kSupportedSchemaVersion) {
        throw std::runtime_error(
            path + ": schema_version=" + std::to_string(m.schema_version)
            + " is newer than this C++ build supports ("
            + std::to_string(kSupportedSchemaVersion) + "). "
            "Rebuild C++ runtime from the matching commit.");
    }
    return m;
}

std::unordered_map<std::string, Tensor>
load_weights_npz(const std::string& dir) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path.push_back('/');
    path += "weights.npz";

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);

    std::unordered_map<std::string, Tensor> out;

    while (f.good()) {
        LocalFileHeader h{};
        f.read(reinterpret_cast<char*>(&h), sizeof(h));
        if (!f.good()) break;
        if (h.signature != 0x04034b50u) break;  // end of local headers
        if (h.method != 0) {
            throw std::runtime_error(
                "weights.npz: compressed entries not supported — "
                "use np.savez (stored) not np.savez_compressed");
        }

        std::string name(h.filename_length, '\0');
        f.read(name.data(), h.filename_length);

        // Read the extra field — we need it to resolve ZIP64 sizes.
        std::vector<char> extra(h.extra_length);
        if (h.extra_length > 0) {
            f.read(extra.data(), h.extra_length);
        }

        // Resolve actual compressed / uncompressed sizes. numpy's zipfile
        // module writes ZIP64 extended info (tag 0x0001) even for small
        // files when the archive is created with allowZip64=True (default).
        std::uint64_t actual_csize = h.compressed_size;
        std::uint64_t actual_usize = h.uncompressed_size;
        if (h.compressed_size == 0xFFFFFFFFu || h.uncompressed_size == 0xFFFFFFFFu) {
            // Parse ZIP64 extra field: tag(2) + size(2) + usize(8) + csize(8)
            bool found = false;
            std::size_t pos = 0;
            while (pos + 4 <= extra.size()) {
                std::uint16_t tag = 0, sz = 0;
                std::memcpy(&tag, extra.data() + pos, 2);
                std::memcpy(&sz,  extra.data() + pos + 2, 2);
                if (tag == 0x0001 && pos + 4 + sz <= extra.size()) {
                    // ZIP64: uncompressed_size(8), compressed_size(8)
                    if (sz >= 16) {
                        std::memcpy(&actual_usize, extra.data() + pos + 4, 8);
                        std::memcpy(&actual_csize, extra.data() + pos + 12, 8);
                    } else if (sz >= 8) {
                        std::memcpy(&actual_usize, extra.data() + pos + 4, 8);
                        actual_csize = actual_usize; // stored → same size
                    }
                    found = true;
                    break;
                }
                pos += 4 + sz;
            }
            if (!found) {
                throw std::runtime_error(
                    "weights.npz: ZIP64 entry without ZIP64 extra field: " + name);
            }
        }

        std::vector<char> raw(actual_csize);
        f.read(raw.data(), static_cast<std::streamsize>(actual_csize));
        if (!f.good()) {
            throw std::runtime_error("weights.npz: truncated entry " + name);
        }

        // Trim .npy suffix from the key.
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) {
            name.resize(name.size() - 4);
        }

        NpyInfo info = parse_npy_header(raw.data(), raw.size());
        if (info.fortran) {
            throw std::runtime_error(
                "weights.npz: fortran-order tensors not supported for key " + name);
        }
        if (info.dtype != "<f4" && info.dtype != "=f4" && info.dtype != "f4") {
            throw std::runtime_error(
                "weights.npz: dtype " + info.dtype + " not supported for key " + name);
        }

        Tensor t;
        t.shape = info.shape;
        std::size_t n = t.numel();
        t.data.resize(n);
        const char* src = raw.data() + info.header_len;
        std::memcpy(t.data.data(), src, n * sizeof(float));

        out.emplace(std::move(name), std::move(t));
    }

    return out;
}

}  // namespace openeva
