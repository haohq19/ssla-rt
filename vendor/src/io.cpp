#include "io.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cassert>

namespace openeva {

// ---------------------------------------------------------------------------
// Minimal .npy parser for float32 arrays (C-order, no compression)
// See: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
// ---------------------------------------------------------------------------

static void npy_parse_header(
    std::ifstream& f,
    std::vector<size_t>& shape,
    std::string& dtype
) {
    // Magic: \x93NUMPY
    char magic[7] = {};
    f.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Not a .npy file");
    }

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len;
    if (major == 1) {
        uint16_t hlen;
        f.read(reinterpret_cast<char*>(&hlen), 2);
        header_len = hlen;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Parse dtype from header string (e.g., "'descr': '<f4'")
    auto find_field = [&](const std::string& key) -> std::string {
        auto pos = header.find(key);
        if (pos == std::string::npos) return "";
        pos = header.find("'", pos + key.size());
        if (pos == std::string::npos) return "";
        auto end = header.find("'", pos + 1);
        return header.substr(pos + 1, end - pos - 1);
    };
    dtype = find_field("'descr':");
    if (dtype.empty()) dtype = find_field("\"descr\":");

    // Parse shape (e.g., "(1000, 4)")
    auto sp = header.find("'shape':");
    if (sp == std::string::npos) sp = header.find("\"shape\":");
    if (sp != std::string::npos) {
        auto lparen = header.find('(', sp);
        auto rparen = header.find(')', sp);
        std::string shape_str = header.substr(lparen + 1, rparen - lparen - 1);
        std::stringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t,") + 1);
            if (!token.empty()) {
                shape.push_back(std::stoull(token));
            }
        }
    }
}


std::vector<Event> load_events_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<size_t> shape;
    std::string dtype;
    npy_parse_header(f, shape, dtype);

    if (shape.size() != 2 || shape[1] != 4) {
        throw std::runtime_error(
            "events.npy must have shape (N, 4), got shape with ndim=" +
            std::to_string(shape.size())
        );
    }
    if (dtype != "<f4" && dtype != "=f4" && dtype != "f4") {
        throw std::runtime_error("events.npy must be float32, got dtype=" + dtype);
    }

    size_t N = shape[0];
    std::vector<float> buf(N * 4);
    f.read(reinterpret_cast<char*>(buf.data()), N * 4 * sizeof(float));

    std::vector<Event> events(N);
    for (size_t i = 0; i < N; ++i) {
        events[i] = { buf[i * 4 + 0], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3] };
    }
    return events;
}


void write_json(const std::string& path, const std::string& json_str) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot write file: " + path);
    }
    f << json_str;
}


std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace openeva
