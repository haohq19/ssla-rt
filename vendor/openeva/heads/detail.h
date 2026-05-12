// detail.h — small helpers shared across `openeva/heads/*.h`.
//
// Pulled out so async_yolov1.h and async_yolox.h can both be included
// in the same translation unit without ODR-violating the must_get_t /
// maybe_get_t lookups.

#pragma once

#include "openeva/event.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace openeva::heads::detail {

inline const Tensor& must_get_t(
    const std::unordered_map<std::string, Tensor>& m,
    const std::string& k,
    const std::string& err_prefix) {
    auto it = m.find(k);
    if (it == m.end()) throw std::runtime_error(err_prefix + ": missing tensor " + k);
    return it->second;
}

inline const Tensor* maybe_get_t(
    const std::unordered_map<std::string, Tensor>& m,
    const std::string& k) {
    auto it = m.find(k);
    return it == m.end() ? nullptr : &it->second;
}

}  // namespace openeva::heads::detail
