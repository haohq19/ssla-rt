#pragma once

#include <cstdint>

namespace openeva {

/// A single polarity event from an event camera.
struct Event {
    float t;   ///< Timestamp in microseconds
    float x;   ///< x pixel coordinate
    float y;   ///< y pixel coordinate
    float p;   ///< Polarity: 0.0 or 1.0
};

}  // namespace openeva
