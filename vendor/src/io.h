#pragma once

#include <string>
#include <vector>

#include "openeva/event.h"   // cpp/include/openeva/event.h

namespace openeva {

/// Load events from an events.npy file (float32, shape [N, 4]).
std::vector<Event> load_events_npy(const std::string& path);

/// Write a JSON string to a file.
void write_json(const std::string& path, const std::string& json_str);

/// Read a text file into a string.
std::string read_file(const std::string& path);

}  // namespace openeva
