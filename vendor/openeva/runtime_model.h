#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "event.h"
#include "output.h"

namespace openeva {

/// Abstract interface every CPU runtime model must implement.
///
/// The authoritative per-event entry point is `step()`. Every `step(e)` call
/// must satisfy the recurrent discipline described in cpp/DEVELOPMENT.md §4:
///
/// - bounded work per event (O(state), not O(cumulative events)),
/// - persistent layer state mutated in place,
/// - task head runs once per call and returns a fresh ModelOutput,
/// - no hidden internal buffering that defers work to a later call.
///
/// The returned `ModelOutput` is a task-agnostic variant (detection,
/// classification, segmentation, or `std::monostate` for no update). Callers
/// `std::visit` or `std::get_if` to dispatch on task.
class RuntimeModel {
public:
    virtual ~RuntimeModel() = default;

    /// Reset all recurrent state buffers. O(state).
    virtual void reset() = 0;

    /// Push one event, incrementally update all recurrent state, run the
    /// task head on the current feature grid, and return the resulting
    /// task-specific output.
    ///
    /// Returned by **const reference** to a model-owned slot: the reference
    /// stays valid until the next step()/reset() call. Callers that only
    /// need the output transiently (benchmark loops) pay zero copies; callers
    /// that must persist it across further step() calls (predict) do their
    /// own copy. Returning by value was costing 32 KB copies per event on
    /// SSLA (the decoded anchor tensor), which showed up as ~1-3 µs per
    /// event in profiling.
    virtual const ModelOutput& step(const Event& e) = 0;

    /// Load exported weights from the given directory (weights.npz + meta.json).
    virtual void load_weights(const std::string& export_dir) = 0;

    /// Configure with random (Xavier-init) weights — no checkpoint required.
    /// `variant_name` selects per-method channel widths (e.g. "ssla_s_det").
    /// FARSE-CNN ignores it (single variant). Default impl throws so methods
    /// that don't support synthetic weights fail loudly rather than silently.
    virtual void configure_random(const std::string& /*variant_name*/,
                                  int /*height*/, int /*width*/,
                                  int /*num_classes*/, std::uint32_t /*seed*/) {
        throw std::runtime_error(
            "configure_random not supported by this runtime ("
            + name() + "); use load_weights with a real export instead.");
    }

    /// Human-readable method name (e.g. "ssla").
    virtual std::string name() const = 0;
};

/// Factory function: create a RuntimeModel by method family name.
/// Valid families: "ssla" (covers SSLA + SSLA2), "farse_cnn", "dagr".
std::unique_ptr<RuntimeModel> create_model(const std::string& method_name);

}  // namespace openeva
