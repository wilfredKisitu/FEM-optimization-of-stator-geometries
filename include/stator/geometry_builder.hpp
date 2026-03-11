#pragma once
#include "stator/params.hpp"
#include "stator/gmsh_backend.hpp"
#include <vector>
#include <string>
#include <memory>

namespace stator {

// ─── SlotProfile ─────────────────────────────────────────────────────────────
// Holds the GMSH tags produced while building one slot.

struct SlotProfile {
    int slot_idx        = -1;
    int slot_surface    = -1;   // main slot air cavity surface tag
    int coil_upper_sf   = -1;   // upper (or only) coil conductor surface
    int coil_lower_sf   = -1;   // lower coil surface (-1 for single-layer)
    int ins_upper_sf    = -1;   // insulation around upper coil
    int ins_lower_sf    = -1;   // insulation around lower coil
    int mouth_curve_bot = -1;   // bore-facing edge (used for BL seed)
    int mouth_curve_top = -1;   // bottom edge of slot (SEMI_CLOSED only)
    double angle        = 0.0;  // rotation angle of this slot (radians)
};

// ─── GeometryBuildResult ─────────────────────────────────────────────────────

struct GeometryBuildResult {
    bool success         = false;
    std::string error_message;

    int yoke_surface     = -1;  // trimmed yoke surface tag
    int bore_curve       = -1;  // inner bore circle tag
    int outer_curve      = -1;  // outer circle tag

    std::vector<SlotProfile> slots; // one per slot
};

// ─── GeometryBuilder ─────────────────────────────────────────────────────────

class GeometryBuilder {
public:
    explicit GeometryBuilder(IGmshBackend* backend);

    // Build the complete 2-D stator cross-section.
    // Throws std::invalid_argument if backend is null.
    GeometryBuildResult build(const StatorParams& p);

    // Rotate a 2-D point by theta (radians).
    static std::pair<double,double> rotate(double x, double y, double theta) noexcept;

private:
    IGmshBackend* backend_;

    // Slot angle for the k-th slot (radians)
    static double slot_angle(int k, int n_slots) noexcept;

    // Add a rotated point at (x,y) local frame → global frame via theta.
    int add_rotated_point(double x, double y, double theta, double mesh_size = 0.0);

    // Build a single slot at angular position k.
    SlotProfile build_single_slot(const StatorParams& p, int k);

    // Shape-specific slot builders (local frame, then rotated).
    SlotProfile build_rectangular(const StatorParams& p, int k);
    SlotProfile build_trapezoidal(const StatorParams& p, int k);
    SlotProfile build_round_bottom(const StatorParams& p, int k);
    SlotProfile build_semi_closed(const StatorParams& p, int k);

    // Build coil conductor surface(s) inside a slot cavity.
    void build_coil_inside_slot(const StatorParams& p, SlotProfile& profile);

    // Build insulation shells around the coil surface(s).
    void build_insulation(const StatorParams& p, SlotProfile& profile);
};

} // namespace stator
