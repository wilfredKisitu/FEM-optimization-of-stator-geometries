#pragma once
#include "stator/params.hpp"
#include <vector>
#include <string>
#include <ostream>
#include <shared_mutex>
#include <stdexcept>

namespace stator {

// ─── RegionType ───────────────────────────────────────────────────────────────
// Integer values are the canonical GMSH physical-group tags.

enum class RegionType : int {
    YOKE          = 100,
    TOOTH         = 101,
    SLOT_AIR      = 200,
    SLOT_INS      = 201,
    COIL_A_POS    = 301,
    COIL_A_NEG    = 302,
    COIL_B_POS    = 303,
    COIL_B_NEG    = 304,
    COIL_C_POS    = 305,
    COIL_C_NEG    = 306,
    BORE_AIR      = 400,
    BOUNDARY_BORE = 500,
    BOUNDARY_OUTER= 501,
    UNKNOWN       = -1
};

// Canonical physical-group integer tag for a RegionType.
inline int canonical_tag(RegionType r) { return static_cast<int>(r); }

const char* to_string(RegionType r);

// ─── SlotWindingAssignment ────────────────────────────────────────────────────

struct SlotWindingAssignment {
    int        slot_idx    = -1;
    RegionType upper_phase = RegionType::UNKNOWN;
    RegionType lower_phase = RegionType::UNKNOWN; // -1 → single-layer (unused)
    int        upper_tag   = -1; // GMSH surface tag for upper coil
    int        lower_tag   = -1; // GMSH surface tag for lower coil (-1 = none)
};

// ─── TopologyRegistry ────────────────────────────────────────────────────────
// Thread-safe registry mapping GMSH entity tags to named physical regions.

class TopologyRegistry {
public:
    explicit TopologyRegistry(int n_slots);

    // ── Registration (write-locked) ───────────────────────────────────────────
    void register_surface(RegionType type, int gmsh_tag, int slot_idx = -1);
    void register_slot_coil(int slot_idx, int upper_tag, int lower_tag = -1);
    void register_boundary_curve(RegionType type, int gmsh_curve);

    // ── Winding layout ────────────────────────────────────────────────────────
    // Throws std::logic_error if coils have not been registered first.
    void assign_winding_layout(WindingType wt);

    // ── Queries (read-locked) ─────────────────────────────────────────────────
    std::vector<int> get_surfaces(RegionType type) const;
    std::vector<int> get_boundary_curves(RegionType type) const;
    const SlotWindingAssignment& get_slot_assignment(int slot_idx) const;
    const std::vector<SlotWindingAssignment>& get_winding_assignments() const;
    int  total_registered_surfaces() const;
    bool winding_assigned() const noexcept;

    // ── Diagnostics ───────────────────────────────────────────────────────────
    void dump(std::ostream& os) const;

private:
    int n_slots_;

    mutable std::shared_mutex mutex_;

    // surface tag lists per region
    std::vector<std::pair<RegionType, int>> surface_records_;  // (type, gmsh_tag)
    std::vector<std::pair<RegionType, int>> boundary_records_; // (type, gmsh_curve)

    // per-slot coil surface tags (before winding assignment)
    std::vector<int> slot_upper_tags_; // size = n_slots_
    std::vector<int> slot_lower_tags_; // size = n_slots_ (-1 = none)

    std::vector<SlotWindingAssignment> winding_assignments_;
    bool winding_assigned_ = false;

    // Helper: compute phase for a given slot index and winding type
    static RegionType phase_for_slot(int slot_idx, WindingType wt, bool lower);
};

} // namespace stator
