#include "stator/topology_registry.hpp"
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <mutex>

namespace stator {

// ─── to_string ────────────────────────────────────────────────────────────────

const char* to_string(RegionType r) {
    switch (r) {
        case RegionType::YOKE:           return "YOKE";
        case RegionType::TOOTH:          return "TOOTH";
        case RegionType::SLOT_AIR:       return "SLOT_AIR";
        case RegionType::SLOT_INS:       return "SLOT_INS";
        case RegionType::COIL_A_POS:     return "COIL_A_POS";
        case RegionType::COIL_A_NEG:     return "COIL_A_NEG";
        case RegionType::COIL_B_POS:     return "COIL_B_POS";
        case RegionType::COIL_B_NEG:     return "COIL_B_NEG";
        case RegionType::COIL_C_POS:     return "COIL_C_POS";
        case RegionType::COIL_C_NEG:     return "COIL_C_NEG";
        case RegionType::BORE_AIR:       return "BORE_AIR";
        case RegionType::BOUNDARY_BORE:  return "BOUNDARY_BORE";
        case RegionType::BOUNDARY_OUTER: return "BOUNDARY_OUTER";
        case RegionType::UNKNOWN:        return "UNKNOWN";
    }
    return "UNKNOWN";
}

// ─── Constructor ──────────────────────────────────────────────────────────────

TopologyRegistry::TopologyRegistry(int n_slots) : n_slots_(n_slots) {
    if (n_slots <= 0)
        throw std::invalid_argument("TopologyRegistry: n_slots must be > 0");
    slot_upper_tags_.assign(n_slots, -1);
    slot_lower_tags_.assign(n_slots, -1);
}

// ─── Registration ─────────────────────────────────────────────────────────────

void TopologyRegistry::register_surface(RegionType type, int gmsh_tag, int /*slot_idx*/) {
    std::unique_lock lock(mutex_);
    surface_records_.push_back({type, gmsh_tag});
}

void TopologyRegistry::register_slot_coil(int slot_idx, int upper_tag, int lower_tag) {
    std::unique_lock lock(mutex_);
    if (slot_idx < 0 || slot_idx >= n_slots_)
        throw std::out_of_range("TopologyRegistry::register_slot_coil: slot_idx out of range");
    slot_upper_tags_[slot_idx] = upper_tag;
    slot_lower_tags_[slot_idx] = lower_tag;
}

void TopologyRegistry::register_boundary_curve(RegionType type, int gmsh_curve) {
    if (type != RegionType::BOUNDARY_BORE && type != RegionType::BOUNDARY_OUTER)
        throw std::invalid_argument(
            "register_boundary_curve: type must be BOUNDARY_BORE or BOUNDARY_OUTER");
    std::unique_lock lock(mutex_);
    boundary_records_.push_back({type, gmsh_curve});
}

// ─── Winding phase assignment ─────────────────────────────────────────────────

// DISTRIBUTED: slot%6 → 0=A+, 1=B−, 2=C+, 3=A−, 4=B+, 5=C−
// CONCENTRATED: slot%6 → 0=A+, 1=A−, 2=B+, 3=B−, 4=C+, 5=C−
// SINGLE/DOUBLE_LAYER: use DISTRIBUTED sequence

RegionType TopologyRegistry::phase_for_slot(int slot_idx, WindingType wt, bool /*lower*/) {
    static const RegionType distributed[6] = {
        RegionType::COIL_A_POS, RegionType::COIL_B_NEG,
        RegionType::COIL_C_POS, RegionType::COIL_A_NEG,
        RegionType::COIL_B_POS, RegionType::COIL_C_NEG
    };
    static const RegionType concentrated[6] = {
        RegionType::COIL_A_POS, RegionType::COIL_A_NEG,
        RegionType::COIL_B_POS, RegionType::COIL_B_NEG,
        RegionType::COIL_C_POS, RegionType::COIL_C_NEG
    };

    int r = slot_idx % 6;
    switch (wt) {
        case WindingType::CONCENTRATED:
            return concentrated[r];
        case WindingType::SINGLE_LAYER:
        case WindingType::DOUBLE_LAYER:
        case WindingType::DISTRIBUTED:
        default:
            return distributed[r];
    }
}

void TopologyRegistry::assign_winding_layout(WindingType wt) {
    std::unique_lock lock(mutex_);
    // Check that at least one coil is registered
    bool any = false;
    for (int i = 0; i < n_slots_; ++i) {
        if (slot_upper_tags_[i] >= 0) { any = true; break; }
    }
    if (!any)
        throw std::logic_error(
            "assign_winding_layout: no coils registered via register_slot_coil()");

    winding_assignments_.clear();
    winding_assignments_.reserve(n_slots_);

    for (int i = 0; i < n_slots_; ++i) {
        SlotWindingAssignment a;
        a.slot_idx    = i;
        a.upper_tag   = slot_upper_tags_[i];
        a.lower_tag   = slot_lower_tags_[i];
        a.upper_phase = phase_for_slot(i, wt, false);
        // For DOUBLE_LAYER both halves get the same phase (simplified model).
        // TODO(v2): full short-pitch coil modelling
        a.lower_phase = (slot_lower_tags_[i] >= 0)
                        ? phase_for_slot(i, wt, true)
                        : RegionType::UNKNOWN;
        winding_assignments_.push_back(a);
    }
    winding_assigned_ = true;
}

// ─── Queries ──────────────────────────────────────────────────────────────────

std::vector<int> TopologyRegistry::get_surfaces(RegionType type) const {
    std::shared_lock lock(mutex_);
    std::vector<int> result;
    for (auto& [t, tag] : surface_records_)
        if (t == type) result.push_back(tag);
    return result;
}

std::vector<int> TopologyRegistry::get_boundary_curves(RegionType type) const {
    std::shared_lock lock(mutex_);
    std::vector<int> result;
    for (auto& [t, tag] : boundary_records_)
        if (t == type) result.push_back(tag);
    return result;
}

const SlotWindingAssignment&
TopologyRegistry::get_slot_assignment(int slot_idx) const {
    std::shared_lock lock(mutex_);
    if (!winding_assigned_)
        throw std::logic_error("get_slot_assignment: winding not yet assigned");
    if (slot_idx < 0 || slot_idx >= static_cast<int>(winding_assignments_.size()))
        throw std::out_of_range("get_slot_assignment: slot_idx out of range");
    return winding_assignments_[slot_idx];
}

const std::vector<SlotWindingAssignment>&
TopologyRegistry::get_winding_assignments() const {
    std::shared_lock lock(mutex_);
    if (!winding_assigned_)
        throw std::logic_error("get_winding_assignments: winding not yet assigned");
    return winding_assignments_;
}

int TopologyRegistry::total_registered_surfaces() const {
    std::shared_lock lock(mutex_);
    return static_cast<int>(surface_records_.size());
}

bool TopologyRegistry::winding_assigned() const noexcept {
    return winding_assigned_;
}

// ─── Diagnostics ──────────────────────────────────────────────────────────────

void TopologyRegistry::dump(std::ostream& os) const {
    std::shared_lock lock(mutex_);
    os << "=== TopologyRegistry (n_slots=" << n_slots_ << ") ===\n";
    os << "  Surfaces (" << surface_records_.size() << "):\n";
    for (auto& [t, tag] : surface_records_)
        os << "    " << to_string(t) << " -> tag " << tag << "\n";
    os << "  Boundary curves (" << boundary_records_.size() << "):\n";
    for (auto& [t, tag] : boundary_records_)
        os << "    " << to_string(t) << " -> curve " << tag << "\n";
    if (winding_assigned_) {
        os << "  Winding assignments:\n";
        for (auto& a : winding_assignments_) {
            os << "    slot[" << a.slot_idx << "]: upper="
               << to_string(a.upper_phase) << "(tag=" << a.upper_tag << ")";
            if (a.lower_tag >= 0)
                os << " lower=" << to_string(a.lower_phase)
                   << "(tag=" << a.lower_tag << ")";
            os << "\n";
        }
    } else {
        os << "  Winding: not yet assigned\n";
    }
}

} // namespace stator
