#include "stator/params.hpp"
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <numbers>

namespace stator {

static constexpr double PI = std::numbers::pi;

// ─── to_string ────────────────────────────────────────────────────────────────

const char* to_string(SlotShape s) {
    switch (s) {
        case SlotShape::RECTANGULAR:  return "RECTANGULAR";
        case SlotShape::TRAPEZOIDAL:  return "TRAPEZOIDAL";
        case SlotShape::ROUND_BOTTOM: return "ROUND_BOTTOM";
        case SlotShape::SEMI_CLOSED:  return "SEMI_CLOSED";
    }
    return "UNKNOWN";
}

const char* to_string(WindingType w) {
    switch (w) {
        case WindingType::SINGLE_LAYER: return "SINGLE_LAYER";
        case WindingType::DOUBLE_LAYER: return "DOUBLE_LAYER";
        case WindingType::CONCENTRATED: return "CONCENTRATED";
        case WindingType::DISTRIBUTED:  return "DISTRIBUTED";
    }
    return "UNKNOWN";
}

const char* to_string(LaminationMaterial m) {
    switch (m) {
        case LaminationMaterial::M270_35A: return "M270_35A";
        case LaminationMaterial::M330_50A: return "M330_50A";
        case LaminationMaterial::M400_50A: return "M400_50A";
        case LaminationMaterial::NO20:     return "NO20";
        case LaminationMaterial::CUSTOM:   return "CUSTOM";
    }
    return "UNKNOWN";
}

// ─── validate_and_derive ──────────────────────────────────────────────────────

void StatorParams::validate_and_derive() {
    // Rule 1: All double dimensions > 0
    if (R_outer <= 0.0)
        throw std::invalid_argument("R_outer must be > 0");
    if (R_inner <= 0.0)
        throw std::invalid_argument("R_inner must be > 0");
    if (airgap_length <= 0.0)
        throw std::invalid_argument("airgap_length must be > 0");
    if (slot_depth <= 0.0)
        throw std::invalid_argument("slot_depth must be > 0");
    if (slot_width_outer <= 0.0)
        throw std::invalid_argument("slot_width_outer must be > 0");
    if (slot_width_inner <= 0.0)
        throw std::invalid_argument("slot_width_inner must be > 0");
    if (coil_depth <= 0.0)
        throw std::invalid_argument("coil_depth must be > 0");
    if (coil_width_outer <= 0.0)
        throw std::invalid_argument("coil_width_outer must be > 0");
    if (coil_width_inner <= 0.0)
        throw std::invalid_argument("coil_width_inner must be > 0");
    if (insulation_thickness <= 0.0)
        throw std::invalid_argument("insulation_thickness must be > 0");
    if (wire_diameter <= 0.0)
        throw std::invalid_argument("wire_diameter must be > 0");
    if (t_lam <= 0.0)
        throw std::invalid_argument("t_lam must be > 0");
    if (mesh_yoke <= 0.0)
        throw std::invalid_argument("mesh_yoke must be > 0");
    if (mesh_slot <= 0.0)
        throw std::invalid_argument("mesh_slot must be > 0");
    if (mesh_coil <= 0.0)
        throw std::invalid_argument("mesh_coil must be > 0");
    if (mesh_ins <= 0.0)
        throw std::invalid_argument("mesh_ins must be > 0");

    // Rule 2: R_inner < R_outer
    if (R_inner >= R_outer)
        throw std::invalid_argument("R_inner must be < R_outer");

    // Rule 3: slot_depth < (R_outer - R_inner)
    if (slot_depth >= (R_outer - R_inner))
        throw std::invalid_argument("slot_depth must be < (R_outer - R_inner)");

    // Rule 4: n_slots >= 6 and even
    if (n_slots < 6)
        throw std::invalid_argument("n_slots must be >= 6");
    if (n_slots % 2 != 0)
        throw std::invalid_argument("n_slots must be even");

    // Rule 5: slot_width_inner < R_inner * 2π / n_slots
    double pitch = 2.0 * PI / static_cast<double>(n_slots);
    if (slot_width_inner >= R_inner * pitch)
        throw std::invalid_argument(
            "slot_width_inner must be < R_inner * 2π/n_slots (tooth pitch arc)");

    // Rule 6: SEMI_CLOSED checks
    if (slot_shape == SlotShape::SEMI_CLOSED) {
        if (slot_opening >= slot_width_inner)
            throw std::invalid_argument(
                "SEMI_CLOSED: slot_opening must be < slot_width_inner");
        if (slot_opening_depth >= slot_depth)
            throw std::invalid_argument(
                "SEMI_CLOSED: slot_opening_depth must be < slot_depth");
    }

    // Rule 7: coil fits inside slot
    double max_coil_depth = slot_depth - slot_opening_depth - 2.0 * insulation_thickness;
    if (coil_depth > max_coil_depth)
        throw std::invalid_argument(
            "coil_depth exceeds available slot space (slot_depth - slot_opening_depth - 2*insulation_thickness)");

    // Rule 8: coil width fits inside slot
    double max_coil_width = slot_width_inner - 2.0 * insulation_thickness;
    if (coil_width_inner > max_coil_width)
        throw std::invalid_argument(
            "coil_width_inner exceeds slot_width_inner - 2*insulation_thickness");

    // Rule 9: n_lam > 0
    if (n_lam <= 0)
        throw std::invalid_argument("n_lam must be > 0");

    // Rule 10: z_spacing >= 0
    if (z_spacing < 0.0)
        throw std::invalid_argument("z_spacing must be >= 0");

    // Rule 11: insulation_coating_thickness >= 0
    if (insulation_coating_thickness < 0.0)
        throw std::invalid_argument("insulation_coating_thickness must be >= 0");

    // Rule 12: CUSTOM material requires material_file
    if (material == LaminationMaterial::CUSTOM && material_file.empty())
        throw std::invalid_argument("material == CUSTOM requires a non-empty material_file");

    // Rule 13: Mesh size ordering (finest to coarsest: ins <= coil <= slot <= yoke)
    if (mesh_ins > mesh_coil)
        throw std::invalid_argument("mesh_ins must be <= mesh_coil");
    if (mesh_coil > mesh_slot)
        throw std::invalid_argument("mesh_coil must be <= mesh_slot");
    if (mesh_slot > mesh_yoke)
        throw std::invalid_argument("mesh_slot must be <= mesh_yoke");

    // Rule 14: tooth_tip_angle >= 0 and < π/4
    if (tooth_tip_angle < 0.0)
        throw std::invalid_argument("tooth_tip_angle must be >= 0");
    if (tooth_tip_angle >= PI / 4.0)
        throw std::invalid_argument("tooth_tip_angle must be < π/4");

    // ── Compute derived quantities ────────────────────────────────────────────
    slot_pitch   = pitch;
    yoke_height  = R_outer - R_inner - slot_depth;
    tooth_width  = R_inner * slot_pitch - slot_width_inner;
    stack_length = static_cast<double>(n_lam) * t_lam
                 + static_cast<double>(n_lam - 1) * z_spacing;

    // Approximate slot area (trapezoidal shape)
    double slot_area = 0.5 * (slot_width_inner + slot_width_outer) * slot_depth;
    // Approximate coil area (per layer — fill_factor represents single-layer occupancy ratio)
    double coil_area = 0.5 * (coil_width_inner + coil_width_outer) * coil_depth;
    fill_factor = (slot_area > 0.0) ? coil_area / slot_area : 0.0;

    // Rule 15: fill_factor in (0, 1)
    if (fill_factor <= 0.0 || fill_factor >= 1.0)
        throw std::logic_error("computed fill_factor is not in (0, 1)");

    // Rule 16: fill_factor vs slot_fill_factor within 5% (warning only)
    double rel_err = std::abs(fill_factor - slot_fill_factor) / slot_fill_factor;
    if (rel_err > 0.05) {
        // Warning only — do not throw
    }
}

// ─── to_json ──────────────────────────────────────────────────────────────────

std::string StatorParams::to_json() const {
    std::ostringstream o;
    o << "{";
    // Section 1
    o << "\"R_outer\":" << R_outer << ","
      << "\"R_inner\":" << R_inner << ","
      << "\"airgap_length\":" << airgap_length << ",";
    // Section 2
    o << "\"n_slots\":" << n_slots << ","
      << "\"slot_depth\":" << slot_depth << ","
      << "\"slot_width_outer\":" << slot_width_outer << ","
      << "\"slot_width_inner\":" << slot_width_inner << ","
      << "\"slot_opening\":" << slot_opening << ","
      << "\"slot_opening_depth\":" << slot_opening_depth << ","
      << "\"tooth_tip_angle\":" << tooth_tip_angle << ","
      << "\"slot_shape\":\"" << to_string(slot_shape) << "\",";
    // Section 3
    o << "\"coil_depth\":" << coil_depth << ","
      << "\"coil_width_outer\":" << coil_width_outer << ","
      << "\"coil_width_inner\":" << coil_width_inner << ","
      << "\"insulation_thickness\":" << insulation_thickness << ","
      << "\"turns_per_coil\":" << turns_per_coil << ","
      << "\"coil_pitch\":" << coil_pitch << ","
      << "\"wire_diameter\":" << wire_diameter << ","
      << "\"slot_fill_factor\":" << slot_fill_factor << ","
      << "\"winding_type\":\"" << to_string(winding_type) << "\",";
    // Section 4
    o << "\"t_lam\":" << t_lam << ","
      << "\"n_lam\":" << n_lam << ","
      << "\"z_spacing\":" << z_spacing << ","
      << "\"insulation_coating_thickness\":" << insulation_coating_thickness << ","
      << "\"material\":\"" << to_string(material) << "\","
      << "\"material_file\":\"" << material_file << "\",";
    // Section 5
    o << "\"mesh_yoke\":" << mesh_yoke << ","
      << "\"mesh_slot\":" << mesh_slot << ","
      << "\"mesh_coil\":" << mesh_coil << ","
      << "\"mesh_ins\":" << mesh_ins << ","
      << "\"mesh_boundary_layers\":" << mesh_boundary_layers << ","
      << "\"mesh_curvature\":" << mesh_curvature << ","
      << "\"mesh_transition_layers\":" << mesh_transition_layers << ",";
    // Section 6 — derived
    o << "\"_derived\":{"
      << "\"yoke_height\":" << yoke_height << ","
      << "\"tooth_width\":" << tooth_width << ","
      << "\"slot_pitch\":" << slot_pitch << ","
      << "\"stack_length\":" << stack_length << ","
      << "\"fill_factor\":" << fill_factor
      << "}}";
    return o.str();
}

// ─── operator<< ──────────────────────────────────────────────────────────────

std::ostream& operator<<(std::ostream& os, const StatorParams& p) {
    os << "=== StatorParams ===\n"
       << "  Core Radii:\n"
       << "    R_outer       = " << p.R_outer       << " m\n"
       << "    R_inner       = " << p.R_inner       << " m\n"
       << "    airgap_length = " << p.airgap_length << " m\n"
       << "  Slot Geometry:\n"
       << "    n_slots            = " << p.n_slots            << "\n"
       << "    slot_depth         = " << p.slot_depth         << " m\n"
       << "    slot_width_outer   = " << p.slot_width_outer   << " m\n"
       << "    slot_width_inner   = " << p.slot_width_inner   << " m\n"
       << "    slot_opening       = " << p.slot_opening       << " m\n"
       << "    slot_opening_depth = " << p.slot_opening_depth << " m\n"
       << "    tooth_tip_angle    = " << p.tooth_tip_angle    << " rad\n"
       << "    slot_shape         = " << to_string(p.slot_shape) << "\n"
       << "  Coil/Winding:\n"
       << "    coil_depth           = " << p.coil_depth           << " m\n"
       << "    coil_width_outer     = " << p.coil_width_outer     << " m\n"
       << "    coil_width_inner     = " << p.coil_width_inner     << " m\n"
       << "    insulation_thickness = " << p.insulation_thickness << " m\n"
       << "    turns_per_coil       = " << p.turns_per_coil       << "\n"
       << "    coil_pitch           = " << p.coil_pitch           << " slots\n"
       << "    wire_diameter        = " << p.wire_diameter        << " m\n"
       << "    slot_fill_factor     = " << p.slot_fill_factor     << "\n"
       << "    winding_type         = " << to_string(p.winding_type) << "\n"
       << "  Lamination Stack:\n"
       << "    t_lam                        = " << p.t_lam                        << " m\n"
       << "    n_lam                        = " << p.n_lam                        << "\n"
       << "    z_spacing                    = " << p.z_spacing                    << " m\n"
       << "    insulation_coating_thickness = " << p.insulation_coating_thickness << " m\n"
       << "    material                     = " << to_string(p.material)          << "\n"
       << "    material_file                = " << p.material_file                << "\n"
       << "  Mesh Sizing:\n"
       << "    mesh_yoke              = " << p.mesh_yoke              << " m\n"
       << "    mesh_slot              = " << p.mesh_slot              << " m\n"
       << "    mesh_coil              = " << p.mesh_coil              << " m\n"
       << "    mesh_ins               = " << p.mesh_ins               << " m\n"
       << "    mesh_boundary_layers   = " << p.mesh_boundary_layers   << "\n"
       << "    mesh_curvature         = " << p.mesh_curvature         << "\n"
       << "    mesh_transition_layers = " << p.mesh_transition_layers << "\n"
       << "  Derived:\n"
       << "    yoke_height  = " << p.yoke_height  << " m\n"
       << "    tooth_width  = " << p.tooth_width  << " m\n"
       << "    slot_pitch   = " << p.slot_pitch   << " rad\n"
       << "    stack_length = " << p.stack_length << " m\n"
       << "    fill_factor  = " << p.fill_factor  << "\n";
    return os;
}

// ─── Factory functions ────────────────────────────────────────────────────────

StatorParams make_reference_params() {
    StatorParams p;
    p.R_outer              = 0.25;
    p.R_inner              = 0.15;
    p.airgap_length        = 0.001;
    p.n_slots              = 36;
    p.slot_depth           = 0.06;
    p.slot_width_outer     = 0.012;
    p.slot_width_inner     = 0.010;
    p.slot_opening         = 0.004;
    p.slot_opening_depth   = 0.003;
    p.tooth_tip_angle      = 0.1;
    p.slot_shape           = SlotShape::SEMI_CLOSED;
    p.coil_depth           = 0.050;
    p.coil_width_outer     = 0.008;
    p.coil_width_inner     = 0.007;
    p.insulation_thickness = 0.001;
    p.turns_per_coil       = 10;
    p.coil_pitch           = 5;
    p.wire_diameter        = 0.001;
    p.slot_fill_factor     = 0.45;
    p.winding_type         = WindingType::DOUBLE_LAYER;
    p.t_lam                        = 0.00035;
    p.n_lam                        = 200;
    p.z_spacing                    = 0.0;
    p.insulation_coating_thickness = 0.00005;
    p.material                     = LaminationMaterial::M270_35A;
    p.material_file                = "";
    p.mesh_yoke              = 0.006;
    p.mesh_slot              = 0.003;
    p.mesh_coil              = 0.0015;
    p.mesh_ins               = 0.0007;
    p.mesh_boundary_layers   = 3;
    p.mesh_curvature         = 0.3;
    p.mesh_transition_layers = 2;
    p.validate_and_derive();
    return p;
}

StatorParams make_minimal_params() {
    StatorParams p;
    p.R_outer              = 0.12;
    p.R_inner              = 0.07;
    p.airgap_length        = 0.001;
    p.n_slots              = 12;
    p.slot_depth           = 0.03;
    p.slot_width_outer     = 0.010;
    p.slot_width_inner     = 0.009;
    p.slot_opening         = 0.003;
    p.slot_opening_depth   = 0.002;
    p.tooth_tip_angle      = 0.0;
    p.slot_shape           = SlotShape::RECTANGULAR;
    p.coil_depth           = 0.025;
    p.coil_width_outer     = 0.007;
    p.coil_width_inner     = 0.006;
    p.insulation_thickness = 0.001;
    p.turns_per_coil       = 8;
    p.coil_pitch           = 3;
    p.wire_diameter        = 0.0012;
    p.slot_fill_factor     = 0.4;
    p.winding_type         = WindingType::SINGLE_LAYER;
    p.t_lam                        = 0.00035;
    p.n_lam                        = 100;
    p.z_spacing                    = 0.0;
    p.insulation_coating_thickness = 0.00005;
    p.material                     = LaminationMaterial::M330_50A;
    p.material_file                = "";
    p.mesh_yoke              = 0.005;
    p.mesh_slot              = 0.002;
    p.mesh_coil              = 0.001;
    p.mesh_ins               = 0.0005;
    p.mesh_boundary_layers   = 2;
    p.mesh_curvature         = 0.3;
    p.mesh_transition_layers = 2;
    p.validate_and_derive();
    return p;
}

} // namespace stator
