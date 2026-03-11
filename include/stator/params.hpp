#pragma once
#include <string>
#include <cstdint>
#include <ostream>

namespace stator {

// ─── Enumerations ─────────────────────────────────────────────────────────────

enum class SlotShape : uint8_t {
    RECTANGULAR  = 0,
    TRAPEZOIDAL  = 1,
    ROUND_BOTTOM = 2,
    SEMI_CLOSED  = 3
};

enum class WindingType : uint8_t {
    SINGLE_LAYER = 0,
    DOUBLE_LAYER = 1,
    CONCENTRATED = 2,
    DISTRIBUTED  = 3
};

enum class LaminationMaterial : uint8_t {
    M270_35A = 0,
    M330_50A = 1,
    M400_50A = 2,
    NO20     = 3,
    CUSTOM   = 4
};

const char* to_string(SlotShape s);
const char* to_string(WindingType w);
const char* to_string(LaminationMaterial m);

// ─── StatorParams ─────────────────────────────────────────────────────────────
// Plain-old-data struct; all SI units (metres, radians).
// Call validate_and_derive() after setting fields.

struct StatorParams {
    // ── Section 1: Core Radii & Air Gap ──────────────────────────────────────
    double R_outer       = 0.25;
    double R_inner       = 0.15;
    double airgap_length = 0.001;

    // ── Section 2: Slot Geometry ──────────────────────────────────────────────
    int       n_slots            = 36;
    double    slot_depth         = 0.06;
    double    slot_width_outer   = 0.012;
    double    slot_width_inner   = 0.010;
    double    slot_opening       = 0.004;
    double    slot_opening_depth = 0.003;
    double    tooth_tip_angle    = 0.1;    // rad
    SlotShape slot_shape         = SlotShape::SEMI_CLOSED;

    // ── Section 3: Coil / Winding ─────────────────────────────────────────────
    double      coil_depth                 = 0.05;
    double      coil_width_outer           = 0.008;
    double      coil_width_inner           = 0.007;
    double      insulation_thickness       = 0.001;
    int         turns_per_coil             = 10;
    int         coil_pitch                 = 5;
    double      wire_diameter              = 0.001;
    double      slot_fill_factor           = 0.45;
    WindingType winding_type               = WindingType::DOUBLE_LAYER;

    // ── Section 4: Lamination Stack ───────────────────────────────────────────
    double             t_lam                        = 0.00035;
    int                n_lam                        = 200;
    double             z_spacing                    = 0.0;
    double             insulation_coating_thickness = 0.00005;
    LaminationMaterial material                     = LaminationMaterial::M270_35A;
    std::string        material_file                = "";

    // ── Section 5: Mesh Sizing ────────────────────────────────────────────────
    double mesh_yoke              = 0.006;
    double mesh_slot              = 0.003;
    double mesh_coil              = 0.0015;
    double mesh_ins               = 0.0007;
    int    mesh_boundary_layers   = 3;
    double mesh_curvature         = 0.3;
    int    mesh_transition_layers = 2;

    // ── Section 6: Derived (read-only; set by validate_and_derive) ────────────
    double yoke_height  = 0.0;  // R_outer - R_inner - slot_depth
    double tooth_width  = 0.0;  // R_inner * slot_pitch - slot_width_inner
    double slot_pitch   = 0.0;  // 2π / n_slots
    double stack_length = 0.0;  // n_lam*t_lam + (n_lam-1)*z_spacing
    double fill_factor  = 0.0;  // coil_area / slot_area

    // Validate all fields and compute derived quantities.
    // Throws std::invalid_argument or std::logic_error with descriptive messages.
    void validate_and_derive();

    // Single-line valid JSON with all user-settable fields + "_derived" sub-object.
    std::string to_json() const;

    friend std::ostream& operator<<(std::ostream& os, const StatorParams& p);
};

// Validated 36-slot reference design (used in tests).
StatorParams make_reference_params();

// Minimal 12-slot design (another test baseline).
StatorParams make_minimal_params();

} // namespace stator
