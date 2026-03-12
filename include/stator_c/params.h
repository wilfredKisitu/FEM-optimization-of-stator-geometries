#ifndef STATOR_C_PARAMS_H
#define STATOR_C_PARAMS_H

#include "stator_c/common.h"

/* ── Enumerations ────────────────────────────────────────────────────────── */

typedef enum {
    SLOT_SHAPE_RECTANGULAR  = 0,
    SLOT_SHAPE_TRAPEZOIDAL  = 1,
    SLOT_SHAPE_ROUND_BOTTOM = 2,
    SLOT_SHAPE_SEMI_CLOSED  = 3
} SlotShape;

typedef enum {
    WINDING_SINGLE_LAYER = 0,
    WINDING_DOUBLE_LAYER = 1,
    WINDING_CONCENTRATED = 2,
    WINDING_DISTRIBUTED  = 3
} WindingType;

typedef enum {
    MATERIAL_M270_35A = 0,
    MATERIAL_M330_50A = 1,
    MATERIAL_M400_50A = 2,
    MATERIAL_NO20     = 3,
    MATERIAL_CUSTOM   = 4
} LaminationMaterial;

const char* stator_slot_shape_to_str(SlotShape s);
const char* stator_winding_type_to_str(WindingType w);
const char* stator_material_to_str(LaminationMaterial m);

/* ── StatorParams ────────────────────────────────────────────────────────── */

typedef struct {
    /* Section 1: Core Radii & Air Gap */
    double R_outer;
    double R_inner;
    double airgap_length;

    /* Section 2: Slot Geometry */
    int       n_slots;
    double    slot_depth;
    double    slot_width_outer;
    double    slot_width_inner;
    double    slot_opening;
    double    slot_opening_depth;
    double    tooth_tip_angle;
    SlotShape slot_shape;

    /* Section 3: Coil / Winding */
    double      coil_depth;
    double      coil_width_outer;
    double      coil_width_inner;
    double      insulation_thickness;
    int         turns_per_coil;
    int         coil_pitch;
    double      wire_diameter;
    double      slot_fill_factor;
    WindingType winding_type;

    /* Section 4: Lamination Stack */
    double             t_lam;
    int                n_lam;
    double             z_spacing;
    double             insulation_coating_thickness;
    LaminationMaterial material;
    char               material_file[256];

    /* Section 5: Mesh Sizing */
    double mesh_yoke;
    double mesh_slot;
    double mesh_coil;
    double mesh_ins;
    int    mesh_boundary_layers;
    double mesh_curvature;
    int    mesh_transition_layers;

    /* Section 6: Derived (read-only; set by validate_and_derive) */
    double yoke_height;
    double tooth_width;
    double slot_pitch;
    double stack_length;
    double fill_factor;
} StatorParams;

/* Initialise with defaults (equivalent to C++ default member init) */
void stator_params_init(StatorParams* p);

/*
 * Validate all fields and compute derived quantities.
 * Returns STATOR_OK on success; writes error message to err_buf if non-NULL.
 */
int stator_params_validate_and_derive(StatorParams* p,
                                       char* err_buf, size_t err_len);

/*
 * Serialise to single-line JSON.
 * Writes into buf (must be at least 4096 bytes for safety).
 * Returns STATOR_OK or STATOR_ERR_NOMEM if buf too small.
 */
int stator_params_to_json(const StatorParams* p, char* buf, size_t buf_len);

/* Print human-readable representation to stdout */
void stator_params_print(const StatorParams* p);

/* Factory functions — validate_and_derive is called inside */
int stator_make_reference_params(StatorParams* out, char* err_buf, size_t err_len);
int stator_make_minimal_params  (StatorParams* out, char* err_buf, size_t err_len);

#endif /* STATOR_C_PARAMS_H */
