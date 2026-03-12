#include "stator_c/params.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define PI 3.14159265358979323846

/* ── to_string helpers ──────────────────────────────────────────────────── */

const char* stator_slot_shape_to_str(SlotShape s) {
    switch (s) {
        case SLOT_SHAPE_RECTANGULAR:  return "RECTANGULAR";
        case SLOT_SHAPE_TRAPEZOIDAL:  return "TRAPEZOIDAL";
        case SLOT_SHAPE_ROUND_BOTTOM: return "ROUND_BOTTOM";
        case SLOT_SHAPE_SEMI_CLOSED:  return "SEMI_CLOSED";
        default:                       return "UNKNOWN";
    }
}

const char* stator_winding_type_to_str(WindingType w) {
    switch (w) {
        case WINDING_SINGLE_LAYER: return "SINGLE_LAYER";
        case WINDING_DOUBLE_LAYER: return "DOUBLE_LAYER";
        case WINDING_CONCENTRATED: return "CONCENTRATED";
        case WINDING_DISTRIBUTED:  return "DISTRIBUTED";
        default:                    return "UNKNOWN";
    }
}

const char* stator_material_to_str(LaminationMaterial m) {
    switch (m) {
        case MATERIAL_M270_35A: return "M270_35A";
        case MATERIAL_M330_50A: return "M330_50A";
        case MATERIAL_M400_50A: return "M400_50A";
        case MATERIAL_NO20:     return "NO20";
        case MATERIAL_CUSTOM:   return "CUSTOM";
        default:                 return "UNKNOWN";
    }
}

/* ── stator_params_init ─────────────────────────────────────────────────── */

void stator_params_init(StatorParams* p) {
    memset(p, 0, sizeof(*p));
    /* Section 1 */
    p->R_outer       = 0.25;
    p->R_inner       = 0.15;
    p->airgap_length = 0.001;
    /* Section 2 */
    p->n_slots            = 36;
    p->slot_depth         = 0.06;
    p->slot_width_outer   = 0.012;
    p->slot_width_inner   = 0.010;
    p->slot_opening       = 0.004;
    p->slot_opening_depth = 0.003;
    p->tooth_tip_angle    = 0.1;
    p->slot_shape         = SLOT_SHAPE_SEMI_CLOSED;
    /* Section 3 */
    p->coil_depth           = 0.05;
    p->coil_width_outer     = 0.008;
    p->coil_width_inner     = 0.007;
    p->insulation_thickness = 0.001;
    p->turns_per_coil       = 10;
    p->coil_pitch           = 5;
    p->wire_diameter        = 0.001;
    p->slot_fill_factor     = 0.45;
    p->winding_type         = WINDING_DOUBLE_LAYER;
    /* Section 4 */
    p->t_lam                        = 0.00035;
    p->n_lam                        = 200;
    p->z_spacing                    = 0.0;
    p->insulation_coating_thickness = 0.00005;
    p->material                     = MATERIAL_M270_35A;
    p->material_file[0]             = '\0';
    /* Section 5 */
    p->mesh_yoke              = 0.006;
    p->mesh_slot              = 0.003;
    p->mesh_coil              = 0.0015;
    p->mesh_ins               = 0.0007;
    p->mesh_boundary_layers   = 3;
    p->mesh_curvature         = 0.3;
    p->mesh_transition_layers = 2;
    /* Section 6 — derived */
    p->yoke_height  = 0.0;
    p->tooth_width  = 0.0;
    p->slot_pitch   = 0.0;
    p->stack_length = 0.0;
    p->fill_factor  = 0.0;
}

/* ── stator_params_validate_and_derive ──────────────────────────────────── */

int stator_params_validate_and_derive(StatorParams* p,
                                       char* err_buf, size_t err_len) {
    /* Rule 1: All double dimensions > 0 */
#define CHECK_POS(field) \
    if (p->field <= 0.0) { \
        STATOR_SET_ERR(err_buf, err_len, #field " must be > 0"); \
        return STATOR_ERR_INVAL; \
    }
    CHECK_POS(R_outer)
    CHECK_POS(R_inner)
    CHECK_POS(airgap_length)
    CHECK_POS(slot_depth)
    CHECK_POS(slot_width_outer)
    CHECK_POS(slot_width_inner)
    CHECK_POS(coil_depth)
    CHECK_POS(coil_width_outer)
    CHECK_POS(coil_width_inner)
    CHECK_POS(insulation_thickness)
    CHECK_POS(wire_diameter)
    CHECK_POS(t_lam)
    CHECK_POS(mesh_yoke)
    CHECK_POS(mesh_slot)
    CHECK_POS(mesh_coil)
    CHECK_POS(mesh_ins)
#undef CHECK_POS

    /* Rule 2: R_inner < R_outer */
    if (p->R_inner >= p->R_outer) {
        STATOR_SET_ERR(err_buf, err_len, "R_inner must be < R_outer");
        return STATOR_ERR_INVAL;
    }

    /* Rule 3: slot_depth < (R_outer - R_inner) */
    if (p->slot_depth >= (p->R_outer - p->R_inner)) {
        STATOR_SET_ERR(err_buf, err_len,
                       "slot_depth must be < (R_outer - R_inner)");
        return STATOR_ERR_INVAL;
    }

    /* Rule 4: n_slots >= 6 and even */
    if (p->n_slots < 6) {
        STATOR_SET_ERR(err_buf, err_len, "n_slots must be >= 6");
        return STATOR_ERR_INVAL;
    }
    if (p->n_slots % 2 != 0) {
        STATOR_SET_ERR(err_buf, err_len, "n_slots must be even");
        return STATOR_ERR_INVAL;
    }

    /* Rule 5: slot_width_inner < R_inner * 2π/n_slots */
    double pitch = 2.0 * PI / (double)p->n_slots;
    if (p->slot_width_inner >= p->R_inner * pitch) {
        STATOR_SET_ERR(err_buf, err_len,
                       "slot_width_inner must be < R_inner * 2pi/n_slots");
        return STATOR_ERR_INVAL;
    }

    /* Rule 6: SEMI_CLOSED checks */
    if (p->slot_shape == SLOT_SHAPE_SEMI_CLOSED) {
        if (p->slot_opening >= p->slot_width_inner) {
            STATOR_SET_ERR(err_buf, err_len,
                           "SEMI_CLOSED: slot_opening must be < slot_width_inner");
            return STATOR_ERR_INVAL;
        }
        if (p->slot_opening_depth >= p->slot_depth) {
            STATOR_SET_ERR(err_buf, err_len,
                           "SEMI_CLOSED: slot_opening_depth must be < slot_depth");
            return STATOR_ERR_INVAL;
        }
    }

    /* Rule 7: coil fits inside slot */
    double max_coil_depth = p->slot_depth - p->slot_opening_depth
                          - 2.0 * p->insulation_thickness;
    if (p->coil_depth > max_coil_depth) {
        STATOR_SET_ERR(err_buf, err_len,
                       "coil_depth exceeds available slot space");
        return STATOR_ERR_INVAL;
    }

    /* Rule 8: coil width fits inside slot */
    double max_coil_width = p->slot_width_inner - 2.0 * p->insulation_thickness;
    if (p->coil_width_inner > max_coil_width) {
        STATOR_SET_ERR(err_buf, err_len,
                       "coil_width_inner exceeds slot_width_inner - 2*insulation_thickness");
        return STATOR_ERR_INVAL;
    }

    /* Rule 9: n_lam > 0 */
    if (p->n_lam <= 0) {
        STATOR_SET_ERR(err_buf, err_len, "n_lam must be > 0");
        return STATOR_ERR_INVAL;
    }

    /* Rule 10: z_spacing >= 0 */
    if (p->z_spacing < 0.0) {
        STATOR_SET_ERR(err_buf, err_len, "z_spacing must be >= 0");
        return STATOR_ERR_INVAL;
    }

    /* Rule 11: insulation_coating_thickness >= 0 */
    if (p->insulation_coating_thickness < 0.0) {
        STATOR_SET_ERR(err_buf, err_len,
                       "insulation_coating_thickness must be >= 0");
        return STATOR_ERR_INVAL;
    }

    /* Rule 12: CUSTOM material requires material_file */
    if (p->material == MATERIAL_CUSTOM && p->material_file[0] == '\0') {
        STATOR_SET_ERR(err_buf, err_len,
                       "material == CUSTOM requires a non-empty material_file");
        return STATOR_ERR_INVAL;
    }

    /* Rule 13: Mesh size ordering (finest to coarsest) */
    if (p->mesh_ins > p->mesh_coil) {
        STATOR_SET_ERR(err_buf, err_len, "mesh_ins must be <= mesh_coil");
        return STATOR_ERR_INVAL;
    }
    if (p->mesh_coil > p->mesh_slot) {
        STATOR_SET_ERR(err_buf, err_len, "mesh_coil must be <= mesh_slot");
        return STATOR_ERR_INVAL;
    }
    if (p->mesh_slot > p->mesh_yoke) {
        STATOR_SET_ERR(err_buf, err_len, "mesh_slot must be <= mesh_yoke");
        return STATOR_ERR_INVAL;
    }

    /* Rule 14: tooth_tip_angle */
    if (p->tooth_tip_angle < 0.0) {
        STATOR_SET_ERR(err_buf, err_len, "tooth_tip_angle must be >= 0");
        return STATOR_ERR_INVAL;
    }
    if (p->tooth_tip_angle >= PI / 4.0) {
        STATOR_SET_ERR(err_buf, err_len, "tooth_tip_angle must be < pi/4");
        return STATOR_ERR_INVAL;
    }

    /* Compute derived quantities */
    p->slot_pitch   = pitch;
    p->yoke_height  = p->R_outer - p->R_inner - p->slot_depth;
    p->tooth_width  = p->R_inner * p->slot_pitch - p->slot_width_inner;
    p->stack_length = (double)p->n_lam * p->t_lam
                    + (double)(p->n_lam - 1) * p->z_spacing;

    double slot_area = 0.5 * (p->slot_width_inner + p->slot_width_outer)
                     * p->slot_depth;
    double coil_area = 0.5 * (p->coil_width_inner + p->coil_width_outer)
                     * p->coil_depth;
    p->fill_factor = (slot_area > 0.0) ? coil_area / slot_area : 0.0;

    /* Rule 15: fill_factor in (0, 1) */
    if (p->fill_factor <= 0.0 || p->fill_factor >= 1.0) {
        STATOR_SET_ERR(err_buf, err_len,
                       "computed fill_factor is not in (0, 1)");
        return STATOR_ERR_LOGIC;
    }

    return STATOR_OK;
}

/* ── stator_params_to_json ──────────────────────────────────────────────── */

int stator_params_to_json(const StatorParams* p, char* buf, size_t buf_len) {
    int n = snprintf(buf, buf_len,
        "{"
        "\"R_outer\":%.17g,"
        "\"R_inner\":%.17g,"
        "\"airgap_length\":%.17g,"
        "\"n_slots\":%d,"
        "\"slot_depth\":%.17g,"
        "\"slot_width_outer\":%.17g,"
        "\"slot_width_inner\":%.17g,"
        "\"slot_opening\":%.17g,"
        "\"slot_opening_depth\":%.17g,"
        "\"tooth_tip_angle\":%.17g,"
        "\"slot_shape\":\"%s\","
        "\"coil_depth\":%.17g,"
        "\"coil_width_outer\":%.17g,"
        "\"coil_width_inner\":%.17g,"
        "\"insulation_thickness\":%.17g,"
        "\"turns_per_coil\":%d,"
        "\"coil_pitch\":%d,"
        "\"wire_diameter\":%.17g,"
        "\"slot_fill_factor\":%.17g,"
        "\"winding_type\":\"%s\","
        "\"t_lam\":%.17g,"
        "\"n_lam\":%d,"
        "\"z_spacing\":%.17g,"
        "\"insulation_coating_thickness\":%.17g,"
        "\"material\":\"%s\","
        "\"material_file\":\"%s\","
        "\"mesh_yoke\":%.17g,"
        "\"mesh_slot\":%.17g,"
        "\"mesh_coil\":%.17g,"
        "\"mesh_ins\":%.17g,"
        "\"mesh_boundary_layers\":%d,"
        "\"mesh_curvature\":%.17g,"
        "\"mesh_transition_layers\":%d,"
        "\"_derived\":{"
        "\"yoke_height\":%.17g,"
        "\"tooth_width\":%.17g,"
        "\"slot_pitch\":%.17g,"
        "\"stack_length\":%.17g,"
        "\"fill_factor\":%.17g"
        "}}",
        p->R_outer, p->R_inner, p->airgap_length,
        p->n_slots, p->slot_depth,
        p->slot_width_outer, p->slot_width_inner,
        p->slot_opening, p->slot_opening_depth, p->tooth_tip_angle,
        stator_slot_shape_to_str(p->slot_shape),
        p->coil_depth, p->coil_width_outer, p->coil_width_inner,
        p->insulation_thickness, p->turns_per_coil, p->coil_pitch,
        p->wire_diameter, p->slot_fill_factor,
        stator_winding_type_to_str(p->winding_type),
        p->t_lam, p->n_lam, p->z_spacing,
        p->insulation_coating_thickness,
        stator_material_to_str(p->material), p->material_file,
        p->mesh_yoke, p->mesh_slot, p->mesh_coil, p->mesh_ins,
        p->mesh_boundary_layers, p->mesh_curvature, p->mesh_transition_layers,
        p->yoke_height, p->tooth_width, p->slot_pitch,
        p->stack_length, p->fill_factor);
    if (n < 0 || (size_t)n >= buf_len) return STATOR_ERR_NOMEM;
    return STATOR_OK;
}

/* ── stator_params_print ────────────────────────────────────────────────── */

void stator_params_print(const StatorParams* p) {
    printf("=== StatorParams ===\n");
    printf("  Core Radii:\n");
    printf("    R_outer       = %g m\n", p->R_outer);
    printf("    R_inner       = %g m\n", p->R_inner);
    printf("    airgap_length = %g m\n", p->airgap_length);
    printf("  Slot Geometry:\n");
    printf("    n_slots            = %d\n",  p->n_slots);
    printf("    slot_depth         = %g m\n", p->slot_depth);
    printf("    slot_width_outer   = %g m\n", p->slot_width_outer);
    printf("    slot_width_inner   = %g m\n", p->slot_width_inner);
    printf("    slot_opening       = %g m\n", p->slot_opening);
    printf("    slot_opening_depth = %g m\n", p->slot_opening_depth);
    printf("    tooth_tip_angle    = %g rad\n", p->tooth_tip_angle);
    printf("    slot_shape         = %s\n", stator_slot_shape_to_str(p->slot_shape));
    printf("  Coil/Winding:\n");
    printf("    coil_depth           = %g m\n", p->coil_depth);
    printf("    coil_width_outer     = %g m\n", p->coil_width_outer);
    printf("    coil_width_inner     = %g m\n", p->coil_width_inner);
    printf("    insulation_thickness = %g m\n", p->insulation_thickness);
    printf("    turns_per_coil       = %d\n",   p->turns_per_coil);
    printf("    coil_pitch           = %d slots\n", p->coil_pitch);
    printf("    wire_diameter        = %g m\n", p->wire_diameter);
    printf("    slot_fill_factor     = %g\n",   p->slot_fill_factor);
    printf("    winding_type         = %s\n", stator_winding_type_to_str(p->winding_type));
    printf("  Lamination Stack:\n");
    printf("    t_lam                        = %g m\n", p->t_lam);
    printf("    n_lam                        = %d\n",   p->n_lam);
    printf("    z_spacing                    = %g m\n", p->z_spacing);
    printf("    insulation_coating_thickness = %g m\n", p->insulation_coating_thickness);
    printf("    material                     = %s\n", stator_material_to_str(p->material));
    printf("    material_file                = %s\n", p->material_file);
    printf("  Mesh Sizing:\n");
    printf("    mesh_yoke              = %g m\n", p->mesh_yoke);
    printf("    mesh_slot              = %g m\n", p->mesh_slot);
    printf("    mesh_coil              = %g m\n", p->mesh_coil);
    printf("    mesh_ins               = %g m\n", p->mesh_ins);
    printf("    mesh_boundary_layers   = %d\n",   p->mesh_boundary_layers);
    printf("    mesh_curvature         = %g\n",   p->mesh_curvature);
    printf("    mesh_transition_layers = %d\n",   p->mesh_transition_layers);
    printf("  Derived:\n");
    printf("    yoke_height  = %g m\n",  p->yoke_height);
    printf("    tooth_width  = %g m\n",  p->tooth_width);
    printf("    slot_pitch   = %g rad\n", p->slot_pitch);
    printf("    stack_length = %g m\n",  p->stack_length);
    printf("    fill_factor  = %g\n",    p->fill_factor);
}

/* ── Factory functions ──────────────────────────────────────────────────── */

int stator_make_reference_params(StatorParams* p,
                                  char* err_buf, size_t err_len) {
    memset(p, 0, sizeof(*p));
    p->R_outer              = 0.25;
    p->R_inner              = 0.15;
    p->airgap_length        = 0.001;
    p->n_slots              = 36;
    p->slot_depth           = 0.06;
    p->slot_width_outer     = 0.012;
    p->slot_width_inner     = 0.010;
    p->slot_opening         = 0.004;
    p->slot_opening_depth   = 0.003;
    p->tooth_tip_angle      = 0.1;
    p->slot_shape           = SLOT_SHAPE_SEMI_CLOSED;
    p->coil_depth           = 0.050;
    p->coil_width_outer     = 0.008;
    p->coil_width_inner     = 0.007;
    p->insulation_thickness = 0.001;
    p->turns_per_coil       = 10;
    p->coil_pitch           = 5;
    p->wire_diameter        = 0.001;
    p->slot_fill_factor     = 0.45;
    p->winding_type         = WINDING_DOUBLE_LAYER;
    p->t_lam                        = 0.00035;
    p->n_lam                        = 200;
    p->z_spacing                    = 0.0;
    p->insulation_coating_thickness = 0.00005;
    p->material                     = MATERIAL_M270_35A;
    p->material_file[0]             = '\0';
    p->mesh_yoke              = 0.006;
    p->mesh_slot              = 0.003;
    p->mesh_coil              = 0.0015;
    p->mesh_ins               = 0.0007;
    p->mesh_boundary_layers   = 3;
    p->mesh_curvature         = 0.3;
    p->mesh_transition_layers = 2;
    return stator_params_validate_and_derive(p, err_buf, err_len);
}

int stator_make_minimal_params(StatorParams* p,
                                char* err_buf, size_t err_len) {
    memset(p, 0, sizeof(*p));
    p->R_outer              = 0.12;
    p->R_inner              = 0.07;
    p->airgap_length        = 0.001;
    p->n_slots              = 12;
    p->slot_depth           = 0.03;
    p->slot_width_outer     = 0.010;
    p->slot_width_inner     = 0.009;
    p->slot_opening         = 0.003;
    p->slot_opening_depth   = 0.002;
    p->tooth_tip_angle      = 0.0;
    p->slot_shape           = SLOT_SHAPE_RECTANGULAR;
    p->coil_depth           = 0.025;
    p->coil_width_outer     = 0.007;
    p->coil_width_inner     = 0.006;
    p->insulation_thickness = 0.001;
    p->turns_per_coil       = 8;
    p->coil_pitch           = 3;
    p->wire_diameter        = 0.0012;
    p->slot_fill_factor     = 0.4;
    p->winding_type         = WINDING_SINGLE_LAYER;
    p->t_lam                        = 0.00035;
    p->n_lam                        = 100;
    p->z_spacing                    = 0.0;
    p->insulation_coating_thickness = 0.00005;
    p->material                     = MATERIAL_M330_50A;
    p->material_file[0]             = '\0';
    p->mesh_yoke              = 0.005;
    p->mesh_slot              = 0.002;
    p->mesh_coil              = 0.001;
    p->mesh_ins               = 0.0005;
    p->mesh_boundary_layers   = 2;
    p->mesh_curvature         = 0.3;
    p->mesh_transition_layers = 2;
    return stator_params_validate_and_derive(p, err_buf, err_len);
}
