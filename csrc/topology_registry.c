#include "stator_c/topology_registry.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── stator_region_to_str ─────────────────────────────────────────────── */

const char* stator_region_to_str(RegionType r) {
    switch (r) {
        case REGION_YOKE:           return "YOKE";
        case REGION_TOOTH:          return "TOOTH";
        case REGION_SLOT_AIR:       return "SLOT_AIR";
        case REGION_SLOT_INS:       return "SLOT_INS";
        case REGION_COIL_A_POS:     return "COIL_A_POS";
        case REGION_COIL_A_NEG:     return "COIL_A_NEG";
        case REGION_COIL_B_POS:     return "COIL_B_POS";
        case REGION_COIL_B_NEG:     return "COIL_B_NEG";
        case REGION_COIL_C_POS:     return "COIL_C_POS";
        case REGION_COIL_C_NEG:     return "COIL_C_NEG";
        case REGION_BORE_AIR:       return "BORE_AIR";
        case REGION_BOUNDARY_BORE:  return "BOUNDARY_BORE";
        case REGION_BOUNDARY_OUTER: return "BOUNDARY_OUTER";
        default:                     return "UNKNOWN";
    }
}

/* ── init / destroy ─────────────────────────────────────────────────────── */

int stator_topo_registry_init(TopologyRegistry* reg, int n_slots,
                                 char* err_buf, size_t err_len) {
    if (n_slots <= 0) {
        STATOR_SET_ERR(err_buf, err_len, "TopologyRegistry: n_slots must be > 0");
        return STATOR_ERR_INVAL;
    }
    memset(reg, 0, sizeof(*reg));
    reg->n_slots = n_slots;
    pthread_rwlock_init(&reg->lock, NULL);

    reg->slot_upper_tags = (int*)malloc((size_t)n_slots * sizeof(int));
    reg->slot_lower_tags = (int*)malloc((size_t)n_slots * sizeof(int));
    reg->winding_assignments =
        (SlotWindingAssignment*)malloc((size_t)n_slots * sizeof(SlotWindingAssignment));

    if (!reg->slot_upper_tags || !reg->slot_lower_tags || !reg->winding_assignments) {
        free(reg->slot_upper_tags);
        free(reg->slot_lower_tags);
        free(reg->winding_assignments);
        STATOR_SET_ERR(err_buf, err_len, "TopologyRegistry: malloc failed");
        return STATOR_ERR_NOMEM;
    }
    for (int i = 0; i < n_slots; i++) {
        reg->slot_upper_tags[i] = -1;
        reg->slot_lower_tags[i] = -1;
    }
    reg->winding_assigned = false;
    return STATOR_OK;
}

void stator_topo_registry_destroy(TopologyRegistry* reg) {
    pthread_rwlock_destroy(&reg->lock);
    free(reg->slot_upper_tags);
    free(reg->slot_lower_tags);
    free(reg->winding_assignments);
    reg->slot_upper_tags      = NULL;
    reg->slot_lower_tags      = NULL;
    reg->winding_assignments  = NULL;
}

/* ── Registration ──────────────────────────────────────────────────────── */

int stator_topo_register_surface(TopologyRegistry* reg, RegionType type,
                                   int gmsh_tag, int slot_idx) {
    (void)slot_idx;
    pthread_rwlock_wrlock(&reg->lock);
    if (reg->n_surface_records >= STATOR_MAX_SURFACE_RECORDS) {
        pthread_rwlock_unlock(&reg->lock);
        return STATOR_ERR_RANGE;
    }
    reg->surface_records[reg->n_surface_records].type     = type;
    reg->surface_records[reg->n_surface_records].gmsh_tag = gmsh_tag;
    reg->n_surface_records++;
    pthread_rwlock_unlock(&reg->lock);
    return STATOR_OK;
}

int stator_topo_register_slot_coil(TopologyRegistry* reg, int slot_idx,
                                     int upper_tag, int lower_tag) {
    pthread_rwlock_wrlock(&reg->lock);
    if (slot_idx < 0 || slot_idx >= reg->n_slots) {
        pthread_rwlock_unlock(&reg->lock);
        return STATOR_ERR_RANGE;
    }
    reg->slot_upper_tags[slot_idx] = upper_tag;
    reg->slot_lower_tags[slot_idx] = lower_tag;
    pthread_rwlock_unlock(&reg->lock);
    return STATOR_OK;
}

int stator_topo_register_boundary_curve(TopologyRegistry* reg, RegionType type,
                                          int gmsh_curve,
                                          char* err_buf, size_t err_len) {
    if (type != REGION_BOUNDARY_BORE && type != REGION_BOUNDARY_OUTER) {
        STATOR_SET_ERR(err_buf, err_len,
                       "register_boundary_curve: type must be BOUNDARY_BORE or BOUNDARY_OUTER");
        return STATOR_ERR_INVAL;
    }
    pthread_rwlock_wrlock(&reg->lock);
    if (reg->n_boundary_records >= STATOR_MAX_BOUNDARY_RECORDS) {
        pthread_rwlock_unlock(&reg->lock);
        return STATOR_ERR_RANGE;
    }
    reg->boundary_records[reg->n_boundary_records].type       = type;
    reg->boundary_records[reg->n_boundary_records].gmsh_curve = gmsh_curve;
    reg->n_boundary_records++;
    pthread_rwlock_unlock(&reg->lock);
    return STATOR_OK;
}

/* ── phase_for_slot ─────────────────────────────────────────────────────── */

static RegionType phase_for_slot(int slot_idx, WindingType wt) {
    static const RegionType distributed[6] = {
        REGION_COIL_A_POS, REGION_COIL_B_NEG,
        REGION_COIL_C_POS, REGION_COIL_A_NEG,
        REGION_COIL_B_POS, REGION_COIL_C_NEG
    };
    static const RegionType concentrated[6] = {
        REGION_COIL_A_POS, REGION_COIL_A_NEG,
        REGION_COIL_B_POS, REGION_COIL_B_NEG,
        REGION_COIL_C_POS, REGION_COIL_C_NEG
    };
    int r = slot_idx % 6;
    if (wt == WINDING_CONCENTRATED) return concentrated[r];
    return distributed[r];
}

/* ── assign_winding_layout ──────────────────────────────────────────────── */

int stator_topo_assign_winding_layout(TopologyRegistry* reg, WindingType wt,
                                        char* err_buf, size_t err_len) {
    pthread_rwlock_wrlock(&reg->lock);
    bool any = false;
    for (int i = 0; i < reg->n_slots; i++) {
        if (reg->slot_upper_tags[i] >= 0) { any = true; break; }
    }
    if (!any) {
        pthread_rwlock_unlock(&reg->lock);
        STATOR_SET_ERR(err_buf, err_len,
                       "assign_winding_layout: no coils registered");
        return STATOR_ERR_LOGIC;
    }
    for (int i = 0; i < reg->n_slots; i++) {
        SlotWindingAssignment* a = &reg->winding_assignments[i];
        a->slot_idx    = i;
        a->upper_tag   = reg->slot_upper_tags[i];
        a->lower_tag   = reg->slot_lower_tags[i];
        a->upper_phase = phase_for_slot(i, wt);
        a->lower_phase = (reg->slot_lower_tags[i] >= 0)
                         ? phase_for_slot(i, wt)
                         : REGION_UNKNOWN;
    }
    reg->winding_assigned = true;
    pthread_rwlock_unlock(&reg->lock);
    return STATOR_OK;
}

/* ── Queries ─────────────────────────────────────────────────────────────── */

int stator_topo_get_surfaces(const TopologyRegistry* reg, RegionType type,
                               int* out_tags, int out_cap) {
    pthread_rwlock_rdlock((pthread_rwlock_t*)&reg->lock);
    int n = 0;
    for (int i = 0; i < reg->n_surface_records && n < out_cap; i++) {
        if (reg->surface_records[i].type == type)
            out_tags[n++] = reg->surface_records[i].gmsh_tag;
    }
    pthread_rwlock_unlock((pthread_rwlock_t*)&reg->lock);
    return n;
}

int stator_topo_get_boundary_curves(const TopologyRegistry* reg, RegionType type,
                                       int* out_curves, int out_cap) {
    pthread_rwlock_rdlock((pthread_rwlock_t*)&reg->lock);
    int n = 0;
    for (int i = 0; i < reg->n_boundary_records && n < out_cap; i++) {
        if (reg->boundary_records[i].type == type)
            out_curves[n++] = reg->boundary_records[i].gmsh_curve;
    }
    pthread_rwlock_unlock((pthread_rwlock_t*)&reg->lock);
    return n;
}

const SlotWindingAssignment* stator_topo_get_slot_assignment(
    const TopologyRegistry* reg, int slot_idx,
    char* err_buf, size_t err_len) {
    if (!reg->winding_assigned) {
        STATOR_SET_ERR(err_buf, err_len, "get_slot_assignment: winding not yet assigned");
        return NULL;
    }
    if (slot_idx < 0 || slot_idx >= reg->n_slots) {
        STATOR_SET_ERR(err_buf, err_len, "get_slot_assignment: slot_idx out of range");
        return NULL;
    }
    return &reg->winding_assignments[slot_idx];
}

int stator_topo_total_surfaces(const TopologyRegistry* reg) {
    pthread_rwlock_rdlock((pthread_rwlock_t*)&reg->lock);
    int n = reg->n_surface_records;
    pthread_rwlock_unlock((pthread_rwlock_t*)&reg->lock);
    return n;
}

bool stator_topo_winding_assigned(const TopologyRegistry* reg) {
    return reg->winding_assigned;
}
