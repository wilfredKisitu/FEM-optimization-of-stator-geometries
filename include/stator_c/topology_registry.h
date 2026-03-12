#ifndef STATOR_C_TOPOLOGY_REGISTRY_H
#define STATOR_C_TOPOLOGY_REGISTRY_H

#include "stator_c/params.h"
#include "stator_c/common.h"
#include <pthread.h>

/* ── RegionType ──────────────────────────────────────────────────────────── */

typedef enum {
    REGION_YOKE          = 100,
    REGION_TOOTH         = 101,
    REGION_SLOT_AIR      = 200,
    REGION_SLOT_INS      = 201,
    REGION_COIL_A_POS    = 301,
    REGION_COIL_A_NEG    = 302,
    REGION_COIL_B_POS    = 303,
    REGION_COIL_B_NEG    = 304,
    REGION_COIL_C_POS    = 305,
    REGION_COIL_C_NEG    = 306,
    REGION_BORE_AIR      = 400,
    REGION_BOUNDARY_BORE = 500,
    REGION_BOUNDARY_OUTER= 501,
    REGION_UNKNOWN       = -1
} RegionType;

static inline int region_canonical_tag(RegionType r) { return (int)r; }
const char* stator_region_to_str(RegionType r);

/* ── SlotWindingAssignment ───────────────────────────────────────────────── */

typedef struct {
    int        slot_idx;
    RegionType upper_phase;
    RegionType lower_phase;
    int        upper_tag;
    int        lower_tag;
} SlotWindingAssignment;

/* ── SurfaceRecord / BoundaryRecord ─────────────────────────────────────── */

#define STATOR_MAX_SURFACE_RECORDS 16384
#define STATOR_MAX_BOUNDARY_RECORDS 64

typedef struct { RegionType type; int gmsh_tag; } SurfaceRecord;
typedef struct { RegionType type; int gmsh_curve; } BoundaryRecord;

/* ── TopologyRegistry ────────────────────────────────────────────────────── */

typedef struct {
    int n_slots;

    pthread_rwlock_t lock;

    SurfaceRecord  surface_records[STATOR_MAX_SURFACE_RECORDS];
    int            n_surface_records;

    BoundaryRecord boundary_records[STATOR_MAX_BOUNDARY_RECORDS];
    int            n_boundary_records;

    int* slot_upper_tags;   /* heap array of size n_slots */
    int* slot_lower_tags;   /* heap array of size n_slots */

    SlotWindingAssignment* winding_assignments; /* heap array of size n_slots */
    bool winding_assigned;
} TopologyRegistry;

/* Allocate / free */
int  stator_topo_registry_init(TopologyRegistry* reg, int n_slots,
                                 char* err_buf, size_t err_len);
void stator_topo_registry_destroy(TopologyRegistry* reg);

/* Registration */
int stator_topo_register_surface(TopologyRegistry* reg, RegionType type,
                                   int gmsh_tag, int slot_idx);
int stator_topo_register_slot_coil(TopologyRegistry* reg, int slot_idx,
                                     int upper_tag, int lower_tag);
int stator_topo_register_boundary_curve(TopologyRegistry* reg, RegionType type,
                                          int gmsh_curve,
                                          char* err_buf, size_t err_len);

/* Winding layout */
int stator_topo_assign_winding_layout(TopologyRegistry* reg, WindingType wt,
                                        char* err_buf, size_t err_len);

/* Queries */
int  stator_topo_get_surfaces(const TopologyRegistry* reg, RegionType type,
                                int* out_tags, int out_cap);
int  stator_topo_get_boundary_curves(const TopologyRegistry* reg, RegionType type,
                                       int* out_curves, int out_cap);
const SlotWindingAssignment* stator_topo_get_slot_assignment(
    const TopologyRegistry* reg, int slot_idx,
    char* err_buf, size_t err_len);
int  stator_topo_total_surfaces(const TopologyRegistry* reg);
bool stator_topo_winding_assigned(const TopologyRegistry* reg);

#endif /* STATOR_C_TOPOLOGY_REGISTRY_H */
