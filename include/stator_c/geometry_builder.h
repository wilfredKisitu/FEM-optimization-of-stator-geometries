#ifndef STATOR_C_GEOMETRY_BUILDER_H
#define STATOR_C_GEOMETRY_BUILDER_H

#include "stator_c/params.h"
#include "stator_c/gmsh_backend.h"

/* ── SlotProfile ─────────────────────────────────────────────────────────── */

typedef struct {
    int    slot_idx;
    int    slot_surface;
    int    coil_upper_sf;
    int    coil_lower_sf;
    int    ins_upper_sf;
    int    ins_lower_sf;
    int    mouth_curve_bot;
    int    mouth_curve_top;
    double angle;
} SlotProfile;

#define STATOR_MAX_SLOTS 512

/* ── GeometryBuildResult ─────────────────────────────────────────────────── */

typedef struct {
    bool        success;
    char        error_message[STATOR_ERR_BUF];
    int         yoke_surface;
    int         bore_curve;
    int         outer_curve;
    SlotProfile slots[STATOR_MAX_SLOTS];
    int         n_slots;
} GeometryBuildResult;

/* ── GeometryBuilder ─────────────────────────────────────────────────────── */

typedef struct {
    GmshBackend* backend;
} GeometryBuilder;

/* Initialise builder (returns STATOR_ERR_INVAL if backend is NULL) */
int  stator_geom_builder_init(GeometryBuilder* gb, GmshBackend* backend,
                               char* err_buf, size_t err_len);

/* Build the complete 2-D stator cross-section */
int  stator_geom_build(GeometryBuilder* gb, const StatorParams* p,
                        GeometryBuildResult* result);

/* Static helpers */
void stator_geom_rotate(double x, double y, double theta,
                         double* out_x, double* out_y);
double stator_geom_slot_angle(int k, int n_slots);

#endif /* STATOR_C_GEOMETRY_BUILDER_H */
