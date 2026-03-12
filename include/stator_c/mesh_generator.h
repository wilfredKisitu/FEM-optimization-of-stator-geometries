#ifndef STATOR_C_MESH_GENERATOR_H
#define STATOR_C_MESH_GENERATOR_H

#include "stator_c/params.h"
#include "stator_c/gmsh_backend.h"
#include "stator_c/topology_registry.h"
#include "stator_c/geometry_builder.h"

/* ── MeshConfig ──────────────────────────────────────────────────────────── */

typedef struct {
    int    algorithm_2d;
    int    algorithm_3d;
    int    smoothing_passes;
    char   optimiser[64];
    double min_quality_threshold;
    bool   periodic;
    int    layers_per_lam;
} MeshConfig;

static inline void stator_mesh_config_init(MeshConfig* c) {
    c->algorithm_2d          = 5;
    c->algorithm_3d          = 10;
    c->smoothing_passes      = 3;
    c->min_quality_threshold = 0.3;
    c->periodic              = false;
    c->layers_per_lam        = 2;
    c->optimiser[0]          = '\0';
    /* default optimiser name */
    const char* def = "Netgen";
    size_t i = 0;
    while (def[i] && i < sizeof(c->optimiser) - 1) { c->optimiser[i] = def[i]; i++; }
    c->optimiser[i] = '\0';
}

/* ── MeshResult ──────────────────────────────────────────────────────────── */

typedef struct {
    bool  success;
    char  error_message[STATOR_ERR_BUF];
    int   n_nodes;
    int   n_elements_2d;
    int   n_elements_3d;
    double min_quality;
    double avg_quality;
    int   n_phys_groups;
} MeshResult;

/* ── MeshGenerator ───────────────────────────────────────────────────────── */

typedef struct {
    GmshBackend* backend;
    MeshConfig   config;
} MeshGenerator;

int  stator_mesh_generator_init(MeshGenerator* mg, GmshBackend* backend,
                                  const MeshConfig* config,
                                  char* err_buf, size_t err_len);

int  stator_mesh_generate(MeshGenerator* mg,
                            const StatorParams* p,
                            const GeometryBuildResult* geo,
                            TopologyRegistry* registry,
                            MeshResult* result);

void stator_mesh_assign_physical_groups(MeshGenerator* mg,
                                          const StatorParams* p,
                                          const GeometryBuildResult* geo,
                                          TopologyRegistry* registry);

#endif /* STATOR_C_MESH_GENERATOR_H */
