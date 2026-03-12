#ifndef STATOR_C_GMSH_BACKEND_H
#define STATOR_C_GMSH_BACKEND_H

#include "stator_c/common.h"

/* ── PhysGroupRecord ────────────────────────────────────────────────────── */

#define STATOR_MAX_TAGS_PER_GROUP 4096

typedef struct {
    int  dim;
    int  tags[STATOR_MAX_TAGS_PER_GROUP];
    int  n_tags;
    char name[64];
    int  tag;
} PhysGroupRecord;

/* ── GmshBackendOps — vtable ─────────────────────────────────────────────── */

typedef struct GmshBackendOps {
    /* Session lifecycle */
    void (*initialize)   (void* impl, const char* model_name);
    void (*synchronize)  (void* impl);
    void (*finalize)     (void* impl);
    void (*set_option)   (void* impl, const char* name, double value);

    /* OCC geometry primitives */
    int  (*add_point)        (void* impl, double x, double y, double z, double ms);
    int  (*add_line)         (void* impl, int start, int end_pt);
    int  (*add_circle)       (void* impl, double cx, double cy, double cz, double r);
    int  (*add_arc)          (void* impl, int start, int centre, int end_pt);
    int  (*add_curve_loop)   (void* impl, const int* tags, int n_tags);
    int  (*add_plane_surface)(void* impl, const int* loop_tags, int n_loop_tags);

    /* Boolean operations — result written into out_pairs / *out_n */
    int  (*boolean_cut)     (void* impl,
                              const IntPair* objects, int n_objects,
                              const IntPair* tools,   int n_tools,
                              int remove_tool,
                              IntPair* out_pairs, int* out_n, int out_cap);
    int  (*boolean_fragment)(void* impl,
                              const IntPair* objects, int n_objects,
                              const IntPair* tools,   int n_tools,
                              IntPair* out_pairs, int* out_n, int out_cap);

    /* Physical groups */
    int  (*add_physical_group)(void* impl, int dim,
                                const int* tags, int n_tags,
                                const char* name, int tag);

    /* Mesh fields */
    int  (*add_math_eval_field)(void* impl, const char* expr);
    int  (*add_constant_field) (void* impl, double value,
                                 const int* surfaces, int n_surfaces);
    void (*set_background_field)(void* impl, int field_tag);

    /* Mesh generation and I/O */
    void (*generate_mesh)  (void* impl, int dim);
    void (*write_mesh)     (void* impl, const char* filename);
    int  (*get_entities_2d)(void* impl, IntPair* out_pairs, int out_cap);
} GmshBackendOps;

/* ── GmshBackend — generic handle ───────────────────────────────────────── */

typedef struct {
    GmshBackendOps* ops;
    void*           impl;
} GmshBackend;

/* Convenience wrappers that forward through the vtable */
static inline void gmsh_initialize(GmshBackend* b, const char* n) {
    b->ops->initialize(b->impl, n);
}
static inline void gmsh_synchronize(GmshBackend* b) {
    b->ops->synchronize(b->impl);
}
static inline void gmsh_finalize(GmshBackend* b) {
    b->ops->finalize(b->impl);
}
static inline void gmsh_set_option(GmshBackend* b, const char* n, double v) {
    b->ops->set_option(b->impl, n, v);
}
static inline int gmsh_add_point(GmshBackend* b, double x, double y, double z, double ms) {
    return b->ops->add_point(b->impl, x, y, z, ms);
}
static inline int gmsh_add_line(GmshBackend* b, int s, int e) {
    return b->ops->add_line(b->impl, s, e);
}
static inline int gmsh_add_circle(GmshBackend* b, double cx, double cy, double cz, double r) {
    return b->ops->add_circle(b->impl, cx, cy, cz, r);
}
static inline int gmsh_add_arc(GmshBackend* b, int s, int c, int e) {
    return b->ops->add_arc(b->impl, s, c, e);
}
static inline int gmsh_add_curve_loop(GmshBackend* b, const int* t, int n) {
    return b->ops->add_curve_loop(b->impl, t, n);
}
static inline int gmsh_add_plane_surface(GmshBackend* b, const int* t, int n) {
    return b->ops->add_plane_surface(b->impl, t, n);
}
static inline int gmsh_add_physical_group(GmshBackend* b, int dim,
    const int* tags, int n, const char* name, int tag) {
    return b->ops->add_physical_group(b->impl, dim, tags, n, name, tag);
}
static inline int gmsh_add_math_eval_field(GmshBackend* b, const char* e) {
    return b->ops->add_math_eval_field(b->impl, e);
}
static inline int gmsh_add_constant_field(GmshBackend* b, double v,
    const int* s, int n) {
    return b->ops->add_constant_field(b->impl, v, s, n);
}
static inline void gmsh_set_background_field(GmshBackend* b, int t) {
    b->ops->set_background_field(b->impl, t);
}
static inline void gmsh_generate_mesh(GmshBackend* b, int dim) {
    b->ops->generate_mesh(b->impl, dim);
}
static inline void gmsh_write_mesh(GmshBackend* b, const char* f) {
    b->ops->write_mesh(b->impl, f);
}

/* ── StubGmshBackend ────────────────────────────────────────────────────── */

#define STUB_MAX_PHYS_GROUPS 1024

typedef struct {
    bool  initialized;
    int   sync_count;
    bool  finalized;
    bool  mesh_generated;
    int   background_field;
    char  last_write_path[512];

    int point_counter;
    int line_counter;
    int surface_counter;
    int curve_loop_counter;
    int field_counter;
    int phys_group_tag_counter;

    PhysGroupRecord phys_groups[STUB_MAX_PHYS_GROUPS];
    int             n_phys_groups;

    IntPair surfaces_2d[STUB_MAX_PHYS_GROUPS];
    int     n_surfaces_2d;
} StubGmshBackendImpl;

/* Allocate / free */
StubGmshBackendImpl* stub_gmsh_impl_new(void);
void                 stub_gmsh_impl_free(StubGmshBackendImpl* impl);
void                 stub_gmsh_impl_reset(StubGmshBackendImpl* impl);

/* Returns a GmshBackend whose .impl points to a heap-allocated StubGmshBackendImpl */
GmshBackend* stub_gmsh_backend_new(void);
void         stub_gmsh_backend_free(GmshBackend* b);

/* ── Factory ─────────────────────────────────────────────────────────────── */
/* Returns stub unless STATOR_WITH_GMSH defined */
GmshBackend* stator_make_default_backend(void);

#endif /* STATOR_C_GMSH_BACKEND_H */
