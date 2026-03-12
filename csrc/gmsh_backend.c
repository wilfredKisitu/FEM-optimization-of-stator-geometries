#include "stator_c/gmsh_backend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── StubGmshBackend ops ─────────────────────────────────────────────────── */

static void stub_initialize(void* impl, const char* model_name) {
    (void)model_name;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    s->initialized = true;
}

static void stub_synchronize(void* impl) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    s->sync_count++;
}

static void stub_finalize(void* impl) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    s->finalized = true;
}

static void stub_set_option(void* impl, const char* name, double value) {
    (void)impl; (void)name; (void)value;
}

static int stub_add_point(void* impl, double x, double y, double z, double ms) {
    (void)x; (void)y; (void)z; (void)ms;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->point_counter;
}

static int stub_add_line(void* impl, int start, int end_pt) {
    (void)start; (void)end_pt;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->line_counter;
}

static int stub_add_circle(void* impl, double cx, double cy, double cz, double r) {
    (void)cx; (void)cy; (void)cz; (void)r;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->line_counter;
}

static int stub_add_arc(void* impl, int start, int centre, int end_pt) {
    (void)start; (void)centre; (void)end_pt;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->line_counter;
}

static int stub_add_curve_loop(void* impl, const int* tags, int n_tags) {
    (void)tags; (void)n_tags;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->curve_loop_counter;
}

static int stub_add_plane_surface(void* impl, const int* loop_tags, int n) {
    (void)loop_tags; (void)n;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    int tag = ++s->surface_counter;
    if (s->n_surfaces_2d < STUB_MAX_PHYS_GROUPS) {
        s->surfaces_2d[s->n_surfaces_2d].first  = 2;
        s->surfaces_2d[s->n_surfaces_2d].second = tag;
        s->n_surfaces_2d++;
    }
    return tag;
}

static int stub_boolean_cut(void* impl,
                              const IntPair* objects, int n_objects,
                              const IntPair* tools,   int n_tools,
                              int remove_tool,
                              IntPair* out_pairs, int* out_n, int out_cap) {
    (void)impl; (void)tools; (void)n_tools; (void)remove_tool;
    /* stub: return objects unchanged */
    int n = n_objects < out_cap ? n_objects : out_cap;
    memcpy(out_pairs, objects, (size_t)n * sizeof(IntPair));
    *out_n = n;
    return STATOR_OK;
}

static int stub_boolean_fragment(void* impl,
                                   const IntPair* objects, int n_objects,
                                   const IntPair* tools,   int n_tools,
                                   IntPair* out_pairs, int* out_n, int out_cap) {
    (void)impl;
    int total = n_objects + n_tools;
    int n = total < out_cap ? total : out_cap;
    int no = n_objects < out_cap ? n_objects : out_cap;
    memcpy(out_pairs, objects, (size_t)no * sizeof(IntPair));
    int nt = (n - no);
    if (nt > 0) memcpy(out_pairs + no, tools, (size_t)nt * sizeof(IntPair));
    *out_n = n;
    return STATOR_OK;
}

static int stub_add_physical_group(void* impl, int dim,
                                    const int* tags, int n_tags,
                                    const char* name, int tag) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    if (s->n_phys_groups >= STUB_MAX_PHYS_GROUPS) return -1;
    PhysGroupRecord* r = &s->phys_groups[s->n_phys_groups++];
    r->dim = dim;
    int n = n_tags < STATOR_MAX_TAGS_PER_GROUP ? n_tags : STATOR_MAX_TAGS_PER_GROUP;
    memcpy(r->tags, tags, (size_t)n * sizeof(int));
    r->n_tags = n;
    strncpy(r->name, name, sizeof(r->name) - 1);
    r->name[sizeof(r->name) - 1] = '\0';
    r->tag = (tag >= 0) ? tag : ++s->phys_group_tag_counter;
    return r->tag;
}

static int stub_add_math_eval_field(void* impl, const char* expr) {
    (void)expr;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->field_counter;
}

static int stub_add_constant_field(void* impl, double value,
                                    const int* surfaces, int n_surfaces) {
    (void)value; (void)surfaces; (void)n_surfaces;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    return ++s->field_counter;
}

static void stub_set_background_field(void* impl, int field_tag) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    s->background_field = field_tag;
}

static void stub_generate_mesh(void* impl, int dim) {
    (void)dim;
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    s->mesh_generated = true;
}

static void stub_write_mesh(void* impl, const char* filename) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    strncpy(s->last_write_path, filename, sizeof(s->last_write_path) - 1);
    s->last_write_path[sizeof(s->last_write_path) - 1] = '\0';
}

static int stub_get_entities_2d(void* impl, IntPair* out_pairs, int out_cap) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)impl;
    int n = s->n_surfaces_2d < out_cap ? s->n_surfaces_2d : out_cap;
    memcpy(out_pairs, s->surfaces_2d, (size_t)n * sizeof(IntPair));
    return n;
}

/* ── vtable singleton ────────────────────────────────────────────────────── */

static GmshBackendOps stub_ops = {
    .initialize         = stub_initialize,
    .synchronize        = stub_synchronize,
    .finalize           = stub_finalize,
    .set_option         = stub_set_option,
    .add_point          = stub_add_point,
    .add_line           = stub_add_line,
    .add_circle         = stub_add_circle,
    .add_arc            = stub_add_arc,
    .add_curve_loop     = stub_add_curve_loop,
    .add_plane_surface  = stub_add_plane_surface,
    .boolean_cut        = stub_boolean_cut,
    .boolean_fragment   = stub_boolean_fragment,
    .add_physical_group = stub_add_physical_group,
    .add_math_eval_field = stub_add_math_eval_field,
    .add_constant_field = stub_add_constant_field,
    .set_background_field = stub_set_background_field,
    .generate_mesh      = stub_generate_mesh,
    .write_mesh         = stub_write_mesh,
    .get_entities_2d    = stub_get_entities_2d,
};

/* ── Public API ─────────────────────────────────────────────────────────── */

void stub_gmsh_impl_reset(StubGmshBackendImpl* s) {
    memset(s, 0, sizeof(*s));
    s->background_field      = -1;
    s->phys_group_tag_counter = 1000;
}

StubGmshBackendImpl* stub_gmsh_impl_new(void) {
    StubGmshBackendImpl* s = (StubGmshBackendImpl*)malloc(sizeof(*s));
    if (!s) return NULL;
    stub_gmsh_impl_reset(s);
    return s;
}

void stub_gmsh_impl_free(StubGmshBackendImpl* s) {
    free(s);
}

GmshBackend* stub_gmsh_backend_new(void) {
    GmshBackend* b = (GmshBackend*)malloc(sizeof(GmshBackend));
    if (!b) return NULL;
    b->ops  = &stub_ops;
    b->impl = stub_gmsh_impl_new();
    if (!b->impl) { free(b); return NULL; }
    return b;
}

void stub_gmsh_backend_free(GmshBackend* b) {
    if (!b) return;
    stub_gmsh_impl_free((StubGmshBackendImpl*)b->impl);
    free(b);
}

GmshBackend* stator_make_default_backend(void) {
#ifdef STATOR_WITH_GMSH
    /* TODO: return real GMSH backend */
    fprintf(stderr, "[stator] STATOR_WITH_GMSH defined but real backend not implemented in C\n");
#else
    fprintf(stderr, "[stator] Warning: using StubGmshBackend\n");
#endif
    return stub_gmsh_backend_new();
}
