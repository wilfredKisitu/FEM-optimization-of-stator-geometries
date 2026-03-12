/* test_stator.c — C test suite for stator_c pipeline.
 * Exit code = number of failures (0 = all pass).
 */

#include "stator_c/params.h"
#include "stator_c/gmsh_backend.h"
#include "stator_c/geometry_builder.h"
#include "stator_c/topology_registry.h"
#include "stator_c/mesh_generator.h"
#include "stator_c/export_engine.h"
#include "stator_c/batch_scheduler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define PI 3.14159265358979323846

/* ── Test framework ──────────────────────────────────────────────────────── */

static int g_pass = 0;
static int g_fail = 0;
static const char *g_current_test = "";

#define TEST(name) static void test_##name(void)
#define RUN(name)               \
    do                          \
    {                           \
        g_current_test = #name; \
        test_##name();          \
    } while (0)

#define EXPECT(cond)                                  \
    do                                                \
    {                                                 \
        if (!(cond))                                  \
        {                                             \
            printf("[FAIL] %s line %d: EXPECT(%s)\n", \
                   g_current_test, __LINE__, #cond);  \
            g_fail++;                                 \
            return;                                   \
        }                                             \
    } while (0)

#define EXPECT_ERR(call)                                          \
    do                                                            \
    {                                                             \
        int _rc = (call);                                         \
        if (_rc == STATOR_OK)                                     \
        {                                                         \
            printf("[FAIL] %s line %d: expected error for: %s\n", \
                   g_current_test, __LINE__, #call);              \
            g_fail++;                                             \
            return;                                               \
        }                                                         \
    } while (0)

#define EXPECT_OK(call)                                                        \
    do                                                                         \
    {                                                                          \
        int _rc = (call);                                                      \
        if (_rc != STATOR_OK)                                                  \
        {                                                                      \
            printf("[FAIL] %s line %d: expected STATOR_OK for: %s (got %d)\n", \
                   g_current_test, __LINE__, #call, _rc);                      \
            g_fail++;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define EXPECT_NEAR(a, b, tol)                                           \
    do                                                                   \
    {                                                                    \
        double _a = (double)(a), _b = (double)(b), _t = (double)(tol);   \
        if (fabs(_a - _b) > _t)                                          \
        {                                                                \
            printf("[FAIL] %s line %d: EXPECT_NEAR %g ~= %g (tol=%g)\n", \
                   g_current_test, __LINE__, _a, _b, _t);                \
            g_fail++;                                                    \
            return;                                                      \
        }                                                                \
    } while (0)

#define PASS()    \
    do            \
    {             \
        g_pass++; \
    } while (0)

/* ─────────────────────────────────────────────────────────────────────────
 * [PARAMS] tests
 * ───────────────────────────────────────────────────────────────────────── */

TEST(params_reference_validates)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_reference_params(&p, err, sizeof(err)));
    EXPECT(p.yoke_height > 0.0);
    EXPECT(p.tooth_width > 0.0);
    EXPECT(p.slot_pitch > 0.0);
    EXPECT(p.stack_length > 0.0);
    EXPECT(p.fill_factor > 0.0 && p.fill_factor < 1.0);
    PASS();
}

TEST(params_minimal_validates)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_minimal_params(&p, err, sizeof(err)));
    EXPECT(p.n_slots == 12);
    EXPECT(p.yoke_height > 0.0);
    PASS();
}

TEST(params_derived_yoke_height)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_reference_params(&p, err, sizeof(err)));
    EXPECT_NEAR(p.yoke_height, p.R_outer - p.R_inner - p.slot_depth, 1e-12);
    PASS();
}

TEST(params_derived_tooth_width)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_reference_params(&p, err, sizeof(err)));
    double expected = p.R_inner * p.slot_pitch - p.slot_width_inner;
    EXPECT_NEAR(p.tooth_width, expected, 1e-12);
    PASS();
}

TEST(params_derived_slot_pitch)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_reference_params(&p, err, sizeof(err)));
    EXPECT_NEAR(p.slot_pitch, 2.0 * PI / p.n_slots, 1e-12);
    PASS();
}

TEST(params_derived_stack_length)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_reference_params(&p, err, sizeof(err)));
    double expected = p.n_lam * p.t_lam + (p.n_lam - 1) * p.z_spacing;
    EXPECT_NEAR(p.stack_length, expected, 1e-12);
    PASS();
}

TEST(params_derived_fill_factor_in_range)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_make_reference_params(&p, err, sizeof(err)));
    EXPECT(p.fill_factor > 0.0 && p.fill_factor < 1.0);
    PASS();
}

TEST(params_rejects_zero_R_outer)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.R_outer = 0.0;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_negative_R_outer)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.R_outer = -0.1;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_R_inner_ge_R_outer)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.R_outer = 0.10;
    p.R_inner = 0.15;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_equal_radii)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.R_inner = p.R_outer;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_too_few_slots)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.n_slots = 4;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_odd_slot_count)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.n_slots = 35;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_slot_depth_exceeds_annulus)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.slot_depth = p.R_outer - p.R_inner;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_slot_too_wide_for_pitch)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.slot_width_inner = p.R_inner * 2.0 * PI / p.n_slots;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_coil_depth_exceeds_slot)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.coil_depth = p.slot_depth;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_coil_too_wide)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.coil_width_inner = p.slot_width_inner;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_mesh_ins_coarser_than_mesh_coil)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.mesh_ins = 0.01;
    p.mesh_coil = 0.001;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_mesh_coil_coarser_than_mesh_slot)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.mesh_coil = 0.01;
    p.mesh_slot = 0.001;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_mesh_slot_coarser_than_mesh_yoke)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.mesh_slot = 0.01;
    p.mesh_yoke = 0.001;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_custom_material_no_file)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.material = MATERIAL_CUSTOM;
    p.material_file[0] = '\0';
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_negative_airgap)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.airgap_length = -0.001;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_rejects_zero_t_lam)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.t_lam = 0.0;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

TEST(params_to_json_contains_all_sections)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    char json[8192];
    EXPECT_OK(stator_params_to_json(&p, json, sizeof(json)));
    EXPECT(strstr(json, "R_outer") != NULL);
    EXPECT(strstr(json, "n_slots") != NULL);
    EXPECT(strstr(json, "_derived") != NULL);
    EXPECT(strstr(json, "fill_factor") != NULL);
    PASS();
}

TEST(params_to_string_all_slot_shapes)
{
    EXPECT(strcmp(stator_slot_shape_to_str(SLOT_SHAPE_RECTANGULAR), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_slot_shape_to_str(SLOT_SHAPE_TRAPEZOIDAL), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_slot_shape_to_str(SLOT_SHAPE_ROUND_BOTTOM), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_slot_shape_to_str(SLOT_SHAPE_SEMI_CLOSED), "UNKNOWN") != 0);
    PASS();
}

TEST(params_to_string_all_winding_types)
{
    EXPECT(strcmp(stator_winding_type_to_str(WINDING_SINGLE_LAYER), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_winding_type_to_str(WINDING_DOUBLE_LAYER), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_winding_type_to_str(WINDING_CONCENTRATED), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_winding_type_to_str(WINDING_DISTRIBUTED), "UNKNOWN") != 0);
    PASS();
}

TEST(params_to_string_all_materials)
{
    EXPECT(strcmp(stator_material_to_str(MATERIAL_M270_35A), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_material_to_str(MATERIAL_M330_50A), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_material_to_str(MATERIAL_M400_50A), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_material_to_str(MATERIAL_NO20), "UNKNOWN") != 0);
    EXPECT(strcmp(stator_material_to_str(MATERIAL_CUSTOM), "UNKNOWN") != 0);
    PASS();
}

TEST(params_rejects_negative_tooth_tip_angle)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.tooth_tip_angle = -0.1;
    EXPECT_ERR(stator_params_validate_and_derive(&p, err, sizeof(err)));
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * [GMSH_BACKEND] tests
 * ───────────────────────────────────────────────────────────────────────── */

TEST(stub_backend_initialize_sets_flag)
{
    GmshBackend *b = stub_gmsh_backend_new();
    EXPECT(b != NULL);
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    EXPECT(!s->initialized);
    gmsh_initialize(b, "test_model");
    EXPECT(s->initialized);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_counters_increment)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    gmsh_initialize(b, "m");
    int p1 = gmsh_add_point(b, 0.0, 0.0, 0.0, 0.0);
    int p2 = gmsh_add_point(b, 1.0, 0.0, 0.0, 0.0);
    EXPECT(p1 == 1);
    EXPECT(p2 == 2);
    EXPECT(s->point_counter == 2);
    int l1 = gmsh_add_line(b, p1, p2);
    EXPECT(l1 == 1);
    EXPECT(s->line_counter == 1);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_sync_count)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    EXPECT(s->sync_count == 0);
    gmsh_synchronize(b);
    gmsh_synchronize(b);
    EXPECT(s->sync_count == 2);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_add_plane_surface_increments)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    int tags[1] = {1};
    int sf1 = gmsh_add_plane_surface(b, tags, 1);
    int sf2 = gmsh_add_plane_surface(b, tags, 1);
    EXPECT(sf1 == 1);
    EXPECT(sf2 == 2);
    EXPECT(s->surface_counter == 2);
    EXPECT(s->n_surfaces_2d == 2);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_write_mesh_records_path)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    gmsh_write_mesh(b, "/tmp/test_stator.msh");
    EXPECT(strcmp(s->last_write_path, "/tmp/test_stator.msh") == 0);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_background_field)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    EXPECT(s->background_field == -1);
    gmsh_set_background_field(b, 42);
    EXPECT(s->background_field == 42);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_physical_groups)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    int tags[2] = {1, 2};
    int t = gmsh_add_physical_group(b, 2, tags, 2, "YOKE", 100);
    EXPECT(t == 100);
    EXPECT(s->n_phys_groups == 1);
    EXPECT(strcmp(s->phys_groups[0].name, "YOKE") == 0);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(stub_backend_reset)
{
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    gmsh_initialize(b, "m");
    gmsh_add_point(b, 0, 0, 0, 0);
    EXPECT(s->point_counter == 1);
    stub_gmsh_impl_reset(s);
    EXPECT(s->point_counter == 0);
    EXPECT(!s->initialized);
    stub_gmsh_backend_free(b);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * [GEOMETRY_BUILDER] tests
 * ───────────────────────────────────────────────────────────────────────── */

TEST(geom_rotate_identity)
{
    double ox, oy;
    stator_geom_rotate(1.0, 0.0, 0.0, &ox, &oy);
    EXPECT_NEAR(ox, 1.0, 1e-12);
    EXPECT_NEAR(oy, 0.0, 1e-12);
    PASS();
}

TEST(geom_rotate_90deg)
{
    double ox, oy;
    stator_geom_rotate(1.0, 0.0, PI / 2.0, &ox, &oy);
    EXPECT_NEAR(ox, 0.0, 1e-12);
    EXPECT_NEAR(oy, 1.0, 1e-12);
    PASS();
}

TEST(geom_slot_angle_zero_for_first)
{
    EXPECT_NEAR(stator_geom_slot_angle(0, 12), 0.0, 1e-12);
    PASS();
}

TEST(geom_slot_angle_full_circle)
{
    /* Sum of all 12 slot angles should equal 2π * (0+1+...+11)/12 => not summed, just check last */
    double last = stator_geom_slot_angle(11, 12);
    EXPECT_NEAR(last, 2.0 * PI * 11.0 / 12.0, 1e-12);
    PASS();
}

TEST(geom_build_rectangular_success)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    p.slot_shape = SLOT_SHAPE_RECTANGULAR;
    stator_params_validate_and_derive(&p, err, sizeof(err));

    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuilder gb;
    EXPECT_OK(stator_geom_builder_init(&gb, b, err, sizeof(err)));
    GeometryBuildResult result;
    EXPECT_OK(stator_geom_build(&gb, &p, &result));
    EXPECT(result.success);
    EXPECT(result.yoke_surface > 0);
    EXPECT(result.n_slots == p.n_slots);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_trapezoidal_success)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    p.slot_shape = SLOT_SHAPE_TRAPEZOIDAL;
    stator_params_validate_and_derive(&p, err, sizeof(err));

    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuilder gb;
    EXPECT_OK(stator_geom_builder_init(&gb, b, err, sizeof(err)));
    GeometryBuildResult result;
    EXPECT_OK(stator_geom_build(&gb, &p, &result));
    EXPECT(result.success);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_round_bottom_success)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    p.slot_shape = SLOT_SHAPE_ROUND_BOTTOM;
    stator_params_validate_and_derive(&p, err, sizeof(err));

    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuilder gb;
    EXPECT_OK(stator_geom_builder_init(&gb, b, err, sizeof(err)));
    GeometryBuildResult result;
    EXPECT_OK(stator_geom_build(&gb, &p, &result));
    EXPECT(result.success);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_semi_closed_success)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    p.slot_shape = SLOT_SHAPE_SEMI_CLOSED;
    stator_params_validate_and_derive(&p, err, sizeof(err));

    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuilder gb;
    EXPECT_OK(stator_geom_builder_init(&gb, b, err, sizeof(err)));
    GeometryBuildResult result;
    EXPECT_OK(stator_geom_build(&gb, &p, &result));
    EXPECT(result.success);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_slot_count_matches)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuilder gb;
    stator_geom_builder_init(&gb, b, err, sizeof(err));
    GeometryBuildResult result;
    stator_geom_build(&gb, &p, &result);
    EXPECT(result.n_slots == p.n_slots);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_synchronize_called)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    GeometryBuilder gb;
    stator_geom_builder_init(&gb, b, err, sizeof(err));
    GeometryBuildResult result;
    stator_geom_build(&gb, &p, &result);
    EXPECT(s->sync_count >= 1);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_produces_points)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    GeometryBuilder gb;
    stator_geom_builder_init(&gb, b, err, sizeof(err));
    GeometryBuildResult result;
    stator_geom_build(&gb, &p, &result);
    EXPECT(s->point_counter > 0);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(geom_build_double_layer_produces_more_surfaces)
{
    StatorParams p1, p2;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p1, err, sizeof(err));
    p1.winding_type = WINDING_SINGLE_LAYER;
    stator_params_validate_and_derive(&p1, err, sizeof(err));

    stator_make_minimal_params(&p2, err, sizeof(err));
    p2.winding_type = WINDING_DOUBLE_LAYER;
    stator_params_validate_and_derive(&p2, err, sizeof(err));

    GmshBackend *b1 = stub_gmsh_backend_new();
    GmshBackend *b2 = stub_gmsh_backend_new();
    StubGmshBackendImpl *s1 = (StubGmshBackendImpl *)b1->impl;
    StubGmshBackendImpl *s2 = (StubGmshBackendImpl *)b2->impl;

    GeometryBuilder gb1, gb2;
    char e[STATOR_ERR_BUF];
    stator_geom_builder_init(&gb1, b1, e, sizeof(e));
    stator_geom_builder_init(&gb2, b2, e, sizeof(e));
    GeometryBuildResult r1, r2;
    stator_geom_build(&gb1, &p1, &r1);
    stator_geom_build(&gb2, &p2, &r2);
    EXPECT(s2->surface_counter > s1->surface_counter);
    stub_gmsh_backend_free(b1);
    stub_gmsh_backend_free(b2);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * [TOPOLOGY_REGISTRY] tests
 * ───────────────────────────────────────────────────────────────────────── */

TEST(topo_init_succeeds)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    EXPECT_OK(stator_topo_registry_init(&reg, 36, err, sizeof(err)));
    EXPECT(reg.n_slots == 36);
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_init_rejects_zero_slots)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    EXPECT_ERR(stator_topo_registry_init(&reg, 0, err, sizeof(err)));
    PASS();
}

TEST(topo_register_surface_and_query)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    stator_topo_register_surface(&reg, REGION_YOKE, 5, -1);
    stator_topo_register_surface(&reg, REGION_YOKE, 7, -1);
    int tags[16];
    int n = stator_topo_get_surfaces(&reg, REGION_YOKE, tags, 16);
    EXPECT(n == 2);
    EXPECT(tags[0] == 5 && tags[1] == 7);
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_register_slot_coil)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    EXPECT_OK(stator_topo_register_slot_coil(&reg, 0, 10, 11));
    EXPECT(reg.slot_upper_tags[0] == 10);
    EXPECT(reg.slot_lower_tags[0] == 11);
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_register_slot_coil_out_of_range)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    EXPECT_ERR(stator_topo_register_slot_coil(&reg, 100, 10, 11));
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_register_boundary_curve)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    EXPECT_OK(stator_topo_register_boundary_curve(&reg, REGION_BOUNDARY_BORE, 3, err, sizeof(err)));
    int curves[4];
    int n = stator_topo_get_boundary_curves(&reg, REGION_BOUNDARY_BORE, curves, 4);
    EXPECT(n == 1);
    EXPECT(curves[0] == 3);
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_register_boundary_curve_invalid_type)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    EXPECT_ERR(stator_topo_register_boundary_curve(&reg, REGION_YOKE, 3, err, sizeof(err)));
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_assign_winding_distributed)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    for (int i = 0; i < 12; i++)
        stator_topo_register_slot_coil(&reg, i, i + 100, -1);
    EXPECT_OK(stator_topo_assign_winding_layout(&reg, WINDING_DISTRIBUTED, err, sizeof(err)));
    EXPECT(stator_topo_winding_assigned(&reg));
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_assign_winding_no_coils_fails)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    EXPECT_ERR(stator_topo_assign_winding_layout(&reg, WINDING_DISTRIBUTED, err, sizeof(err)));
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_winding_slot0_distributed_is_A_pos)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 6, err, sizeof(err));
    for (int i = 0; i < 6; i++)
        stator_topo_register_slot_coil(&reg, i, i + 100, -1);
    stator_topo_assign_winding_layout(&reg, WINDING_DISTRIBUTED, err, sizeof(err));
    const SlotWindingAssignment *a = stator_topo_get_slot_assignment(&reg, 0, err, sizeof(err));
    EXPECT(a != NULL);
    EXPECT(a->upper_phase == REGION_COIL_A_POS);
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(topo_total_registered_surfaces)
{
    TopologyRegistry reg;
    char err[STATOR_ERR_BUF];
    stator_topo_registry_init(&reg, 12, err, sizeof(err));
    stator_topo_register_surface(&reg, REGION_YOKE, 1, -1);
    stator_topo_register_surface(&reg, REGION_SLOT_AIR, 2, 0);
    stator_topo_register_surface(&reg, REGION_SLOT_AIR, 3, 1);
    EXPECT(stator_topo_total_surfaces(&reg) == 3);
    stator_topo_registry_destroy(&reg);
    PASS();
}

TEST(region_to_str_all)
{
    EXPECT(strcmp(stator_region_to_str(REGION_YOKE), "YOKE") == 0);
    EXPECT(strcmp(stator_region_to_str(REGION_TOOTH), "TOOTH") == 0);
    EXPECT(strcmp(stator_region_to_str(REGION_SLOT_AIR), "SLOT_AIR") == 0);
    EXPECT(strcmp(stator_region_to_str(REGION_SLOT_INS), "SLOT_INS") == 0);
    EXPECT(strcmp(stator_region_to_str(REGION_COIL_A_POS), "COIL_A_POS") == 0);
    EXPECT(strcmp(stator_region_to_str(REGION_BOUNDARY_BORE), "BOUNDARY_BORE") == 0);
    EXPECT(strcmp(stator_region_to_str(REGION_BOUNDARY_OUTER), "BOUNDARY_OUTER") == 0);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * [MESH_GENERATOR] tests
 * ───────────────────────────────────────────────────────────────────────── */

static void make_full_geo(const StatorParams *p, GmshBackend *b,
                          GeometryBuildResult *result)
{
    GeometryBuilder gb;
    char e[STATOR_ERR_BUF];
    stator_geom_builder_init(&gb, b, e, sizeof(e));
    stator_geom_build(&gb, p, result);
}

TEST(mesh_generate_success)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuildResult geo;
    make_full_geo(&p, b, &geo);

    TopologyRegistry reg;
    stator_topo_registry_init(&reg, p.n_slots, err, sizeof(err));

    MeshGenerator mg;
    EXPECT_OK(stator_mesh_generator_init(&mg, b, NULL, err, sizeof(err)));
    MeshResult result;
    EXPECT_OK(stator_mesh_generate(&mg, &p, &geo, &reg, &result));
    EXPECT(result.success);
    EXPECT(result.n_nodes > 0);
    EXPECT(result.n_elements_2d > 0);
    stator_topo_registry_destroy(&reg);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(mesh_generate_records_mesh_generated)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    GeometryBuildResult geo;
    make_full_geo(&p, b, &geo);
    TopologyRegistry reg;
    stator_topo_registry_init(&reg, p.n_slots, err, sizeof(err));
    MeshGenerator mg;
    stator_mesh_generator_init(&mg, b, NULL, err, sizeof(err));
    MeshResult result;
    stator_mesh_generate(&mg, &p, &geo, &reg, &result);
    EXPECT(s->mesh_generated);
    stator_topo_registry_destroy(&reg);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(mesh_assign_phys_groups_registers_yoke)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    GeometryBuildResult geo;
    make_full_geo(&p, b, &geo);
    TopologyRegistry reg;
    stator_topo_registry_init(&reg, p.n_slots, err, sizeof(err));
    MeshGenerator mg;
    stator_mesh_generator_init(&mg, b, NULL, err, sizeof(err));
    stator_mesh_assign_physical_groups(&mg, &p, &geo, &reg);
    EXPECT(s->n_phys_groups > 0);
    stator_topo_registry_destroy(&reg);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(mesh_generate_sets_background_field)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    StubGmshBackendImpl *s = (StubGmshBackendImpl *)b->impl;
    GeometryBuildResult geo;
    make_full_geo(&p, b, &geo);
    TopologyRegistry reg;
    stator_topo_registry_init(&reg, p.n_slots, err, sizeof(err));
    MeshGenerator mg;
    stator_mesh_generator_init(&mg, b, NULL, err, sizeof(err));
    MeshResult result;
    stator_mesh_generate(&mg, &p, &geo, &reg, &result);
    EXPECT(s->background_field >= 0);
    stator_topo_registry_destroy(&reg);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(mesh_generate_fails_on_bad_geo)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    GeometryBuildResult geo;
    memset(&geo, 0, sizeof(geo));
    geo.success = false;
    snprintf(geo.error_message, sizeof(geo.error_message), "simulated failure");

    TopologyRegistry reg;
    stator_topo_registry_init(&reg, p.n_slots, err, sizeof(err));
    MeshGenerator mg;
    stator_mesh_generator_init(&mg, b, NULL, err, sizeof(err));
    MeshResult result;
    int rc = stator_mesh_generate(&mg, &p, &geo, &reg, &result);
    EXPECT(rc != STATOR_OK);
    EXPECT(!result.success);
    stator_topo_registry_destroy(&reg);
    stub_gmsh_backend_free(b);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * [EXPORT_ENGINE] tests
 * ───────────────────────────────────────────────────────────────────────── */

TEST(sha256_known_vector)
{
    /* SHA-256("abc") = ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469f492c347e2eed7495 */
    char out[65];
    stator_sha256("abc", 3, out);
    EXPECT(strncmp(out, "ba7816bf", 8) == 0);
    PASS();
}

TEST(sha256_empty_string)
{
    /* SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 */
    char out[65];
    stator_sha256("", 0, out);
    EXPECT(strncmp(out, "e3b0c442", 8) == 0);
    PASS();
}

TEST(export_compute_stem_format)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    char stem[64];
    EXPECT_OK(stator_export_compute_stem(&p, stem, sizeof(stem)));
    EXPECT(strncmp(stem, "stator_", 7) == 0);
    EXPECT(strlen(stem) == 15); /* "stator_" + 8 hex chars */
    PASS();
}

TEST(export_compute_stem_deterministic)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    char s1[64], s2[64];
    stator_export_compute_stem(&p, s1, sizeof(s1));
    stator_export_compute_stem(&p, s2, sizeof(s2));
    EXPECT(strcmp(s1, s2) == 0);
    PASS();
}

TEST(export_write_vtk_creates_file)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    ExportEngine ee;
    stator_export_engine_init(&ee, b, NULL, 0);
    ExportConfig cfg;
    stator_export_config_init(&cfg);
    cfg.formats = EXPORT_VTK;
    snprintf(cfg.output_dir, sizeof(cfg.output_dir), "/tmp");

    MeshResult mesh;
    memset(&mesh, 0, sizeof(mesh));
    mesh.success = true;
    mesh.n_nodes = 10;
    mesh.n_elements_2d = 20;

    ExportResult results[4];
    int n = 0;
    EXPECT_OK(stator_export_write_all_sync(&ee, &p, &mesh, &cfg, results, 4, &n));
    EXPECT(n == 1);
    EXPECT(results[0].success);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(export_write_json_creates_file)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&p, err, sizeof(err));
    GmshBackend *b = stub_gmsh_backend_new();
    ExportEngine ee;
    stator_export_engine_init(&ee, b, NULL, 0);
    ExportConfig cfg;
    stator_export_config_init(&cfg);
    cfg.formats = EXPORT_JSON;
    snprintf(cfg.output_dir, sizeof(cfg.output_dir), "/tmp");

    MeshResult mesh;
    memset(&mesh, 0, sizeof(mesh));
    mesh.success = true;
    mesh.n_nodes = 10;
    mesh.n_elements_2d = 20;

    ExportResult results[4];
    int n = 0;
    EXPECT_OK(stator_export_write_all_sync(&ee, &p, &mesh, &cfg, results, 4, &n));
    EXPECT(n == 1);
    EXPECT(results[0].success);
    stub_gmsh_backend_free(b);
    PASS();
}

TEST(export_outputs_exist_false_when_missing)
{
    StatorParams p;
    char err[STATOR_ERR_BUF];
    stator_make_reference_params(&p, err, sizeof(err));
    ExportConfig cfg;
    stator_export_config_init(&cfg);
    snprintf(cfg.output_dir, sizeof(cfg.output_dir), "/tmp/stator_test_nonexistent_dir_xyz");
    cfg.formats = EXPORT_MSH;
    EXPECT(!stator_export_outputs_exist(&p, &cfg));
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * [BATCH_SCHEDULER] tests
 * ───────────────────────────────────────────────────────────────────────── */

TEST(batch_execute_job_succeeds)
{
    BatchJob job;
    memset(&job, 0, sizeof(job));
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&job.params, err, sizeof(err));
    strncpy(job.job_id, "test_job_0", sizeof(job.job_id) - 1);
    stator_export_config_init(&job.export_config);
    snprintf(job.export_config.output_dir, sizeof(job.export_config.output_dir),
             "/tmp/stator_test_batch");
    job.export_config.formats = EXPORT_JSON;
    stator_mesh_config_init(&job.mesh_config);

    char status_path[256];
    snprintf(status_path, sizeof(status_path),
             "/tmp/stator_test_batch_status_%d.json", (int)getpid());

    int rc = stator_batch_execute_job(&job, status_path);
    EXPECT(rc == 0);
    PASS();
}

TEST(batch_run_single_job)
{
    BatchJob job;
    memset(&job, 0, sizeof(job));
    char err[STATOR_ERR_BUF];
    stator_make_minimal_params(&job.params, err, sizeof(err));
    strncpy(job.job_id, "batch_run_test", sizeof(job.job_id) - 1);
    stator_export_config_init(&job.export_config);
    snprintf(job.export_config.output_dir, sizeof(job.export_config.output_dir),
             "/tmp/stator_batch_run_test");
    job.export_config.formats = EXPORT_JSON;
    stator_mesh_config_init(&job.mesh_config);

    BatchScheduler s;
    stator_batch_scheduler_init(&s);
    BatchSchedulerConfig cfg;
    stator_batch_sched_config_init(&cfg);
    cfg.max_parallel = 1;
    cfg.write_summary = false;
    cfg.skip_existing = false;

    BatchResult result;
    int rc = stator_batch_run(&s, &job, 1, &cfg, &result);
    EXPECT(rc == STATOR_OK);
    EXPECT(result.success);
    PASS();
}

TEST(batch_empty_jobs_returns_ok)
{
    BatchScheduler s;
    stator_batch_scheduler_init(&s);
    BatchSchedulerConfig cfg;
    stator_batch_sched_config_init(&cfg);
    BatchResult results[1];
    int rc = stator_batch_run(&s, NULL, 0, &cfg, results);
    EXPECT(rc == STATOR_OK);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────
 * main
 * ───────────────────────────────────────────────────────────────────────── */

int main(void)
{
    /* [PARAMS] */
    RUN(params_reference_validates);
    RUN(params_minimal_validates);
    RUN(params_derived_yoke_height);
    RUN(params_derived_tooth_width);
    RUN(params_derived_slot_pitch);
    RUN(params_derived_stack_length);
    RUN(params_derived_fill_factor_in_range);
    RUN(params_rejects_zero_R_outer);
    RUN(params_rejects_negative_R_outer);
    RUN(params_rejects_R_inner_ge_R_outer);
    RUN(params_rejects_equal_radii);
    RUN(params_rejects_too_few_slots);
    RUN(params_rejects_odd_slot_count);
    RUN(params_rejects_slot_depth_exceeds_annulus);
    RUN(params_rejects_slot_too_wide_for_pitch);
    RUN(params_rejects_coil_depth_exceeds_slot);
    RUN(params_rejects_coil_too_wide);
    RUN(params_rejects_mesh_ins_coarser_than_mesh_coil);
    RUN(params_rejects_mesh_coil_coarser_than_mesh_slot);
    RUN(params_rejects_mesh_slot_coarser_than_mesh_yoke);
    RUN(params_rejects_custom_material_no_file);
    RUN(params_rejects_negative_airgap);
    RUN(params_rejects_zero_t_lam);
    RUN(params_to_json_contains_all_sections);
    RUN(params_to_string_all_slot_shapes);
    RUN(params_to_string_all_winding_types);
    RUN(params_to_string_all_materials);
    RUN(params_rejects_negative_tooth_tip_angle);

    /* [GMSH_BACKEND] */
    RUN(stub_backend_initialize_sets_flag);
    RUN(stub_backend_counters_increment);
    RUN(stub_backend_sync_count);
    RUN(stub_backend_add_plane_surface_increments);
    RUN(stub_backend_write_mesh_records_path);
    RUN(stub_backend_background_field);
    RUN(stub_backend_physical_groups);
    RUN(stub_backend_reset);

    /* [GEOMETRY_BUILDER] */
    RUN(geom_rotate_identity);
    RUN(geom_rotate_90deg);
    RUN(geom_slot_angle_zero_for_first);
    RUN(geom_slot_angle_full_circle);
    RUN(geom_build_rectangular_success);
    RUN(geom_build_trapezoidal_success);
    RUN(geom_build_round_bottom_success);
    RUN(geom_build_semi_closed_success);
    RUN(geom_build_slot_count_matches);
    RUN(geom_build_synchronize_called);
    RUN(geom_build_produces_points);
    RUN(geom_build_double_layer_produces_more_surfaces);

    /* [TOPOLOGY_REGISTRY] */
    RUN(topo_init_succeeds);
    RUN(topo_init_rejects_zero_slots);
    RUN(topo_register_surface_and_query);
    RUN(topo_register_slot_coil);
    RUN(topo_register_slot_coil_out_of_range);
    RUN(topo_register_boundary_curve);
    RUN(topo_register_boundary_curve_invalid_type);
    RUN(topo_assign_winding_distributed);
    RUN(topo_assign_winding_no_coils_fails);
    RUN(topo_winding_slot0_distributed_is_A_pos);
    RUN(topo_total_registered_surfaces);
    RUN(region_to_str_all);

    /* [MESH_GENERATOR] */
    RUN(mesh_generate_success);
    RUN(mesh_generate_records_mesh_generated);
    RUN(mesh_assign_phys_groups_registers_yoke);
    RUN(mesh_generate_sets_background_field);
    RUN(mesh_generate_fails_on_bad_geo);

    /* [EXPORT_ENGINE] */
    RUN(sha256_known_vector);
    RUN(sha256_empty_string);
    RUN(export_compute_stem_format);
    RUN(export_compute_stem_deterministic);
    RUN(export_write_vtk_creates_file);
    RUN(export_write_json_creates_file);
    RUN(export_outputs_exist_false_when_missing);

    /* [BATCH_SCHEDULER] */
    RUN(batch_execute_job_succeeds);
    RUN(batch_run_single_job);
    RUN(batch_empty_jobs_returns_ok);

    printf("\n=== Results: PASS=%d FAIL=%d ===\n", g_pass, g_fail);
    return g_fail;
}
