/**
 * single_geometry_c.c — Generate one stator geometry using the C API.
 *
 * Demonstrates the full pipeline in pure C:
 *   1. Build and validate StatorParams
 *   2. Create a GMSH stub backend
 *   3. Build the 2-D cross-section geometry
 *   4. Register topology (physical regions)
 *   5. Generate the mesh
 *   6. Export to JSON + VTK
 *   7. Print a geometry summary to stdout
 *
 * Build (from project root):
 *   mkdir -p build && cd build
 *   cmake -DSTATOR_BUILD_EXAMPLES=ON ..
 *   make single_geometry_c
 *
 * Run:
 *   ./single_geometry_c
 *   ./single_geometry_c --output /tmp/my_stator
 *   ./single_geometry_c --slots 48 --lam 400
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stator_c/params.h"
#include "stator_c/gmsh_backend.h"
#include "stator_c/geometry_builder.h"
#include "stator_c/topology_registry.h"
#include "stator_c/mesh_generator.h"
#include "stator_c/export_engine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── CLI parsing ─────────────────────────────────────────────────────────── */

static const char *USAGE =
    "Usage: single_geometry_c [options]\n"
    "\n"
    "Options:\n"
    "  --output <dir>   Output directory   (default: /tmp/stator_single_c)\n"
    "  --slots  <n>     Number of slots    (default: 36)\n"
    "  --lam    <n>     Number of lam.     (default: 200)\n"
    "  --help           Show this message\n";

typedef struct
{
    char output[512];
    int n_slots;
    int n_lam;
} Args;

static Args parse_args(int argc, char *argv[])
{
    Args a;
    strncpy(a.output, "/tmp/stator_single_c", sizeof(a.output) - 1);
    a.output[sizeof(a.output) - 1] = '\0';
    a.n_slots = 36;
    a.n_lam = 200;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            puts(USAGE);
            exit(0);
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            strncpy(a.output, argv[++i], sizeof(a.output) - 1);
        }
        else if (strcmp(argv[i], "--slots") == 0 && i + 1 < argc)
        {
            a.n_slots = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--lam") == 0 && i + 1 < argc)
        {
            a.n_lam = atoi(argv[++i]);
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n%s", argv[i], USAGE);
            exit(1);
        }
    }
    return a;
}

/* ── Print separator ─────────────────────────────────────────────────────── */

static void sep(void) { puts("─────────────────────────────────────────────────────────────"); }

/* ── Build StatorParams ──────────────────────────────────────────────────── */

static StatorParams make_params(int n_slots, int n_lam)
{
    StatorParams p;
    char err[512];
    stator_make_reference_params(&p, err, sizeof(err)); /* validated defaults */

    /* Override user-selected fields */
    p.n_slots = n_slots;
    p.n_lam = n_lam;

    /* Scale slot widths to maintain roughly constant tooth/slot ratio */
    double arc = M_PI * p.R_inner / n_slots;
    p.slot_width_outer = arc * 0.55;
    p.slot_width_inner = arc * 0.45;
    p.slot_opening = arc * 0.20;

    /* Coil widths must fit inside slot after insulation (each side) */
    double ins = p.insulation_thickness;
    p.coil_width_outer = p.slot_width_outer - 2.0 * ins - 0.0001;
    p.coil_width_inner = p.slot_width_inner - 2.0 * ins - 0.0001;

    return p;
}

/* ── Geometry report ─────────────────────────────────────────────────────── */

static void print_report(const StatorParams *p)
{
    sep();
    printf("  STATOR GEOMETRY — C PIPELINE REPORT\n");
    sep();
    printf("  %-32s %8.1f mm\n", "Outer radius", p->R_outer * 1e3);
    printf("  %-32s %8.1f mm\n", "Inner radius", p->R_inner * 1e3);
    printf("  %-32s %8.2f mm\n", "Air-gap", p->airgap_length * 1e3);
    printf("  %-32s %9d\n", "Slots", p->n_slots);
    printf("  %-32s %12s\n", "Slot shape",
           stator_slot_shape_to_str(p->slot_shape));
    printf("  %-32s %12s\n", "Winding type",
           stator_winding_type_to_str(p->winding_type));
    printf("  %-32s %4d x %.3f mm\n", "Laminations",
           p->n_lam, p->t_lam * 1e3);
    printf("  %-32s %12s\n", "Material",
           stator_material_to_str(p->material));
    puts("");
    printf("  %-32s %8.2f mm  (derived)\n", "Yoke height", p->yoke_height * 1e3);
    printf("  %-32s %8.3f mm  (derived)\n", "Tooth width", p->tooth_width * 1e3);
    printf("  %-32s %8.2f °   (derived)\n", "Slot pitch",
           p->slot_pitch * 180.0 / M_PI);
    printf("  %-32s %8.1f mm  (derived)\n", "Stack length", p->stack_length * 1e3);
    printf("  %-32s %8.3f     (%4.1f %%)\n", "Fill factor",
           p->fill_factor, p->fill_factor * 100.0);
    puts("");
    printf("  %-32s %8.2f mm\n", "Mesh — yoke", p->mesh_yoke * 1e3);
    printf("  %-32s %8.2f mm\n", "Mesh — slot", p->mesh_slot * 1e3);
    printf("  %-32s %8.2f mm\n", "Mesh — coil", p->mesh_coil * 1e3);
    printf("  %-32s %8.2f mm\n", "Mesh — insul", p->mesh_ins * 1e3);
    sep();
}

/* ── SHA-256 fingerprint ─────────────────────────────────────────────────── */

static void print_fingerprint(const StatorParams *p)
{
    char json_buf[8192];
    stator_params_to_json(p, json_buf, sizeof(json_buf));

    char hash[65];
    stator_sha256(json_buf, strlen(json_buf), hash);
    printf("  SHA-256  : %.20s…\n", hash);
}

/* ── Make output dir ─────────────────────────────────────────────────────── */

static int ensure_dir(const char *path)
{
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    return system(cmd);
}

/* ═════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    Args args = parse_args(argc, argv);
    char err[512];

    printf("\n=== Stator Single Geometry — C Example ===\n\n");

    /* ── 1. Build + validate params ──────────────────────────────────────── */
    printf("Step 1/6  Validating parameters …\n");
    StatorParams p = make_params(args.n_slots, args.n_lam);

    if (stator_params_validate_and_derive(&p, err, sizeof(err)) != STATOR_OK)
    {
        fprintf(stderr, "  ERROR: %s\n", err);
        return 1;
    }
    printf("  OK\n");
    print_report(&p);

    /* ── 2. SHA-256 fingerprint ───────────────────────────────────────────── */
    print_fingerprint(&p);

    /* ── 3. Backend + initialise GMSH model ──────────────────────────────── */
    printf("\nStep 2/6  Creating GMSH backend …\n");
    GmshBackend *backend = stator_make_default_backend();
    if (!backend)
    {
        fprintf(stderr, "  ERROR: failed to allocate backend\n");
        return 1;
    }

    char stem[64];
    stator_export_compute_stem(&p, stem, sizeof(stem));

    char model_name[80];
    snprintf(model_name, sizeof(model_name), "stator_%s", stem);
    gmsh_initialize(backend, model_name);
    printf("  Model: %s\n", model_name);

    /* ── 4. Build 2-D geometry ───────────────────────────────────────────── */
    printf("\nStep 3/6  Building geometry (%d slots, %s) …\n",
           p.n_slots, stator_slot_shape_to_str(p.slot_shape));

    GeometryBuilder gb;
    if (stator_geom_builder_init(&gb, backend, err, sizeof(err)) != STATOR_OK)
    {
        fprintf(stderr, "  ERROR: %s\n", err);
        stub_gmsh_backend_free(backend);
        return 1;
    }

    GeometryBuildResult geo;
    if (stator_geom_build(&gb, &p, &geo) != STATOR_OK || !geo.success)
    {
        fprintf(stderr, "  ERROR: %s\n", geo.error_message);
        stub_gmsh_backend_free(backend);
        return 1;
    }
    printf("  Yoke surface   : tag %d\n", geo.yoke_surface);
    printf("  Bore curve     : tag %d\n", geo.bore_curve);
    printf("  Slot profiles  : %d built\n", geo.n_slots);

    /* ── 5. Register topology ────────────────────────────────────────────── */
    printf("\nStep 4/6  Registering topology …\n");

    TopologyRegistry reg;
    if (stator_topo_registry_init(&reg, p.n_slots, err, sizeof(err)) != STATOR_OK)
    {
        fprintf(stderr, "  ERROR: %s\n", err);
        stub_gmsh_backend_free(backend);
        return 1;
    }

    /* Register yoke */
    stator_topo_register_surface(&reg, REGION_YOKE, geo.yoke_surface, -1);

    /* Register each slot's surfaces */
    for (int i = 0; i < geo.n_slots; i++)
    {
        const SlotProfile *s = &geo.slots[i];
        stator_topo_register_surface(&reg, REGION_SLOT_AIR, s->slot_surface, i);
        stator_topo_register_surface(&reg, REGION_COIL_A_POS, s->coil_upper_sf, i);
        stator_topo_register_surface(&reg, REGION_COIL_B_POS, s->coil_lower_sf, i);
        stator_topo_register_surface(&reg, REGION_SLOT_INS, s->ins_upper_sf, i);
    }

    /* Assign distributed winding layout */
    stator_topo_assign_winding_layout(&reg, p.winding_type, err, sizeof(err));
    printf("  Winding layout : %s\n", stator_winding_type_to_str(p.winding_type));

    /* ── 6. Generate mesh ────────────────────────────────────────────────── */
    printf("\nStep 5/6  Generating mesh …\n");

    MeshConfig mc;
    stator_mesh_config_init(&mc);
    /* Mesh sizing comes from StatorParams; MeshConfig controls algorithm */
    mc.algorithm_2d = 5;  /* Frontal-Delaunay 2D */
    mc.algorithm_3d = 10; /* HXT 3D             */
    mc.smoothing_passes = p.mesh_boundary_layers;

    MeshGenerator mg;
    if (stator_mesh_generator_init(&mg, backend, &mc, err, sizeof(err)) != STATOR_OK)
    {
        fprintf(stderr, "  ERROR: %s\n", err);
        stator_topo_registry_destroy(&reg);
        stub_gmsh_backend_free(backend);
        return 1;
    }

    MeshResult mesh;
    if (stator_mesh_generate(&mg, &p, &geo, &reg, &mesh) != STATOR_OK)
    {
        fprintf(stderr, "  ERROR: mesh generation failed\n");
        stator_topo_registry_destroy(&reg);
        stub_gmsh_backend_free(backend);
        return 1;
    }
    printf("  Nodes       : %d\n", mesh.n_nodes);
    printf("  Elements 2D : %d\n", mesh.n_elements_2d);
    printf("  Elements 3D : %d\n", mesh.n_elements_3d);
    printf("  Min quality : %.4f\n", mesh.min_quality);
    printf("  Avg quality : %.4f\n", mesh.avg_quality);

    /* ── 7. Export ───────────────────────────────────────────────────────── */
    printf("\nStep 6/6  Exporting results …\n");
    ensure_dir(args.output);

    ExportConfig ec;
    stator_export_config_init(&ec);
    strncpy(ec.output_dir, args.output, sizeof(ec.output_dir) - 1);
    ec.formats = EXPORT_VTK | EXPORT_JSON;

    ExportEngine ee;
    stator_export_engine_init(&ee, backend, NULL, 0);

    ExportResult results[4];
    int n_results = 0;
    stator_export_write_all_sync(&ee, &p, &mesh, &ec, results, 4, &n_results);

    for (int i = 0; i < n_results; i++)
    {
        if (results[i].success)
            printf("  %-6s → %s  (%.1f ms)\n",
                   results[i].format == EXPORT_VTK ? "VTK" : results[i].format == EXPORT_JSON ? "JSON"
                                                         : results[i].format == EXPORT_MSH    ? "MSH"
                                                                                              : "HDF5",
                   results[i].path, results[i].write_time_ms);
        else
            printf("  FAIL  %s\n", results[i].error_message);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    gmsh_finalize(backend);
    stator_topo_registry_destroy(&reg);
    stub_gmsh_backend_free(backend);

    printf("\nDone.  Output → %s\n\n", args.output);
    return 0;
}
