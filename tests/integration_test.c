/**
 * integration_test.c  —  3-D stator geometry + mesh visualisation
 *
 * Builds a full stator cross-section in 2-D (yoke ring, 36 semi-closed slots,
 * double-layer coils, slot-liner insulation), extrudes it into 1000 lamination
 * layers (each 0.35 mm → 350 mm total stack), generates a 3-D tetrahedral
 * mesh, then opens the GMSH FLTK interactive viewer for visual validation.
 *
 * Build:
 *   mkdir -p build && cd build
 *   cmake -DSTATOR_BUILD_INTEGRATION=ON .. && make integration_test
 *
 * Run:
 *   ./integration_test
 */

#include <gmshc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Geometry constants ─────────────────────────────────────────────────── */
#define N_SLOTS              36
#define R_OUTER              0.250   /* m  outer stator radius    */
#define R_INNER              0.151   /* m  bore (tooth-tip face)  */
#define SLOT_DEPTH           0.060   /* m                         */
#define SLOT_WIDTH           0.0120  /* m  slot body width        */
#define SLOT_OPENING         0.0040  /* m  tooth-tip gap          */
#define SLOT_OPENING_DEPTH   0.003   /* m  depth of gap region    */
#define INS_THICK            0.001   /* m  slot-liner insulation  */

/* Lamination stack */
#define N_LAM                1000
#define T_LAM                0.00035 /* m per lamination          */
#define STACK_LEN            (N_LAM * T_LAM)   /* 0.350 m         */

/* Mesh characteristic lengths */
#define MS_YOKE   0.010
#define MS_SLOT   0.005
#define MS_COIL   0.003
#define MS_INS    0.0015

/* ── Error helper ───────────────────────────────────────────────────────── */
#define CHECK(ierr) \
    do { if ((ierr) != 0) { \
        fprintf(stderr, "GMSH error %d at line %d\n", (ierr), __LINE__); \
        gmshFinalize(NULL); exit(1); } } while(0)

/* Append (dim,tag) pair to a flat int array; *n is pair-count */
static void push_dt(int **arr, size_t *n, int dim, int tag)
{
    *arr = realloc(*arr, (*n + 1) * 2 * sizeof(int));
    (*arr)[(*n) * 2]     = dim;
    (*arr)[(*n) * 2 + 1] = tag;
    (*n)++;
}

/* ── Build one slot profile at angle theta=0, then rotate ──────────────── */
typedef struct {
    int slot_surf;   /* full slot outline (merged body + opening) */
    int ins_surf;    /* slot-liner ring                           */
    int coil_a;      /* lower coil winding                        */
    int coil_b;      /* upper coil winding                        */
} SlotSurfs;

static SlotSurfs build_slot(double theta)
{
    int ierr = 0;
    int *outDT = NULL; size_t outDT_n = 0;
    int **mapDT = NULL; size_t *mapDT_n = NULL; size_t mapDT_nn = 0;

    double cx = 0.0;

    /* ---- 1. Slot body rectangle (full depth, full width) ---- */
    int body = gmshModelOccAddRectangle(
        cx - SLOT_WIDTH * 0.5, R_INNER, 0.0,
        SLOT_WIDTH, SLOT_DEPTH,
        -1, 0.0, &ierr); CHECK(ierr);

    /* ---- 2. Tooth-tip opening rectangle ---- */
    int opening = gmshModelOccAddRectangle(
        cx - SLOT_OPENING * 0.5, R_INNER - SLOT_OPENING_DEPTH, 0.0,
        SLOT_OPENING, SLOT_OPENING_DEPTH,
        -1, 0.0, &ierr); CHECK(ierr);

    /* Fuse body + opening → single slot outline */
    {
        int obj[2]  = {2, body};
        int tool[2] = {2, opening};
        gmshModelOccFuse(obj, 2, tool, 2,
                         &outDT, &outDT_n,
                         &mapDT, &mapDT_n, &mapDT_nn,
                         -1, 1, 1, &ierr); CHECK(ierr);
    }
    int slot_surf = (outDT_n >= 2) ? outDT[1] : body;
    gmshFree(outDT); outDT = NULL; outDT_n = 0;
    for (size_t i = 0; i < mapDT_nn; i++) gmshFree(mapDT[i]);
    gmshFree(mapDT); gmshFree(mapDT_n); mapDT = NULL; mapDT_n = NULL;

    /* ---- 3. Insulation liner: annular ring at slot perimeter ---- */
    double ins_y = R_INNER + SLOT_OPENING_DEPTH;
    double ins_h = SLOT_DEPTH - SLOT_OPENING_DEPTH;
    double ins_w = SLOT_WIDTH;

    int ins_outer = gmshModelOccAddRectangle(
        cx - ins_w * 0.5, ins_y, 0.0,
        ins_w, ins_h,
        -1, 0.0, &ierr); CHECK(ierr);

    int ins_inner = gmshModelOccAddRectangle(
        cx - ins_w * 0.5 + INS_THICK, ins_y + INS_THICK, 0.0,
        ins_w - 2.0 * INS_THICK, ins_h - 2.0 * INS_THICK,
        -1, 0.0, &ierr); CHECK(ierr);

    /* Insulation = ins_outer − ins_inner */
    {
        int obj[2]  = {2, ins_outer};
        int tool[2] = {2, ins_inner};
        gmshModelOccCut(obj, 2, tool, 2,
                        &outDT, &outDT_n,
                        &mapDT, &mapDT_n, &mapDT_nn,
                        -1, 1, 1, &ierr); CHECK(ierr);
    }
    int ins_surf = (outDT_n >= 2) ? outDT[1] : ins_outer;
    gmshFree(outDT); outDT = NULL; outDT_n = 0;
    for (size_t i = 0; i < mapDT_nn; i++) gmshFree(mapDT[i]);
    gmshFree(mapDT); gmshFree(mapDT_n); mapDT = NULL; mapDT_n = NULL;

    /* ---- 4. Coil A (lower layer) & Coil B (upper layer) ---- */
    double win_y = ins_y + INS_THICK;
    double win_w = ins_w - 2.0 * INS_THICK;
    double win_h = ins_h - 2.0 * INS_THICK;
    double coil_h = (win_h - INS_THICK) * 0.5;  /* half window, leave gap between coils */

    int coil_a = gmshModelOccAddRectangle(
        cx - win_w * 0.5, win_y, 0.0,
        win_w, coil_h,
        -1, 0.0, &ierr); CHECK(ierr);

    int coil_b = gmshModelOccAddRectangle(
        cx - win_w * 0.5, win_y + coil_h + INS_THICK, 0.0,
        win_w, coil_h,
        -1, 0.0, &ierr); CHECK(ierr);

    /* ---- 5. Rotate all four surfaces by theta about Z ---- */
    int all4[8] = {2, slot_surf, 2, ins_surf, 2, coil_a, 2, coil_b};
    gmshModelOccRotate(all4, 8,
                       0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0,
                       theta, &ierr); CHECK(ierr);

    SlotSurfs ss = { slot_surf, ins_surf, coil_a, coil_b };
    return ss;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  main
 * ════════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    int ierr = 0;

    gmshInitialize(0, NULL, 1, 0, &ierr); CHECK(ierr);
    gmshModelAdd("stator_3d_1000lam", &ierr); CHECK(ierr);
    gmshOptionSetNumber("General.Verbosity", 3, &ierr); CHECK(ierr);

    /* ── 1. Yoke ring ──────────────────────────────────────────────────── */
    printf("=== Building 2-D cross-section ===\n");

    /* addDisk: xc,yc,zc, rx,ry, tag, zAxis,zAxis_n, xAxis,xAxis_n, ierr */
    int d_outer = gmshModelOccAddDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER,
                                      -1, NULL, 0, NULL, 0, &ierr); CHECK(ierr);
    int d_inner = gmshModelOccAddDisk(0.0, 0.0, 0.0, R_INNER, R_INNER,
                                      -1, NULL, 0, NULL, 0, &ierr); CHECK(ierr);

    int *outDT = NULL; size_t outDT_n = 0;
    int **mapDT = NULL; size_t *mapDT_n = NULL; size_t mapDT_nn = 0;

    /* Yoke ring = outer disk − inner disk */
    {
        int obj[2]  = {2, d_outer};
        int tool[2] = {2, d_inner};
        gmshModelOccCut(obj, 2, tool, 2,
                        &outDT, &outDT_n,
                        &mapDT, &mapDT_n, &mapDT_nn,
                        -1, 1, 1, &ierr); CHECK(ierr);
    }
    int yoke_ring = (outDT_n >= 2) ? outDT[1] : d_outer;
    gmshFree(outDT); outDT = NULL; outDT_n = 0;
    for (size_t i = 0; i < mapDT_nn; i++) gmshFree(mapDT[i]);
    gmshFree(mapDT); gmshFree(mapDT_n); mapDT = NULL; mapDT_n = NULL;

    /* ── 2. Build 36 slot cross-sections ────────────────────────────── */
    printf("    Building %d slots (body + insulation + coils) ...\n", N_SLOTS);
    double slot_pitch = 2.0 * M_PI / N_SLOTS;
    SlotSurfs slots[N_SLOTS];
    for (int i = 0; i < N_SLOTS; i++)
        slots[i] = build_slot(i * slot_pitch);

    /* ── 3. Synchronise before boolean operations ───────────────────── */
    gmshModelOccSynchronize(&ierr); CHECK(ierr);

    /* ── 4. Fragment: merge yoke + all slot geometry ────────────────── */
    printf("    Fragmenting yoke + slot bodies ...\n");

    /* Fragment object list: yoke + all slot outlines  */
    int *frag_obj  = NULL; size_t frag_obj_n  = 0;
    int *frag_tool = NULL; size_t frag_tool_n = 0;

    push_dt(&frag_obj, &frag_obj_n, 2, yoke_ring);
    for (int i = 0; i < N_SLOTS; i++) {
        push_dt(&frag_obj,  &frag_obj_n,  2, slots[i].slot_surf);
        push_dt(&frag_tool, &frag_tool_n, 2, slots[i].ins_surf);
        push_dt(&frag_tool, &frag_tool_n, 2, slots[i].coil_a);
        push_dt(&frag_tool, &frag_tool_n, 2, slots[i].coil_b);
    }

    gmshModelOccFragment(frag_obj,  frag_obj_n  * 2,
                         frag_tool, frag_tool_n * 2,
                         &outDT, &outDT_n,
                         &mapDT, &mapDT_n, &mapDT_nn,
                         -1, 1, 1, &ierr); CHECK(ierr);

    free(frag_obj); free(frag_tool);
    gmshFree(outDT);
    for (size_t i = 0; i < mapDT_nn; i++) gmshFree(mapDT[i]);
    gmshFree(mapDT); gmshFree(mapDT_n);

    gmshModelOccSynchronize(&ierr); CHECK(ierr);

    /* ── 5. Physical groups (2-D, before extrusion) ─────────────────── */
    printf("    Assigning physical groups ...\n");

    /* Yoke */
    {
        int tags[1] = {yoke_ring};
        gmshModelAddPhysicalGroup(2, tags, 1, 1, "Yoke", &ierr); CHECK(ierr);
    }

    /* Slots: collect all slot_surf tags */
    {
        int *tags = malloc(N_SLOTS * sizeof(int));
        for (int i = 0; i < N_SLOTS; i++) tags[i] = slots[i].slot_surf;
        gmshModelAddPhysicalGroup(2, tags, N_SLOTS, 2, "SlotBody", &ierr); CHECK(ierr);
        free(tags);
    }

    /* Insulations */
    {
        int *tags = malloc(N_SLOTS * sizeof(int));
        for (int i = 0; i < N_SLOTS; i++) tags[i] = slots[i].ins_surf;
        gmshModelAddPhysicalGroup(2, tags, N_SLOTS, 3, "Insulation", &ierr); CHECK(ierr);
        free(tags);
    }

    /* Coil A (phase A lower) */
    {
        int *tags = malloc(N_SLOTS * sizeof(int));
        for (int i = 0; i < N_SLOTS; i++) tags[i] = slots[i].coil_a;
        gmshModelAddPhysicalGroup(2, tags, N_SLOTS, 4, "CoilA", &ierr); CHECK(ierr);
        free(tags);
    }

    /* Coil B (phase B upper) */
    {
        int *tags = malloc(N_SLOTS * sizeof(int));
        for (int i = 0; i < N_SLOTS; i++) tags[i] = slots[i].coil_b;
        gmshModelAddPhysicalGroup(2, tags, N_SLOTS, 5, "CoilB", &ierr); CHECK(ierr);
        free(tags);
    }

    /* ── 6. Collect all 2-D surfaces and extrude to 3-D ─────────────── */
    printf("=== Extruding to 3-D (%d laminations × %.2f mm = %.0f mm stack) ===\n",
           N_LAM, T_LAM * 1000.0, STACK_LEN * 1000.0);

    /* gmshModelGetEntities(dimTags, dimTags_n, dim, ierr)
     * returns flat (dim,tag) pairs for the given dim.                    */
    int *surf_dt = NULL; size_t surf_dt_n = 0;
    gmshModelGetEntities(&surf_dt, &surf_dt_n, 2, &ierr); CHECK(ierr);

    /* surf_dt is already filtered to dim=2; pairs are (2,tag)  */
    printf("    Extruding %zu 2-D surfaces ...\n", surf_dt_n / 2);

    /* One extrusion step, N_LAM layers, uniform distribution */
    int numElem[1]    = { N_LAM };
    double heights[1] = { 1.0 };

    int *extOut = NULL; size_t extOut_n = 0;
    gmshModelOccExtrude(surf_dt, surf_dt_n,
                        0.0, 0.0, STACK_LEN,
                        &extOut, &extOut_n,
                        numElem, 1,
                        heights, 1,
                        0,   /* recombine=0 → tetrahedra */
                        &ierr); CHECK(ierr);

    gmshFree(surf_dt);
    gmshFree(extOut);

    gmshModelOccSynchronize(&ierr); CHECK(ierr);

    /* ── 7. 3-D physical group: all volumes ─────────────────────────── */
    printf("    Assigning 3-D physical group ...\n");
    {
        int *vol_dt = NULL; size_t vol_dt_n = 0;
        gmshModelGetEntities(&vol_dt, &vol_dt_n, 3, &ierr); CHECK(ierr);

        size_t nv = vol_dt_n / 2;
        int *vtags = malloc(nv * sizeof(int));
        for (size_t k = 0; k < nv; k++)
            vtags[k] = vol_dt[k * 2 + 1];

        gmshFree(vol_dt);
        printf("    Total volumes: %zu\n", nv);

        if (nv > 0)
            gmshModelAddPhysicalGroup(3, vtags, nv, 10, "StatorVolume", &ierr);
        CHECK(ierr);
        free(vtags);
    }

    /* ── 8. Mesh options ─────────────────────────────────────────────── */
    printf("=== Setting mesh options ===\n");
    gmshOptionSetNumber("Mesh.CharacteristicLengthMin", MS_INS  * 0.5, &ierr); CHECK(ierr);
    gmshOptionSetNumber("Mesh.CharacteristicLengthMax", MS_YOKE,       &ierr); CHECK(ierr);
    gmshOptionSetNumber("Mesh.Algorithm",      5, &ierr); CHECK(ierr);  /* Delaunay 2-D */
    gmshOptionSetNumber("Mesh.Algorithm3D",    4, &ierr); CHECK(ierr);  /* Frontal 3-D  */
    gmshOptionSetNumber("Mesh.Optimize",       1, &ierr); CHECK(ierr);
    gmshOptionSetNumber("Mesh.OptimizeNetgen", 1, &ierr); CHECK(ierr);

    /* ── 9. Generate 3-D mesh ────────────────────────────────────────── */
    printf("=== Generating 3-D mesh (may take a few minutes) ===\n");
    gmshModelMeshGenerate(3, &ierr); CHECK(ierr);
    printf("    Mesh generation complete.\n");

    /* ── 10. Save mesh ───────────────────────────────────────────────── */
    const char *out_path = "/tmp/stator_3d_1000lam.msh";
    gmshWrite(out_path, &ierr); CHECK(ierr);
    printf("    Mesh written: %s\n", out_path);

    /* ── 11. FLTK interactive viewer ─────────────────────────────────── */
    printf("=== Opening GMSH FLTK viewer — close window to exit ===\n");
    gmshOptionSetNumber("Mesh.SurfaceFaces", 1, &ierr); CHECK(ierr);
    gmshOptionSetNumber("Mesh.VolumeEdges",  0, &ierr); CHECK(ierr);

    gmshFltkInitialize(&ierr); CHECK(ierr);
    gmshFltkRun(&ierr); CHECK(ierr);

    gmshFinalize(NULL);
    printf("Done.\n");
    return 0;
}
