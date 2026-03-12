// integration_test.cpp — 3-D stator geometry integration test with live visualisation.
//
// Pipeline:
//   1. Load a validated 12-slot StatorParams (make_minimal_params)
//   2. Build the 2-D annular cross-section (yoke + tooth body)
//   3. Cut semi-closed slots around the inner bore
//   4. Extrude the cross-section along Z by stack_length → full 3-D lamination block
//   5. Generate a 3-D tetrahedral mesh
//   6. Open the GMSH FLTK interactive window for visual inspection
//
// Build (from repo root):
//   mkdir -p build && cd build
//   cmake -DSTATOR_BUILD_INTEGRATION=ON ..
//   make integration_test
//   ./integration_test
//
// Controls in the visualisation window:
//   Left-drag   : rotate    |  Right-drag : zoom
//   Middle-drag : pan       |  q / Esc    : quit

#include <gmsh.h>
#include "stator/params.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

using namespace stator;
static constexpr double PI = std::numbers::pi;

// ── Geometry helpers ──────────────────────────────────────────────────────────

// Rotate point (x, y) by `angle` radians around the origin.
static std::pair<double,double> rot2d(double x, double y, double angle) {
    return { x * std::cos(angle) - y * std::sin(angle),
             x * std::sin(angle) + y * std::cos(angle) };
}

// Build the stator annulus: a solid disk at R_outer minus the bore hole at R_inner.
// Returns the tag of the resulting surface (dim=2).
static int build_annulus(double r_outer, double r_inner)
{
    int outer = gmsh::model::occ::addDisk(0.0, 0.0, 0.0, r_outer, r_outer);
    int inner = gmsh::model::occ::addDisk(0.0, 0.0, 0.0, r_inner, r_inner);
    gmsh::vectorpair result;
    std::vector<gmsh::vectorpair> result_map;
    gmsh::model::occ::cut({{2, outer}}, {{2, inner}}, result, result_map, -1, true, true);
    if (result.empty()) throw std::runtime_error("Annulus cut returned no surfaces");
    return result[0].second;
}

// Build a single slot cutout at angular index `k`.
// Shape: rectangular slot body with a narrow tooth-tip opening at the bore.
//
//   bore (R_inner) ──►  | opening |
//                        ___________
//                       |           |   ← slot body (R_inner+δ .. R_inner+slot_depth)
//                       |___________|
//
static int build_slot(const StatorParams& p, int k)
{
    double angle = (2.0 * PI / p.n_slots) * k;

    // Radial extents
    double r0 = p.R_inner;                         // bore face
    double r1 = p.R_inner + p.slot_opening_depth;  // end of tooth tip
    double r2 = p.R_inner + p.slot_depth;          // slot back

    // Half-widths at each level
    double hw_opening = p.slot_opening   * 0.5;
    double hw_body    = p.slot_width_inner * 0.5;

    // Semi-closed shape:  8 vertices in local (unrotated) coordinates.
    //
    //   6---5
    //   |   |
    //   7   4
    //   \   /   ← tooth tips taper to opening
    //   1   2  ← at bore face, r0
    //   centre origin
    //
    //  x-axis  tangential;  y-axis  radial outward.
    struct V { double x, y; };
    std::vector<V> pts = {
        {-hw_opening, r0},   // 0 – left bore edge
        { hw_opening, r0},   // 1 – right bore edge
        { hw_body,    r1},   // 2 – right top of tooth tip
        { hw_body,    r2},   // 3 – right slot back
        {-hw_body,    r2},   // 4 – left slot back
        {-hw_body,    r1},   // 5 – left top of tooth tip
    };

    // Add rotated points
    std::vector<int> ptags;
    ptags.reserve(pts.size());
    for (auto& v : pts) {
        auto [rx, ry] = rot2d(v.x, v.y, angle);
        ptags.push_back(gmsh::model::occ::addPoint(rx, ry, 0.0, 0.0));
    }

    // Lines forming the closed polygon
    int n = static_cast<int>(ptags.size());
    std::vector<int> ltags;
    ltags.reserve(n);
    for (int i = 0; i < n; ++i)
        ltags.push_back(gmsh::model::occ::addLine(ptags[i], ptags[(i+1) % n]));

    int loop = gmsh::model::occ::addCurveLoop(ltags);
    return gmsh::model::occ::addPlaneSurface({loop});
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::cout << "\n=== 3-D Stator Integration Test ===\n\n";

    // ── 1. Validated stator parameters ───────────────────────────────────────
    StatorParams p = make_minimal_params();   // 12-slot design, fully derived

    std::cout << "[PARAMS] Design summary:\n"
              << "         n_slots      = " << p.n_slots       << "\n"
              << "         R_outer      = " << p.R_outer       << " m\n"
              << "         R_inner      = " << p.R_inner       << " m\n"
              << "         slot_depth   = " << p.slot_depth    << " m\n"
              << "         yoke_height  = " << p.yoke_height   << " m\n"
              << "         stack_length = " << p.stack_length  << " m  ("
              << p.n_lam << " lams × " << p.t_lam * 1e3 << " mm)\n\n";

    // ── 2. Initialise GMSH session ────────────────────────────────────────────
    gmsh::initialize(argc, argv);
    gmsh::option::setNumber("General.Terminal",   1);   // echo GMSH log to stdout
    gmsh::option::setNumber("General.Verbosity",  3);
    gmsh::model::add("stator_3d");

    try {
        // ── 3. Build 2-D annular cross-section ────────────────────────────────
        std::cout << "[GEOM]  Building annular yoke (R_inner=" << p.R_inner
                  << " m → R_outer=" << p.R_outer << " m)...\n";
        int annulus = build_annulus(p.R_outer, p.R_inner);

        // ── 4. Build slot cutouts ─────────────────────────────────────────────
        std::cout << "[GEOM]  Creating " << p.n_slots << " slot profiles...\n";
        gmsh::vectorpair slot_dimtags;
        for (int k = 0; k < p.n_slots; ++k) {
            int s = build_slot(p, k);
            slot_dimtags.push_back({2, s});
        }
        gmsh::model::occ::synchronize();

        // ── 5. Boolean cut: yoke − slots ──────────────────────────────────────
        std::cout << "[GEOM]  Cutting slots from yoke (boolean cut)...\n";
        gmsh::vectorpair cross_section;
        std::vector<gmsh::vectorpair> cross_section_map;
        gmsh::model::occ::cut({{2, annulus}}, slot_dimtags, cross_section,
                               cross_section_map, -1, true, true);
        gmsh::model::occ::synchronize();

        if (cross_section.empty())
            throw std::runtime_error("Slot boolean cut produced no remaining surfaces");

        std::cout << "[OK]    2-D cross-section: " << cross_section.size()
                  << " surface fragment(s)\n";

        // ── 6. Extrude 2-D → 3-D ──────────────────────────────────────────────
        std::cout << "[GEOM]  Extruding " << p.stack_length * 1e3
                  << " mm along +Z...\n";
        gmsh::vectorpair extruded;
        gmsh::model::occ::extrude(cross_section, 0.0, 0.0, p.stack_length, extruded);
        gmsh::model::occ::synchronize();

        // Collect extruded volumes
        std::vector<int> vol_tags;
        for (auto& [dim, tag] : extruded)
            if (dim == 3) vol_tags.push_back(tag);

        std::cout << "[OK]    3-D volumes: " << vol_tags.size() << "\n\n";

        // ── 7. Physical groups ─────────────────────────────────────────────────
        if (!vol_tags.empty())
            gmsh::model::addPhysicalGroup(3, vol_tags, -1, "StatorLamination");

        // Tag bottom face (z=0) and top face (z=stack_length) for BCs
        std::vector<std::pair<int,int>> all_bnd;
        gmsh::model::getBoundary(extruded, all_bnd, false, false, true);
        std::vector<int> face_tags;
        for (auto& [d, t] : all_bnd) if (d == 2) face_tags.push_back(t);
        if (!face_tags.empty())
            gmsh::model::addPhysicalGroup(2, face_tags, -1, "StatorSurfaces");

        // ── 8. Mesh sizing ────────────────────────────────────────────────────
        // Global bounds: fine near slots, coarser in yoke
        gmsh::option::setNumber("Mesh.CharacteristicLengthMin",
                                 p.mesh_coil * 0.5);           // ~0.75 mm
        gmsh::option::setNumber("Mesh.CharacteristicLengthMax",
                                 p.mesh_yoke * 2.0);           // ~12 mm
        gmsh::option::setNumber("Mesh.CharacteristicLengthFromCurvature", 1);
        gmsh::option::setNumber("Mesh.MinimumCirclePoints", 20);

        gmsh::option::setNumber("Mesh.Algorithm",   6);   // 2D Frontal-Delaunay
        gmsh::option::setNumber("Mesh.Algorithm3D", 1);   // 3D Delaunay
        gmsh::option::setNumber("Mesh.Optimize",    1);
        gmsh::option::setNumber("Mesh.Smoothing",   5);

        // ── 9. Generate 3-D mesh ──────────────────────────────────────────────
        std::cout << "[MESH]  Generating 3-D mesh (this may take ~10-30 s)...\n";
        gmsh::model::mesh::generate(3);

        // Mesh statistics
        std::vector<std::size_t> ntags;
        std::vector<double> coords, param;
        gmsh::model::mesh::getNodes(ntags, coords, param);

        std::vector<int> etypes;
        std::vector<std::vector<std::size_t>> etags, enodes;
        gmsh::model::mesh::getElements(etypes, etags, enodes);

        std::size_t n_elems = 0;
        for (auto& et : etags) n_elems += et.size();

        std::cout << "[OK]    Mesh statistics:\n"
                  << "         Nodes:    " << ntags.size()  << "\n"
                  << "         Elements: " << n_elems        << "\n\n";

        // ── 10. Interactive GMSH visualisation ────────────────────────────────
        // Display options for a clear 3-D view
        gmsh::option::setNumber("Geometry.Surfaces",       0);
        gmsh::option::setNumber("Geometry.Volumes",        0);
        gmsh::option::setNumber("Mesh.SurfaceEdges",       1);
        gmsh::option::setNumber("Mesh.SurfaceFaces",       1);
        gmsh::option::setNumber("Mesh.VolumeEdges",        0);   // cleaner
        gmsh::option::setNumber("Mesh.VolumeFaces",        0);
        gmsh::option::setNumber("Mesh.ColorCarousel",      2);   // colour by entity
        gmsh::option::setNumber("General.Axes",            1);   // coordinate axes
        gmsh::option::setNumber("General.RotationX",      70.0); // nice isometric angle
        gmsh::option::setNumber("General.RotationY",       0.0);
        gmsh::option::setNumber("General.RotationZ",      20.0);

        std::cout << "╔══════════════════════════════════════════════╗\n"
                  << "║   GMSH 3-D Stator Visualisation              ║\n"
                  << "║                                              ║\n"
                  << "║   Mouse controls:                            ║\n"
                  << "║     Left-drag   → rotate                     ║\n"
                  << "║     Right-drag  → zoom                       ║\n"
                  << "║     Middle-drag → pan                        ║\n"
                  << "║                                              ║\n"
                  << "║   Keyboard:                                  ║\n"
                  << "║     q / Esc     → quit                       ║\n"
                  << "║     m           → toggle mesh display        ║\n"
                  << "║     t           → toggle surface fill        ║\n"
                  << "║     x/y/z       → snap view to axis          ║\n"
                  << "╚══════════════════════════════════════════════╝\n\n";

        gmsh::fltk::run();

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        gmsh::finalize();
        return 1;
    }

    // ── 11. Cleanup ───────────────────────────────────────────────────────────
    gmsh::finalize();
    std::cout << "=== Integration test complete. ===\n\n";
    return 0;
}
