#pragma once
#include "stator/params.hpp"
#include "stator/gmsh_backend.hpp"
#include "stator/topology_registry.hpp"
#include "stator/geometry_builder.hpp"
#include <string>

namespace stator {

// ─── MeshConfig ───────────────────────────────────────────────────────────────

struct MeshConfig {
    int         algorithm_2d          = 5;      // 5=Delaunay, 6=Frontal-Delaunay, 8=Delaunay-quads
    int         algorithm_3d          = 10;     // 10=HXT, 1=Delaunay, 4=Frontal
    int         smoothing_passes      = 3;
    std::string optimiser             = "Netgen";
    double      min_quality_threshold = 0.3;
    bool        periodic              = false;
    int         layers_per_lam        = 2;
};

// ─── MeshResult ──────────────────────────────────────────────────────────────

struct MeshResult {
    bool        success       = false;
    std::string error_message;
    int         n_nodes       = 0;
    int         n_elements_2d = 0;
    int         n_elements_3d = 0;
    double      min_quality   = 0.0;
    double      avg_quality   = 0.0;
    int         n_phys_groups = 0;
};

// ─── MeshGenerator ───────────────────────────────────────────────────────────

class MeshGenerator {
public:
    MeshGenerator(IGmshBackend* backend, const MeshConfig& config = {});

    // Generate mesh for the given geometry and register physical groups.
    MeshResult generate(const StatorParams& p,
                        const GeometryBuildResult& geo,
                        TopologyRegistry& registry);

    // Only assign physical groups (no mesh generation).
    void assign_physical_groups(const StatorParams& p,
                                 const GeometryBuildResult& geo,
                                 TopologyRegistry& registry);

private:
    IGmshBackend* backend_;
    MeshConfig    config_;

    // Layer A: per-surface constant size fields.
    void add_constant_fields(const StatorParams& p,
                              const TopologyRegistry& registry);

    // Layer B: mouth transition Distance+Threshold fields.
    void add_mouth_transition_fields(const StatorParams& p,
                                      const GeometryBuildResult& geo);

    // Layer C: bore boundary-layer field.
    void add_bore_boundary_layer(const StatorParams& p,
                                  const TopologyRegistry& registry);

    // Combine all fields into a Min field and set as background.
    void combine_and_set_background(const std::vector<int>& field_tags);
};

} // namespace stator
