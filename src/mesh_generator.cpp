#include "stator/mesh_generator.hpp"
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stator {

// ─── Constructor ──────────────────────────────────────────────────────────────

MeshGenerator::MeshGenerator(IGmshBackend* backend, const MeshConfig& config)
    : backend_(backend), config_(config) {
    if (!backend_)
        throw std::invalid_argument("MeshGenerator: backend must not be null");
}

// ─── assign_physical_groups ───────────────────────────────────────────────────

void MeshGenerator::assign_physical_groups(const StatorParams& p,
                                            const GeometryBuildResult& geo,
                                            TopologyRegistry& registry) {
    // Register yoke surface
    if (geo.yoke_surface >= 0) {
        registry.register_surface(RegionType::YOKE, geo.yoke_surface);
        backend_->add_physical_group(2, {geo.yoke_surface},
            to_string(RegionType::YOKE),
            canonical_tag(RegionType::YOKE));
    }

    // Register boundary curves
    if (geo.bore_curve >= 0) {
        registry.register_boundary_curve(RegionType::BOUNDARY_BORE, geo.bore_curve);
        backend_->add_physical_group(1, {geo.bore_curve},
            to_string(RegionType::BOUNDARY_BORE),
            canonical_tag(RegionType::BOUNDARY_BORE));
    }
    if (geo.outer_curve >= 0) {
        registry.register_boundary_curve(RegionType::BOUNDARY_OUTER, geo.outer_curve);
        backend_->add_physical_group(1, {geo.outer_curve},
            to_string(RegionType::BOUNDARY_OUTER),
            canonical_tag(RegionType::BOUNDARY_OUTER));
    }

    // Register per-slot surfaces
    for (int k = 0; k < static_cast<int>(geo.slots.size()); ++k) {
        const auto& sp = geo.slots[k];

        if (sp.slot_surface >= 0)
            registry.register_surface(RegionType::SLOT_AIR, sp.slot_surface, k);

        if (sp.coil_upper_sf >= 0 || sp.coil_lower_sf >= 0)
            registry.register_slot_coil(k, sp.coil_upper_sf, sp.coil_lower_sf);

        if (sp.ins_upper_sf >= 0)
            registry.register_surface(RegionType::SLOT_INS, sp.ins_upper_sf, k);
        if (sp.ins_lower_sf >= 0)
            registry.register_surface(RegionType::SLOT_INS, sp.ins_lower_sf, k);
    }

    // Assign winding layout
    registry.assign_winding_layout(p.winding_type);

    // Create GMSH physical groups for coil regions per slot
    // Aggregate tags by region type
    std::vector<int> slot_air_tags;
    std::vector<int> slot_ins_tags;
    std::vector<int> coil_a_pos, coil_a_neg, coil_b_pos, coil_b_neg, coil_c_pos, coil_c_neg;

    slot_air_tags = registry.get_surfaces(RegionType::SLOT_AIR);
    slot_ins_tags = registry.get_surfaces(RegionType::SLOT_INS);

    for (const auto& wa : registry.get_winding_assignments()) {
        auto add_tag = [](std::vector<int>& v, int tag) {
            if (tag >= 0) v.push_back(tag);
        };
        switch (wa.upper_phase) {
            case RegionType::COIL_A_POS: add_tag(coil_a_pos, wa.upper_tag); break;
            case RegionType::COIL_A_NEG: add_tag(coil_a_neg, wa.upper_tag); break;
            case RegionType::COIL_B_POS: add_tag(coil_b_pos, wa.upper_tag); break;
            case RegionType::COIL_B_NEG: add_tag(coil_b_neg, wa.upper_tag); break;
            case RegionType::COIL_C_POS: add_tag(coil_c_pos, wa.upper_tag); break;
            case RegionType::COIL_C_NEG: add_tag(coil_c_neg, wa.upper_tag); break;
            default: break;
        }
        switch (wa.lower_phase) {
            case RegionType::COIL_A_POS: add_tag(coil_a_pos, wa.lower_tag); break;
            case RegionType::COIL_A_NEG: add_tag(coil_a_neg, wa.lower_tag); break;
            case RegionType::COIL_B_POS: add_tag(coil_b_pos, wa.lower_tag); break;
            case RegionType::COIL_B_NEG: add_tag(coil_b_neg, wa.lower_tag); break;
            case RegionType::COIL_C_POS: add_tag(coil_c_pos, wa.lower_tag); break;
            case RegionType::COIL_C_NEG: add_tag(coil_c_neg, wa.lower_tag); break;
            default: break;
        }
    }

    auto add_group = [&](RegionType rt, const std::vector<int>& tags, int dim = 2) {
        if (!tags.empty())
            backend_->add_physical_group(dim, tags, to_string(rt), canonical_tag(rt));
    };

    add_group(RegionType::SLOT_AIR,  slot_air_tags);
    add_group(RegionType::SLOT_INS,  slot_ins_tags);
    add_group(RegionType::COIL_A_POS, coil_a_pos);
    add_group(RegionType::COIL_A_NEG, coil_a_neg);
    add_group(RegionType::COIL_B_POS, coil_b_pos);
    add_group(RegionType::COIL_B_NEG, coil_b_neg);
    add_group(RegionType::COIL_C_POS, coil_c_pos);
    add_group(RegionType::COIL_C_NEG, coil_c_neg);
}

// ─── add_constant_fields (Layer A) ───────────────────────────────────────────

void MeshGenerator::add_constant_fields(const StatorParams& p,
                                          const TopologyRegistry& registry) {
    auto make_field = [&](RegionType rt, double size) {
        auto tags = registry.get_surfaces(rt);
        if (!tags.empty())
            backend_->add_constant_field(size, tags);
    };
    make_field(RegionType::YOKE,      p.mesh_yoke);
    make_field(RegionType::TOOTH,     p.mesh_yoke);
    make_field(RegionType::SLOT_AIR,  p.mesh_slot);
    make_field(RegionType::SLOT_INS,  p.mesh_ins);
    // All coil regions at mesh_coil
    for (auto rt : {RegionType::COIL_A_POS, RegionType::COIL_A_NEG,
                    RegionType::COIL_B_POS, RegionType::COIL_B_NEG,
                    RegionType::COIL_C_POS, RegionType::COIL_C_NEG}) {
        make_field(rt, p.mesh_coil);
    }
}

// ─── add_mouth_transition_fields (Layer B) ───────────────────────────────────

void MeshGenerator::add_mouth_transition_fields(const StatorParams& p,
                                                  const GeometryBuildResult& geo) {
    // Collect all mouth curve tags
    std::vector<int> mouth_curves;
    for (const auto& sp : geo.slots) {
        if (sp.mouth_curve_bot >= 0) mouth_curves.push_back(sp.mouth_curve_bot);
    }
    if (mouth_curves.empty()) return;

    // TODO(v2): RealGmshBackend per-field GMSH API calls
    // For stub: create a math-eval field representing the threshold
    std::ostringstream expr;
    expr << "Threshold{" << p.mesh_slot << "," << p.mesh_yoke
         << "," << p.slot_depth / 4.0 << "}";
    backend_->add_math_eval_field(expr.str());
}

// ─── add_bore_boundary_layer (Layer C) ───────────────────────────────────────

void MeshGenerator::add_bore_boundary_layer(const StatorParams& p,
                                              const TopologyRegistry& registry) {
    // TODO(v2): RealGmshBackend per-field GMSH API calls
    // For stub: represent as a math-eval field
    std::ostringstream expr;
    expr << "BoundaryLayer{size=" << p.mesh_ins
         << ",ratio=1.2,NbLayers=" << p.mesh_boundary_layers << "}";
    backend_->add_math_eval_field(expr.str());
}

// ─── combine_and_set_background ──────────────────────────────────────────────

void MeshGenerator::combine_and_set_background(const std::vector<int>& field_tags) {
    if (field_tags.empty()) return;
    // Min field represented as a math-eval combining all layer fields
    std::ostringstream expr;
    expr << "Min{";
    for (int i = 0; i < static_cast<int>(field_tags.size()); ++i) {
        if (i) expr << ",";
        expr << "F" << field_tags[i];
    }
    expr << "}";
    int min_field = backend_->add_math_eval_field(expr.str());
    backend_->set_background_field(min_field);
}

// ─── generate ─────────────────────────────────────────────────────────────────

MeshResult MeshGenerator::generate(const StatorParams& p,
                                    const GeometryBuildResult& geo,
                                    TopologyRegistry& registry) {
    MeshResult result;
    try {
        if (!geo.success) {
            result.success = false;
            result.error_message = "Geometry build failed: " + geo.error_message;
            return result;
        }

        // Assign physical groups
        assign_physical_groups(p, geo, registry);
        result.n_phys_groups = static_cast<int>(registry.get_surfaces(RegionType::YOKE).size());

        // Size fields — collect tags
        std::vector<int> all_field_tags;
        {
            auto fa = backend_->add_constant_field(p.mesh_yoke,
                            registry.get_surfaces(RegionType::YOKE));
            if (fa) all_field_tags.push_back(fa);
        }
        add_constant_fields(p, registry);
        add_mouth_transition_fields(p, geo);
        add_bore_boundary_layer(p, registry);
        // Use a single min-field for background
        combine_and_set_background({1}); // stub: field tag 1 representative

        // Set mesh algorithm options
        backend_->set_option("Mesh.Algorithm",  static_cast<double>(config_.algorithm_2d));
        backend_->set_option("Mesh.Smoothing",  static_cast<double>(config_.smoothing_passes));

        // Generate 2D mesh
        backend_->generate_mesh(2);

        // If multi-lamination, generate 3D by extrusion
        if (p.n_lam > 1) {
            // TODO(v2): actual gmsh extrude calls
            backend_->generate_mesh(3);
            result.n_elements_3d = p.n_lam * 10; // stub
        }

        result.success       = true;
        result.n_nodes        = 100;  // stub values
        result.n_elements_2d  = 200;
        result.min_quality    = 0.5;
        result.avg_quality    = 0.8;
        result.n_phys_groups  = backend_->get_entities_2d().size();

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    return result;
}

} // namespace stator
