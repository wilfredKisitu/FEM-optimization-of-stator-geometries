#include "stator/gmsh_backend.hpp"
#include <iostream>
#include <stdexcept>

namespace stator {

// ─── StubGmshBackend ─────────────────────────────────────────────────────────

void StubGmshBackend::reset() {
    initialized_      = false;
    sync_count_       = 0;
    finalized_        = false;
    mesh_generated_   = false;
    background_field_ = -1;
    last_write_path_.clear();
    point_counter_        = 0;
    line_counter_         = 0;
    surface_counter_      = 0;
    curve_loop_counter_   = 0;
    field_counter_        = 0;
    phys_group_tag_counter_ = 1000;
    phys_groups_.clear();
    surfaces_2d_.clear();
}

void StubGmshBackend::initialize(const std::string& /*model_name*/) {
    initialized_ = true;
}

void StubGmshBackend::synchronize() { ++sync_count_; }

void StubGmshBackend::finalize() { finalized_ = true; }

void StubGmshBackend::set_option(const std::string& /*name*/, double /*value*/) {}

int StubGmshBackend::add_point(double, double, double, double) {
    return ++point_counter_;
}

int StubGmshBackend::add_line(int, int) { return ++line_counter_; }

int StubGmshBackend::add_circle(double, double, double, double) {
    return ++line_counter_;
}

int StubGmshBackend::add_arc(int, int, int) { return ++line_counter_; }

int StubGmshBackend::add_curve_loop(const std::vector<int>&) {
    return ++curve_loop_counter_;
}

int StubGmshBackend::add_plane_surface(const std::vector<int>&) {
    int tag = ++surface_counter_;
    surfaces_2d_.push_back({2, tag});
    return tag;
}

std::vector<std::pair<int,int>> StubGmshBackend::boolean_cut(
    const std::vector<std::pair<int,int>>& objects,
    const std::vector<std::pair<int,int>>&, bool)
{
    return objects; // stub: unchanged
}

std::vector<std::pair<int,int>> StubGmshBackend::boolean_fragment(
    const std::vector<std::pair<int,int>>& objects,
    const std::vector<std::pair<int,int>>& tools)
{
    auto r = objects;
    r.insert(r.end(), tools.begin(), tools.end());
    return r;
}

int StubGmshBackend::add_physical_group(int dim, const std::vector<int>& tags,
                                         const std::string& name, int tag) {
    int t = (tag >= 0) ? tag : ++phys_group_tag_counter_;
    phys_groups_.push_back({dim, tags, name, t});
    return t;
}

int StubGmshBackend::add_math_eval_field(const std::string&) {
    return ++field_counter_;
}

int StubGmshBackend::add_constant_field(double, const std::vector<int>&) {
    return ++field_counter_;
}

void StubGmshBackend::set_background_field(int field_tag) {
    background_field_ = field_tag;
}

void StubGmshBackend::generate_mesh(int) { mesh_generated_ = true; }

void StubGmshBackend::write_mesh(const std::string& filename) {
    last_write_path_ = filename;
}

std::vector<std::pair<int,int>> StubGmshBackend::get_entities_2d() {
    return surfaces_2d_;
}

// ─── Factory ──────────────────────────────────────────────────────────────────

std::unique_ptr<IGmshBackend> make_default_backend() {
#ifdef STATOR_WITH_GMSH
    extern std::unique_ptr<IGmshBackend> make_real_gmsh_backend();
    return make_real_gmsh_backend();
#else
    std::cerr << "[stator] Warning: STATOR_WITH_GMSH not defined; using StubGmshBackend.\n";
    return std::make_unique<StubGmshBackend>();
#endif
}

} // namespace stator
