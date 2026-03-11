#pragma once
#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace stator {

// ─── PhysGroupRecord ──────────────────────────────────────────────────────────

struct PhysGroupRecord {
    int              dim;
    std::vector<int> tags;
    std::string      name;
    int              tag;
};

// ─── IGmshBackend ─────────────────────────────────────────────────────────────
// Abstract interface decoupling geometry code from GMSH C library.
// Unit tests use StubGmshBackend; production uses RealGmshBackend.

class IGmshBackend {
public:
    virtual ~IGmshBackend() = default;

    // ── Session lifecycle ────────────────────────────────────────────────────
    virtual void initialize(const std::string& model_name) = 0;
    virtual void synchronize() = 0;
    virtual void finalize() = 0;
    virtual void set_option(const std::string& name, double value) = 0;

    // ── OCC geometry primitives ──────────────────────────────────────────────
    virtual int add_point(double x, double y, double z, double mesh_size = 0.0) = 0;
    virtual int add_line(int start, int end) = 0;
    virtual int add_circle(double cx, double cy, double cz, double radius) = 0;
    virtual int add_arc(int start, int centre, int end) = 0;
    virtual int add_curve_loop(const std::vector<int>& tags) = 0;
    virtual int add_plane_surface(const std::vector<int>& loop_tags) = 0;

    // ── Boolean operations ───────────────────────────────────────────────────
    virtual std::vector<std::pair<int,int>> boolean_cut(
        const std::vector<std::pair<int,int>>& objects,
        const std::vector<std::pair<int,int>>& tools,
        bool remove_tool = true) = 0;

    virtual std::vector<std::pair<int,int>> boolean_fragment(
        const std::vector<std::pair<int,int>>& objects,
        const std::vector<std::pair<int,int>>& tools) = 0;

    // ── Physical groups ──────────────────────────────────────────────────────
    virtual int add_physical_group(int dim, const std::vector<int>& tags,
                                   const std::string& name, int tag = -1) = 0;

    // ── Mesh fields ──────────────────────────────────────────────────────────
    virtual int  add_math_eval_field(const std::string& expr) = 0;
    virtual int  add_constant_field(double value, const std::vector<int>& surfaces) = 0;
    virtual void set_background_field(int field_tag) = 0;

    // ── Mesh generation and I/O ──────────────────────────────────────────────
    virtual void generate_mesh(int dim) = 0;
    virtual void write_mesh(const std::string& filename) = 0;
    virtual std::vector<std::pair<int,int>> get_entities_2d() = 0;
};

// ─── StubGmshBackend ─────────────────────────────────────────────────────────
// Fully functional stub for unit testing without GMSH installed.

class StubGmshBackend : public IGmshBackend {
public:
    StubGmshBackend() { reset(); }

    // ── Session lifecycle ────────────────────────────────────────────────────
    void initialize(const std::string& model_name) override;
    void synchronize() override;
    void finalize() override;
    void set_option(const std::string& name, double value) override;

    // ── OCC geometry primitives ──────────────────────────────────────────────
    int add_point(double x, double y, double z, double mesh_size = 0.0) override;
    int add_line(int start, int end) override;
    int add_circle(double cx, double cy, double cz, double radius) override;
    int add_arc(int start, int centre, int end) override;
    int add_curve_loop(const std::vector<int>& tags) override;
    int add_plane_surface(const std::vector<int>& loop_tags) override;

    // ── Boolean operations ───────────────────────────────────────────────────
    std::vector<std::pair<int,int>> boolean_cut(
        const std::vector<std::pair<int,int>>& objects,
        const std::vector<std::pair<int,int>>& tools,
        bool remove_tool = true) override;

    std::vector<std::pair<int,int>> boolean_fragment(
        const std::vector<std::pair<int,int>>& objects,
        const std::vector<std::pair<int,int>>& tools) override;

    // ── Physical groups ──────────────────────────────────────────────────────
    int add_physical_group(int dim, const std::vector<int>& tags,
                           const std::string& name, int tag = -1) override;

    // ── Mesh fields ──────────────────────────────────────────────────────────
    int  add_math_eval_field(const std::string& expr) override;
    int  add_constant_field(double value, const std::vector<int>& surfaces) override;
    void set_background_field(int field_tag) override;

    // ── Mesh generation and I/O ──────────────────────────────────────────────
    void generate_mesh(int dim) override;
    void write_mesh(const std::string& filename) override;
    std::vector<std::pair<int,int>> get_entities_2d() override;

    // ── Inspection / test helpers ────────────────────────────────────────────
    int  point_count()         const { return point_counter_;  }
    int  line_count()          const { return line_counter_;   }
    int  surface_count()       const { return surface_counter_; }
    int  field_count()         const { return field_counter_;  }
    int  physical_group_count()const { return static_cast<int>(phys_groups_.size()); }
    bool was_initialized()     const { return initialized_;    }
    bool was_synchronized()    const { return sync_count_ > 0; }
    bool was_finalized()       const { return finalized_;      }
    int  sync_count()          const { return sync_count_;     }
    bool mesh_generated()      const { return mesh_generated_; }
    int  background_field()    const { return background_field_; }
    const std::string& last_write_path() const { return last_write_path_; }
    const std::vector<PhysGroupRecord>& physical_groups() const { return phys_groups_; }

    // Reset all state to allow reuse between test cases.
    void reset();

private:
    bool        initialized_    = false;
    int         sync_count_     = 0;
    bool        finalized_      = false;
    bool        mesh_generated_ = false;
    int         background_field_ = -1;
    std::string last_write_path_;

    int point_counter_   = 0;
    int line_counter_    = 0;
    int surface_counter_ = 0;
    int curve_loop_counter_ = 0;
    int field_counter_   = 0;
    int phys_group_tag_counter_ = 1000;

    std::vector<PhysGroupRecord>           phys_groups_;
    std::vector<std::pair<int,int>>        surfaces_2d_;
};

// ─── Factory ──────────────────────────────────────────────────────────────────
// Returns RealGmshBackend when STATOR_WITH_GMSH defined, else StubGmshBackend.

std::unique_ptr<IGmshBackend> make_default_backend();

} // namespace stator
