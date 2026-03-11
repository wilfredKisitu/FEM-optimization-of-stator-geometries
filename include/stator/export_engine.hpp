#pragma once
#include "stator/params.hpp"
#include "stator/mesh_generator.hpp"
#include "stator/gmsh_backend.hpp"
#include <string>
#include <vector>
#include <future>

namespace stator {

// ─── ExportFormat bitmask enum ───────────────────────────────────────────────

enum class ExportFormat : uint32_t {
    NONE = 0,
    MSH  = 1u << 0,
    VTK  = 1u << 1,
    HDF5 = 1u << 2,
    JSON = 1u << 3,
    ALL  = MSH | VTK | HDF5 | JSON
};

inline ExportFormat operator|(ExportFormat a, ExportFormat b) {
    return static_cast<ExportFormat>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline ExportFormat operator&(ExportFormat a, ExportFormat b) {
    return static_cast<ExportFormat>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline bool has_format(ExportFormat mask, ExportFormat f) {
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(f)) != 0;
}

// ─── ExportConfig ─────────────────────────────────────────────────────────────

struct ExportConfig {
    ExportFormat formats    = ExportFormat::ALL;
    std::string  output_dir = ".";
    int          msh_version = 4;   // 2 or 4
};

// ─── ExportResult ─────────────────────────────────────────────────────────────

struct ExportResult {
    bool        success      = false;
    ExportFormat format      = ExportFormat::NONE;
    std::string path;
    std::string error_message;
    double      write_time_ms = 0.0;
};

// ─── ExportEngine ─────────────────────────────────────────────────────────────

class ExportEngine {
public:
    explicit ExportEngine(IGmshBackend* backend);

    // Compute the output stem: "stator_" + sha256(params.to_json()).substr(0,8)
    static std::string compute_stem(const StatorParams& p);

    // Check whether all output files for the given formats already exist.
    static bool outputs_exist(const StatorParams& p, const ExportConfig& cfg);

    // Write all requested formats synchronously (joins all async tasks first).
    std::vector<ExportResult> write_all_sync(const StatorParams& p,
                                              const MeshResult& mesh,
                                              const ExportConfig& cfg);

    // Launch async tasks and return futures (caller must join before finalize).
    std::vector<std::future<ExportResult>> write_all_async(const StatorParams& p,
                                                            const MeshResult& mesh,
                                                            const ExportConfig& cfg);

private:
    IGmshBackend* backend_;

    ExportResult write_msh (const StatorParams& p, const ExportConfig& cfg,
                             const std::string& stem);
    ExportResult write_vtk (const StatorParams& p, const MeshResult& mesh,
                             const ExportConfig& cfg, const std::string& stem);
    ExportResult write_hdf5(const StatorParams& p, const MeshResult& mesh,
                             const ExportConfig& cfg, const std::string& stem);
    ExportResult write_json(const StatorParams& p, const MeshResult& mesh,
                             const ExportConfig& cfg, const std::string& stem);
};

// SHA-256 utility (FIPS 180-4 self-contained implementation).
std::string sha256(const std::string& data);

} // namespace stator
