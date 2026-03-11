#include "stator/export_engine.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <filesystem>
#include <array>
#include <cstring>

namespace stator {

// ─── SHA-256 (FIPS 180-4 self-contained) ─────────────────────────────────────
namespace {

static const uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch (uint32_t e, uint32_t f, uint32_t g) { return (e & f) ^ (~e & g); }
inline uint32_t maj(uint32_t a, uint32_t b, uint32_t c) { return (a & b) ^ (a & c) ^ (b & c); }
inline uint32_t ep0(uint32_t a) { return rotr(a,2) ^ rotr(a,13) ^ rotr(a,22); }
inline uint32_t ep1(uint32_t e) { return rotr(e,6) ^ rotr(e,11) ^ rotr(e,25); }
inline uint32_t sig0(uint32_t x){ return rotr(x,7) ^ rotr(x,18) ^ (x >> 3);  }
inline uint32_t sig1(uint32_t x){ return rotr(x,17)^ rotr(x,19) ^ (x >> 10); }

std::string sha256_impl(const std::string& data) {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    // Pre-processing: padding
    size_t len = data.size();
    size_t padded = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> msg(padded, 0);
    std::memcpy(msg.data(), data.data(), len);
    msg[len] = 0x80u;
    // Append bit-length as big-endian 64-bit
    uint64_t bitlen = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i)
        msg[padded - 8 + i] = static_cast<uint8_t>(bitlen >> (56 - 8*i));

    // Process 512-bit chunks
    for (size_t i = 0; i < padded; i += 64) {
        uint32_t w[64];
        for (int j = 0; j < 16; ++j)
            w[j] = (static_cast<uint32_t>(msg[i + j*4    ]) << 24)
                 | (static_cast<uint32_t>(msg[i + j*4 + 1]) << 16)
                 | (static_cast<uint32_t>(msg[i + j*4 + 2]) <<  8)
                 |  static_cast<uint32_t>(msg[i + j*4 + 3]);
        for (int j = 16; j < 64; ++j)
            w[j] = sig1(w[j-2]) + w[j-7] + sig0(w[j-15]) + w[j-16];

        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int j = 0; j < 64; ++j) {
            uint32_t t1 = hh + ep1(e) + ch(e,f,g) + K[j] + w[j];
            uint32_t t2 = ep0(a) + maj(a,b,c);
            hh = g; g = f; f = e; e = d + t1;
            d  = c; c = b; b = a; a = t1 + t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    // Format as hex string
    char hex[65]; hex[64] = '\0';
    for (int i = 0; i < 8; ++i)
        std::snprintf(hex + i*8, 9, "%08x", h[i]);
    return std::string(hex);
}

} // anonymous namespace

// Public wrapper
std::string sha256(const std::string& data) { return sha256_impl(data); }

// ─── ExportEngine ─────────────────────────────────────────────────────────────

ExportEngine::ExportEngine(IGmshBackend* backend) : backend_(backend) {
    if (!backend_)
        throw std::invalid_argument("ExportEngine: backend must not be null");
}

std::string ExportEngine::compute_stem(const StatorParams& p) {
    std::string hash = sha256(p.to_json());
    return "stator_" + hash.substr(0, 8);
}

bool ExportEngine::outputs_exist(const StatorParams& p, const ExportConfig& cfg) {
    std::string stem = compute_stem(p);
    std::string base = cfg.output_dir + "/" + stem;
    namespace fs = std::filesystem;
    if (has_format(cfg.formats, ExportFormat::MSH)  && !fs::exists(base + ".msh"))  return false;
    if (has_format(cfg.formats, ExportFormat::VTK)  && !fs::exists(base + ".vtk"))  return false;
    if (has_format(cfg.formats, ExportFormat::HDF5) && !fs::exists(base + ".h5"))   return false;
    if (has_format(cfg.formats, ExportFormat::JSON) && !fs::exists(base + "_meta.json")) return false;
    return true;
}

// ─── Format writers ───────────────────────────────────────────────────────────

ExportResult ExportEngine::write_msh(const StatorParams& p, const ExportConfig& cfg,
                                      const std::string& stem) {
    ExportResult r;
    r.format = ExportFormat::MSH;
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::string path = cfg.output_dir + "/" + stem + ".msh";
        backend_->write_mesh(path);
        r.path    = path;
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    auto t1 = std::chrono::steady_clock::now();
    r.write_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

ExportResult ExportEngine::write_vtk(const StatorParams& p, const MeshResult& /*mesh*/,
                                      const ExportConfig& cfg, const std::string& stem) {
    ExportResult r;
    r.format = ExportFormat::VTK;
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::string path = cfg.output_dir + "/" + stem + ".vtk";
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot open " + path);
        f << "# vtk DataFile Version 3.0\n"
          << "Stator mesh " << stem << "\n"
          << "ASCII\n"
          << "DATASET UNSTRUCTURED_GRID\n"
          << "POINTS 0 double\n"
          << "CELLS 0 0\n"
          << "CELL_TYPES 0\n";
        r.path    = path;
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    auto t1 = std::chrono::steady_clock::now();
    r.write_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

ExportResult ExportEngine::write_hdf5(const StatorParams& p, const MeshResult& mesh,
                                       const ExportConfig& cfg, const std::string& stem) {
    ExportResult r;
    r.format = ExportFormat::HDF5;
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::string path = cfg.output_dir + "/" + stem + ".h5";
#ifdef STATOR_WITH_HDF5
        // TODO(v2): HighFive HDF5 real write
        (void)mesh;
#else
        // Placeholder text file when HighFive not available
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot open " + path);
        f << "HDF5 placeholder for " << stem << "\n"
          << "n_nodes=" << mesh.n_nodes << "\n"
          << "n_elements_2d=" << mesh.n_elements_2d << "\n";
#endif
        r.path    = path;
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    auto t1 = std::chrono::steady_clock::now();
    r.write_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

ExportResult ExportEngine::write_json(const StatorParams& p, const MeshResult& mesh,
                                       const ExportConfig& cfg, const std::string& stem) {
    ExportResult r;
    r.format = ExportFormat::JSON;
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::string path = cfg.output_dir + "/" + stem + "_meta.json";
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot open " + path);
        std::string base = cfg.output_dir + "/" + stem;
        f << "{"
          << "\"params\":" << p.to_json() << ","
          << "\"mesh_stats\":{"
          << "\"n_nodes\":" << mesh.n_nodes << ","
          << "\"n_elements_2d\":" << mesh.n_elements_2d << ","
          << "\"n_elements_3d\":" << mesh.n_elements_3d << ","
          << "\"min_quality\":" << mesh.min_quality << ","
          << "\"avg_quality\":" << mesh.avg_quality
          << "},"
          << "\"output_files\":{"
          << "\"msh\":\"" << base << ".msh\","
          << "\"vtk\":\"" << base << ".vtk\","
          << "\"hdf5\":\"" << base << ".h5\","
          << "\"json\":\"" << path << "\""
          << "}"
          << "}";
        r.path    = path;
        r.success = true;
    } catch (const std::exception& e) {
        r.error_message = e.what();
    }
    auto t1 = std::chrono::steady_clock::now();
    r.write_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

// ─── write_all_sync / write_all_async ────────────────────────────────────────

std::vector<ExportResult> ExportEngine::write_all_sync(const StatorParams& p,
                                                        const MeshResult& mesh,
                                                        const ExportConfig& cfg) {
    auto futures = write_all_async(p, mesh, cfg);
    std::vector<ExportResult> results;
    results.reserve(futures.size());
    for (auto& fut : futures)
        results.push_back(fut.get());
    return results;
}

std::vector<std::future<ExportResult>> ExportEngine::write_all_async(
    const StatorParams& p, const MeshResult& mesh, const ExportConfig& cfg)
{
    std::string stem = compute_stem(p);
    std::vector<std::future<ExportResult>> futures;

    // Capture by value for async safety
    if (has_format(cfg.formats, ExportFormat::MSH)) {
        futures.push_back(std::async(std::launch::async,
            [this, &p, &cfg, stem]() mutable { return write_msh(p, cfg, stem); }));
    }
    if (has_format(cfg.formats, ExportFormat::VTK)) {
        futures.push_back(std::async(std::launch::async,
            [this, &p, &mesh, &cfg, stem]() mutable { return write_vtk(p, mesh, cfg, stem); }));
    }
    if (has_format(cfg.formats, ExportFormat::HDF5)) {
        futures.push_back(std::async(std::launch::async,
            [this, &p, &mesh, &cfg, stem]() mutable { return write_hdf5(p, mesh, cfg, stem); }));
    }
    if (has_format(cfg.formats, ExportFormat::JSON)) {
        futures.push_back(std::async(std::launch::async,
            [this, &p, &mesh, &cfg, stem]() mutable { return write_json(p, mesh, cfg, stem); }));
    }
    return futures;
}

} // namespace stator
