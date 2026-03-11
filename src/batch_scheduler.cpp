#include "stator/batch_scheduler.hpp"
#include "stator/gmsh_backend.hpp"
#include "stator/geometry_builder.hpp"
#include "stator/topology_registry.hpp"
#include "stator/mesh_generator.hpp"
#include "stator/export_engine.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <filesystem>

// POSIX
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

namespace stator {

// ─── execute_job ──────────────────────────────────────────────────────────────
// Runs inside the child process. Returns 0 on success.

int BatchScheduler::execute_job(const BatchJob& job, const std::string& status_path) {
    std::string error_msg;
    std::string msh_path, vtk_path, hdf5_path, json_path;
    bool success = false;

    try {
        // Step 1: validate params
        BatchJob mutable_job = job;
        mutable_job.params.validate_and_derive();

        // Step 2: backend
        auto backend = make_default_backend();

        // Step 3: initialize
        std::string stem = ExportEngine::compute_stem(mutable_job.params);
        backend->initialize("stator_" + stem);

        // Step 4: geometry
        GeometryBuilder builder(backend.get());
        auto geo = builder.build(mutable_job.params);
        if (!geo.success)
            throw std::runtime_error("Geometry build failed: " + geo.error_message);

        // Step 5-6: topology + winding
        TopologyRegistry registry(mutable_job.params.n_slots);
        // (physical group assignment happens inside MeshGenerator::generate)

        // Step 7-8: mesh
        MeshGenerator mesher(backend.get(), mutable_job.mesh_config);
        auto mesh_result = mesher.generate(mutable_job.params, geo, registry);
        if (!mesh_result.success)
            throw std::runtime_error("Mesh generation failed: " + mesh_result.error_message);

        // Steps 9-10: export
        namespace fs = std::filesystem;
        fs::create_directories(mutable_job.export_config.output_dir);
        ExportEngine exporter(backend.get());
        auto export_results = exporter.write_all_sync(
            mutable_job.params, mesh_result, mutable_job.export_config);

        std::string base = mutable_job.export_config.output_dir + "/" + stem;
        for (auto& er : export_results) {
            if (!er.success) continue;
            if (er.format == ExportFormat::MSH)  msh_path  = er.path;
            if (er.format == ExportFormat::VTK)  vtk_path  = er.path;
            if (er.format == ExportFormat::HDF5) hdf5_path = er.path;
            if (er.format == ExportFormat::JSON) json_path = er.path;
        }

        // Step 11: finalize
        backend->finalize();
        success = true;

    } catch (const std::exception& e) {
        error_msg = e.what();
    } catch (...) {
        error_msg = "Unknown exception in execute_job";
    }

    // Write status JSON
    try {
        std::ofstream f(status_path);
        f << "{"
          << "\"job_id\":\"" << job.job_id << "\","
          << "\"success\":" << (success ? "true" : "false") << ","
          << "\"error\":\"" << error_msg << "\","
          << "\"msh_path\":\"" << msh_path << "\","
          << "\"vtk_path\":\"" << vtk_path << "\","
          << "\"hdf5_path\":\"" << hdf5_path << "\","
          << "\"json_path\":\"" << json_path << "\""
          << "}";
    } catch (...) {
        // Cannot write status — fall through
    }

    return success ? 0 : 1;
}

// ─── read_status_json ─────────────────────────────────────────────────────────

BatchResult BatchScheduler::read_status_json(const std::string& path,
                                              const std::string& job_id) {
    BatchResult r;
    r.job_id = job_id;
    try {
        std::ifstream f(path);
        if (!f) { r.error = "Cannot open status file: " + path; return r; }
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        // Minimal JSON field extraction (no external JSON lib)
        auto extract = [&](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\":\"";
            auto pos = content.find(search);
            if (pos == std::string::npos) return "";
            pos += search.size();
            auto end = content.find('"', pos);
            return content.substr(pos, end - pos);
        };
        auto extract_bool = [&](const std::string& key) -> bool {
            std::string search = "\"" + key + "\":";
            auto pos = content.find(search);
            if (pos == std::string::npos) return false;
            pos += search.size();
            return content.substr(pos, 4) == "true";
        };

        r.success   = extract_bool("success");
        r.error     = extract("error");
        r.msh_path  = extract("msh_path");
        r.vtk_path  = extract("vtk_path");
        r.hdf5_path = extract("hdf5_path");
        r.json_path = extract("json_path");
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

// ─── write_summary ────────────────────────────────────────────────────────────

void BatchScheduler::write_summary(const std::string& output_dir,
                                    const std::vector<BatchResult>& results) {
    try {
        namespace fs = std::filesystem;
        fs::create_directories(output_dir);
        std::ofstream f(output_dir + "/batch_summary.json");
        f << "[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i) f << ",";
            const auto& r = results[i];
            f << "{"
              << "\"job_id\":\"" << r.job_id << "\","
              << "\"success\":" << (r.success ? "true" : "false") << ","
              << "\"error\":\"" << r.error << "\","
              << "\"msh_path\":\"" << r.msh_path << "\""
              << "}";
        }
        f << "]";
    } catch (...) {}
}

// ─── cancel ───────────────────────────────────────────────────────────────────

void BatchScheduler::cancel() {
    cancel_flag_.store(true);
}

// ─── run ──────────────────────────────────────────────────────────────────────

std::vector<BatchResult> BatchScheduler::run(const std::vector<BatchJob>& jobs,
                                              const BatchSchedulerConfig& config) {
    if (jobs.empty()) return {};

    running_.store(true);
    cancel_flag_.store(false);

    int max_par = config.max_parallel;
    if (max_par <= 0)
        max_par = static_cast<int>(std::thread::hardware_concurrency());
    if (max_par <= 0) max_par = 1;

    std::vector<BatchResult> results(jobs.size());
    std::string tmp_dir = std::filesystem::temp_directory_path().string() + "/stator_batch";
    std::filesystem::create_directories(tmp_dir);

    // Active child bookkeeping: (pid, job_index, start_time)
    struct ActiveChild {
        pid_t pid;
        int   job_idx;
        std::chrono::steady_clock::time_point start;
    };
    std::vector<ActiveChild> active;

    int jobs_done = 0;
    int n_jobs    = static_cast<int>(jobs.size());

    auto reap_one = [&](bool blocking) {
        for (auto it = active.begin(); it != active.end(); ) {
            // Check timeout
            if (config.job_timeout_sec > 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - it->start).count();
                if (elapsed >= config.job_timeout_sec)
                    kill(it->pid, SIGKILL);
            }
            int status = 0;
            pid_t ret = waitpid(it->pid, &status, blocking ? 0 : WNOHANG);
            if (ret == it->pid) {
                // Child finished — read status JSON
                std::string sfile = tmp_dir + "/job_" + std::to_string(it->job_idx) + ".json";
                results[it->job_idx] = read_status_json(sfile, jobs[it->job_idx].job_id);
                results[it->job_idx].job_id = jobs[it->job_idx].job_id;
                ++jobs_done;
                if (progress_cb_)
                    progress_cb_(jobs_done, n_jobs,
                                  results[it->job_idx].success,
                                  results[it->job_idx].job_id);
                it = active.erase(it);
                return true;
            } else {
                ++it;
            }
        }
        return false;
    };

    for (int i = 0; i < n_jobs; ++i) {
        if (cancel_flag_.load()) {
            results[i].job_id = jobs[i].job_id;
            results[i].error  = "cancelled";
            continue;
        }

        // Check skip_existing
        if (config.skip_existing &&
            ExportEngine::outputs_exist(jobs[i].params, jobs[i].export_config)) {
            results[i].job_id = jobs[i].job_id;
            results[i].success = true;
            ++jobs_done;
            if (progress_cb_)
                progress_cb_(jobs_done, n_jobs, true, jobs[i].job_id);
            continue;
        }

        // Wait for a slot if at capacity
        while (static_cast<int>(active.size()) >= max_par) {
            reap_one(false);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::string status_path = tmp_dir + "/job_" + std::to_string(i) + ".json";
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            int rc = execute_job(jobs[i], status_path);
            _exit(rc);
        } else if (pid > 0) {
            active.push_back({pid, i, std::chrono::steady_clock::now()});
        } else {
            results[i].job_id = jobs[i].job_id;
            results[i].error  = "fork() failed";
        }
    }

    // Wait for all remaining children
    while (!active.empty()) {
        if (!reap_one(false))
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Send SIGTERM to any children still active if cancelled
    for (auto& ac : active)
        kill(ac.pid, SIGTERM);

    // Write summary
    if (config.write_summary && !jobs.empty()) {
        std::string out_dir = jobs[0].export_config.output_dir;
        write_summary(out_dir, results);
    }

    running_.store(false);
    return results;
}

} // namespace stator
