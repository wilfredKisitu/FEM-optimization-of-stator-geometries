#pragma once
#include "stator/params.hpp"
#include "stator/mesh_generator.hpp"
#include "stator/export_engine.hpp"
#include <string>
#include <vector>
#include <functional>
#include <atomic>

namespace stator {

// ─── BatchJob ─────────────────────────────────────────────────────────────────

struct BatchJob {
    StatorParams params;
    ExportConfig export_config;
    MeshConfig   mesh_config;
    std::string  job_id;         // caller-assigned identifier, echoed in status JSON
};

// ─── BatchSchedulerConfig ─────────────────────────────────────────────────────

struct BatchSchedulerConfig {
    int  max_parallel    = 0;    // 0 = auto-detect via hardware_concurrency()
    bool skip_existing   = true; // idempotent re-runs
    int  job_timeout_sec = 300;  // SIGKILL child after this; 0 = no timeout
    bool write_summary   = true; // write batch_summary.json to output_dir
};

// ─── BatchResult ──────────────────────────────────────────────────────────────

struct BatchResult {
    std::string job_id;
    bool        success   = false;
    std::string error;
    std::string msh_path;
    std::string vtk_path;
    std::string hdf5_path;
    std::string json_path;
};

// ─── ProgressCallback ─────────────────────────────────────────────────────────
using ProgressCallback = std::function<void(int jobs_done, int jobs_total,
                                             bool success, const std::string& job_id)>;

// ─── BatchScheduler ───────────────────────────────────────────────────────────

class BatchScheduler {
public:
    BatchScheduler() = default;

    void set_progress_callback(ProgressCallback cb) { progress_cb_ = std::move(cb); }

    // Run a batch of jobs with fork()-based parallelism.
    std::vector<BatchResult> run(const std::vector<BatchJob>& jobs,
                                  const BatchSchedulerConfig& config = {});

    // Signal all active children to stop.
    void cancel();

    bool is_running() const noexcept { return running_.load(); }

    // Execute a single job (runs inside child process).
    // Returns 0 on success, non-zero on error.
    // Writes status JSON to status_path.
    static int execute_job(const BatchJob& job, const std::string& status_path);

private:
    ProgressCallback   progress_cb_;
    std::atomic<bool>  cancel_flag_{false};
    std::atomic<bool>  running_{false};

    // Write batch_summary.json
    static void write_summary(const std::string& output_dir,
                               const std::vector<BatchResult>& results);

    // Read a status JSON file into a BatchResult.
    static BatchResult read_status_json(const std::string& path,
                                         const std::string& job_id);
};

} // namespace stator
