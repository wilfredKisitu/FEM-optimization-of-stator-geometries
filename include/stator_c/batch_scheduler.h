#ifndef STATOR_C_BATCH_SCHEDULER_H
#define STATOR_C_BATCH_SCHEDULER_H

#include "stator_c/params.h"
#include "stator_c/mesh_generator.h"
#include "stator_c/export_engine.h"

/* ── BatchJob ────────────────────────────────────────────────────────────── */

typedef struct {
    StatorParams  params;
    ExportConfig  export_config;
    MeshConfig    mesh_config;
    char          job_id[128];
} BatchJob;

/* ── BatchSchedulerConfig ────────────────────────────────────────────────── */

typedef struct {
    int  max_parallel;
    bool skip_existing;
    int  job_timeout_sec;
    bool write_summary;
} BatchSchedulerConfig;

static inline void stator_batch_sched_config_init(BatchSchedulerConfig* c) {
    c->max_parallel    = 0;
    c->skip_existing   = true;
    c->job_timeout_sec = 300;
    c->write_summary   = true;
}

/* ── BatchResult ─────────────────────────────────────────────────────────── */

typedef struct {
    char job_id[128];
    bool success;
    char error[STATOR_ERR_BUF];
    char msh_path[512];
    char vtk_path[512];
    char hdf5_path[512];
    char json_path[512];
} BatchResult;

/* ── ProgressCallback ────────────────────────────────────────────────────── */

typedef void (*ProgressCallback)(int jobs_done, int jobs_total,
                                  bool success, const char* job_id,
                                  void* user_data);

/* ── BatchScheduler ──────────────────────────────────────────────────────── */

typedef struct {
    ProgressCallback progress_cb;
    void*            progress_user_data;
    volatile int     cancel_flag;
    volatile int     running;
} BatchScheduler;

static inline void stator_batch_scheduler_init(BatchScheduler* s) {
    s->progress_cb        = NULL;
    s->progress_user_data = NULL;
    s->cancel_flag        = 0;
    s->running            = 0;
}

void stator_batch_scheduler_set_callback(BatchScheduler* s,
                                           ProgressCallback cb, void* user_data);

/* Run jobs with fork()-based parallelism.
   results must be pre-allocated array of n_jobs BatchResults. */
int stator_batch_run(BatchScheduler* s,
                      const BatchJob* jobs, int n_jobs,
                      const BatchSchedulerConfig* config,
                      BatchResult* results);

void stator_batch_cancel(BatchScheduler* s);

/* Execute a single job (called in child process).
   Returns 0 on success, writes status JSON to status_path. */
int stator_batch_execute_job(const BatchJob* job, const char* status_path);

#endif /* STATOR_C_BATCH_SCHEDULER_H */
