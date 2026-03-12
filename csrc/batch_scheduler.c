#include "stator_c/batch_scheduler.h"
#include "stator_c/geometry_builder.h"
#include "stator_c/topology_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

/* ── execute_job ─────────────────────────────────────────────────────────── */

int stator_batch_execute_job(const BatchJob* job, const char* status_path) {
    char error_msg[STATOR_ERR_BUF] = "";
    char msh_path[512]  = "";
    char vtk_path[512]  = "";
    char hdf5_path[512] = "";
    char json_path[512] = "";
    bool success = false;
    char err_buf[STATOR_ERR_BUF];

    BatchJob mj = *job; /* mutable copy */

    /* Step 1: validate */
    if (stator_params_validate_and_derive(&mj.params, err_buf, sizeof(err_buf)) != STATOR_OK) {
        snprintf(error_msg, sizeof(error_msg), "%s", err_buf);
        goto write_status;
    }

    /* Step 2-3: backend + initialize */
    {
        GmshBackend* backend = stator_make_default_backend();
        if (!backend) {
            snprintf(error_msg, sizeof(error_msg), "Failed to create backend");
            goto write_status;
        }

        char stem[64];
        stator_export_compute_stem(&mj.params, stem, sizeof(stem));
        char model_name[80];
        snprintf(model_name, sizeof(model_name), "stator_%s", stem);
        gmsh_initialize(backend, model_name);

        /* Step 4: geometry */
        GeometryBuilder gb;
        if (stator_geom_builder_init(&gb, backend, err_buf, sizeof(err_buf)) != STATOR_OK) {
            snprintf(error_msg, sizeof(error_msg), "%s", err_buf);
            stub_gmsh_backend_free(backend);
            goto write_status;
        }

        GeometryBuildResult geo;
        if (stator_geom_build(&gb, &mj.params, &geo) != STATOR_OK || !geo.success) {
            snprintf(error_msg, sizeof(error_msg),
                     "Geometry build failed: %s", geo.error_message);
            stub_gmsh_backend_free(backend);
            goto write_status;
        }

        /* Step 5: topology */
        TopologyRegistry reg;
        if (stator_topo_registry_init(&reg, mj.params.n_slots,
                                        err_buf, sizeof(err_buf)) != STATOR_OK) {
            snprintf(error_msg, sizeof(error_msg), "%s", err_buf);
            stub_gmsh_backend_free(backend);
            goto write_status;
        }

        /* Step 6-7: mesh */
        MeshGenerator mg;
        if (stator_mesh_generator_init(&mg, backend, &mj.mesh_config,
                                         err_buf, sizeof(err_buf)) != STATOR_OK) {
            snprintf(error_msg, sizeof(error_msg), "%s", err_buf);
            stator_topo_registry_destroy(&reg);
            stub_gmsh_backend_free(backend);
            goto write_status;
        }

        MeshResult mesh;
        if (stator_mesh_generate(&mg, &mj.params, &geo, &reg, &mesh) != STATOR_OK) {
            snprintf(error_msg, sizeof(error_msg), "Mesh generation failed");
            stator_topo_registry_destroy(&reg);
            stub_gmsh_backend_free(backend);
            goto write_status;
        }

        /* Step 8: export */
        /* Ensure output dir exists */
        char mkdir_cmd[600];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p '%s'", mj.export_config.output_dir);
        system(mkdir_cmd);

        ExportEngine ee;
        stator_export_engine_init(&ee, backend, NULL, 0);

        ExportResult results[4];
        int n_results = 0;
        stator_export_write_all_sync(&ee, &mj.params, &mesh,
                                      &mj.export_config,
                                      results, 4, &n_results);

        for (int i = 0; i < n_results; i++) {
            if (!results[i].success) continue;
            if (results[i].format == EXPORT_MSH)
                strncpy(msh_path,  results[i].path, sizeof(msh_path)  - 1);
            if (results[i].format == EXPORT_VTK)
                strncpy(vtk_path,  results[i].path, sizeof(vtk_path)  - 1);
            if (results[i].format == EXPORT_HDF5)
                strncpy(hdf5_path, results[i].path, sizeof(hdf5_path) - 1);
            if (results[i].format == EXPORT_JSON)
                strncpy(json_path, results[i].path, sizeof(json_path) - 1);
        }

        /* Step 9: finalize */
        gmsh_finalize(backend);
        stator_topo_registry_destroy(&reg);
        stub_gmsh_backend_free(backend);
        success = true;
    }

write_status:
    {
        FILE* f = fopen(status_path, "w");
        if (f) {
            fprintf(f,
                "{\"job_id\":\"%s\","
                "\"success\":%s,"
                "\"error\":\"%s\","
                "\"msh_path\":\"%s\","
                "\"vtk_path\":\"%s\","
                "\"hdf5_path\":\"%s\","
                "\"json_path\":\"%s\"}",
                job->job_id,
                success ? "true" : "false",
                error_msg, msh_path, vtk_path, hdf5_path, json_path);
            fclose(f);
        }
    }
    return success ? 0 : 1;
}

/* ── read_status_json ────────────────────────────────────────────────────── */

static void read_status_json(const char* path, const char* job_id,
                               BatchResult* r) {
    memset(r, 0, sizeof(*r));
    strncpy(r->job_id, job_id, sizeof(r->job_id) - 1);

    FILE* f = fopen(path, "r");
    if (!f) {
        snprintf(r->error, sizeof(r->error), "Cannot open status file: %s", path);
        return;
    }
    char content[4096];
    size_t n = fread(content, 1, sizeof(content) - 1, f);
    fclose(f);
    content[n] = '\0';

    /* Extract bool field */
    char* pos = strstr(content, "\"success\":");
    if (pos) {
        pos += strlen("\"success\":");
        r->success = (strncmp(pos, "true", 4) == 0);
    }

    /* Extract string field helper */
#define EXTRACT(key, dest, dest_sz) do { \
    char search[64]; \
    snprintf(search, sizeof(search), "\"" key "\":\""); \
    char* p2 = strstr(content, search); \
    if (p2) { \
        p2 += strlen(search); \
        char* end = strchr(p2, '"'); \
        if (end) { \
            size_t len = (size_t)(end - p2); \
            if (len >= (dest_sz)) len = (dest_sz) - 1; \
            strncpy((dest), p2, len); \
            (dest)[len] = '\0'; \
        } \
    } \
} while(0)

    EXTRACT("error",     r->error,     sizeof(r->error));
    EXTRACT("msh_path",  r->msh_path,  sizeof(r->msh_path));
    EXTRACT("vtk_path",  r->vtk_path,  sizeof(r->vtk_path));
    EXTRACT("hdf5_path", r->hdf5_path, sizeof(r->hdf5_path));
    EXTRACT("json_path", r->json_path, sizeof(r->json_path));
#undef EXTRACT
}

/* ── write_summary ───────────────────────────────────────────────────────── */

static void write_summary(const char* output_dir,
                            const BatchResult* results, int n) {
    char path[600];
    snprintf(path, sizeof(path), "%s/batch_summary.json", output_dir);
    FILE* f = fopen(path, "w");
    if (!f) return;
    fputc('[', f);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', f);
        fprintf(f,
            "{\"job_id\":\"%s\","
            "\"success\":%s,"
            "\"error\":\"%s\","
            "\"msh_path\":\"%s\"}",
            results[i].job_id,
            results[i].success ? "true" : "false",
            results[i].error,
            results[i].msh_path);
    }
    fputs("]", f);
    fclose(f);
}

/* ── set_callback / cancel ───────────────────────────────────────────────── */

void stator_batch_scheduler_set_callback(BatchScheduler* s,
                                           ProgressCallback cb, void* user_data) {
    s->progress_cb        = cb;
    s->progress_user_data = user_data;
}

void stator_batch_cancel(BatchScheduler* s) {
    s->cancel_flag = 1;
}

/* ── run ─────────────────────────────────────────────────────────────────── */

int stator_batch_run(BatchScheduler* s,
                      const BatchJob* jobs, int n_jobs,
                      const BatchSchedulerConfig* config,
                      BatchResult* results) {
    if (n_jobs <= 0) return STATOR_OK;

    s->running     = 1;
    s->cancel_flag = 0;

    int max_par = config->max_parallel;
    if (max_par <= 0) {
        long nproc = sysconf(_SC_NPROCESSORS_ONLN);
        max_par = (nproc > 0) ? (int)nproc : 1;
    }
    if (max_par <= 0) max_par = 1;

    char tmp_dir[256];
    snprintf(tmp_dir, sizeof(tmp_dir), "/tmp/stator_batch_%d", (int)getpid());
    char mkdir_cmd[300];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p '%s'", tmp_dir);
    system(mkdir_cmd);

    /* Active children */
    typedef struct { pid_t pid; int job_idx; time_t start; } ActiveChild;
    ActiveChild* active = (ActiveChild*)calloc((size_t)n_jobs, sizeof(ActiveChild));
    if (!active) { s->running = 0; return STATOR_ERR_NOMEM; }
    int n_active = 0;

    int jobs_done = 0;
    memset(results, 0, (size_t)n_jobs * sizeof(BatchResult));

    /* Try to reap one non-blocking; returns 1 if reaped */
    int blocking = 0;
    (void)blocking;

    /* Inner reap lambda equivalent */
#define REAP_NONBLOCKING() do { \
    for (int _ri = 0; _ri < n_active; ) { \
        if (config->job_timeout_sec > 0) { \
            time_t elapsed = time(NULL) - active[_ri].start; \
            if (elapsed >= config->job_timeout_sec) \
                kill(active[_ri].pid, SIGKILL); \
        } \
        int _status = 0; \
        pid_t _ret = waitpid(active[_ri].pid, &_status, WNOHANG); \
        if (_ret == active[_ri].pid) { \
            char _sfile[512]; \
            snprintf(_sfile, sizeof(_sfile), "%s/job_%d.json", \
                     tmp_dir, active[_ri].job_idx); \
            read_status_json(_sfile, jobs[active[_ri].job_idx].job_id, \
                              &results[active[_ri].job_idx]); \
            strncpy(results[active[_ri].job_idx].job_id, \
                    jobs[active[_ri].job_idx].job_id, \
                    sizeof(results[0].job_id) - 1); \
            jobs_done++; \
            if (s->progress_cb) \
                s->progress_cb(jobs_done, n_jobs, \
                                results[active[_ri].job_idx].success, \
                                results[active[_ri].job_idx].job_id, \
                                s->progress_user_data); \
            active[_ri] = active[--n_active]; \
        } else { \
            _ri++; \
        } \
    } \
} while(0)

    for (int i = 0; i < n_jobs; i++) {
        if (s->cancel_flag) {
            strncpy(results[i].job_id, jobs[i].job_id, sizeof(results[i].job_id) - 1);
            snprintf(results[i].error, sizeof(results[i].error), "cancelled");
            continue;
        }
        if (config->skip_existing &&
            stator_export_outputs_exist(&jobs[i].params, &jobs[i].export_config)) {
            strncpy(results[i].job_id, jobs[i].job_id, sizeof(results[i].job_id) - 1);
            results[i].success = true;
            jobs_done++;
            if (s->progress_cb)
                s->progress_cb(jobs_done, n_jobs, true, jobs[i].job_id,
                                s->progress_user_data);
            continue;
        }

        /* Wait for a slot */
        while (n_active >= max_par) {
            REAP_NONBLOCKING();
            struct timespec ts = {0, 100000000L}; /* 100ms */
            nanosleep(&ts, NULL);
        }

        char status_path[512];
        snprintf(status_path, sizeof(status_path), "%s/job_%d.json", tmp_dir, i);

        pid_t pid = fork();
        if (pid == 0) {
            int rc = stator_batch_execute_job(&jobs[i], status_path);
            _exit(rc);
        } else if (pid > 0) {
            active[n_active].pid     = pid;
            active[n_active].job_idx = i;
            active[n_active].start   = time(NULL);
            n_active++;
        } else {
            strncpy(results[i].job_id, jobs[i].job_id, sizeof(results[i].job_id) - 1);
            snprintf(results[i].error, sizeof(results[i].error), "fork() failed");
        }
    }

    /* Drain remaining children */
    while (n_active > 0) {
        REAP_NONBLOCKING();
        if (n_active > 0) {
            struct timespec ts = {0, 100000000L};
            nanosleep(&ts, NULL);
        }
    }
#undef REAP_NONBLOCKING

    if (config->write_summary && n_jobs > 0)
        write_summary(jobs[0].export_config.output_dir, results, n_jobs);

    free(active);
    s->running = 0;
    return STATOR_OK;
}
