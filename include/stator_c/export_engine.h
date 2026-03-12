#ifndef STATOR_C_EXPORT_ENGINE_H
#define STATOR_C_EXPORT_ENGINE_H

#include "stator_c/params.h"
#include "stator_c/gmsh_backend.h"
#include "stator_c/mesh_generator.h"

/* ── ExportFormat bitmask ────────────────────────────────────────────────── */

#define EXPORT_NONE 0u
#define EXPORT_MSH  (1u << 0)
#define EXPORT_VTK  (1u << 1)
#define EXPORT_HDF5 (1u << 2)
#define EXPORT_JSON (1u << 3)
#define EXPORT_ALL  (EXPORT_MSH | EXPORT_VTK | EXPORT_HDF5 | EXPORT_JSON)

typedef uint32_t ExportFormat;

static inline bool export_has_format(ExportFormat mask, ExportFormat f) {
    return (mask & f) != 0u;
}

/* ── ExportConfig ────────────────────────────────────────────────────────── */

typedef struct {
    ExportFormat formats;
    char         output_dir[512];
    int          msh_version;
} ExportConfig;

static inline void stator_export_config_init(ExportConfig* c) {
    c->formats     = EXPORT_ALL;
    c->output_dir[0] = '.';
    c->output_dir[1] = '\0';
    c->msh_version = 4;
}

/* ── ExportResult ────────────────────────────────────────────────────────── */

typedef struct {
    bool         success;
    ExportFormat format;
    char         path[512];
    char         error_message[STATOR_ERR_BUF];
    double       write_time_ms;
} ExportResult;

/* ── ExportEngine ────────────────────────────────────────────────────────── */

typedef struct {
    GmshBackend* backend;
} ExportEngine;

int stator_export_engine_init(ExportEngine* ee, GmshBackend* backend,
                                char* err_buf, size_t err_len);

/* Compute "stator_" + sha256(params_json).substr(0,8) into out_stem (>=32 bytes) */
int stator_export_compute_stem(const StatorParams* p, char* out_stem, size_t stem_len);

/* Check if all output files already exist */
bool stator_export_outputs_exist(const StatorParams* p, const ExportConfig* cfg);

/* Write all requested formats synchronously */
int stator_export_write_all_sync(ExportEngine* ee,
                                  const StatorParams* p,
                                  const MeshResult* mesh,
                                  const ExportConfig* cfg,
                                  ExportResult* out_results,
                                  int out_cap,
                                  int* out_n);

/* SHA-256: writes 64-char hex + NUL into out (>=65 bytes) */
void stator_sha256(const char* data, size_t data_len, char* out);

#endif /* STATOR_C_EXPORT_ENGINE_H */
