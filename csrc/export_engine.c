#include "stator_c/export_engine.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* ── SHA-256 (FIPS 180-4 self-contained) ───────────────────────────────── */

static const uint32_t K256[64] = {
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

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(e,f,g)  (((e) & (f)) ^ (~(e) & (g)))
#define MAJ(a,b,c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))
#define EP0(a) (ROTR(a,2)  ^ ROTR(a,13) ^ ROTR(a,22))
#define EP1(e) (ROTR(e,6)  ^ ROTR(e,11) ^ ROTR(e,25))
#define SIG0(x)(ROTR(x,7)  ^ ROTR(x,18) ^ ((x) >> 3))
#define SIG1(x)(ROTR(x,17) ^ ROTR(x,19) ^ ((x) >> 10))

void stator_sha256(const char* data, size_t len, char* out) {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    size_t padded = ((len + 8) / 64 + 1) * 64;
    unsigned char* msg = (unsigned char*)calloc(padded, 1);
    if (!msg) { memset(out, '0', 64); out[64] = '\0'; return; }
    memcpy(msg, data, len);
    msg[len] = 0x80u;
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        msg[padded - 8 + i] = (unsigned char)(bitlen >> (56 - 8*i));

    for (size_t i = 0; i < padded; i += 64) {
        uint32_t w[64];
        for (int j = 0; j < 16; j++)
            w[j] = ((uint32_t)msg[i+j*4  ] << 24)
                 | ((uint32_t)msg[i+j*4+1] << 16)
                 | ((uint32_t)msg[i+j*4+2] <<  8)
                 |  (uint32_t)msg[i+j*4+3];
        for (int j = 16; j < 64; j++)
            w[j] = SIG1(w[j-2]) + w[j-7] + SIG0(w[j-15]) + w[j-16];

        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int j = 0; j < 64; j++) {
            uint32_t t1 = hh + EP1(e) + CH(e,f,g) + K256[j] + w[j];
            uint32_t t2 = EP0(a) + MAJ(a,b,c);
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    free(msg);

    for (int i = 0; i < 8; i++)
        snprintf(out + i*8, 9, "%08x", h[i]);
    out[64] = '\0';
}

/* ── ExportEngine ────────────────────────────────────────────────────────── */

int stator_export_engine_init(ExportEngine* ee, GmshBackend* backend,
                                char* err_buf, size_t err_len) {
    if (!backend) {
        STATOR_SET_ERR(err_buf, err_len, "ExportEngine: backend must not be null");
        return STATOR_ERR_INVAL;
    }
    ee->backend = backend;
    return STATOR_OK;
}

int stator_export_compute_stem(const StatorParams* p, char* out_stem, size_t stem_len) {
    char json_buf[8192];
    if (stator_params_to_json(p, json_buf, sizeof(json_buf)) != STATOR_OK)
        return STATOR_ERR_NOMEM;
    char hash[65];
    stator_sha256(json_buf, strlen(json_buf), hash);
    int n = snprintf(out_stem, stem_len, "stator_%.8s", hash);
    if (n < 0 || (size_t)n >= stem_len) return STATOR_ERR_NOMEM;
    return STATOR_OK;
}

bool stator_export_outputs_exist(const StatorParams* p, const ExportConfig* cfg) {
    char stem[64];
    if (stator_export_compute_stem(p, stem, sizeof(stem)) != STATOR_OK) return false;
    char path[1024];

    if (export_has_format(cfg->formats, EXPORT_MSH)) {
        snprintf(path, sizeof(path), "%s/%s.msh", cfg->output_dir, stem);
        FILE* f = fopen(path, "r"); if (!f) return false; fclose(f);
    }
    if (export_has_format(cfg->formats, EXPORT_VTK)) {
        snprintf(path, sizeof(path), "%s/%s.vtk", cfg->output_dir, stem);
        FILE* f = fopen(path, "r"); if (!f) return false; fclose(f);
    }
    if (export_has_format(cfg->formats, EXPORT_HDF5)) {
        snprintf(path, sizeof(path), "%s/%s.h5", cfg->output_dir, stem);
        FILE* f = fopen(path, "r"); if (!f) return false; fclose(f);
    }
    if (export_has_format(cfg->formats, EXPORT_JSON)) {
        snprintf(path, sizeof(path), "%s/%s_meta.json", cfg->output_dir, stem);
        FILE* f = fopen(path, "r"); if (!f) return false; fclose(f);
    }
    return true;
}

/* ── Format writers ─────────────────────────────────────────────────────── */

static ExportResult write_msh(ExportEngine* ee, const StatorParams* p,
                                const ExportConfig* cfg, const char* stem) {
    ExportResult r;
    memset(&r, 0, sizeof(r));
    r.format = EXPORT_MSH;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    snprintf(r.path, sizeof(r.path), "%s/%s.msh", cfg->output_dir, stem);
    gmsh_write_mesh(ee->backend, r.path);
    r.success = true;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    r.write_time_ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    return r;
}

static ExportResult write_vtk(const StatorParams* p, const MeshResult* mesh,
                                const ExportConfig* cfg, const char* stem) {
    (void)p;
    ExportResult r;
    memset(&r, 0, sizeof(r));
    r.format = EXPORT_VTK;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    snprintf(r.path, sizeof(r.path), "%s/%s.vtk", cfg->output_dir, stem);
    FILE* f = fopen(r.path, "w");
    if (!f) {
        snprintf(r.error_message, sizeof(r.error_message),
                 "Cannot open %s", r.path);
        goto done;
    }
    fprintf(f, "# vtk DataFile Version 3.0\n"
               "Stator mesh %s\n"
               "ASCII\n"
               "DATASET UNSTRUCTURED_GRID\n"
               "POINTS 0 double\n"
               "CELLS 0 0\n"
               "CELL_TYPES 0\n", stem);
    fclose(f);
    r.success = true;
done:
    clock_gettime(CLOCK_MONOTONIC, &t1);
    r.write_time_ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    (void)mesh;
    return r;
}

static ExportResult write_hdf5(const StatorParams* p, const MeshResult* mesh,
                                 const ExportConfig* cfg, const char* stem) {
    (void)p;
    ExportResult r;
    memset(&r, 0, sizeof(r));
    r.format = EXPORT_HDF5;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    snprintf(r.path, sizeof(r.path), "%s/%s.h5", cfg->output_dir, stem);
    FILE* f = fopen(r.path, "w");
    if (!f) {
        snprintf(r.error_message, sizeof(r.error_message),
                 "Cannot open %s", r.path);
        goto done;
    }
    fprintf(f, "HDF5 placeholder for %s\nn_nodes=%d\nn_elements_2d=%d\n",
            stem, mesh->n_nodes, mesh->n_elements_2d);
    fclose(f);
    r.success = true;
done:
    clock_gettime(CLOCK_MONOTONIC, &t1);
    r.write_time_ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    return r;
}

static ExportResult write_json(const StatorParams* p, const MeshResult* mesh,
                                 const ExportConfig* cfg, const char* stem) {
    ExportResult r;
    memset(&r, 0, sizeof(r));
    r.format = EXPORT_JSON;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    snprintf(r.path, sizeof(r.path), "%s/%s_meta.json", cfg->output_dir, stem);
    FILE* f = fopen(r.path, "w");
    if (!f) {
        snprintf(r.error_message, sizeof(r.error_message),
                 "Cannot open %s", r.path);
        goto done;
    }
    char json_buf[8192];
    stator_params_to_json(p, json_buf, sizeof(json_buf));
    fprintf(f,
        "{\"params\":%s,"
        "\"mesh_stats\":{\"n_nodes\":%d,\"n_elements_2d\":%d,"
        "\"n_elements_3d\":%d,\"min_quality\":%g,\"avg_quality\":%g},"
        "\"output_files\":{"
        "\"msh\":\"%s/%s.msh\","
        "\"vtk\":\"%s/%s.vtk\","
        "\"hdf5\":\"%s/%s.h5\","
        "\"json\":\"%s\"}"
        "}",
        json_buf,
        mesh->n_nodes, mesh->n_elements_2d,
        mesh->n_elements_3d, mesh->min_quality, mesh->avg_quality,
        cfg->output_dir, stem,
        cfg->output_dir, stem,
        cfg->output_dir, stem,
        r.path);
    fclose(f);
    r.success = true;
done:
    clock_gettime(CLOCK_MONOTONIC, &t1);
    r.write_time_ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
    return r;
}

int stator_export_write_all_sync(ExportEngine* ee,
                                  const StatorParams* p,
                                  const MeshResult* mesh,
                                  const ExportConfig* cfg,
                                  ExportResult* out_results,
                                  int out_cap,
                                  int* out_n) {
    char stem[64];
    if (stator_export_compute_stem(p, stem, sizeof(stem)) != STATOR_OK)
        return STATOR_ERR_NOMEM;

    *out_n = 0;
    if (export_has_format(cfg->formats, EXPORT_MSH) && *out_n < out_cap)
        out_results[(*out_n)++] = write_msh(ee, p, cfg, stem);
    if (export_has_format(cfg->formats, EXPORT_VTK) && *out_n < out_cap)
        out_results[(*out_n)++] = write_vtk(p, mesh, cfg, stem);
    if (export_has_format(cfg->formats, EXPORT_HDF5) && *out_n < out_cap)
        out_results[(*out_n)++] = write_hdf5(p, mesh, cfg, stem);
    if (export_has_format(cfg->formats, EXPORT_JSON) && *out_n < out_cap)
        out_results[(*out_n)++] = write_json(p, mesh, cfg, stem);
    return STATOR_OK;
}
