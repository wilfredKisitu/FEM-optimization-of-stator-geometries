/**
 * batch_scheduler_c.c — Fork-based batch scheduler example in pure C.
 *
 * Demonstrates:
 *   1. Building a parameter sweep (slot count × lamination count)
 *   2. Wiring a ProgressCallback to print a live progress bar
 *   3. Running stator_batch_run with fork()-based parallelism
 *   4. Reading BatchResult arrays and printing a summary table
 *   5. Writing a batch_summary.json to the output directory
 *
 * Build (from project root):
 *   mkdir -p build && cd build
 *   cmake -DSTATOR_BUILD_EXAMPLES=ON ..
 *   make batch_scheduler_c
 *
 * Run:
 *   ./batch_scheduler_c
 *   ./batch_scheduler_c --output /tmp/stator_batch_c --parallel 4
 *   ./batch_scheduler_c --dry-run
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "stator_c/params.h"
#include "stator_c/batch_scheduler.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Sweep definition ────────────────────────────────────────────────────── */

static const int SLOT_COUNTS[]   = {24, 36, 48, 60, 72};
static const int N_SLOT_COUNTS   = 5;
static const int LAM_COUNTS[]    = {100, 200, 400};
static const int N_LAM_COUNTS    = 3;
#define N_JOBS (N_SLOT_COUNTS * N_LAM_COUNTS)   /* 15 */

/* ── CLI ─────────────────────────────────────────────────────────────────── */

static const char* USAGE =
    "Usage: batch_scheduler_c [options]\n"
    "\n"
    "Options:\n"
    "  --output   <dir>  Output directory    (default: /tmp/stator_batch_c)\n"
    "  --parallel <n>    Max parallel jobs   (default: 0 = #CPUs)\n"
    "  --timeout  <s>    Per-job timeout (s) (default: 120)\n"
    "  --dry-run         Validate only, skip generation\n"
    "  --help            Show this message\n";

typedef struct {
    char output[512];
    int  max_parallel;
    int  timeout_sec;
    int  dry_run;
} Args;

static Args parse_args(int argc, char* argv[])
{
    Args a;
    strncpy(a.output, "/tmp/stator_batch_c", sizeof(a.output) - 1);
    a.output[sizeof(a.output) - 1] = '\0';
    a.max_parallel = 0;
    a.timeout_sec  = 120;
    a.dry_run      = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0)     { puts(USAGE); exit(0); }
        if (strcmp(argv[i], "--dry-run") == 0)  { a.dry_run = 1; continue; }
        if (strcmp(argv[i], "--output")   == 0 && i + 1 < argc)
            strncpy(a.output, argv[++i], sizeof(a.output) - 1);
        else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc)
            a.max_parallel = atoi(argv[++i]);
        else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
            a.timeout_sec = atoi(argv[++i]);
        else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n%s", argv[i], USAGE);
            exit(1);
        }
    }
    return a;
}

/* ── Build a single StatorParams for a sweep point ──────────────────────── */

static StatorParams make_sweep_params(int n_slots, int n_lam)
{
    StatorParams p;
    char _err[512];
    stator_make_reference_params(&p, _err, sizeof(_err));

    p.n_slots = n_slots;
    p.n_lam   = n_lam;

    /* Scale slot widths; coil widths derived after insulation clearance */
    double arc          = M_PI * p.R_inner / n_slots;
    p.slot_width_outer  = arc * 0.55;
    p.slot_width_inner  = arc * 0.45;
    p.slot_opening      = arc * 0.20;
    double ins          = p.insulation_thickness;
    p.coil_width_outer  = p.slot_width_outer - 2.0 * ins - 0.0001;
    p.coil_width_inner  = p.slot_width_inner - 2.0 * ins - 0.0001;

    /* Vary slot shape and winding type with slot count */
    if      (n_slots <= 24) { p.slot_shape  = SLOT_SHAPE_RECTANGULAR;
                               p.winding_type = WINDING_CONCENTRATED; }
    else if (n_slots <= 48) { p.slot_shape  = SLOT_SHAPE_SEMI_CLOSED;
                               p.winding_type = WINDING_DOUBLE_LAYER; }
    else                    { p.slot_shape  = SLOT_SHAPE_TRAPEZOIDAL;
                               p.winding_type = WINDING_DISTRIBUTED; }

    return p;
}

/* ── Progress callback ───────────────────────────────────────────────────── */

typedef struct {
    int   total;
    time_t t_start;
} ProgCtx;

static void progress_cb(int done, int total, bool success,
                          const char* job_id, void* user_data)
{
    ProgCtx* ctx = (ProgCtx*)user_data;
    double elapsed = difftime(time(NULL), ctx->t_start);

    int bar_len = 28;
    int filled  = (total > 0) ? (done * bar_len / total) : 0;

    printf("\r  [");
    for (int i = 0; i < bar_len; i++)
        putchar(i < filled ? '#' : '.');
    printf("] %3d/%d  %s  %.0fs ",
           done, total,
           success ? "OK  " : "FAIL",
           elapsed);

    /* Print job_id truncated to 20 chars so the line stays stable */
    printf("%-20.20s", job_id);
    fflush(stdout);

    if (done == total) putchar('\n');
}

/* ── Separator ───────────────────────────────────────────────────────────── */

static void sep(void) {
    puts("─────────────────────────────────────────────────────────────────────");
}

/* ── Print summary table ─────────────────────────────────────────────────── */

static void print_summary(const BatchJob* jobs, const BatchResult* results,
                            int n, const StatorParams* derived)
{
    (void)jobs;
    sep();
    printf("  %-18s %5s %5s %7s %9s %5s\n",
           "JOB ID", "SLOTS", "LAM", "FILL %", "STACK mm", "OK");
    sep();
    for (int i = 0; i < n; i++) {
        const StatorParams* p = &derived[i];
        printf("  %-18s %5d %5d %7.1f %9.1f  %s\n",
               results[i].job_id,
               p->n_slots,
               p->n_lam,
               p->fill_factor * 100.0,
               p->stack_length * 1e3,
               results[i].success ? "✓" : "✗");
        if (!results[i].success && results[i].error[0])
            printf("    ↳ %s\n", results[i].error);
    }
    sep();

    int n_ok = 0;
    for (int i = 0; i < n; i++) if (results[i].success) n_ok++;
    printf("  %d/%d succeeded\n", n_ok, n);
    sep();
}

/* ── Ensure output directory ─────────────────────────────────────────────── */

static void ensure_dir(const char* path)
{
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    system(cmd);
}

/* ═════════════════════════════════════════════════════════════════════════ */

int main(int argc, char* argv[])
{
    Args args = parse_args(argc, argv);
    char err[512];

    printf("\n=== Stator Batch Scheduler — C Example ===\n");
    printf("    %d slot counts × %d stack depths = %d jobs\n\n",
           N_SLOT_COUNTS, N_LAM_COUNTS, N_JOBS);

    /* ── 1. Build jobs ───────────────────────────────────────────────────── */
    BatchJob       jobs[N_JOBS];
    StatorParams   derived[N_JOBS];   /* store validated params for report */
    int n = 0;

    for (int si = 0; si < N_SLOT_COUNTS; si++) {
        for (int li = 0; li < N_LAM_COUNTS; li++, n++) {
            int ns = SLOT_COUNTS[si];
            int nl = LAM_COUNTS[li];

            memset(&jobs[n], 0, sizeof(jobs[n]));
            snprintf(jobs[n].job_id, sizeof(jobs[n].job_id),
                     "s%d_lam%d", ns, nl);

            jobs[n].params = make_sweep_params(ns, nl);
            stator_mesh_config_init(&jobs[n].mesh_config);
            stator_export_config_init(&jobs[n].export_config);
            strncpy(jobs[n].export_config.output_dir, args.output,
                    sizeof(jobs[n].export_config.output_dir) - 1);
            jobs[n].export_config.formats = EXPORT_VTK | EXPORT_JSON;
        }
    }

    /* ── 2. Validate all ─────────────────────────────────────────────────── */
    printf("Validating %d configurations …\n", n);
    int n_valid = 0;
    for (int i = 0; i < n; i++) {
        int rc = stator_params_validate_and_derive(&jobs[i].params,
                                                    err, sizeof(err));
        derived[i] = jobs[i].params;
        if (rc == STATOR_OK) {
            n_valid++;
            printf("  %-20s OK  (fill=%.1f%%  stack=%.0f mm)\n",
                   jobs[i].job_id,
                   jobs[i].params.fill_factor * 100.0,
                   jobs[i].params.stack_length * 1e3);
        } else {
            printf("  %-20s FAIL  %s\n", jobs[i].job_id, err);
            /* Mark as invalid so the scheduler skips it */
            jobs[i].params.n_slots = 0;
        }
    }
    printf("  %d/%d valid\n\n", n_valid, n);

    if (args.dry_run) {
        printf("[--dry-run] Skipping generation.\n\n");
        BatchResult dummy[N_JOBS];
        memset(dummy, 0, sizeof(dummy));
        for (int i = 0; i < n; i++)
            strncpy(dummy[i].job_id, jobs[i].job_id, sizeof(dummy[i].job_id) - 1);
        print_summary(jobs, dummy, n, derived);
        return 0;
    }

    /* ── 3. Create output directory ──────────────────────────────────────── */
    ensure_dir(args.output);

    /* ── 4. Run batch ────────────────────────────────────────────────────── */
    printf("Running batch (max_parallel=%d, timeout=%ds) …\n",
           args.max_parallel, args.timeout_sec);

    BatchScheduler      sched;
    BatchSchedulerConfig cfg;
    BatchResult          results[N_JOBS];

    stator_batch_scheduler_init(&sched);
    stator_batch_sched_config_init(&cfg);
    cfg.max_parallel    = args.max_parallel;
    cfg.skip_existing   = true;
    cfg.job_timeout_sec = args.timeout_sec;
    cfg.write_summary   = true;

    ProgCtx prog_ctx = { .total = n, .t_start = time(NULL) };
    stator_batch_scheduler_set_callback(&sched, progress_cb, &prog_ctx);

    time_t t0 = time(NULL);
    int rc = stator_batch_run(&sched, jobs, n, &cfg, results);
    double elapsed = difftime(time(NULL), t0);

    if (rc != STATOR_OK) {
        fprintf(stderr, "stator_batch_run returned error %d\n", rc);
        return 1;
    }

    /* ── 5. Print summary ────────────────────────────────────────────────── */
    printf("\nCompleted in %.0f s\n\n", elapsed);
    print_summary(jobs, results, n, derived);

    /* ── 6. Report output paths for first successful job ─────────────────── */
    printf("\nSample output paths:\n");
    for (int i = 0; i < n; i++) {
        if (results[i].success) {
            if (results[i].vtk_path[0])
                printf("  VTK  : %s\n", results[i].vtk_path);
            if (results[i].json_path[0])
                printf("  JSON : %s\n", results[i].json_path);
            break;
        }
    }
    printf("\nBatch summary : %s/batch_summary.json\n", args.output);
    printf("\nDone.\n\n");
    return 0;
}
