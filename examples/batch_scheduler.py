"""batch_scheduler.py — Run a parameter sweep with the batch scheduler.

Demonstrates:
  1. Building a parameter sweep (slot count × lamination count grid)
  2. Running generate_batch with a progress callback
  3. Parsing results and producing a summary table
  4. Plotting a fill-factor vs slot-count chart with matplotlib

Usage:
    cd /path/to/FEM
    python examples/batch_scheduler.py
    python examples/batch_scheduler.py --output /tmp/stator_batch --no-plot
    python examples/batch_scheduler.py --dry-run      # validate only, no output
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import stator_pipeline as sp


# ── CLI ───────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(description="Batch scheduler parameter sweep")
    p.add_argument("--output",   default="/tmp/stator_batch",
                   help="Output directory  (default: /tmp/stator_batch)")
    p.add_argument("--no-plot",  action="store_true",
                   help="Skip matplotlib chart")
    p.add_argument("--dry-run",  action="store_true",
                   help="Validate configs only, skip generation")
    return p.parse_args()


# ── Parameter sweep definition ────────────────────────────────────────────────

SLOT_COUNTS       = [24, 36, 48, 60, 72]    # 5 different slot counts
LAMINATION_COUNTS = [100, 200, 400]          # 3 stack depths → 15 jobs total

WINDING_MAP = {
    24: sp.WindingType.CONCENTRATED,
    36: sp.WindingType.DOUBLE_LAYER,
    48: sp.WindingType.DOUBLE_LAYER,
    60: sp.WindingType.DISTRIBUTED,
    72: sp.WindingType.DISTRIBUTED,
}

SHAPE_MAP = {
    24: sp.SlotShape.RECTANGULAR,
    36: sp.SlotShape.SEMI_CLOSED,
    48: sp.SlotShape.SEMI_CLOSED,
    60: sp.SlotShape.TRAPEZOIDAL,
    72: sp.SlotShape.TRAPEZOIDAL,
}


def make_sweep() -> list[tuple[str, sp.StatorConfig]]:
    """Return [(job_id, config)] for all sweep points."""
    jobs = []
    for n in SLOT_COUNTS:
        for lam in LAMINATION_COUNTS:
            arc   = math.pi * 0.151 / n
            ins   = 0.001  # insulation_thickness
            sw_o  = round(arc * 0.55, 6)
            sw_i  = round(arc * 0.45, 6)
            # Coil widths must clear insulation on each side
            cw_o  = round(sw_o - 2 * ins - 0.0001, 6)
            cw_i  = round(sw_i - 2 * ins - 0.0001, 6)

            cfg = sp.StatorConfig(
                R_outer=0.250, R_inner=0.151, airgap_length=0.001,
                n_slots=n,
                slot_depth=0.060,
                slot_width_outer=sw_o,
                slot_width_inner=sw_i,
                slot_opening=round(arc * 0.20, 6),
                slot_opening_depth=0.002,
                tooth_tip_angle=0.05,
                slot_shape=SHAPE_MAP[n],
                coil_depth=0.050,
                coil_width_outer=cw_o,
                coil_width_inner=cw_i,
                insulation_thickness=ins,
                turns_per_coil=max(4, 60 // n),
                coil_pitch=max(1, n // 6),
                wire_diameter=0.0010,
                slot_fill_factor=0.45,
                winding_type=WINDING_MAP[n],
                t_lam=0.00035,
                n_lam=lam,
                z_spacing=0.0,
                insulation_coating_thickness=0.00005,
                material=sp.LaminationMaterial.M270_35A,
                mesh_yoke=0.007, mesh_slot=0.004, mesh_coil=0.002,
                mesh_ins=0.001, mesh_boundary_layers=2,
                mesh_curvature=0.3, mesh_transition_layers=2,
            )
            jobs.append((f"s{n}_lam{lam}", cfg))
    return jobs


# ── Progress callback ─────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total: int):
        self.total   = total
        self.done    = 0
        self.t_start = time.time()

    def __call__(self, done: int, total: int, job_id: str):
        self.done = done
        elapsed   = time.time() - self.t_start
        bar_len   = 30
        filled    = int(bar_len * done / total)
        bar       = "█" * filled + "░" * (bar_len - filled)
        pct       = done / total * 100
        eta_s     = (elapsed / done) * (total - done) if done else 0
        print(f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  "
              f"eta {eta_s:.0f}s  ({job_id})",
              end="", flush=True)
        if done == total:
            print()


# ── Validation pass ───────────────────────────────────────────────────────────

def validate_all(jobs: list[tuple[str, sp.StatorConfig]]) -> list[dict]:
    print(f"\nValidating {len(jobs)} configurations …")
    rows = []
    n_fail = 0
    for job_id, cfg in jobs:
        v = sp.validate_config(cfg)
        rows.append({"job_id": job_id, "cfg": cfg, "v": v})
        status = "OK  " if v["success"] else f"FAIL  {v['error'][:60]}"
        print(f"  {job_id:<18} {status}")
        if not v["success"]:
            n_fail += 1
    print(f"\n  {len(jobs) - n_fail}/{len(jobs)} valid")
    return rows


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(rows: list[dict], results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print(f"  {'JOB ID':<18} {'SLOTS':>5} {'LAM':>5} "
          f"{'FILL%':>6} {'STACK mm':>9} {'YOKE mm':>8} {'OK':>4}")
    print("-" * 80)
    result_map = {r["job_id"]: r for r in results}
    for row in rows:
        jid = row["job_id"]
        cfg = row["cfg"]
        v   = row["v"]
        r   = result_map.get(jid, {})
        ok  = "✓" if r.get("success") else "✗"
        if v["success"]:
            print(f"  {jid:<18} {cfg.n_slots:>5} {cfg.n_lam:>5} "
                  f"{v['fill_factor']*100:>6.1f} "
                  f"{v['stack_length']*1e3:>9.1f} "
                  f"{v['yoke_height']*1e3:>8.2f} "
                  f"{ok:>4}")
        else:
            print(f"  {jid:<18} {'--':>5} {'--':>5} {'--':>6} {'--':>9} "
                  f"{'--':>8} {ok:>4}")
    print("=" * 80)


# ── Matplotlib chart ──────────────────────────────────────────────────────────

def plot_sweep(rows: list[dict], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("\n  [skip] matplotlib not installed")
        return

    # Group by lamination count
    lam_groups: dict[int, dict] = {}
    for row in rows:
        if not row["v"]["success"]:
            continue
        lam = row["cfg"].n_lam
        if lam not in lam_groups:
            lam_groups[lam] = {"slots": [], "fill": [], "stack": [], "yoke": []}
        g = lam_groups[lam]
        g["slots"].append(row["cfg"].n_slots)
        g["fill"].append(row["v"]["fill_factor"] * 100)
        g["stack"].append(row["v"]["stack_length"] * 1e3)
        g["yoke"].append(row["v"]["yoke_height"] * 1e3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colours = cm.viridis(np.linspace(0.2, 0.85, len(lam_groups)))

    # ── Left: fill factor vs slot count ──────────────────────────────────────
    ax = axes[0]
    for (lam, g), col in zip(sorted(lam_groups.items()), colours):
        ax.plot(g["slots"], g["fill"], "o-", color=col,
                label=f"{lam} lam ({lam * 0.35:.0f} mm)")
    ax.set_xlabel("Number of slots")
    ax.set_ylabel("Slot fill factor  (%)")
    ax.set_title("Fill Factor vs Slot Count")
    ax.legend(title="Stack", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(SLOT_COUNTS)

    # ── Right: yoke height vs slot count ─────────────────────────────────────
    ax = axes[1]
    for (lam, g), col in zip(sorted(lam_groups.items()), colours):
        ax.plot(g["slots"], g["yoke"], "s--", color=col,
                label=f"{lam} lam")
    ax.set_xlabel("Number of slots")
    ax.set_ylabel("Yoke height  (mm)")
    ax.set_title("Yoke Height vs Slot Count")
    ax.legend(title="Stack", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(SLOT_COUNTS)

    plt.suptitle("Stator Parameter Sweep — 15 configurations", fontsize=13)
    plt.tight_layout()

    png = os.path.join(output_dir, "batch_sweep_chart.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {png}")
    plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _args()
    os.makedirs(args.output, exist_ok=True)

    # 1. Build sweep
    sweep = make_sweep()
    print(f"\nParameter sweep: {len(SLOT_COUNTS)} slot counts × "
          f"{len(LAMINATION_COUNTS)} stack depths = {len(sweep)} jobs")

    # 2. Validate all
    rows = validate_all(sweep)

    if args.dry_run:
        print("\n  [--dry-run] Skipping generation.")
        print_summary(rows, [])
        if not args.no_plot:
            plot_sweep(rows, args.output)
        return

    # 3. Run batch — only valid configs
    valid_cfgs = [row["cfg"] for row in rows if row["v"]["success"]]
    valid_ids  = [row["job_id"] for row in rows if row["v"]["success"]]
    n_valid    = len(valid_cfgs)

    print(f"\nRunning batch ({n_valid} jobs) …")
    bar = ProgressBar(n_valid)

    t0 = time.time()
    results = sp.generate_batch(
        valid_cfgs,
        output_dir=args.output,
        formats="JSON",
        progress_callback=lambda done, total, job_id: bar(done, total, job_id),
    )
    elapsed = time.time() - t0

    # Restore job_ids from our names (generate_batch uses batch_N internally)
    for i, r in enumerate(results):
        r["job_id"] = valid_ids[i]

    # Fill in skipped (invalid) jobs
    invalid_results = [
        {"job_id": row["job_id"], "success": False,
         "error": row["v"].get("error", "validation failed")}
        for row in rows if not row["v"]["success"]
    ]
    all_results = results + invalid_results

    n_ok   = sum(1 for r in results if r.get("success"))
    n_fail = n_valid - n_ok
    print(f"\n  Completed in {elapsed:.1f}s  |  "
          f"{n_ok} succeeded  |  {n_fail} failed")

    # 4. Write batch summary JSON
    summary_path = os.path.join(args.output, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Summary → {summary_path}")

    # 5. Print table
    print_summary(rows, all_results)

    # 6. Chart
    if not args.no_plot:
        print("\nRendering sweep chart …")
        plot_sweep(rows, args.output)

    print(f"\nDone.  Output → {args.output}\n")


if __name__ == "__main__":
    main()
