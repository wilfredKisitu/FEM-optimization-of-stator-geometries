"""single_geometry.py — Generate and visualise one stator geometry.

Demonstrates:
  1. Building a StatorConfig (36-slot, double-layer, semi-closed)
  2. Validating parameters and reading all derived geometry values
  3. Running the single-job pipeline (produces JSON metadata)
  4. Drawing the 2-D cross-section with matplotlib (colour-coded regions)
  5. Printing a full geometry + mesh sizing summary

Usage:
    cd /path/to/FEM
    python examples/single_geometry.py
    python examples/single_geometry.py --output /tmp/my_stator --no-plot
    python examples/single_geometry.py --slots 48 --lam 1000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import stator_pipeline as sp


# ── CLI ───────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(description="Single stator geometry example")
    p.add_argument("--output",  default="/tmp/stator_single",
                   help="Output directory  (default: /tmp/stator_single)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib visualisation")
    p.add_argument("--slots",   type=int, default=36,
                   help="Number of stator slots  (default: 36)")
    p.add_argument("--lam",     type=int, default=200,
                   help="Number of laminations   (default: 200)")
    return p.parse_args()


# ── Config factory ────────────────────────────────────────────────────────────

def make_config(n_slots: int = 36, n_lam: int = 200) -> sp.StatorConfig:
    """36-slot (or custom), double-layer, semi-closed, M270-35A silicon steel."""
    return sp.StatorConfig(
        R_outer=0.250, R_inner=0.151, airgap_length=0.001,
        n_slots=n_slots,
        slot_depth=0.060, slot_width_outer=0.012, slot_width_inner=0.010,
        slot_opening=0.004, slot_opening_depth=0.003, tooth_tip_angle=0.10,
        slot_shape=sp.SlotShape.SEMI_CLOSED,
        coil_depth=0.050, coil_width_outer=0.008, coil_width_inner=0.007,
        insulation_thickness=0.001, turns_per_coil=10, coil_pitch=5,
        wire_diameter=0.001, slot_fill_factor=0.45,
        winding_type=sp.WindingType.DOUBLE_LAYER,
        t_lam=0.00035, n_lam=n_lam, z_spacing=0.0,
        insulation_coating_thickness=0.00005,
        material=sp.LaminationMaterial.M270_35A,
        mesh_yoke=0.006, mesh_slot=0.003, mesh_coil=0.0015, mesh_ins=0.0007,
        mesh_boundary_layers=3, mesh_curvature=0.3, mesh_transition_layers=2,
    )


# ── Validation report ─────────────────────────────────────────────────────────

_SHAPE_NAMES    = {0: "RECTANGULAR", 1: "TRAPEZOIDAL",
                   2: "ROUND_BOTTOM", 3: "SEMI_CLOSED"}
_WINDING_NAMES  = {0: "SINGLE_LAYER", 1: "DOUBLE_LAYER",
                   2: "CONCENTRATED", 3: "DISTRIBUTED"}
_MATERIAL_NAMES = {0: "M270-35A", 1: "M330-50A", 2: "M400-50A",
                   3: "NO20", 4: "CUSTOM"}

def print_report(cfg: sp.StatorConfig, v: dict) -> None:
    W = 62
    print("\n" + "=" * W)
    print("  STATOR GEOMETRY — VALIDATION REPORT")
    print("=" * W)
    print(f"  {'Outer radius':<30} {cfg.R_outer * 1e3:>8.1f} mm")
    print(f"  {'Inner radius':<30} {cfg.R_inner * 1e3:>8.1f} mm")
    print(f"  {'Air-gap':<30} {cfg.airgap_length * 1e3:>8.2f} mm")
    print(f"  {'Slots':<30} {cfg.n_slots:>9}")
    print(f"  {'Slot shape':<30} {_SHAPE_NAMES[cfg.slot_shape]:>12}")
    print(f"  {'Winding type':<30} {_WINDING_NAMES[cfg.winding_type]:>12}")
    print(f"  {'Laminations':<30} {cfg.n_lam:>4} × {cfg.t_lam*1e3:.3f} mm")
    print(f"  {'Material':<30} {_MATERIAL_NAMES[cfg.material]:>12}")
    print("-" * W)
    print(f"  {'Yoke height':<30} {v['yoke_height'] * 1e3:>8.2f} mm")
    print(f"  {'Tooth width':<30} {v['tooth_width'] * 1e3:>8.3f} mm")
    print(f"  {'Slot pitch':<30} {math.degrees(v['slot_pitch']):>8.2f} °  "
          f"({v['slot_pitch']:.4f} rad)")
    print(f"  {'Stack length':<30} {v['stack_length'] * 1e3:>8.1f} mm")
    print(f"  {'Fill factor':<30} {v['fill_factor']:>8.3f}  "
          f"({v['fill_factor']*100:.1f} %)")
    print("-" * W)
    print(f"  {'Mesh — yoke':<30} {cfg.mesh_yoke * 1e3:>8.2f} mm")
    print(f"  {'Mesh — slot':<30} {cfg.mesh_slot * 1e3:>8.2f} mm")
    print(f"  {'Mesh — coil':<30} {cfg.mesh_coil * 1e3:>8.2f} mm")
    print(f"  {'Mesh — insulation':<30} {cfg.mesh_ins * 1e3:>8.2f} mm")
    print("=" * W)


# ── Matplotlib visualisation ──────────────────────────────────────────────────

def visualise(cfg: sp.StatorConfig, v: dict, output_dir: str) -> None:
    """Render the 2-D cross-section using pure matplotlib geometry."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("\n  [skip] matplotlib not installed — skipping visualisation")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    R_o   = cfg.R_outer
    R_i   = cfg.R_inner
    depth = cfg.slot_depth
    hw    = cfg.slot_width_outer / 2
    hop   = cfg.slot_opening / 2
    op_d  = cfg.slot_opening_depth
    ins   = cfg.insulation_thickness
    n     = cfg.n_slots
    pitch = 2 * math.pi / n

    # ── Yoke ring ─────────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((0, 0), R_o, color="#4a90d9", zorder=1))
    ax.add_patch(plt.Circle((0, 0), R_i, color="white",   zorder=2))

    PHASE_LOWER = ["#e63946", "#2a9d8f", "#f4a261"]
    PHASE_UPPER = ["#ff8fa3", "#80cdc1", "#ffd6a5"]
    INS_COL     = "#f5e642"

    def rot(x, y, th):
        c, s = math.cos(th), math.sin(th)
        return (x * c - y * s, x * s + y * c)

    def rect(corners, color, z):
        ax.add_patch(plt.Polygon(corners, color=color, zorder=z))

    for k in range(n):
        th  = k * pitch
        ph  = k % 3

        # Slot opening (narrow gap below bore)
        rect([rot(-hop, R_i - op_d, th), rot(hop, R_i - op_d, th),
              rot(hop, R_i, th),         rot(-hop, R_i, th)],
             "white", 3)

        # Slot body
        rect([rot(-hw, R_i, th), rot(hw, R_i, th),
              rot(hw, R_i + depth, th), rot(-hw, R_i + depth, th)],
             "white", 3)

        # Insulation liner (outer yellow, inner white punch-out)
        y0, y1 = R_i + op_d, R_i + depth
        rect([rot(-hw,       y0, th), rot(hw,       y0, th),
              rot(hw,        y1, th), rot(-hw,       y1, th)],
             INS_COL, 4)
        rect([rot(-hw + ins, y0 + ins, th), rot(hw - ins, y0 + ins, th),
              rot(hw - ins,  y1 - ins, th), rot(-hw + ins, y1 - ins, th)],
             "white", 5)

        # Winding window
        wy0 = R_i + op_d + ins
        wy1 = R_i + depth - ins
        wh  = (wy1 - wy0 - ins) / 2
        wx  = hw - ins

        # Lower layer
        rect([rot(-wx, wy0,        th), rot(wx, wy0,        th),
              rot(wx, wy0 + wh,    th), rot(-wx, wy0 + wh,    th)],
             PHASE_LOWER[ph], 6)

        # Upper layer (next phase)
        yb0 = wy0 + wh + ins
        rect([rot(-wx, yb0,        th), rot(wx, yb0,        th),
              rot(wx, yb0 + wh,    th), rot(-wx, yb0 + wh,   th)],
             PHASE_UPPER[ph], 6)

    # ── Bore circle ───────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((0, 0), R_i, fill=False,
                             edgecolor="#264653", linewidth=1.5,
                             linestyle="--", zorder=7))

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(handles=[
        mpatches.Patch(color="#4a90d9",       label="Yoke"),
        mpatches.Patch(color=INS_COL,          label="Slot insulation"),
        mpatches.Patch(color=PHASE_LOWER[0],   label="Phase A — lower coil"),
        mpatches.Patch(color=PHASE_UPPER[0],   label="Phase A — upper coil"),
        mpatches.Patch(color=PHASE_LOWER[1],   label="Phase B — lower coil"),
        mpatches.Patch(color=PHASE_LOWER[2],   label="Phase C — lower coil"),
    ], loc="lower right", fontsize=9, framealpha=0.9)

    lim = R_o * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(
        f"{n}-slot stator  |  Rₒ = {R_o*1e3:.0f} mm  Rᵢ = {R_i*1e3:.0f} mm\n"
        f"Stack: {cfg.n_lam} lam × {cfg.t_lam*1e3:.3f} mm = "
        f"{v['stack_length']*1e3:.1f} mm  |  "
        f"Fill factor: {v['fill_factor']*100:.1f} %",
        fontsize=11,
    )
    ax.set_xlabel("x  (m)"); ax.set_ylabel("y  (m)")

    png = os.path.join(output_dir, "stator_cross_section.png")
    plt.tight_layout()
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"\n  Cross-section saved → {png}")
    plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _args()
    os.makedirs(args.output, exist_ok=True)

    # 1. Build config
    cfg = make_config(n_slots=args.slots, n_lam=args.lam)

    # 2. Validate
    print("\nValidating parameters …")
    v = sp.validate_config(cfg)
    if not v["success"]:
        print(f"  ERROR: {v['error']}")
        sys.exit(1)
    print("  OK")

    # 3. Print report
    print_report(cfg, v)

    # 4. Pipeline (JSON metadata)
    print("\nRunning pipeline …")
    out = sp.generate_single(cfg, args.output, formats="JSON")
    if not out["success"]:
        print(f"  ERROR: {out['error']}")
        sys.exit(1)

    print(f"  Stem      : {out['stem']}")
    if "json_path" in out:
        print(f"  Metadata  : {out['json_path']}")

    # 5. SHA-256 fingerprint
    param_str = json.dumps(
        {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, default=float
    )
    fp = sp.sha256(param_str)
    print(f"  SHA-256   : {fp[:20]}…")

    # 6. Visualise
    if not args.no_plot:
        print("\nRendering cross-section …")
        visualise(cfg, v, args.output)
    else:
        print("\n  [--no-plot] skipping visualisation")

    print(f"\nDone.  Output → {args.output}\n")


if __name__ == "__main__":
    main()
