"""single_geometry.py — Generate and visualise one stator geometry.

See examples/single_geometry.md for full parameter rationale and design notes.

Usage:
    python examples/single_geometry.py
    python examples/single_geometry.py --output /tmp/my_stator --no-plot
    python examples/single_geometry.py --slots 48 --lam 1400
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stator_pipeline as sp


# CLI

def _args():
    p = argparse.ArgumentParser(description="Single stator geometry example")
    p.add_argument("--output",  default="/tmp/stator_single", help="Output directory  (default: /tmp/stator_single)")
    p.add_argument("--no-plot", action="store_true", help="Skip gmsh visualisation")
    p.add_argument("--slots",   type=int, default=48,   help="Number of stator slots  (default: 48)")
    p.add_argument("--lam",     type=int, default=1400, help="Number of laminations   (default: 1400)")
    return p.parse_args()


# Config factory

def make_config(n_slots: int = 48, n_lam: int = 1400) -> sp.StatorConfig:
    return sp.StatorConfig(
        # --- Radial geometry -------------------------------------------------
        R_outer=0.650,
        R_inner=0.420,
        airgap_length=0.003,
        n_slots=n_slots,
        # --- Slot geometry ---------------------------------------------------
        slot_depth=0.115, slot_width_outer=0.022, slot_width_inner=0.019,
        slot_opening=0.008, slot_opening_depth=0.006, tooth_tip_angle=0.08,
        slot_shape=sp.SlotShape.SEMI_CLOSED,
        # --- Coil / winding --------------------------------------------------
        # coil_depth <= slot_depth - slot_opening_depth - 2*insulation_thickness
        # 0.103      <= 0.115      - 0.006              - 2*0.003
        coil_depth=0.103, coil_width_outer=0.016, coil_width_inner=0.013,
        insulation_thickness=0.003,
        turns_per_coil=6,
        coil_pitch=11,
        wire_diameter=0.004,
        slot_fill_factor=0.38,
        winding_type=sp.WindingType.DOUBLE_LAYER,
        # --- Lamination stack ------------------------------------------------
        t_lam=0.00050,
        n_lam=n_lam,
        z_spacing=0.0,
        insulation_coating_thickness=0.0001,
        material=sp.LaminationMaterial.M330_50A,
        # --- Mesh sizing -----------------------------------------------------
        mesh_yoke=0.020, mesh_slot=0.010, mesh_coil=0.006, mesh_ins=0.003,
        mesh_boundary_layers=3, mesh_curvature=0.3, mesh_transition_layers=2,
    )


#  Validation report 

_SHAPE_NAMES    = {0: "RECTANGULAR", 1: "TRAPEZOIDAL", 2: "ROUND_BOTTOM", 3: "SEMI_CLOSED"}
_WINDING_NAMES  = {0: "SINGLE_LAYER", 1: "DOUBLE_LAYER",2: "CONCENTRATED", 3: "DISTRIBUTED"}
_MATERIAL_NAMES = {0: "M270-35A", 1: "M330-50A", 2: "M400-50A", 3: "NO20", 4: "CUSTOM"}

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


# Gmsh visualisation

def visualise(cfg: sp.StatorConfig, v: dict, output_dir: str) -> None:
    """Render the 2-D cross-section using the gmsh API."""
    try:
        import gmsh
    except ImportError:
        print("\n  [skip] gmsh not installed — skipping visualisation")
        return

    def _rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    YOKE_COL    = "#4a90d9"
    PHASE_LOWER = ["#e63946", "#2a9d8f", "#f4a261"]
    PHASE_UPPER = ["#ff8fa3", "#80cdc1", "#ffd6a5"]
    INS_COL     = "#f5e642"

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("stator_vis")
    occ = gmsh.model.occ

    R_o, R_i = cfg.R_outer, cfg.R_inner
    n         = cfg.n_slots
    pitch     = 2 * math.pi / n
    depth     = cfg.slot_depth
    hw        = cfg.slot_width_outer / 2
    op_d      = cfg.slot_opening_depth
    ins       = cfg.insulation_thickness

    # Yoke annulus via boolean cut
    outer_tag = occ.addDisk(0, 0, 0, R_o, R_o)
    inner_tag = occ.addDisk(0, 0, 0, R_i, R_i)
    yoke_result, _ = occ.cut([(2, outer_tag)], [(2, inner_tag)], removeTool=True)
    yoke_tags = [t for _, t in yoke_result]

    ins_tags     = []
    coil_lo_tags = [[], [], []]
    coil_hi_tags = [[], [], []]

    for k in range(n):
        th = k * pitch
        ph = k % 3

        wy0 = R_i + op_d + ins
        wy1 = R_i + depth - ins
        wh  = (wy1 - wy0 - ins) / 2
        wx  = hw - ins

        def _rect(x0: float, y0: float, dx: float, dy: float, _th: float = th) -> int:
            tag = occ.addRectangle(x0, y0, 0, dx, dy)
            occ.rotate([(2, tag)], 0, 0, 0, 0, 0, 1, _th)
            return tag

        ins_tags.append(_rect(-hw + ins, R_i + op_d + ins, 2 * (hw - ins), wy1 - wy0))
        coil_lo_tags[ph].append(_rect(-wx, wy0,            2 * wx, wh))
        coil_hi_tags[ph].append(_rect(-wx, wy0 + wh + ins, 2 * wx, wh))

    occ.synchronize()

    # Physical groups and colours
    if yoke_tags:
        gmsh.model.addPhysicalGroup(2, yoke_tags, name="Yoke")
        gmsh.model.setColor([(2, t) for t in yoke_tags], *_rgb(YOKE_COL), 255)

    if ins_tags:
        gmsh.model.addPhysicalGroup(2, ins_tags, name="Insulation")
        gmsh.model.setColor([(2, t) for t in ins_tags], *_rgb(INS_COL), 255)

    for ph, name in enumerate("ABC"):
        if coil_lo_tags[ph]:
            gmsh.model.addPhysicalGroup(2, coil_lo_tags[ph], name=f"Coil_{name}_lower")
            gmsh.model.setColor([(2, t) for t in coil_lo_tags[ph]], *_rgb(PHASE_LOWER[ph]), 255)
        if coil_hi_tags[ph]:
            gmsh.model.addPhysicalGroup(2, coil_hi_tags[ph], name=f"Coil_{name}_upper")
            gmsh.model.setColor([(2, t) for t in coil_hi_tags[ph]], *_rgb(PHASE_UPPER[ph]), 255)

    gmsh.option.setNumber("Geometry.SurfaceLabels", 0)
    gmsh.option.setColor("General.Background", 255, 255, 255)

    png = os.path.join(output_dir, "stator_cross_section.png")
    try:
        gmsh.fltk.initialize()
        gmsh.option.setNumber("General.GraphicsWidth",  900)
        gmsh.option.setNumber("General.GraphicsHeight", 900)
        gmsh.write(png)
        print(f"\n  Cross-section saved → {png}")
        gmsh.fltk.run()
    except Exception:
        gmsh.write(png)
        print(f"\n  Cross-section saved → {png}")
    finally:
        gmsh.finalize()


# 3-D lamination stack visualisation

def visualise_3d(cfg: sp.StatorConfig, v: dict, output_dir: str) -> None:
    """Render a 3-D lamination stack using gmsh OCC extrusion.

    Builds ``n_vis = min(n_lam, 8)`` individual laminations, each extruded by
    ``t_lam`` along Z, with a small visible air-gap between them so the
    stacked structure is apparent.  Yoke is rendered semi-transparent so the
    internal coils and insulation are visible.
    """
    try:
        import gmsh
    except ImportError:
        print("\n  [skip] gmsh not installed — skipping 3-D visualisation")
        return

    def _rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    YOKE_COL    = "#4a90d9"
    PHASE_LOWER = ["#e63946", "#2a9d8f", "#f4a261"]
    PHASE_UPPER = ["#ff8fa3", "#80cdc1", "#ffd6a5"]
    INS_COL     = "#f5e642"

    R_o   = cfg.R_outer
    R_i   = cfg.R_inner
    n     = cfg.n_slots
    pitch = 2 * math.pi / n
    depth = cfg.slot_depth
    hw    = cfg.slot_width_outer / 2
    op_d  = cfg.slot_opening_depth
    ins   = cfg.insulation_thickness
    t_lam = cfg.t_lam
    # Ensure inter-lamination gap is visible in the GUI
    vis_gap = max(cfg.z_spacing, 0.15 * t_lam)
    n_vis   = min(cfg.n_lam, 8)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("stator_3d_stack")
    occ = gmsh.model.occ

    all_yoke_vols  = []
    all_ins_vols   = []
    coil_lo_vols   = [[], [], []]
    coil_hi_vols   = [[], [], []]

    for lam_idx in range(n_vis):
        z0 = lam_idx * (t_lam + vis_gap)

        # Yoke annulus: outer disk – inner bore (no slot cut keeps it simple)
        outer_tag = occ.addDisk(0, 0, z0, R_o, R_o)
        inner_tag = occ.addDisk(0, 0, z0, R_i, R_i)
        yoke_res, _ = occ.cut([(2, outer_tag)], [(2, inner_tag)], removeTool=True)
        yoke_tags_2d = [t for _, t in yoke_res]

        ins_tags_2d = []
        coil_lo_2d  = [[], [], []]
        coil_hi_2d  = [[], [], []]

        for k in range(n):
            th = k * pitch
            ph = k % 3
            wy0 = R_i + op_d + ins
            wy1 = R_i + depth - ins
            wh  = (wy1 - wy0 - ins) / 2
            wx  = hw - ins

            def _rect(x0: float, y0: float, dx: float, dy: float,
                      _th: float = th, _z: float = z0) -> int:
                tag = occ.addRectangle(x0, y0, _z, dx, dy)
                occ.rotate([(2, tag)], 0, 0, _z, 0, 0, 1, _th)
                return tag

            ins_tags_2d.append(
                _rect(-hw + ins, R_i + op_d + ins, 2 * (hw - ins), wy1 - wy0))
            coil_lo_2d[ph].append(_rect(-wx, wy0,            2 * wx, wh))
            coil_hi_2d[ph].append(_rect(-wx, wy0 + wh + ins, 2 * wx, wh))

        # Extrude 2-D surfaces by t_lam along Z → collect volume tags
        def _extrude(tags_2d: list[int]) -> list[int]:
            vols = []
            for tag in tags_2d:
                res = occ.extrude([(2, tag)], 0, 0, t_lam)
                vols += [t for dim, t in res if dim == 3]
            return vols

        all_yoke_vols += _extrude(yoke_tags_2d)
        all_ins_vols  += _extrude(ins_tags_2d)
        for ph in range(3):
            coil_lo_vols[ph] += _extrude(coil_lo_2d[ph])
            coil_hi_vols[ph] += _extrude(coil_hi_2d[ph])

    occ.synchronize()

    # Physical groups + colours (yoke semi-transparent so coils show through)
    if all_yoke_vols:
        gmsh.model.addPhysicalGroup(3, all_yoke_vols, name="Yoke")
        gmsh.model.setColor([(3, t) for t in all_yoke_vols],
                            *_rgb(YOKE_COL), 140)          # alpha=140

    if all_ins_vols:
        gmsh.model.addPhysicalGroup(3, all_ins_vols, name="Insulation")
        gmsh.model.setColor([(3, t) for t in all_ins_vols],
                            *_rgb(INS_COL), 255)

    for ph, name in enumerate("ABC"):
        if coil_lo_vols[ph]:
            gmsh.model.addPhysicalGroup(3, coil_lo_vols[ph],
                                        name=f"Coil_{name}_lower")
            gmsh.model.setColor([(3, t) for t in coil_lo_vols[ph]],
                                *_rgb(PHASE_LOWER[ph]), 255)
        if coil_hi_vols[ph]:
            gmsh.model.addPhysicalGroup(3, coil_hi_vols[ph],
                                        name=f"Coil_{name}_upper")
            gmsh.model.setColor([(3, t) for t in coil_hi_vols[ph]],
                                *_rgb(PHASE_UPPER[ph]), 255)

    # Dark background, enable transparency rendering
    gmsh.option.setColor("General.Background",  30,  30,  40)
    gmsh.option.setColor("General.Foreground", 220, 220, 220)
    gmsh.option.setNumber("Geometry.SurfaceLabels", 0)
    gmsh.option.setNumber("General.SmallAxes", 1)

    stack_mm = n_vis * t_lam * 1e3
    print(f"\n  Showing {n_vis}/{cfg.n_lam} laminations  "
          f"(t_lam={cfg.t_lam*1e3:.3f} mm, "
          f"vis stack ≈ {stack_mm:.2f} mm)")

    png = os.path.join(output_dir, "stator_3d_stack.png")
    try:
        gmsh.fltk.initialize()
        gmsh.option.setNumber("General.GraphicsWidth",  1024)
        gmsh.option.setNumber("General.GraphicsHeight",  800)
        # Isometric-ish view angle
        gmsh.option.setNumber("General.RotationX",  55)
        gmsh.option.setNumber("General.RotationY",   0)
        gmsh.option.setNumber("General.RotationZ",  25)
        gmsh.write(png)
        print(f"  3-D stack saved → {png}")
        gmsh.fltk.run()
    except Exception:
        gmsh.write(png)
        print(f"  3-D stack saved → {png}")
    finally:
        gmsh.finalize()


# Main

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
        print("\nRendering 2-D cross-section …")
        visualise(cfg, v, args.output)
        print("\nRendering 3-D lamination stack …")
        visualise_3d(cfg, v, args.output)
    else:
        print("\n  [--no-plot] skipping visualisation")

    print(f"\nDone.  Output → {args.output}\n")


if __name__ == "__main__":
    main()
