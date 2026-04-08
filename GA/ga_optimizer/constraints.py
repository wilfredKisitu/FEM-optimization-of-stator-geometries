"""constraints.py — Geometric feasibility checks applied before FEA is called.

Geometric constraints are O(1) and prevent wasting a FEA evaluation on an
unphysical stator.  Individuals that fail are assigned INFEASIBLE_OBJECTIVES
and are kept in the population (subject to selection pressure) rather than
discarded, preserving genetic diversity.
"""

from __future__ import annotations

import math

import numpy as np

from .chromosome import decode_chromosome


class GeometricConstraintViolation(Exception):
    """Raised when a chromosome decodes to a geometrically infeasible stator."""


def check_geometric_constraints(genes: np.ndarray, config: dict) -> None:
    """Check all O(1) geometric constraints before invoking mesh generation.

    Parameters
    ----------
    genes:
        (N_GENES,) gene vector.
    config:
        Full GA config dict.  Reads the ``"constraints"`` sub-dict.

    Raises
    ------
    GeometricConstraintViolation
        On the first violated constraint.  Callers should catch this and
        return :data:`INFEASIBLE_OBJECTIVES` without calling FEA.
    """
    try:
        params = decode_chromosome(genes)
    except ValueError as exc:
        raise GeometricConstraintViolation(f"Chromosome decode failed: {exc}") from exc

    c = config["constraints"]

    OD   = params["outer_diameter"]
    ID   = params["inner_diameter"]
    air_gap = (OD - ID) / 2.0

    # ── 1. Minimum air-gap clearance ────────────────────────────────────
    min_ag = float(c.get("min_air_gap_m", 8e-4))
    if air_gap < min_ag:
        raise GeometricConstraintViolation(
            f"Air gap {air_gap * 1e3:.2f} mm < minimum {min_ag * 1e3:.2f} mm"
        )

    # ── 2. Minimum slot width ────────────────────────────────────────────
    slot_pitch = math.pi * ID / params["num_slots"]
    slot_w = slot_pitch - params["tooth_width"]
    min_slot_w = float(c.get("min_slot_width_m", 3e-3))
    if slot_w < min_slot_w:
        raise GeometricConstraintViolation(
            f"Slot width {slot_w * 1e3:.2f} mm < minimum {min_slot_w * 1e3:.2f} mm"
        )

    # ── 3. Minimum yoke height for flux containment ──────────────────────
    min_yoke = float(c.get("min_yoke_height_m", 8e-3))
    if params["yoke_height"] < min_yoke:
        raise GeometricConstraintViolation(
            f"Yoke height {params['yoke_height'] * 1e3:.2f} mm < "
            f"minimum {min_yoke * 1e3:.2f} mm"
        )

    # ── 4. Conductors per slot must be even (two-layer winding) ──────────
    if params["conductors_per_slot"] % 2 != 0:
        raise GeometricConstraintViolation(
            f"conductors_per_slot ({params['conductors_per_slot']}) must be even"
        )

    # ── 5. Slot opening must be smaller than slot width ──────────────────
    if params["slot_opening"] >= slot_w:
        raise GeometricConstraintViolation(
            f"slot_opening ({params['slot_opening'] * 1e3:.2f} mm) >= "
            f"slot_width ({slot_w * 1e3:.2f} mm)"
        )

    # ── 6. Radial fit: slot_depth + yoke_height must fit radial build ────
    radial_build = (OD - ID) / 2.0
    if params["slot_depth"] + params["yoke_height"] > radial_build:
        raise GeometricConstraintViolation(
            f"slot_depth + yoke_height ({(params['slot_depth'] + params['yoke_height'])*1e3:.1f} mm) "
            f"> radial_build ({radial_build*1e3:.1f} mm)"
        )

    # ── 7. Slot must be deeper than its opening ──────────────────────────
    if params["slot_depth"] <= params["slot_opening"]:
        raise GeometricConstraintViolation(
            f"slot_depth ({params['slot_depth']*1e3:.2f} mm) must be > "
            f"slot_opening ({params['slot_opening']*1e3:.2f} mm)"
        )
