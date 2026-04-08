"""evaluation/evaluator.py — Full evaluation pipeline for one individual.

The evaluation chain for a single chromosome is:

  1. Geometric constraint check (O(1) — rejects unphysical chromosomes)
  2. Cache lookup (skips duplicate evaluations)
  3. Chromosome decode → StatorParams (mesh-generation parameters)
  4. StatorParams validation via stator_pipeline.validate_and_derive
  5. Construct StatorMeshInput using synthetic mesh (no external mesh file)
  6. Run FEA pipeline (EM → Thermal → Structural)
  7. Extract ObjectiveVector from PipelineResults
  8. Store result in cache

The stator_pipeline and FEA pipeline are called **only through their public
interfaces** — this module never imports internal implementation details.
"""

from __future__ import annotations

import logging
import sys
import os

import numpy as np

from ..chromosome import decode_chromosome
from ..constraints import check_geometric_constraints, GeometricConstraintViolation
from ..objectives import extract_objectives, INFEASIBLE_OBJECTIVES, ObjectiveVector
from .cache import EvaluationCache

log = logging.getLogger(__name__)


def evaluate_individual(
    genes: np.ndarray,
    generation: int,
    individual_index: int,
    config: dict,
    cache: EvaluationCache,
    fea_config_path: str,
    fea_output_dir: str,
) -> ObjectiveVector:
    """Evaluate one individual: geometry check → mesh → FEA → objectives.

    Parameters
    ----------
    genes:
        (N_GENES,) float array — gene vector.
    generation:
        Current GA generation number (used for stator_id naming).
    individual_index:
        Index within the current evaluation batch (used for stator_id naming).
    config:
        Full GA config dict (passed through to objective extractor and
        constraint checker).
    cache:
        :class:`EvaluationCache` instance — shared within a process.
    fea_config_path:
        Path to the FEA YAML config file.
    fea_output_dir:
        Root directory for FEA result JSON files.

    Returns
    -------
    ObjectiveVector
        Fully populated objective vector.  Returns :data:`INFEASIBLE_OBJECTIVES`
        on any failure (geometric, mesh, or FEA error).
    """
    # ── Step 1: Geometric feasibility ────────────────────────────────────
    try:
        check_geometric_constraints(genes, config)
    except GeometricConstraintViolation as exc:
        log.debug(
            "Gen %d ind %d: geometric constraint failed — %s",
            generation, individual_index, exc,
        )
        return INFEASIBLE_OBJECTIVES

    # ── Step 2: Cache lookup ─────────────────────────────────────────────
    cached = cache.get(genes)
    if cached is not None:
        log.debug("Gen %d ind %d: cache hit", generation, individual_index)
        return cached

    # ── Step 3: Decode chromosome ────────────────────────────────────────
    try:
        params = decode_chromosome(genes)
    except ValueError as exc:
        log.warning("Gen %d ind %d: decode failed — %s", generation, individual_index, exc)
        return INFEASIBLE_OBJECTIVES

    stator_id = f"ga_gen{generation:04d}_ind{individual_index:04d}"
    log.debug("Evaluating %s", stator_id)

    try:
        obj = _run_fea_pipeline(
            params=params,
            stator_id=stator_id,
            config=config,
            fea_config_path=fea_config_path,
            fea_output_dir=fea_output_dir,
        )
    except Exception as exc:
        log.error(
            "Gen %d ind %d (%s): evaluation error — %s",
            generation, individual_index, stator_id, exc,
            exc_info=True,
        )
        return INFEASIBLE_OBJECTIVES

    cache.put(genes, obj)
    return obj


# ---------------------------------------------------------------------------
# Internal: bridge between chromosome decode and FEA pipeline
# ---------------------------------------------------------------------------

def _run_fea_pipeline(
    params: dict,
    stator_id: str,
    config: dict,
    fea_config_path: str,
    fea_output_dir: str,
) -> ObjectiveVector:
    """Build StatorMeshInput from decoded chromosome and run FEA.

    The stator_pipeline validates geometry and computes derived dimensions.
    The FEA pipeline uses the "synthetic" mesh mode — no external mesh file
    is required, making parallelism simple (no shared filesystem state).

    Parameters
    ----------
    params:
        Output of ``decode_chromosome``.
    stator_id:
        Unique identifier for this evaluation.
    config:
        GA config dict.
    fea_config_path, fea_output_dir:
        Forwarded to ``run_fea_pipeline``.
    """
    # Lazy imports so worker processes do not pay the import cost on startup
    # if they are never used (e.g. when the cache always hits).
    import sys, os
    # Ensure project root is on the path for workers
    _ensure_project_path()

    from stator_pipeline.params import (
        StatorParams, SlotShape, WindingType, validate_and_derive,
    )
    from fea_pipeline.io.schema import StatorMeshInput
    from fea_pipeline.orchestrator import run_fea_pipeline

    op = config["operating_point"]
    mats = config.get("materials", {
        "stator_core": "M250-35A",
        "winding": "copper_class_F",
        "air_gap": "air",
    })

    OD   = params["outer_diameter"]
    ID   = params["inner_diameter"]
    axial = params["axial_length"]
    n_slots = params["num_slots"]
    n_poles = params["num_poles"]

    # ── Build StatorParams for validation (derives yoke_height, tooth_width, etc.)
    # Map decoded params → StatorParams fields
    radial_build = (OD - ID) / 2.0
    R_outer = OD / 2.0
    R_inner = ID / 2.0

    slot_d = params["slot_depth"]
    yoke_h = params["yoke_height"]
    tooth_w = params["tooth_width"]
    slot_op = params["slot_opening"]
    fill = params["fill_factor"]

    # Derive slot widths from tooth width and slot pitch
    slot_pitch_inner = np.pi * ID / n_slots
    slot_pitch_outer = np.pi * OD / n_slots
    slot_w_inner = slot_pitch_inner - tooth_w
    slot_w_outer = max(slot_w_inner + 0.001, slot_pitch_outer - tooth_w)

    # Insulation thickness: 1 mm for synthetic mesh
    ins_thick = 0.001
    # coil depth: leave 6 mm for slot opening depth + insulation headroom
    slot_opening_depth = min(0.006, slot_d * 0.05)
    coil_depth = max(0.005, slot_d - slot_opening_depth - 2 * ins_thick)

    coil_w_inner = max(0.002, slot_w_inner - 2 * ins_thick)
    coil_w_outer = max(coil_w_inner + 0.001, slot_w_outer - 2 * ins_thick)

    # lamination stack: assume 0.5 mm per lamination
    t_lam = 0.00050
    n_lam = max(10, int(round(axial / t_lam)))

    sp = StatorParams(
        R_outer=R_outer,
        R_inner=R_inner,
        airgap_length=slot_op,
        n_slots=n_slots,
        slot_depth=slot_d,
        slot_width_outer=max(0.003, slot_w_outer),
        slot_width_inner=max(0.002, slot_w_inner),
        slot_opening=slot_op,
        slot_opening_depth=slot_opening_depth,
        tooth_tip_angle=0.05,
        slot_shape=SlotShape.SEMI_CLOSED,
        coil_depth=coil_depth,
        coil_width_outer=coil_w_outer,
        coil_width_inner=coil_w_inner,
        insulation_thickness=ins_thick,
        turns_per_coil=max(2, params["conductors_per_slot"] // 2),
        coil_pitch=max(1, n_slots // (n_poles * 3 // 2)),
        wire_diameter=0.002,
        slot_fill_factor=fill,
        winding_type=WindingType.DOUBLE_LAYER,
        t_lam=t_lam,
        n_lam=n_lam,
        z_spacing=0.0,
        insulation_coating_thickness=0.00005,
    )

    try:
        validated = validate_and_derive(sp)
    except ValueError as exc:
        raise ValueError(f"StatorParams validation failed: {exc}") from exc

    # ── Construct StatorMeshInput for FEA ────────────────────────────────
    stator_input = StatorMeshInput(
        stator_id=stator_id,
        geometry_source="ga_optimizer",
        mesh_format="synthetic",     # no external mesh file needed
        outer_diameter=OD,
        inner_diameter=ID,
        axial_length=axial,
        num_slots=n_slots,
        num_poles=n_poles,
        slot_opening=slot_op,
        tooth_width=tooth_w,
        yoke_height=yoke_h,
        slot_depth=slot_d,
        winding_type="distributed",
        num_layers=2,
        conductors_per_slot=params["conductors_per_slot"],
        winding_factor=0.866,
        fill_factor=fill,
        material_map=mats,
        rated_current_rms=float(op.get("current_A", 50.0)),
        rated_speed_rpm=float(op.get("speed_rpm", 3000.0)),
        rated_torque=float(op.get("torque_Nm", 50.0)),
        dc_bus_voltage=float(op.get("voltage_V", 400.0)),
    )

    # ── Run FEA ─────────────────────────────────────────────────────────
    ind_output_dir = os.path.join(fea_output_dir, stator_id)
    fea_results = run_fea_pipeline(
        stator_input=stator_input,
        config_path=fea_config_path,
        output_dir=ind_output_dir,
    )

    # ── Extract objectives ───────────────────────────────────────────────
    return extract_objectives(fea_results, params, config)


def _ensure_project_path() -> None:
    """Add project root to sys.path if not already present (for workers)."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    if root not in sys.path:
        sys.path.insert(0, root)
