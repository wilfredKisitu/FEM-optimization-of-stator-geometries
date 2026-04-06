"""Top-level pipeline orchestrator.

Receives a :class:`StatorMeshInput`, runs all three FEA stages in sequence,
and writes results to disk.

Usage::

    from fea_pipeline.orchestrator import run_fea_pipeline
    from fea_pipeline.io.schema import StatorMeshInput

    inp = StatorMeshInput(stator_id="my_stator", ...)
    results = run_fea_pipeline(inp, config_path="configs/default.yaml",
                               output_dir="results/")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io.schema import StatorMeshInput
from .io.mesh_reader import load_stator_geometry
from .io.result_writer import write_results
from .electromagnetic.solver import run_electromagnetic_analysis
from .thermal.solver import run_thermal_analysis
from .structural.solver import run_structural_analysis

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResults:
    """Holds the output of all three analysis stages plus cross-physics metrics."""
    em_results: dict
    thermal_results: dict
    structural_results: dict
    coupled_metrics: dict


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fea_pipeline(
    stator_input: StatorMeshInput,
    config_path: str = "configs/default.yaml",
    output_dir: str = "results/",
) -> PipelineResults:
    """Run the full three-stage FEA pipeline.

    Parameters
    ----------
    stator_input:
        Validated :class:`StatorMeshInput` from the upstream mesh module.
    config_path:
        Path to a YAML configuration file.  If the file does not exist, built-in
        defaults are used silently.
    output_dir:
        Root directory for result files.

    Returns
    -------
    PipelineResults
        All stage results and coupled metrics.
    """
    config = _load_config(config_path)
    _configure_logging(config.get("pipeline", {}).get("log_level", "INFO"))

    log.info("=== FEA Pipeline  stator_id=%s ===", stator_input.stator_id)

    # --- Load / synthesise mesh ---
    log.info("Loading stator geometry…")
    mesh, region_meshes = load_stator_geometry(stator_input)
    log.info("  %d nodes, %d elements", mesh.n_nodes, mesh.n_elements)

    # --- Stage 1: Electromagnetic ---
    log.info("Stage 1 — Electromagnetic analysis")
    em_results = run_electromagnetic_analysis(
        mesh=mesh,
        regions=region_meshes,
        stator=stator_input,
        config=config.get("electromagnetic", {}),
    )
    log.info(
        "  torque=%.3f N·m  total_loss=%.1f W  η=%.4f",
        em_results["torque_Nm"],
        em_results["total_loss_W"],
        em_results["efficiency"],
    )

    # --- Stage 2: Thermal ---
    log.info("Stage 2 — Thermal analysis")
    thermal_results = run_thermal_analysis(
        mesh=mesh,
        regions=region_meshes,
        stator=stator_input,
        em_results=em_results,
        config=config.get("thermal", {}),
    )
    log.info(
        "  T_peak=%.1f K (%.1f °C)  margin=%.1f K",
        thermal_results["peak_temperature_K"],
        thermal_results["peak_temperature_C"],
        thermal_results["thermal_margin_K"],
    )

    # --- Stage 3: Structural ---
    log.info("Stage 3 — Structural analysis")
    structural_results = run_structural_analysis(
        mesh=mesh,
        regions=region_meshes,
        stator=stator_input,
        em_results=em_results,
        thermal_results=thermal_results,
        config=config.get("structural", {}),
    )
    log.info(
        "  max_vm=%.3e Pa  SF=%.2f  max_disp=%.3e m",
        structural_results["max_von_mises_Pa"],
        structural_results["safety_factor"],
        structural_results["max_displacement_m"],
    )

    # --- Cross-physics metrics ---
    coupled_metrics = _compute_coupled_metrics(em_results, thermal_results,
                                               structural_results)

    results = PipelineResults(
        em_results=em_results,
        thermal_results=thermal_results,
        structural_results=structural_results,
        coupled_metrics=coupled_metrics,
    )

    # --- Write to disk ---
    os.makedirs(output_dir, exist_ok=True)
    write_results(results, output_dir, stator_input.stator_id)
    log.info("Results written → %s/%s/", output_dir, stator_input.stator_id)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    """Load YAML config; return empty dict if file not found."""
    p = Path(path)
    if not p.exists():
        log.debug("Config file %s not found — using built-in defaults.", path)
        return _default_config()
    with open(p) as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _default_config() -> dict:
    """Minimal built-in configuration used when no YAML is provided."""
    return {
        "pipeline": {"log_level": "INFO"},
        "electromagnetic": {
            "nonlinear": {"enabled": True, "max_iterations": 20, "tolerance": 1e-5},
            "loss_model": "steinmetz",
            "materials": {"temperature_dependent": False},
        },
        "thermal": {
            "analysis_type": "steady_state",
            "cooling": {
                "type": "water_jacket",
                "coolant_temperature_K": 313.15,
                "h_outer": 500.0,
                "h_endturn": 80.0,
                "contact_resistance": 1e-4,
            },
            "insulation": {"class": "F", "max_temperature_K": 428.15},
            "anisotropy": {"k_in_plane": 28.0, "k_through_plane": 1.0},
            "convergence_tolerance": 1e-8,
        },
        "structural": {
            "reference_temperature_K": 293.15,
            "electromagnetic_loads": True,
            "thermal_loads": True,
            "materials": {
                "stator_core": {
                    "youngs_modulus_Pa": 2.0e11,
                    "poisson_ratio": 0.28,
                    "density_kg_m3": 7650,
                    "thermal_expansion_1_K": 12.0e-6,
                    "yield_strength_Pa": 3.5e8,
                    "ultimate_strength_Pa": 5.0e8,
                    "fatigue_limit_Pa": 2.0e8,
                },
                "winding_equivalent": {
                    "youngs_modulus_Pa": 3.0e9,
                    "poisson_ratio": 0.35,
                    "density_kg_m3": 3500,
                    "thermal_expansion_1_K": 18.0e-6,
                },
            },
            "fatigue": {
                "method": "goodman",
                "stress_concentration_factor": 1.5,
                "surface_finish_factor": 0.85,
                "reliability_factor": 0.897,
            },
            "modal": {"enabled": True, "num_modes": 6},
        },
        "output": {"format": "json", "write_fields": True},
    }


def _compute_coupled_metrics(
    em: dict, thermal: dict, structural: dict
) -> dict:
    """Compute cross-physics summary metrics."""
    total_loss  = em["total_loss_W"]
    peak_T      = thermal["peak_temperature_K"]
    max_vm      = structural["max_von_mises_Pa"]
    yield_Pa    = structural["yield_strength_Pa"]

    return {
        "total_loss_W":             total_loss,
        "peak_temperature_K":       peak_T,
        "max_von_mises_Pa":         max_vm,
        "thermal_derating_factor":  _thermal_derating(peak_T),
        "safety_factor":            (yield_Pa / max_vm) if max_vm > 0 else float("inf"),
    }


def _thermal_derating(T_peak_K: float) -> float:
    """Linear derating above 120 °C (393 K) for class-F insulation."""
    T_max = 393.0
    if T_peak_K <= T_max:
        return 1.0
    return max(0.0, 1.0 - (T_peak_K - T_max) / 50.0)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)-8s %(name)s — %(message)s",
    )
