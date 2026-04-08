"""io/result_writer.py — Write Pareto front results to JSON, CSV, and HTML plots.

Produces the final output of one GA run:
  - ``pareto_front.json``      — full solution data
  - ``pareto_front.csv``       — tabular form (easy import into spreadsheets)
  - ``hypervolume_history.json``— per-generation HV indicator
  - ``run_metadata.json``      — config snapshot and timestamps

Plotly is optional: if not installed, the HTML plots are skipped with a warning.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..pareto.archive import ParetoArchive

log = logging.getLogger(__name__)
_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_pareto_results(
    archive: "ParetoArchive",
    output_dir: str,
    config: dict,
    state=None,          # Optional[GAState] — for hypervolume history
    run_id: str | None = None,
) -> dict[str, str]:
    """Write all result files to *output_dir*.

    Parameters
    ----------
    archive:
        Final :class:`ParetoArchive` with all non-dominated solutions.
    output_dir:
        Root output directory.
    config:
        Full GA config (snapshot stored in metadata).
    state:
        :class:`GAState` — optional, used for hypervolume history export.
    run_id:
        Optional unique identifier for this run.  Auto-generated from the
        current UTC timestamp if not provided.

    Returns
    -------
    dict[str, str]
        Mapping of result type → absolute file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    if run_id is None:
        run_id = "ga_run_" + datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")

    ts = datetime.now(timezone.utc).isoformat()
    written: dict[str, str] = {}

    # ── Build solution list ──────────────────────────────────────────────
    solutions = _build_solutions(archive)

    # ── pareto_front.json ───────────────────────────────────────────────
    pf_data = {
        "run_id":                run_id,
        "version":               _VERSION,
        "timestamp_utc":         ts,
        "final_generation":      state.generation if state else -1,
        "total_fea_evaluations": state.total_evaluations if state else -1,
        "pareto_front_size":     archive.size,
        "solutions":             solutions,
    }
    pf_path = out / "pareto_front.json"
    _write_json(pf_data, pf_path)
    written["pareto_front_json"] = str(pf_path)

    # ── pareto_front.csv ────────────────────────────────────────────────
    csv_path = out / "pareto_front.csv"
    _write_csv(solutions, csv_path)
    written["pareto_front_csv"] = str(csv_path)

    # ── hypervolume_history.json ────────────────────────────────────────
    if state is not None and state.hypervolume_history:
        hv_data = {
            "run_id": run_id,
            "hypervolume_history":   state.hypervolume_history,
            "archive_size_history":  state.archive_size_history,
        }
        hv_path = out / "hypervolume_history.json"
        _write_json(hv_data, hv_path)
        written["hypervolume_history"] = str(hv_path)

    # ── run_metadata.json ───────────────────────────────────────────────
    meta = {
        "run_id":        run_id,
        "version":       _VERSION,
        "timestamp_utc": ts,
        "config":        _sanitise(config),
    }
    meta_path = out / "run_metadata.json"
    _write_json(meta, meta_path)
    written["run_metadata"] = str(meta_path)

    # ── Plotly HTML plots (optional) ────────────────────────────────────
    try:
        import plotly  # noqa: F401
        _write_plots(solutions, plots_dir)
        written["plots_dir"] = str(plots_dir)
    except ImportError:
        log.info("plotly not installed — skipping Pareto front plots.")

    log.info("Results written to %s (%d Pareto solutions)", out, archive.size)
    return written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_solutions(archive: "ParetoArchive") -> list[dict]:
    """Convert archive members to serialisable dicts."""
    solutions = []
    for member in archive.members:
        obj = member.objectives
        decoded: dict = {}
        try:
            from ..chromosome import decode_chromosome
            decoded = {
                k: v for k, v in decode_chromosome(member.genes).items()
                if not k.startswith("_")
            }
        except Exception:
            pass

        solutions.append({
            "rank":          0,
            "stator_id":     member.stator_id or "unknown",
            "genes":         member.genes.tolist(),
            "decoded_params": _sanitise(decoded),
            "objectives": {
                "efficiency":         -float(obj.neg_efficiency),
                "total_loss_W":       float(obj.total_loss_W),
                "power_density_W_m3": -float(obj.neg_power_density),
            },
            "constraints": {
                "temperature_violation_K": float(obj.temperature_violation_K),
                "safety_factor_violation": float(obj.safety_factor_violation),
                "feasible":                obj.is_feasible,
            },
        })
    return solutions


def _write_csv(solutions: list[dict], path: Path) -> None:
    if not solutions:
        return
    fieldnames = [
        "stator_id", "efficiency", "total_loss_W", "power_density_W_m3",
        "temperature_violation_K", "safety_factor_violation", "feasible",
        "outer_diameter", "inner_diameter", "axial_length",
        "num_slots", "num_poles", "fill_factor",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sol in solutions:
            row = {
                "stator_id":   sol["stator_id"],
                **sol["objectives"],
                **sol["constraints"],
                **sol.get("decoded_params", {}),
            }
            writer.writerow(row)


def _write_plots(solutions: list[dict], plots_dir: Path) -> None:
    """Generate interactive Plotly scatter plots of the Pareto front."""
    import plotly.graph_objects as go

    if not solutions:
        return

    effs  = [-s["objectives"]["efficiency"] for s in solutions]      # negate back
    losses= [s["objectives"]["total_loss_W"] for s in solutions]
    pds   = [s["objectives"]["power_density_W_m3"] / 1e6 for s in solutions]
    ids_  = [s["stator_id"] for s in solutions]

    # Efficiency vs Loss
    fig1 = go.Figure(go.Scatter(
        x=losses, y=effs,
        mode="markers",
        marker=dict(size=8, color=pds, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Power Density [MW/m³]")),
        text=ids_,
        hovertemplate="ID: %{text}<br>Loss: %{x:.0f} W<br>η: %{y:.4f}<extra></extra>",
    ))
    fig1.update_layout(
        title="Pareto Front — Efficiency vs Total Loss",
        xaxis_title="Total Loss [W]",
        yaxis_title="Efficiency η",
    )
    fig1.write_html(str(plots_dir / "pareto_front_2d_eff_loss.html"))

    # Power Density vs Loss
    fig2 = go.Figure(go.Scatter(
        x=losses, y=pds,
        mode="markers",
        marker=dict(size=8, color=effs, colorscale="Plasma",
                    showscale=True, colorbar=dict(title="Efficiency η")),
        text=ids_,
        hovertemplate="ID: %{text}<br>Loss: %{x:.0f} W<br>PD: %{y:.3f} MW/m³<extra></extra>",
    ))
    fig2.update_layout(
        title="Pareto Front — Power Density vs Total Loss",
        xaxis_title="Total Loss [W]",
        yaxis_title="Power Density [MW/m³]",
    )
    fig2.write_html(str(plots_dir / "pareto_front_2d_pd_loss.html"))

    # 3-D scatter
    fig3 = go.Figure(go.Scatter3d(
        x=losses, y=effs, z=pds,
        mode="markers",
        marker=dict(size=5, color=effs, colorscale="RdYlGn"),
        text=ids_,
    ))
    fig3.update_layout(
        title="Pareto Front — 3-D Objective Space",
        scene=dict(
            xaxis_title="Total Loss [W]",
            yaxis_title="Efficiency η",
            zaxis_title="Power Density [MW/m³]",
        ),
    )
    fig3.write_html(str(plots_dir / "pareto_front_3d.html"))


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _sanitise(obj):
    """Recursively convert numpy types to Python primitives."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
