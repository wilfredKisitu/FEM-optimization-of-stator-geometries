"""Result writer — serialises pipeline outputs to JSON and optionally HDF5."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_VERSION = "0.1.0"


def write_results(
    results: "PipelineResults",  # noqa: F821
    output_dir: str,
    stator_id: str,
) -> dict[str, str]:
    """Write all pipeline results to ``output_dir/<stator_id>/``.

    Parameters
    ----------
    results:
        PipelineResults returned by the orchestrator.
    output_dir:
        Root output directory.
    stator_id:
        Unique identifier used as sub-directory name.

    Returns
    -------
    dict mapping result-type → written file path.
    """
    base = Path(output_dir) / stator_id
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    written: dict[str, str] = {}

    # metadata
    meta_path = base / "metadata.json"
    _write_json(
        {
            "stator_id": stator_id,
            "solver_version": _VERSION,
            "timestamp_utc": ts,
        },
        meta_path,
    )
    written["metadata"] = str(meta_path)

    # electromagnetic scalars
    em_dir = base / "electromagnetic"
    em_dir.mkdir(exist_ok=True)
    em_scalars = _extract_scalars(results.em_results, "electromagnetic", stator_id, ts)
    em_path = em_dir / "scalars.json"
    _write_json(em_scalars, em_path)
    written["em_scalars"] = str(em_path)

    # thermal scalars
    th_dir = base / "thermal"
    th_dir.mkdir(exist_ok=True)
    th_scalars = _extract_scalars(results.thermal_results, "thermal", stator_id, ts)
    th_path = th_dir / "scalars.json"
    _write_json(th_scalars, th_path)
    written["thermal_scalars"] = str(th_path)

    # structural scalars + natural frequencies
    st_dir = base / "structural"
    st_dir.mkdir(exist_ok=True)
    st_scalars = _extract_scalars(results.structural_results, "structural", stator_id, ts)
    st_path = st_dir / "scalars.json"
    _write_json(st_scalars, st_path)
    written["structural_scalars"] = str(st_path)

    nat_freq = results.structural_results.get("natural_frequencies_Hz")
    if nat_freq is not None:
        nf_path = st_dir / "natural_frequencies.json"
        _write_json({"natural_frequencies_Hz": _to_python(nat_freq)}, nf_path)
        written["natural_frequencies"] = str(nf_path)

    # coupled metrics
    cm_path = base / "coupled_metrics.json"
    _write_json(_to_python(results.coupled_metrics), cm_path)
    written["coupled_metrics"] = str(cm_path)

    log.info("Results written to %s", base)
    return written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_scalars(stage_dict: dict, stage: str, sid: str, ts: str) -> dict:
    """Strip non-serialisable fields (numpy arrays, FEAMesh objects)."""
    scalars: dict[str, Any] = {}
    for k, v in stage_dict.items():
        if k in ("domain", "A_field", "B_field", "T_field", "u_field",
                 "von_mises_field", "principal_stress_field",
                 "loss_density_map", "copper_loss_density_map"):
            continue
        scalars[k] = _to_python(v)

    return {
        "stage": stage,
        "stator_id": sid,
        "solver_version": _VERSION,
        "timestamp_utc": ts,
        "results": scalars,
    }


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _write_json(data: Any, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
