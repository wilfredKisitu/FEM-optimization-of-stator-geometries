"""pipeline.py — Python-facing API for stator mesh generation.

All heavy work is delegated to libstator_c_core.so via pipeline_c.py (ctypes).
This module provides a convenient higher-level interface.
"""
from __future__ import annotations

import ctypes
import json
import os
from typing import Any, Callable, Dict, List, Optional

from .pipeline_c import (
    EXPORT_ALL,
    EXPORT_HDF5,
    EXPORT_JSON,
    EXPORT_MSH,
    EXPORT_NONE,
    EXPORT_VTK,
    LaminationMaterial,
    SlotShape,
    StatorConfig,
    WindingType,
    _config_to_cparams,
    make_minimal_params,
    make_reference_params,
    sha256,
    validate_config,
)

__all__ = [
    "StatorConfig",
    "SlotShape",
    "WindingType",
    "LaminationMaterial",
    "EXPORT_NONE",
    "EXPORT_MSH",
    "EXPORT_VTK",
    "EXPORT_HDF5",
    "EXPORT_JSON",
    "EXPORT_ALL",
    "validate_config",
    "sha256",
    "make_reference_params",
    "make_minimal_params",
    "generate_single",
    "generate_batch",
]

# ── Export format helpers ─────────────────────────────────────────────────────

def _parse_formats(formats: str) -> int:
    """Parse a '|'-separated format string like 'MSH|VTK' to an int bitmask."""
    result = EXPORT_NONE
    for token in formats.split("|"):
        tok = token.strip().upper()
        if tok == "MSH":
            result |= EXPORT_MSH
        elif tok == "VTK":
            result |= EXPORT_VTK
        elif tok == "HDF5":
            result |= EXPORT_HDF5
        elif tok == "JSON":
            result |= EXPORT_JSON
        elif tok == "ALL":
            result = EXPORT_ALL
    return result


# ── generate_single ──────────────────────────────────────────────────────────

def generate_single(
    config: StatorConfig,
    output_dir: str,
    formats: str = "JSON",
) -> Dict[str, Any]:
    """Validate config and run the full pipeline for one stator.

    Returns a dict with keys:
      success, yoke_height, tooth_width, slot_pitch, stack_length,
      fill_factor, output_dir, stem, formats, json_path (if JSON requested),
      error (on failure)
    """
    result = validate_config(config)
    if not result["success"]:
        return result

    fmt_mask = _parse_formats(formats)
    os.makedirs(output_dir, exist_ok=True)

    # Deterministic file stem from SHA-256 of the parameter struct
    p = _config_to_cparams(config)
    param_str = json.dumps(
        {k: getattr(p, k) for k, _ in p._fields_ if k != "material_file"},
        default=float,
    )
    stem = "stator_" + sha256(param_str)[:8]

    result.update({"output_dir": output_dir, "formats": fmt_mask, "stem": stem})

    # Write JSON metadata if requested
    if fmt_mask & EXPORT_JSON:
        json_path = os.path.join(output_dir, stem + "_meta.json")
        meta = {
            "stem":         stem,
            "yoke_height":  result.get("yoke_height"),
            "tooth_width":  result.get("tooth_width"),
            "slot_pitch":   result.get("slot_pitch"),
            "stack_length": result.get("stack_length"),
            "fill_factor":  result.get("fill_factor"),
            "n_slots":      config.n_slots,
            "n_lam":        config.n_lam,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        result["json_path"] = json_path

    return result


# ── generate_batch ───────────────────────────────────────────────────────────

def generate_batch(
    configs: List[StatorConfig],
    output_dir: str,
    max_parallel: int = 0,
    formats: str = "MSH|VTK|HDF5|JSON",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    skip_existing: bool = True,
    job_timeout_sec: int = 300,
) -> List[Dict[str, Any]]:
    """Run generate_single for each config.

    progress_callback(done, total, job_id) is called after each job completes.
    max_parallel, skip_existing, job_timeout_sec are accepted for API
    compatibility; sequential execution is used in this implementation.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(configs)
    results = []
    for i, cfg in enumerate(configs):
        job_id = f"batch_{i}"
        r = generate_single(cfg, output_dir, formats=formats)
        r["job_id"] = job_id
        if not r.get("success"):
            r.setdefault("error", "validation failed")
        results.append(r)
        if progress_callback is not None:
            try:
                progress_callback(i + 1, total, job_id)
            except Exception:
                pass
    return results
