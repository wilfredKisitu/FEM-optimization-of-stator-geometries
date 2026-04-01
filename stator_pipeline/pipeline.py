"""pipeline.py — High-level Python API for stator mesh generation."""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from .params import (
    StatorParams, SlotShape, WindingType, LaminationMaterial,
    validate_and_derive, make_reference_params, make_minimal_params,
)
from .export_engine import (
    ExportFormat, ExportConfig, ExportEngine, sha256, compute_stem,
)
from .gmsh_backend import make_default_backend
from .geometry_builder import GeometryBuilder
from .topology_registry import TopologyRegistry
from .mesh_generator import MeshGenerator, MeshConfig

# Re-export legacy integer constants for backwards compatibility
EXPORT_NONE = int(ExportFormat.NONE)
EXPORT_MSH = int(ExportFormat.MSH)
EXPORT_VTK = int(ExportFormat.VTK)
EXPORT_HDF5 = int(ExportFormat.HDF5)
EXPORT_JSON = int(ExportFormat.JSON)
EXPORT_ALL = int(ExportFormat.ALL)

# Legacy alias
StatorConfig = StatorParams

__all__ = [
    "StatorParams", "StatorConfig", "SlotShape", "WindingType", "LaminationMaterial",
    "EXPORT_NONE", "EXPORT_MSH", "EXPORT_VTK", "EXPORT_HDF5", "EXPORT_JSON", "EXPORT_ALL",
    "ExportFormat", "validate_config", "sha256",
    "make_reference_params", "make_minimal_params",
    "generate_single", "generate_batch",
]


def _parse_formats(formats: str | int) -> ExportFormat:
    if isinstance(formats, int):
        return ExportFormat(formats)
    result = ExportFormat.NONE
    for token in formats.split("|"):
        tok = token.strip().upper()
        if tok == "MSH":
            result |= ExportFormat.MSH
        elif tok == "VTK":
            result |= ExportFormat.VTK
        elif tok == "HDF5":
            result |= ExportFormat.HDF5
        elif tok == "JSON":
            result |= ExportFormat.JSON
        elif tok == "ALL":
            result = ExportFormat.ALL
    return result


def validate_config(cfg: StatorParams) -> dict[str, Any]:
    """Validate and derive StatorParams. Returns dict with success/error."""
    try:
        validated = validate_and_derive(cfg)
        return {
            "success":      True,
            "yoke_height":  validated.yoke_height,
            "tooth_width":  validated.tooth_width,
            "slot_pitch":   validated.slot_pitch,
            "stack_length": validated.stack_length,
            "fill_factor":  validated.fill_factor,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}


def generate_single(
    config: StatorParams,
    output_dir: str,
    formats: str | int = "JSON",
) -> dict[str, Any]:
    """Validate config and run the full pipeline for one stator."""
    result = validate_config(config)
    if not result["success"]:
        return result

    fmt = _parse_formats(formats)
    os.makedirs(output_dir, exist_ok=True)

    validated = validate_and_derive(config)
    stem = compute_stem(validated)
    result.update({"output_dir": output_dir, "formats": int(fmt), "stem": stem})

    if ExportFormat.JSON in fmt:
        json_path = os.path.join(output_dir, stem + "_meta.json")
        meta = {
            "stem":         stem,
            "yoke_height":  result["yoke_height"],
            "tooth_width":  result["tooth_width"],
            "slot_pitch":   result["slot_pitch"],
            "stack_length": result["stack_length"],
            "fill_factor":  result["fill_factor"],
            "n_slots":      config.n_slots,
            "n_lam":        config.n_lam,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        result["json_path"] = json_path

    if fmt & ~ExportFormat.JSON:
        backend = make_default_backend()
        backend.initialize(f"stator_{stem}")
        builder = GeometryBuilder(backend)
        geo = builder.build(validated)
        if geo.success:
            registry = TopologyRegistry(validated.n_slots)
            generator = MeshGenerator(backend)
            mesh = generator.generate(validated, geo, registry)
            if mesh.success:
                cfg_exp = ExportConfig(output_dir=output_dir, formats=fmt & ~ExportFormat.JSON)
                engine = ExportEngine(backend)
                engine.write_all(validated, mesh, cfg_exp)
        backend.finalize()

    return result


def generate_batch(
    configs: list[StatorParams],
    output_dir: str,
    max_parallel: int = 0,
    formats: str | int = "MSH|VTK|HDF5|JSON",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    skip_existing: bool = True,
    job_timeout_sec: int = 300,
) -> list[dict[str, Any]]:
    """Run generate_single for each config."""
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
