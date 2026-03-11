"""pipeline.py — Python-facing API for stator mesh generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any

try:
    import _stator_core as _core
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False


@dataclass
class StatorConfig:
    """Python dataclass mirroring StatorParams. All SI units."""
    # Section 1 — Core Radii & Air Gap
    R_outer: float = 0.25
    R_inner: float = 0.15
    airgap_length: float = 0.001

    # Section 2 — Slot Geometry
    n_slots: int = 36
    slot_depth: float = 0.06
    slot_width_outer: float = 0.012
    slot_width_inner: float = 0.010
    slot_opening: float = 0.004
    slot_opening_depth: float = 0.003
    tooth_tip_angle: float = 0.1
    slot_shape: str = "SEMI_CLOSED"

    # Section 3 — Coil / Winding
    coil_depth: float = 0.05
    coil_width_outer: float = 0.008
    coil_width_inner: float = 0.007
    insulation_thickness: float = 0.001
    turns_per_coil: int = 10
    coil_pitch: int = 5
    wire_diameter: float = 0.001
    slot_fill_factor: float = 0.45
    winding_type: str = "DOUBLE_LAYER"

    # Section 4 — Lamination Stack
    t_lam: float = 0.00035
    n_lam: int = 200
    z_spacing: float = 0.0
    insulation_coating_thickness: float = 0.00005
    material: str = "M270_35A"
    material_file: str = ""

    # Section 5 — Mesh Sizing
    mesh_yoke: float = 0.006
    mesh_slot: float = 0.003
    mesh_coil: float = 0.0015
    mesh_ins: float = 0.0007
    mesh_boundary_layers: int = 3
    mesh_curvature: float = 0.3
    mesh_transition_layers: int = 2


def _config_to_params(cfg: StatorConfig):
    """Convert a StatorConfig to a _core.StatorParams."""
    if not _CORE_AVAILABLE:
        raise ImportError("_stator_core not available; build with -DSTATOR_WITH_PYTHON=ON")
    p = _core.StatorParams()
    p.R_outer        = cfg.R_outer
    p.R_inner        = cfg.R_inner
    p.airgap_length  = cfg.airgap_length
    p.n_slots        = cfg.n_slots
    p.slot_depth     = cfg.slot_depth
    p.slot_width_outer  = cfg.slot_width_outer
    p.slot_width_inner  = cfg.slot_width_inner
    p.slot_opening      = cfg.slot_opening
    p.slot_opening_depth = cfg.slot_opening_depth
    p.tooth_tip_angle   = cfg.tooth_tip_angle
    p.slot_shape        = getattr(_core.SlotShape, cfg.slot_shape)
    p.coil_depth        = cfg.coil_depth
    p.coil_width_outer  = cfg.coil_width_outer
    p.coil_width_inner  = cfg.coil_width_inner
    p.insulation_thickness = cfg.insulation_thickness
    p.turns_per_coil    = cfg.turns_per_coil
    p.coil_pitch        = cfg.coil_pitch
    p.wire_diameter     = cfg.wire_diameter
    p.slot_fill_factor  = cfg.slot_fill_factor
    p.winding_type      = getattr(_core.WindingType, cfg.winding_type)
    p.t_lam             = cfg.t_lam
    p.n_lam             = cfg.n_lam
    p.z_spacing         = cfg.z_spacing
    p.insulation_coating_thickness = cfg.insulation_coating_thickness
    p.material          = getattr(_core.LaminationMaterial, cfg.material)
    p.material_file     = cfg.material_file
    p.mesh_yoke         = cfg.mesh_yoke
    p.mesh_slot         = cfg.mesh_slot
    p.mesh_coil         = cfg.mesh_coil
    p.mesh_ins          = cfg.mesh_ins
    p.mesh_boundary_layers   = cfg.mesh_boundary_layers
    p.mesh_curvature         = cfg.mesh_curvature
    p.mesh_transition_layers = cfg.mesh_transition_layers
    p.validate_and_derive()
    return p


def _parse_formats(formats: str):
    """Parse a '|'-separated format string like 'MSH|VTK' to ExportFormat."""
    result = _core.ExportFormat.NONE
    for token in formats.split("|"):
        token = token.strip().upper()
        if token == "MSH":  result = result | _core.ExportFormat.MSH
        elif token == "VTK": result = result | _core.ExportFormat.VTK
        elif token == "HDF5": result = result | _core.ExportFormat.HDF5
        elif token == "JSON": result = result | _core.ExportFormat.JSON
        elif token == "ALL": result = _core.ExportFormat.ALL
    return result


def generate_single(
    config: StatorConfig,
    output_dir: str,
    formats: str = "JSON|HDF5",
) -> Dict[str, Any]:
    """Generate mesh for one stator config. Returns dict with output paths."""
    params = _config_to_params(config)
    job = _core.BatchJob()
    job.params = params
    job.job_id = "single"
    job.export_config.output_dir = output_dir
    job.export_config.formats    = _parse_formats(formats)

    import tempfile, os
    status_path = os.path.join(tempfile.gettempdir(), "stator_single_status.json")
    rc = _core.BatchScheduler.execute_job(job, status_path)
    result = {"success": rc == 0, "output_dir": output_dir}
    try:
        with open(status_path) as f:
            import json
            data = json.load(f)
            result.update(data)
    except Exception:
        pass
    return result


def generate_batch(
    configs: List[StatorConfig],
    output_dir: str,
    max_parallel: int = 0,
    formats: str = "MSH|VTK|HDF5|JSON",
    progress_callback: Optional[Callable] = None,
    skip_existing: bool = True,
    job_timeout_sec: int = 300,
) -> List[Dict[str, Any]]:
    """Generate meshes for a batch. Returns list of result dicts."""
    sched = _core.BatchScheduler()
    if progress_callback is not None:
        sched.set_progress_callback(progress_callback)

    jobs = []
    for i, cfg in enumerate(configs):
        job = _core.BatchJob()
        job.params = _config_to_params(cfg)
        job.job_id = f"batch_{i}"
        job.export_config.output_dir = output_dir
        job.export_config.formats    = _parse_formats(formats)
        jobs.append(job)

    sched_cfg = _core.BatchSchedulerConfig()
    sched_cfg.max_parallel    = max_parallel
    sched_cfg.skip_existing   = skip_existing
    sched_cfg.job_timeout_sec = job_timeout_sec

    results = sched.run(jobs, sched_cfg)
    return [
        {
            "job_id":    r.job_id,
            "success":   r.success,
            "error":     r.error,
            "msh_path":  r.msh_path,
            "vtk_path":  r.vtk_path,
            "hdf5_path": r.hdf5_path,
            "json_path": r.json_path,
        }
        for r in results
    ]
