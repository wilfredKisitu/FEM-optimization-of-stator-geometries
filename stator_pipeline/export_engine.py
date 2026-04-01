"""export_engine.py — Pure Python export using hashlib."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import IntFlag
from typing import Optional

from .params import StatorParams
from .gmsh_backend import GmshBackend
from .mesh_generator import MeshResult


class ExportFormat(IntFlag):
    NONE = 0
    MSH = 1 << 0
    VTK = 1 << 1
    HDF5 = 1 << 2
    JSON = 1 << 3
    ALL = MSH | VTK | HDF5 | JSON


@dataclass
class ExportConfig:
    output_dir: str = "/tmp/stator_out"
    formats: ExportFormat = ExportFormat.JSON


@dataclass
class ExportResult:
    format: ExportFormat = ExportFormat.NONE
    path: str = ""
    success: bool = False
    write_time_ms: float = 0.0
    error_message: str = ""


def sha256(data: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_stem(p: StatorParams) -> str:
    """Deterministic 8-char stem from SHA-256 of the params JSON."""
    d = {
        "R_outer": p.R_outer, "R_inner": p.R_inner, "airgap_length": p.airgap_length,
        "n_slots": p.n_slots, "slot_depth": p.slot_depth,
        "slot_width_outer": p.slot_width_outer, "slot_width_inner": p.slot_width_inner,
        "slot_opening": p.slot_opening, "slot_opening_depth": p.slot_opening_depth,
        "tooth_tip_angle": p.tooth_tip_angle, "slot_shape": int(p.slot_shape),
        "coil_depth": p.coil_depth, "coil_width_outer": p.coil_width_outer,
        "coil_width_inner": p.coil_width_inner,
        "insulation_thickness": p.insulation_thickness,
        "turns_per_coil": p.turns_per_coil, "coil_pitch": p.coil_pitch,
        "wire_diameter": p.wire_diameter, "slot_fill_factor": p.slot_fill_factor,
        "winding_type": int(p.winding_type),
        "t_lam": p.t_lam, "n_lam": p.n_lam, "z_spacing": p.z_spacing,
        "insulation_coating_thickness": p.insulation_coating_thickness,
        "material": int(p.material), "material_file": p.material_file,
        "mesh_yoke": p.mesh_yoke, "mesh_slot": p.mesh_slot,
        "mesh_coil": p.mesh_coil, "mesh_ins": p.mesh_ins,
        "mesh_boundary_layers": p.mesh_boundary_layers,
        "mesh_curvature": p.mesh_curvature,
        "mesh_transition_layers": p.mesh_transition_layers,
    }
    return "stator_" + sha256(json.dumps(d, sort_keys=True))[:8]


def outputs_exist(p: StatorParams, cfg: ExportConfig) -> bool:
    stem = compute_stem(p)
    if ExportFormat.MSH in cfg.formats and not os.path.isfile(os.path.join(cfg.output_dir, stem + ".msh")):
        return False
    if ExportFormat.VTK in cfg.formats and not os.path.isfile(os.path.join(cfg.output_dir, stem + ".vtk")):
        return False
    if ExportFormat.HDF5 in cfg.formats and not os.path.isfile(os.path.join(cfg.output_dir, stem + ".h5")):
        return False
    if ExportFormat.JSON in cfg.formats and not os.path.isfile(os.path.join(cfg.output_dir, stem + "_meta.json")):
        return False
    return True


class ExportEngine:
    def __init__(self, backend: GmshBackend) -> None:
        self.backend = backend

    def write_all(self, p: StatorParams, mesh: MeshResult, cfg: ExportConfig) -> list[ExportResult]:
        stem = compute_stem(p)
        results: list[ExportResult] = []

        if ExportFormat.MSH in cfg.formats:
            results.append(self._write_msh(cfg.output_dir, stem))
        if ExportFormat.VTK in cfg.formats:
            results.append(self._write_vtk(cfg.output_dir, stem, mesh))
        if ExportFormat.HDF5 in cfg.formats:
            results.append(self._write_hdf5(cfg.output_dir, stem, mesh))
        if ExportFormat.JSON in cfg.formats:
            results.append(self._write_json(cfg.output_dir, stem, p, mesh))
        return results

    def _write_msh(self, output_dir: str, stem: str) -> ExportResult:
        path = os.path.join(output_dir, stem + ".msh")
        t0 = time.monotonic()
        self.backend.write_mesh(path)
        return ExportResult(
            format=ExportFormat.MSH, path=path, success=True,
            write_time_ms=(time.monotonic() - t0) * 1000,
        )

    def _write_vtk(self, output_dir: str, stem: str, mesh: MeshResult) -> ExportResult:
        path = os.path.join(output_dir, stem + ".vtk")
        t0 = time.monotonic()
        try:
            with open(path, "w") as f:
                f.write(
                    "# vtk DataFile Version 3.0\n"
                    f"Stator mesh {stem}\n"
                    "ASCII\n"
                    "DATASET UNSTRUCTURED_GRID\n"
                    "POINTS 0 double\n"
                    "CELLS 0 0\n"
                    "CELL_TYPES 0\n"
                )
            return ExportResult(format=ExportFormat.VTK, path=path, success=True,
                                write_time_ms=(time.monotonic() - t0) * 1000)
        except OSError as e:
            return ExportResult(format=ExportFormat.VTK, path=path, error_message=str(e))

    def _write_hdf5(self, output_dir: str, stem: str, mesh: MeshResult) -> ExportResult:
        path = os.path.join(output_dir, stem + ".h5")
        t0 = time.monotonic()
        try:
            with open(path, "w") as f:
                f.write(f"HDF5 placeholder for {stem}\nn_nodes={mesh.n_nodes}\nn_elements_2d={mesh.n_elements_2d}\n")
            return ExportResult(format=ExportFormat.HDF5, path=path, success=True,
                                write_time_ms=(time.monotonic() - t0) * 1000)
        except OSError as e:
            return ExportResult(format=ExportFormat.HDF5, path=path, error_message=str(e))

    def _write_json(self, output_dir: str, stem: str, p: StatorParams, mesh: MeshResult) -> ExportResult:
        path = os.path.join(output_dir, stem + "_meta.json")
        t0 = time.monotonic()
        try:
            meta = {
                "stem": stem,
                "params": {
                    "R_outer": p.R_outer, "R_inner": p.R_inner,
                    "n_slots": p.n_slots, "slot_depth": p.slot_depth,
                    "slot_shape": p.slot_shape.name,
                    "winding_type": p.winding_type.name,
                    "n_lam": p.n_lam, "t_lam": p.t_lam,
                    "material": p.material.name,
                },
                "_derived": {
                    "yoke_height": p.yoke_height, "tooth_width": p.tooth_width,
                    "slot_pitch": p.slot_pitch, "stack_length": p.stack_length,
                    "fill_factor": p.fill_factor,
                },
                "mesh_stats": {
                    "n_nodes": mesh.n_nodes, "n_elements_2d": mesh.n_elements_2d,
                    "n_elements_3d": mesh.n_elements_3d,
                    "min_quality": mesh.min_quality, "avg_quality": mesh.avg_quality,
                },
                "output_files": {
                    "msh": os.path.join(output_dir, stem + ".msh"),
                    "vtk": os.path.join(output_dir, stem + ".vtk"),
                    "hdf5": os.path.join(output_dir, stem + ".h5"),
                    "json": path,
                },
            }
            with open(path, "w") as f:
                json.dump(meta, f, indent=2)
            return ExportResult(format=ExportFormat.JSON, path=path, success=True,
                                write_time_ms=(time.monotonic() - t0) * 1000)
        except OSError as e:
            return ExportResult(format=ExportFormat.JSON, path=path, error_message=str(e))
