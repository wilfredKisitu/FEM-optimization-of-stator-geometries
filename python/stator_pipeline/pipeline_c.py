"""pipeline_c.py — ctypes-based Python API wrapping libstator_c_core.so"""
from __future__ import annotations
import ctypes
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any

# ── Locate shared library ─────────────────────────────────────────────────────
def _find_lib() -> Optional[str]:
    candidates = [
        pathlib.Path(__file__).parent.parent.parent / "build" / "libstator_c_core.so",
        pathlib.Path(__file__).parent.parent.parent / "build" / "libstator_c_core.dylib",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Try LD_LIBRARY_PATH / system
    try:
        return ctypes.util.find_library("stator_c_core")
    except Exception:
        return None

_lib_path = _find_lib()
_lib: Optional[ctypes.CDLL] = None
if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError:
        _lib = None

# ── Enum constants ────────────────────────────────────────────────────────────
class SlotShape:
    RECTANGULAR  = 0
    TRAPEZOIDAL  = 1
    ROUND_BOTTOM = 2
    SEMI_CLOSED  = 3

class WindingType:
    SINGLE_LAYER = 0
    DOUBLE_LAYER = 1
    CONCENTRATED = 2
    DISTRIBUTED  = 3

class LaminationMaterial:
    M270_35A = 0
    M330_50A = 1
    M400_50A = 2
    NO20     = 3
    CUSTOM   = 4

EXPORT_NONE = 0
EXPORT_MSH  = 1 << 0
EXPORT_VTK  = 1 << 1
EXPORT_HDF5 = 1 << 2
EXPORT_JSON = 1 << 3
EXPORT_ALL  = EXPORT_MSH | EXPORT_VTK | EXPORT_HDF5 | EXPORT_JSON

STATOR_OK = 0

# ── ctypes structs ────────────────────────────────────────────────────────────

class _StatorParams(ctypes.Structure):
    _fields_ = [
        # Section 1
        ("R_outer",       ctypes.c_double),
        ("R_inner",       ctypes.c_double),
        ("airgap_length", ctypes.c_double),
        # Section 2
        ("n_slots",            ctypes.c_int),
        ("slot_depth",         ctypes.c_double),
        ("slot_width_outer",   ctypes.c_double),
        ("slot_width_inner",   ctypes.c_double),
        ("slot_opening",       ctypes.c_double),
        ("slot_opening_depth", ctypes.c_double),
        ("tooth_tip_angle",    ctypes.c_double),
        ("slot_shape",         ctypes.c_int),
        # Section 3
        ("coil_depth",                 ctypes.c_double),
        ("coil_width_outer",           ctypes.c_double),
        ("coil_width_inner",           ctypes.c_double),
        ("insulation_thickness",       ctypes.c_double),
        ("turns_per_coil",             ctypes.c_int),
        ("coil_pitch",                 ctypes.c_int),
        ("wire_diameter",              ctypes.c_double),
        ("slot_fill_factor",           ctypes.c_double),
        ("winding_type",               ctypes.c_int),
        # Section 4
        ("t_lam",                        ctypes.c_double),
        ("n_lam",                        ctypes.c_int),
        ("z_spacing",                    ctypes.c_double),
        ("insulation_coating_thickness", ctypes.c_double),
        ("material",                     ctypes.c_int),
        ("material_file",                ctypes.c_char * 256),
        # Section 5
        ("mesh_yoke",              ctypes.c_double),
        ("mesh_slot",              ctypes.c_double),
        ("mesh_coil",              ctypes.c_double),
        ("mesh_ins",               ctypes.c_double),
        ("mesh_boundary_layers",   ctypes.c_int),
        ("mesh_curvature",         ctypes.c_double),
        ("mesh_transition_layers", ctypes.c_int),
        # Section 6 — derived
        ("yoke_height",  ctypes.c_double),
        ("tooth_width",  ctypes.c_double),
        ("slot_pitch",   ctypes.c_double),
        ("stack_length", ctypes.c_double),
        ("fill_factor",  ctypes.c_double),
    ]


# ── StatorConfig dataclass (user-facing) ─────────────────────────────────────

@dataclass
class StatorConfig:
    """Python dataclass mirroring StatorParams. All SI units."""
    R_outer: float = 0.25
    R_inner: float = 0.15
    airgap_length: float = 0.001
    n_slots: int = 36
    slot_depth: float = 0.06
    slot_width_outer: float = 0.012
    slot_width_inner: float = 0.010
    slot_opening: float = 0.004
    slot_opening_depth: float = 0.003
    tooth_tip_angle: float = 0.1
    slot_shape: int = SlotShape.SEMI_CLOSED
    coil_depth: float = 0.05
    coil_width_outer: float = 0.008
    coil_width_inner: float = 0.007
    insulation_thickness: float = 0.001
    turns_per_coil: int = 10
    coil_pitch: int = 5
    wire_diameter: float = 0.001
    slot_fill_factor: float = 0.45
    winding_type: int = WindingType.DOUBLE_LAYER
    t_lam: float = 0.00035
    n_lam: int = 200
    z_spacing: float = 0.0
    insulation_coating_thickness: float = 0.00005
    material: int = LaminationMaterial.M270_35A
    material_file: str = ""
    mesh_yoke: float = 0.006
    mesh_slot: float = 0.003
    mesh_coil: float = 0.0015
    mesh_ins: float = 0.0007
    mesh_boundary_layers: int = 3
    mesh_curvature: float = 0.3
    mesh_transition_layers: int = 2


def _config_to_cparams(cfg: StatorConfig) -> _StatorParams:
    p = _StatorParams()
    p.R_outer       = cfg.R_outer
    p.R_inner       = cfg.R_inner
    p.airgap_length = cfg.airgap_length
    p.n_slots       = cfg.n_slots
    p.slot_depth    = cfg.slot_depth
    p.slot_width_outer   = cfg.slot_width_outer
    p.slot_width_inner   = cfg.slot_width_inner
    p.slot_opening       = cfg.slot_opening
    p.slot_opening_depth = cfg.slot_opening_depth
    p.tooth_tip_angle    = cfg.tooth_tip_angle
    p.slot_shape         = cfg.slot_shape
    p.coil_depth         = cfg.coil_depth
    p.coil_width_outer   = cfg.coil_width_outer
    p.coil_width_inner   = cfg.coil_width_inner
    p.insulation_thickness = cfg.insulation_thickness
    p.turns_per_coil     = cfg.turns_per_coil
    p.coil_pitch         = cfg.coil_pitch
    p.wire_diameter      = cfg.wire_diameter
    p.slot_fill_factor   = cfg.slot_fill_factor
    p.winding_type       = cfg.winding_type
    p.t_lam              = cfg.t_lam
    p.n_lam              = cfg.n_lam
    p.z_spacing          = cfg.z_spacing
    p.insulation_coating_thickness = cfg.insulation_coating_thickness
    p.material           = cfg.material
    p.material_file      = cfg.material_file.encode()[:255]
    p.mesh_yoke          = cfg.mesh_yoke
    p.mesh_slot          = cfg.mesh_slot
    p.mesh_coil          = cfg.mesh_coil
    p.mesh_ins           = cfg.mesh_ins
    p.mesh_boundary_layers   = cfg.mesh_boundary_layers
    p.mesh_curvature         = cfg.mesh_curvature
    p.mesh_transition_layers = cfg.mesh_transition_layers
    return p


def validate_config(cfg: StatorConfig) -> Dict[str, Any]:
    """Validate a StatorConfig via the C library. Returns dict with success/error."""
    if _lib is None:
        return {"success": False, "error": "libstator_c_core not loaded"}
    p = _config_to_cparams(cfg)
    err_buf = ctypes.create_string_buffer(512)
    _lib.stator_params_validate_and_derive.restype  = ctypes.c_int
    _lib.stator_params_validate_and_derive.argtypes = [
        ctypes.POINTER(_StatorParams), ctypes.c_char_p, ctypes.c_size_t
    ]
    rc = _lib.stator_params_validate_and_derive(ctypes.byref(p), err_buf, 512)
    if rc == STATOR_OK:
        return {
            "success":      True,
            "yoke_height":  p.yoke_height,
            "tooth_width":  p.tooth_width,
            "slot_pitch":   p.slot_pitch,
            "stack_length": p.stack_length,
            "fill_factor":  p.fill_factor,
        }
    else:
        return {"success": False, "error": err_buf.value.decode()}


def sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string via the C library."""
    if _lib is None:
        raise ImportError("libstator_c_core not loaded")
    _lib.stator_sha256.restype  = None
    _lib.stator_sha256.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    out = ctypes.create_string_buffer(65)
    encoded = data.encode()
    _lib.stator_sha256(encoded, len(encoded), out)
    return out.value.decode()


def make_reference_params() -> StatorConfig:
    """Return a validated reference 36-slot StatorConfig."""
    return StatorConfig(
        R_outer=0.25, R_inner=0.15, airgap_length=0.001,
        n_slots=36, slot_depth=0.06,
        slot_width_outer=0.012, slot_width_inner=0.010,
        slot_opening=0.004, slot_opening_depth=0.003,
        tooth_tip_angle=0.1, slot_shape=SlotShape.SEMI_CLOSED,
        coil_depth=0.050, coil_width_outer=0.008, coil_width_inner=0.007,
        insulation_thickness=0.001, turns_per_coil=10, coil_pitch=5,
        wire_diameter=0.001, slot_fill_factor=0.45,
        winding_type=WindingType.DOUBLE_LAYER,
        t_lam=0.00035, n_lam=200, z_spacing=0.0,
        insulation_coating_thickness=0.00005,
        material=LaminationMaterial.M270_35A, material_file="",
        mesh_yoke=0.006, mesh_slot=0.003, mesh_coil=0.0015, mesh_ins=0.0007,
        mesh_boundary_layers=3, mesh_curvature=0.3, mesh_transition_layers=2,
    )


def make_minimal_params() -> StatorConfig:
    """Return a validated minimal 12-slot StatorConfig."""
    return StatorConfig(
        R_outer=0.12, R_inner=0.07, airgap_length=0.001,
        n_slots=12, slot_depth=0.03,
        slot_width_outer=0.010, slot_width_inner=0.009,
        slot_opening=0.003, slot_opening_depth=0.002,
        tooth_tip_angle=0.0, slot_shape=SlotShape.RECTANGULAR,
        coil_depth=0.025, coil_width_outer=0.007, coil_width_inner=0.006,
        insulation_thickness=0.001, turns_per_coil=8, coil_pitch=3,
        wire_diameter=0.0012, slot_fill_factor=0.4,
        winding_type=WindingType.SINGLE_LAYER,
        t_lam=0.00035, n_lam=100, z_spacing=0.0,
        insulation_coating_thickness=0.00005,
        material=LaminationMaterial.M330_50A, material_file="",
        mesh_yoke=0.005, mesh_slot=0.002, mesh_coil=0.001, mesh_ins=0.0005,
        mesh_boundary_layers=2, mesh_curvature=0.3, mesh_transition_layers=2,
    )
