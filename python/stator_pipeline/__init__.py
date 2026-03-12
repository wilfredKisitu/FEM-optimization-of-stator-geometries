"""stator_pipeline — Python interface to the stator mesh construction pipeline.

Backed by libstator_c_core.so via ctypes (no C++ extension required).
"""
from .pipeline import (
    StatorConfig,
    SlotShape,
    WindingType,
    LaminationMaterial,
    EXPORT_NONE,
    EXPORT_MSH,
    EXPORT_VTK,
    EXPORT_HDF5,
    EXPORT_JSON,
    EXPORT_ALL,
    validate_config,
    sha256,
    make_reference_params,
    make_minimal_params,
    generate_single,
    generate_batch,
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
