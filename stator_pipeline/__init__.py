"""stator_pipeline — Pure-Python FEM stator mesh generation pipeline."""
from .params import (
    StatorParams, SlotShape, WindingType, LaminationMaterial,
    validate_and_derive, make_reference_params, make_minimal_params,
)
from .export_engine import ExportFormat, sha256
from .pipeline import (
    StatorConfig,
    EXPORT_NONE, EXPORT_MSH, EXPORT_VTK, EXPORT_HDF5, EXPORT_JSON, EXPORT_ALL,
    validate_config, generate_single, generate_batch,
)
from .visualiser import StatorVisualiser

__all__ = [
    "StatorParams", "StatorConfig", "SlotShape", "WindingType", "LaminationMaterial",
    "ExportFormat",
    "EXPORT_NONE", "EXPORT_MSH", "EXPORT_VTK", "EXPORT_HDF5", "EXPORT_JSON", "EXPORT_ALL",
    "validate_and_derive", "validate_config", "sha256",
    "make_reference_params", "make_minimal_params",
    "generate_single", "generate_batch",
    "StatorVisualiser",
]
