"""fea_pipeline — multi-physics FEA pipeline for stator geometry analysis."""

from .orchestrator import run_fea_pipeline, PipelineResults
from .io.schema import StatorMeshInput
from .io.mesh_reader import load_stator_geometry
from .electromagnetic.solver import run_electromagnetic_analysis
from .thermal.solver import run_thermal_analysis
from .structural.solver import run_structural_analysis

__version__ = "0.1.0"

__all__ = [
    "run_fea_pipeline",
    "PipelineResults",
    "StatorMeshInput",
    "load_stator_geometry",
    "run_electromagnetic_analysis",
    "run_thermal_analysis",
    "run_structural_analysis",
]
