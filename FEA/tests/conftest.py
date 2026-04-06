"""Shared pytest fixtures for all test levels."""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Make sure the FEA package is importable from the test root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline.utils.mesh_utils import FEAMesh, make_annular_mesh


# ---------------------------------------------------------------------------
# Minimal stator input (6-slot, synthetic mesh)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_stator() -> StatorMeshInput:
    """6-slot, 4-pole, small stator — no real mesh file needed."""
    return StatorMeshInput(
        stator_id="test_simple",
        geometry_source="test_fixture",
        mesh_file_path="",
        mesh_format="synthetic",
        outer_diameter=0.200,
        inner_diameter=0.120,
        axial_length=0.080,
        num_slots=6,
        num_poles=4,
        slot_opening=0.005,
        tooth_width=0.012,
        yoke_height=0.015,
        slot_depth=0.025,
        winding_type="distributed",
        num_layers=2,
        conductors_per_slot=20,
        winding_factor=0.866,
        fill_factor=0.45,
        wire_diameter=0.001,
        region_tags={"stator_core": 1, "winding": 2, "air_gap": 3},
        material_map={
            "stator_core": "M250-35A",
            "winding": "copper_class_F",
            "air_gap": "air",
        },
        rated_current_rms=50.0,
        rated_speed_rpm=3000.0,
        rated_torque=50.0,
        dc_bus_voltage=400.0,
        min_element_quality=0.3,
        max_element_size=0.010,
        num_elements=1000,
        num_nodes=600,
    )


# ---------------------------------------------------------------------------
# Small synthetic FEAMesh (annular, 3 regions)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_mesh() -> FEAMesh:
    """Structured annular mesh: r=0.06–0.10, three regions."""
    return make_annular_mesh(
        r_inner=0.060,
        r_outer=0.100,
        region_radii=[
            (0.060, 0.068, 3),   # air_gap
            (0.068, 0.082, 2),   # winding
            (0.082, 0.100, 1),   # stator_core
        ],
        n_radial=4,
        n_theta=24,
    )


# ---------------------------------------------------------------------------
# Default config dicts (mirrors default.yaml sections)
# ---------------------------------------------------------------------------

@pytest.fixture
def em_config() -> dict:
    return {
        "nonlinear": {"enabled": True, "max_iterations": 5, "tolerance": 1e-4},
        "loss_model": "steinmetz",
        "materials": {"temperature_dependent": False},
    }


@pytest.fixture
def thermal_config() -> dict:
    return {
        "analysis_type": "steady_state",
        "cooling": {
            "type": "water_jacket",
            "coolant_temperature_K": 313.15,
            "h_outer": 500.0,
            "h_endturn": 80.0,
            "contact_resistance": 1e-4,
        },
        "insulation": {"class": "F", "max_temperature_K": 428.15},
        "anisotropy": {"k_in_plane": 28.0, "k_through_plane": 1.0},
        "convergence_tolerance": 1e-8,
    }


@pytest.fixture
def structural_config() -> dict:
    return {
        "reference_temperature_K": 293.15,
        "electromagnetic_loads": True,
        "thermal_loads": True,
        "materials": {
            "stator_core": {
                "youngs_modulus_Pa": 2.0e11,
                "poisson_ratio": 0.28,
                "density_kg_m3": 7650,
                "thermal_expansion_1_K": 12.0e-6,
                "yield_strength_Pa": 3.5e8,
                "ultimate_strength_Pa": 5.0e8,
                "fatigue_limit_Pa": 2.0e8,
            },
            "winding_equivalent": {
                "youngs_modulus_Pa": 3.0e9,
                "poisson_ratio": 0.35,
                "density_kg_m3": 3500,
                "thermal_expansion_1_K": 18.0e-6,
            },
        },
        "fatigue": {
            "method": "goodman",
            "stress_concentration_factor": 1.5,
            "surface_finish_factor": 0.85,
            "reliability_factor": 0.897,
        },
        "modal": {"enabled": True, "num_modes": 4},
    }
