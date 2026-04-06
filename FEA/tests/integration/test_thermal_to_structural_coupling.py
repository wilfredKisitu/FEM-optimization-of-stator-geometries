"""Integration tests — Thermal → Structural coupling."""

from __future__ import annotations

import numpy as np
import pytest

from fea_pipeline.io.mesh_reader import load_stator_geometry
from fea_pipeline.electromagnetic.solver import run_electromagnetic_analysis
from fea_pipeline.thermal.solver import run_thermal_analysis
from fea_pipeline.structural.solver import run_structural_analysis


class TestThermalStructuralCoupling:
    def _run_all(self, simple_stator, em_config, thermal_config, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        th = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        st = run_structural_analysis(
            mesh, regions, simple_stator, em, th, structural_config
        )
        return mesh, em, th, st

    def test_higher_temperature_increases_displacement(
        self, simple_stator, em_config, thermal_config, structural_config
    ):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)

        th_ambient = {
            "T_field": np.full(mesh.n_nodes, 293.15),
            "domain": mesh,
        }
        th_hot = {
            "T_field": np.full(mesh.n_nodes, 450.0),
            "domain": mesh,
        }

        st_cold = run_structural_analysis(
            mesh, regions, simple_stator, em, th_ambient, structural_config
        )
        st_hot = run_structural_analysis(
            mesh, regions, simple_stator, em, th_hot, structural_config
        )

        assert st_hot["max_displacement_m"] >= st_cold["max_displacement_m"]

    def test_safety_factor_finite(
        self, simple_stator, em_config, thermal_config, structural_config
    ):
        _, _, _, st = self._run_all(
            simple_stator, em_config, thermal_config, structural_config
        )
        assert np.isfinite(st["safety_factor"])

    def test_von_mises_field_all_finite(
        self, simple_stator, em_config, thermal_config, structural_config
    ):
        _, _, _, st = self._run_all(
            simple_stator, em_config, thermal_config, structural_config
        )
        assert np.all(np.isfinite(st["von_mises_field"]))

    def test_principal_stresses_shape(
        self, simple_stator, em_config, thermal_config, structural_config
    ):
        mesh, _, _, st = self._run_all(
            simple_stator, em_config, thermal_config, structural_config
        )
        assert st["principal_stress_field"].shape == (mesh.n_elements, 2)

    def test_fatigue_life_positive(
        self, simple_stator, em_config, thermal_config, structural_config
    ):
        _, _, _, st = self._run_all(
            simple_stator, em_config, thermal_config, structural_config
        )
        assert st["fatigue_life_cycles"] > 0.0
