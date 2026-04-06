"""Integration tests — EM → Thermal coupling."""

from __future__ import annotations

import numpy as np
import pytest

from fea_pipeline.io.mesh_reader import load_stator_geometry
from fea_pipeline.electromagnetic.solver import run_electromagnetic_analysis
from fea_pipeline.thermal.solver import run_thermal_analysis


class TestEMThermalCoupling:
    def test_em_losses_positive(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert em["total_loss_W"] > 0.0

    def test_peak_temperature_above_coolant(
        self, simple_stator, em_config, thermal_config
    ):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        th = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        coolant_T = thermal_config["cooling"]["coolant_temperature_K"]
        assert th["peak_temperature_K"] > coolant_T

    def test_winding_temperature_above_coolant(
        self, simple_stator, em_config, thermal_config
    ):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        th = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        coolant_T = thermal_config["cooling"]["coolant_temperature_K"]
        assert th["winding_average_temperature_K"] >= coolant_T

    def test_higher_losses_give_higher_temperature(
        self, simple_stator, em_config, thermal_config
    ):
        mesh, regions = load_stator_geometry(simple_stator)

        em_low = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        th_low = run_thermal_analysis(mesh, regions, simple_stator, em_low, thermal_config)

        # Artificial high-loss case
        em_high = dict(em_low)
        em_high["loss_density_map"] = em_low["loss_density_map"] * 5.0
        em_high["copper_loss_density_map"] = (
            em_low["copper_loss_density_map"] * 5.0
            if isinstance(em_low["copper_loss_density_map"], np.ndarray)
            else em_low["copper_loss_density_map"] * 5.0
        )
        em_high["total_loss_W"] = em_low["total_loss_W"] * 5.0

        th_high = run_thermal_analysis(mesh, regions, simple_stator, em_high, thermal_config)

        assert th_high["peak_temperature_K"] > th_low["peak_temperature_K"]

    def test_t_field_is_finite(self, simple_stator, em_config, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        th = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        assert np.all(np.isfinite(th["T_field"]))

    def test_same_mesh_used_in_both_stages(
        self, simple_stator, em_config, thermal_config
    ):
        mesh, regions = load_stator_geometry(simple_stator)
        em = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        th = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        assert em["domain"] is mesh
        assert th["domain"] is mesh
