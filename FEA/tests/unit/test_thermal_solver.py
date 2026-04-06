"""Unit tests for the thermal analysis stage."""

from __future__ import annotations

import numpy as np
import pytest

from fea_pipeline.thermal.boundary_conditions import (
    apply_thermal_boundary_conditions,
    get_boundary_segment_lengths,
)
from fea_pipeline.thermal.heat_sources import map_em_losses_to_heat_sources
from fea_pipeline.thermal.postprocessor import (
    identify_hot_spots,
    compute_winding_average_temperature,
    compute_temperature_uniformity,
)
from fea_pipeline.thermal.solver import run_thermal_analysis
from fea_pipeline.io.mesh_reader import load_stator_geometry
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestThermalBoundaryConditions:
    def test_raises_on_missing_cooling_key(self, simple_mesh, simple_stator):
        n = simple_mesh.n_nodes
        K = sp.eye(n, format="lil")
        F = np.zeros(n)
        with pytest.raises(KeyError, match="cooling"):
            apply_thermal_boundary_conditions(simple_mesh, K, F, simple_stator, config={})

    def test_water_jacket_modifies_diagonal(self, simple_mesh, simple_stator, thermal_config):
        n = simple_mesh.n_nodes
        K = sp.eye(n, format="lil") * 1.0
        F = np.zeros(n)
        K2, F2 = apply_thermal_boundary_conditions(
            simple_mesh, K, F, simple_stator, thermal_config
        )
        # Outer nodes must have elevated diagonal (Robin BC adds h*L_i)
        outer_nodes = simple_mesh.boundary_node_sets["outer"]
        K2_arr = K2.toarray()
        for node in outer_nodes:
            assert K2_arr[node, node] > 1.0  # original diagonal was 1

    def test_boundary_segment_lengths_sum_to_perimeter(self, simple_mesh):
        lengths = get_boundary_segment_lengths(simple_mesh, "outer")
        outer_nodes = simple_mesh.boundary_node_sets["outer"]
        r_outer = np.linalg.norm(simple_mesh.nodes[outer_nodes], axis=1).mean()
        expected_perimeter = 2 * np.pi * r_outer
        total = float(lengths.sum())
        assert abs(total - expected_perimeter) / expected_perimeter < 0.05

    def test_fixed_temperature_bc_sets_dirichlet(self, simple_mesh, simple_stator):
        cfg = {
            "cooling": {
                "type": "fixed_temperature",
                "coolant_temperature_K": 313.15,
            },
            "insulation": {"max_temperature_K": 428.15},
        }
        n = simple_mesh.n_nodes
        K = sp.eye(n, format="lil") * 2.0
        F = np.zeros(n)
        K2, F2 = apply_thermal_boundary_conditions(
            simple_mesh, K, F, simple_stator, cfg
        )
        outer_nodes = simple_mesh.boundary_node_sets["outer"]
        K2_arr = K2.toarray()
        for node in outer_nodes:
            assert K2_arr[node, node] == pytest.approx(1.0)
            assert F2[node] == pytest.approx(313.15)


# ---------------------------------------------------------------------------
# Heat sources
# ---------------------------------------------------------------------------

class TestHeatSources:
    def _make_em_results(self, mesh):
        return {
            "loss_density_map": np.ones(mesh.n_elements) * 5000.0,
            "copper_loss_density_map": np.ones(mesh.n_elements) * 2000.0,
            "total_loss_W": 100.0,
            "domain": mesh,
        }

    def test_returns_correct_length(self, simple_mesh, simple_stator):
        em = self._make_em_results(simple_mesh)
        q = map_em_losses_to_heat_sources(
            simple_mesh, em, simple_stator, simple_stator.axial_length
        )
        assert len(q) == simple_mesh.n_elements

    def test_air_gap_has_zero_heat(self, simple_mesh, simple_stator):
        em = self._make_em_results(simple_mesh)
        q = map_em_losses_to_heat_sources(
            simple_mesh, em, simple_stator, simple_stator.axial_length
        )
        ag_id = simple_stator.region_tags["air_gap"]
        ag_mask = simple_mesh.region_ids == ag_id
        assert np.all(q[ag_mask] == 0.0)

    def test_winding_has_positive_heat(self, simple_mesh, simple_stator):
        em = self._make_em_results(simple_mesh)
        q = map_em_losses_to_heat_sources(
            simple_mesh, em, simple_stator, simple_stator.axial_length
        )
        wnd_id = simple_stator.region_tags["winding"]
        wnd_mask = simple_mesh.region_ids == wnd_id
        if wnd_mask.any():
            assert np.any(q[wnd_mask] > 0.0)

    def test_scalar_copper_density_handled(self, simple_mesh, simple_stator):
        em = {
            "loss_density_map": np.ones(simple_mesh.n_elements) * 3000.0,
            "copper_loss_density_map": 1500.0,  # scalar
            "total_loss_W": 50.0,
            "domain": simple_mesh,
        }
        q = map_em_losses_to_heat_sources(
            simple_mesh, em, simple_stator, simple_stator.axial_length
        )
        assert len(q) == simple_mesh.n_elements

    def test_dict_copper_density_handled(self, simple_mesh, simple_stator):
        em = {
            "loss_density_map": np.ones(simple_mesh.n_elements) * 3000.0,
            "copper_loss_density_map": {"spatial_W_per_m3": 1500.0},
            "total_loss_W": 50.0,
            "domain": simple_mesh,
        }
        q = map_em_losses_to_heat_sources(
            simple_mesh, em, simple_stator, simple_stator.axial_length
        )
        assert len(q) == simple_mesh.n_elements


# ---------------------------------------------------------------------------
# Postprocessor
# ---------------------------------------------------------------------------

class TestThermalPostprocessor:
    def test_identify_hot_spots_returns_dict(self):
        T = np.linspace(300, 450, 100)
        result = identify_hot_spots(T, threshold_fraction=0.95)
        assert "peak_T_K" in result
        assert "n_hotspot_nodes" in result
        assert "threshold_T_K" in result

    def test_peak_temperature_correct(self):
        T = np.array([300.0, 350.0, 400.0, 450.0])
        result = identify_hot_spots(T)
        assert result["peak_T_K"] == pytest.approx(450.0)

    def test_hotspot_count_reasonable(self):
        T = np.array([300.0, 300.0, 300.0, 450.0])
        result = identify_hot_spots(T, threshold_fraction=0.95)
        assert result["n_hotspot_nodes"] >= 1

    def test_winding_average_temperature(self, simple_mesh):
        T = np.full(simple_mesh.n_nodes, 350.0)
        avg = compute_winding_average_temperature(T, simple_mesh, winding_region_id=2)
        assert avg == pytest.approx(350.0, rel=1e-3)

    def test_temperature_uniformity_constant_field(self, simple_mesh):
        T = np.full(simple_mesh.n_nodes, 400.0)
        std = compute_temperature_uniformity(T, simple_mesh, region_id=1)
        assert std == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Full thermal solver
# ---------------------------------------------------------------------------

class TestThermalSolver:
    def _make_em_results(self, mesh):
        return {
            "loss_density_map": np.ones(mesh.n_elements) * 1e4,
            "copper_loss_density_map": np.ones(mesh.n_elements) * 5e3,
            "total_loss_W": 500.0,
            "domain": mesh,
            "B_field": {"B_x": np.zeros(mesh.n_elements),
                        "B_y": np.zeros(mesh.n_elements),
                        "B_mag": np.zeros(mesh.n_elements)},
        }

    def test_solver_runs(self, simple_stator, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        result = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        assert "peak_temperature_K" in result

    def test_peak_above_coolant(self, simple_stator, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        result = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        coolant_T = thermal_config["cooling"]["coolant_temperature_K"]
        assert result["peak_temperature_K"] > coolant_T

    def test_t_field_length(self, simple_stator, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        result = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        assert len(result["T_field"]) == mesh.n_nodes

    def test_no_negative_temperatures(self, simple_stator, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        result = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        assert np.all(result["T_field"] > 0.0)

    def test_thermal_margin_key_present(self, simple_stator, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        result = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        assert "thermal_margin_K" in result

    def test_zero_losses_returns_ambient(self, simple_stator, thermal_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = {
            "loss_density_map": np.zeros(mesh.n_elements),
            "copper_loss_density_map": 0.0,
            "total_loss_W": 0.0,
            "domain": mesh,
        }
        result = run_thermal_analysis(mesh, regions, simple_stator, em, thermal_config)
        coolant_T = thermal_config["cooling"]["coolant_temperature_K"]
        # With zero heat, temperature should be near coolant temperature
        assert result["peak_temperature_K"] <= coolant_T + 5.0
