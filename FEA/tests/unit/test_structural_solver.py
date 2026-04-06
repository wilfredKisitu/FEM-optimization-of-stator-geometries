"""Unit tests for the structural analysis stage."""

from __future__ import annotations

import numpy as np
import pytest

from fea_pipeline.structural.postprocessor import (
    compute_von_mises,
    compute_principal_stresses,
    compute_fatigue_life,
    compute_natural_frequencies,
)
from fea_pipeline.structural.load_mapper import (
    compute_thermal_expansion_load,
    compute_maxwell_stress_load,
)
from fea_pipeline.structural.boundary_conditions import (
    apply_structural_boundary_conditions,
)
from fea_pipeline.structural.solver import run_structural_analysis
from fea_pipeline.io.mesh_reader import load_stator_geometry
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Von Mises stress
# ---------------------------------------------------------------------------

class TestVonMises:
    def _build_material_arrays(self, mesh):
        n = mesh.n_elements
        return (
            np.full(n, 2e11),   # E
            np.full(n, 0.28),   # nu
            np.full(n, 12e-6),  # alpha
        )

    def test_zero_displacement_zero_stress(self, simple_mesh):
        E, nu, alpha = self._build_material_arrays(simple_mesh)
        u = np.zeros(2 * simple_mesh.n_nodes)
        T = np.full(simple_mesh.n_nodes, 293.15)
        vm = compute_von_mises(u, simple_mesh, E, nu, alpha, T, T_ref_K=293.15)
        assert np.allclose(vm, 0.0, atol=1e-6)

    def test_von_mises_non_negative(self, simple_mesh):
        E, nu, alpha = self._build_material_arrays(simple_mesh)
        rng = np.random.default_rng(42)
        u = rng.standard_normal(2 * simple_mesh.n_nodes) * 1e-5
        T = np.full(simple_mesh.n_nodes, 293.15)
        vm = compute_von_mises(u, simple_mesh, E, nu, alpha, T, T_ref_K=293.15)
        assert np.all(vm >= 0.0)

    def test_von_mises_shape(self, simple_mesh):
        E, nu, alpha = self._build_material_arrays(simple_mesh)
        u = np.zeros(2 * simple_mesh.n_nodes)
        T = np.full(simple_mesh.n_nodes, 293.15)
        vm = compute_von_mises(u, simple_mesh, E, nu, alpha, T, T_ref_K=293.15)
        assert vm.shape == (simple_mesh.n_elements,)

    def test_thermal_mismatch_gives_nonzero_stress(self, simple_mesh):
        E, nu, alpha = self._build_material_arrays(simple_mesh)
        u = np.zeros(2 * simple_mesh.n_nodes)
        T_hot = np.full(simple_mesh.n_nodes, 350.0)  # 57 K above reference
        vm = compute_von_mises(u, simple_mesh, E, nu, alpha, T_hot, T_ref_K=293.15)
        # Thermal expansion constrained → non-zero stress
        assert np.any(vm > 0.0)


# ---------------------------------------------------------------------------
# Principal stresses
# ---------------------------------------------------------------------------

class TestPrincipalStresses:
    def test_shape(self, simple_mesh):
        E = np.full(simple_mesh.n_elements, 2e11)
        nu = np.full(simple_mesh.n_elements, 0.28)
        alpha = np.full(simple_mesh.n_elements, 12e-6)
        u = np.zeros(2 * simple_mesh.n_nodes)
        T = np.full(simple_mesh.n_nodes, 293.15)
        ps = compute_principal_stresses(u, simple_mesh, E, nu, alpha, T, 293.15)
        assert ps.shape == (simple_mesh.n_elements, 2)

    def test_principal_stress_ordering(self, simple_mesh):
        E = np.full(simple_mesh.n_elements, 2e11)
        nu = np.full(simple_mesh.n_elements, 0.28)
        alpha = np.full(simple_mesh.n_elements, 12e-6)
        rng = np.random.default_rng(7)
        u = rng.standard_normal(2 * simple_mesh.n_nodes) * 1e-5
        T = np.full(simple_mesh.n_nodes, 293.15)
        ps = compute_principal_stresses(u, simple_mesh, E, nu, alpha, T, 293.15)
        # sigma1 >= sigma2
        assert np.all(ps[:, 0] >= ps[:, 1] - 1e-6)


# ---------------------------------------------------------------------------
# Fatigue life
# ---------------------------------------------------------------------------

class TestFatigueLife:
    def _config(self):
        return {
            "materials": {
                "stator_core": {
                    "yield_strength_Pa": 3.5e8,
                    "ultimate_strength_Pa": 5.0e8,
                    "fatigue_limit_Pa": 2.0e8,
                }
            },
            "fatigue": {
                "method": "goodman",
                "stress_concentration_factor": 1.5,
                "surface_finish_factor": 0.85,
                "reliability_factor": 0.897,
            },
        }

    def test_zero_stress_gives_infinite_life(self):
        vm = np.zeros(100)
        life = compute_fatigue_life(vm, self._config())
        assert life >= 1e10

    def test_high_stress_gives_finite_life(self):
        vm = np.full(100, 4.0e8)   # above fatigue limit
        life = compute_fatigue_life(vm, self._config())
        assert life < 1e12

    def test_life_decreases_with_stress(self):
        vm_low  = np.full(100, 1.0e8)
        vm_high = np.full(100, 3.5e8)
        life_low  = compute_fatigue_life(vm_low,  self._config())
        life_high = compute_fatigue_life(vm_high, self._config())
        assert life_low > life_high


# ---------------------------------------------------------------------------
# Natural frequencies
# ---------------------------------------------------------------------------

class TestNaturalFrequencies:
    def test_returns_array(self, simple_mesh, structural_config):
        E = np.full(simple_mesh.n_elements, 2e11)
        nu = np.full(simple_mesh.n_elements, 0.28)
        rho = np.full(simple_mesh.n_elements, 7650.0)
        freqs = compute_natural_frequencies(simple_mesh, E, nu, rho, structural_config)
        assert isinstance(freqs, np.ndarray)

    def test_frequencies_are_positive(self, simple_mesh, structural_config):
        E = np.full(simple_mesh.n_elements, 2e11)
        nu = np.full(simple_mesh.n_elements, 0.28)
        rho = np.full(simple_mesh.n_elements, 7650.0)
        freqs = compute_natural_frequencies(simple_mesh, E, nu, rho, structural_config)
        assert np.all(freqs >= 0.0)

    def test_frequencies_are_sorted(self, simple_mesh, structural_config):
        E = np.full(simple_mesh.n_elements, 2e11)
        nu = np.full(simple_mesh.n_elements, 0.28)
        rho = np.full(simple_mesh.n_elements, 7650.0)
        freqs = compute_natural_frequencies(simple_mesh, E, nu, rho, structural_config)
        if len(freqs) > 1:
            assert np.all(np.diff(freqs) >= -1e-3)


# ---------------------------------------------------------------------------
# Load mapper
# ---------------------------------------------------------------------------

class TestLoadMapper:
    def test_thermal_load_zero_for_uniform_ref_temperature(self, simple_mesh):
        n = simple_mesh.n_elements
        E = np.full(n, 2e11)
        nu = np.full(n, 0.28)
        alpha = np.full(n, 12e-6)
        T_ref = 293.15
        T = np.full(simple_mesh.n_nodes, T_ref)  # no temperature rise
        F = compute_thermal_expansion_load(simple_mesh, T, E, nu, alpha, T_ref)
        assert np.allclose(F, 0.0, atol=1e-6)

    def test_thermal_load_length(self, simple_mesh):
        n = simple_mesh.n_elements
        E = np.full(n, 2e11)
        nu = np.full(n, 0.28)
        alpha = np.full(n, 12e-6)
        T = np.full(simple_mesh.n_nodes, 350.0)
        F = compute_thermal_expansion_load(simple_mesh, T, E, nu, alpha, 293.15)
        assert len(F) == 2 * simple_mesh.n_nodes

    def test_maxwell_stress_load_length(self, simple_mesh):
        n = simple_mesh.n_elements
        B_field = {
            "B_x":   np.random.default_rng(1).standard_normal(n) * 0.5,
            "B_y":   np.random.default_rng(2).standard_normal(n) * 0.5,
            "B_mag": np.ones(n) * 0.7,
        }
        F = compute_maxwell_stress_load(simple_mesh, B_field, {})
        assert len(F) == 2 * simple_mesh.n_nodes

    def test_maxwell_stress_disabled_gives_zero(self, simple_mesh):
        B_field = {
            "B_x": np.ones(simple_mesh.n_elements),
            "B_y": np.ones(simple_mesh.n_elements),
            "B_mag": np.ones(simple_mesh.n_elements),
        }
        F = compute_maxwell_stress_load(
            simple_mesh, B_field, {"electromagnetic_loads": False}
        )
        assert np.allclose(F, 0.0)


# ---------------------------------------------------------------------------
# Full structural solver
# ---------------------------------------------------------------------------

class TestStructuralSolver:
    def _make_em_results(self, mesh):
        return {
            "B_field": {
                "B_x":   np.zeros(mesh.n_elements),
                "B_y":   np.zeros(mesh.n_elements),
                "B_mag": np.zeros(mesh.n_elements),
            },
            "domain": mesh,
        }

    def _make_thermal_results(self, mesh):
        return {
            "T_field": np.full(mesh.n_nodes, 320.0),
            "domain": mesh,
        }

    def test_solver_runs(self, simple_stator, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        th = self._make_thermal_results(mesh)
        result = run_structural_analysis(mesh, regions, simple_stator,
                                         em, th, structural_config)
        assert "max_von_mises_Pa" in result

    def test_safety_factor_positive(self, simple_stator, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        th = self._make_thermal_results(mesh)
        result = run_structural_analysis(mesh, regions, simple_stator,
                                         em, th, structural_config)
        assert result["safety_factor"] > 0.0

    def test_displacement_non_negative(self, simple_stator, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        th = self._make_thermal_results(mesh)
        result = run_structural_analysis(mesh, regions, simple_stator,
                                         em, th, structural_config)
        assert result["max_displacement_m"] >= 0.0

    def test_u_field_length(self, simple_stator, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        th = self._make_thermal_results(mesh)
        result = run_structural_analysis(mesh, regions, simple_stator,
                                         em, th, structural_config)
        assert len(result["u_field"]) == 2 * mesh.n_nodes

    def test_von_mises_field_length(self, simple_stator, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        th = self._make_thermal_results(mesh)
        result = run_structural_analysis(mesh, regions, simple_stator,
                                         em, th, structural_config)
        assert len(result["von_mises_field"]) == mesh.n_elements

    def test_natural_frequencies_present(self, simple_stator, structural_config):
        mesh, regions = load_stator_geometry(simple_stator)
        em = self._make_em_results(mesh)
        th = self._make_thermal_results(mesh)
        result = run_structural_analysis(mesh, regions, simple_stator,
                                         em, th, structural_config)
        assert "natural_frequencies_Hz" in result
