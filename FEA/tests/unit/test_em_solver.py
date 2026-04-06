"""Unit tests for the electromagnetic stage."""

from __future__ import annotations

import numpy as np
import pytest

from fea_pipeline.electromagnetic.loss_calculator import (
    steinmetz_iron_loss,
    compute_iron_losses,
    compute_copper_losses,
)
from fea_pipeline.electromagnetic.material_library import (
    get_material_properties,
    interpolate_reluctivity,
    MATERIAL_DB,
)
from fea_pipeline.electromagnetic.postprocessor import (
    extract_flux_density,
    compute_efficiency,
)
from fea_pipeline.electromagnetic.boundary_conditions import (
    apply_dirichlet_bcs,
    get_em_boundary_nodes,
)
from fea_pipeline.electromagnetic.solver import run_electromagnetic_analysis
from fea_pipeline.io.mesh_reader import load_stator_geometry
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Steinmetz iron loss
# ---------------------------------------------------------------------------

class TestSteinmetzLoss:
    def test_zero_frequency_gives_zero(self):
        loss = steinmetz_iron_loss(B_peak=1.5, freq_Hz=0.0,
                                   kh=143.0, ke=0.53, alpha=2.0)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_zero_field_gives_zero(self):
        loss = steinmetz_iron_loss(B_peak=0.0, freq_Hz=50.0,
                                   kh=143.0, ke=0.53, alpha=2.0)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_known_value_at_50hz_1T(self):
        # P = 143*50*1.0^2 + 0.53*50²*1.0² = 7150 + 1325 = 8475 W/m³
        loss = steinmetz_iron_loss(B_peak=1.0, freq_Hz=50.0,
                                   kh=143.0, ke=0.53, alpha=2.0)
        assert loss == pytest.approx(8475.0, rel=1e-4)

    def test_b_squared_scaling(self):
        """Doubling B at alpha=2 should quadruple loss approximately."""
        l1 = steinmetz_iron_loss(0.5, 50.0, 143.0, 0.53, 2.0)
        l2 = steinmetz_iron_loss(1.0, 50.0, 143.0, 0.53, 2.0)
        assert l2 / l1 == pytest.approx(4.0, rel=0.01)

    def test_loss_increases_with_frequency(self):
        l50  = steinmetz_iron_loss(1.0, 50.0,  143.0, 0.53, 2.0)
        l100 = steinmetz_iron_loss(1.0, 100.0, 143.0, 0.53, 2.0)
        assert l100 > l50


# ---------------------------------------------------------------------------
# Iron loss integration
# ---------------------------------------------------------------------------

class TestComputeIronLosses:
    def test_returns_expected_keys(self, simple_mesh):
        n = simple_mesh.n_elements
        B_elem = np.ones(n) * 1.0
        _, _, area = simple_mesh.gradient_operators()
        result = compute_iron_losses(
            B_elem=B_elem,
            region_ids=simple_mesh.region_ids,
            areas=area,
            axial_length=0.08,
            freq_Hz=50.0,
            material_id="M250-35A",
        )
        for key in ("total", "eddy", "hysteresis", "spatial_W_per_m3"):
            assert key in result

    def test_total_is_sum_of_components(self, simple_mesh):
        _, _, area = simple_mesh.gradient_operators()
        r = compute_iron_losses(
            B_elem=np.ones(simple_mesh.n_elements) * 1.2,
            region_ids=simple_mesh.region_ids,
            areas=area,
            axial_length=0.08,
            freq_Hz=50.0,
            material_id="M250-35A",
        )
        assert r["total"] == pytest.approx(r["eddy"] + r["hysteresis"], rel=1e-6)

    def test_zero_field_gives_zero_losses(self, simple_mesh):
        _, _, area = simple_mesh.gradient_operators()
        r = compute_iron_losses(
            B_elem=np.zeros(simple_mesh.n_elements),
            region_ids=simple_mesh.region_ids,
            areas=area,
            axial_length=0.08,
            freq_Hz=50.0,
            material_id="M250-35A",
        )
        assert r["total"] == pytest.approx(0.0, abs=1e-10)

    def test_spatial_array_length(self, simple_mesh):
        _, _, area = simple_mesh.gradient_operators()
        r = compute_iron_losses(
            B_elem=np.ones(simple_mesh.n_elements) * 0.8,
            region_ids=simple_mesh.region_ids,
            areas=area,
            axial_length=0.08,
            freq_Hz=50.0,
            material_id="M250-35A",
        )
        assert len(r["spatial_W_per_m3"]) == simple_mesh.n_elements


# ---------------------------------------------------------------------------
# Copper losses
# ---------------------------------------------------------------------------

class TestComputeCopperLosses:
    def test_returns_positive_total(self, simple_stator):
        r = compute_copper_losses(simple_stator, B_avg_winding=0.5, config={})
        assert r["total"] > 0.0

    def test_spatial_density_positive(self, simple_stator):
        r = compute_copper_losses(simple_stator, B_avg_winding=0.5, config={})
        assert r["spatial_W_per_m3"] > 0.0

    def test_higher_current_gives_higher_loss(self, simple_stator):
        r = compute_copper_losses(simple_stator, B_avg_winding=0.5, config={})
        assert r["total"] > 0


# ---------------------------------------------------------------------------
# Material library
# ---------------------------------------------------------------------------

class TestMaterialLibrary:
    def test_all_standard_materials_present(self):
        for mat in ("M250-35A", "M330-50A", "copper_class_F", "air"):
            props = get_material_properties(mat)
            assert isinstance(props, dict)

    def test_missing_material_raises_key_error(self):
        with pytest.raises(KeyError):
            get_material_properties("nonexistent_material_xyz")

    def test_bh_curve_present_for_iron(self):
        props = get_material_properties("M250-35A")
        assert "BH_curve" in props
        assert len(props["BH_curve"]) >= 5

    def test_reluctivity_zero_b(self):
        nu = interpolate_reluctivity(0.0, "M250-35A")
        assert nu > 0.0
        assert np.isfinite(nu)

    def test_reluctivity_at_1T(self):
        nu = interpolate_reluctivity(1.0, "M250-35A")
        assert nu > 0.0

    def test_reluctivity_increases_with_saturation(self):
        """Above saturation knee, reluctivity should increase with B."""
        nu_low  = interpolate_reluctivity(0.5,  "M250-35A")
        nu_high = interpolate_reluctivity(1.8,  "M250-35A")
        # Deep saturation → higher H/B → higher reluctivity
        assert nu_high > nu_low

    def test_reluctivity_air_is_1_over_mu0(self):
        import math
        MU_0 = 4 * math.pi * 1e-7
        nu_air = interpolate_reluctivity(1.0, "air")
        assert nu_air == pytest.approx(1.0 / MU_0, rel=1e-4)

    def test_reluctivity_array_input(self):
        B_arr = np.array([0.1, 0.5, 1.0, 1.5])
        nu_arr = interpolate_reluctivity(B_arr, "M250-35A")
        assert nu_arr.shape == (4,)
        assert np.all(nu_arr > 0)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestEMBoundaryConditions:
    def test_outer_nodes_returned(self, simple_mesh):
        bc_nodes, bc_vals = get_em_boundary_nodes(simple_mesh, {})
        assert len(bc_nodes) > 0

    def test_bc_values_are_zero(self, simple_mesh):
        bc_nodes, bc_vals = get_em_boundary_nodes(simple_mesh, {})
        assert np.all(bc_vals == 0.0)

    def test_apply_dirichlet_bcs_modifies_diagonal(self, simple_mesh):
        n = simple_mesh.n_nodes
        K = sp.eye(n, format="lil") * 2.0
        F = np.ones(n)
        bc_nodes = np.array([0, 1, 2])
        bc_vals  = np.zeros(3)
        K2, F2 = apply_dirichlet_bcs(K, F, bc_nodes, bc_vals)
        # Diagonal of BC nodes should be 1
        K_dense = K2.toarray()
        for node in bc_nodes:
            assert K_dense[node, node] == pytest.approx(1.0)
            assert F2[node] == pytest.approx(0.0)

    def test_apply_dirichlet_bcs_non_bc_rows_unchanged(self, simple_mesh):
        n = simple_mesh.n_nodes
        K = sp.eye(n, format="lil") * 3.0
        F = np.ones(n) * 5.0
        K2, F2 = apply_dirichlet_bcs(K, F, np.array([0]), np.array([0.0]))
        # Non-BC nodes should still have F=5
        assert F2[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Postprocessor
# ---------------------------------------------------------------------------

class TestEMPostprocessor:
    def test_extract_flux_density_returns_correct_keys(self, simple_mesh):
        A_z = np.zeros(simple_mesh.n_nodes)
        result = extract_flux_density(A_z, simple_mesh)
        assert "B_x" in result
        assert "B_y" in result
        assert "B_mag" in result

    def test_zero_potential_gives_zero_field(self, simple_mesh):
        A_z = np.zeros(simple_mesh.n_nodes)
        result = extract_flux_density(A_z, simple_mesh)
        assert np.allclose(result["B_mag"], 0.0)

    def test_b_mag_is_non_negative(self, simple_mesh):
        rng = np.random.default_rng(42)
        A_z = rng.standard_normal(simple_mesh.n_nodes) * 1e-3
        result = extract_flux_density(A_z, simple_mesh)
        assert np.all(result["B_mag"] >= 0.0)

    def test_efficiency_zero_loss(self, simple_stator):
        eta = compute_efficiency(50.0, simple_stator, 0.0)
        assert eta == pytest.approx(1.0, abs=1e-4)

    def test_efficiency_bounded(self, simple_stator):
        eta = compute_efficiency(50.0, simple_stator, 1000.0)
        assert 0.0 < eta <= 1.0


# ---------------------------------------------------------------------------
# Full EM solver
# ---------------------------------------------------------------------------

class TestEMSolver:
    def test_solver_runs_without_error(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert result["torque_Nm"] is not None

    def test_total_loss_positive(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert result["total_loss_W"] > 0.0

    def test_efficiency_in_range(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert 0.0 < result["efficiency"] <= 1.0

    def test_a_field_length(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert len(result["A_field"]) == mesh.n_nodes

    def test_b_field_dict_present(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert isinstance(result["B_field"], dict)
        assert "B_mag" in result["B_field"]

    def test_loss_maps_correct_length(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert len(result["loss_density_map"]) == mesh.n_elements

    def test_domain_is_mesh(self, simple_stator, em_config):
        mesh, regions = load_stator_geometry(simple_stator)
        result = run_electromagnetic_analysis(mesh, regions, simple_stator, em_config)
        assert result["domain"] is mesh
