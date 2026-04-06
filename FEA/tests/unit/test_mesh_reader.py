"""Unit tests for the mesh reader module."""

from __future__ import annotations

import numpy as np
import pytest

from fea_pipeline.io.mesh_reader import load_stator_geometry, _synthesise_mesh
from fea_pipeline.utils.mesh_utils import FEAMesh


class TestSyntheticMesh:
    def test_returns_fea_mesh(self, simple_stator):
        mesh, regions = load_stator_geometry(simple_stator)
        assert isinstance(mesh, FEAMesh)

    def test_node_count_positive(self, simple_stator):
        mesh, _ = load_stator_geometry(simple_stator)
        assert mesh.n_nodes > 0

    def test_element_count_positive(self, simple_stator):
        mesh, _ = load_stator_geometry(simple_stator)
        assert mesh.n_elements > 0

    def test_all_elements_have_positive_area(self, simple_stator):
        mesh, _ = load_stator_geometry(simple_stator)
        areas = mesh.element_areas()
        assert np.all(areas > 0), f"Non-positive areas: {areas[areas <= 0]}"

    def test_region_submeshes_keyed_by_name(self, simple_stator):
        _, regions = load_stator_geometry(simple_stator)
        for key in ("stator_core", "winding", "air_gap"):
            assert key in regions

    def test_region_submeshes_are_fea_mesh(self, simple_stator):
        _, regions = load_stator_geometry(simple_stator)
        for name, sub in regions.items():
            assert isinstance(sub, FEAMesh), f"{name} is not FEAMesh"

    def test_all_three_regions_have_elements(self, simple_stator):
        _, regions = load_stator_geometry(simple_stator)
        for name, sub in regions.items():
            assert sub.n_elements > 0, f"Region {name} has no elements"

    def test_boundary_node_sets_present(self, simple_stator):
        mesh, _ = load_stator_geometry(simple_stator)
        assert "outer" in mesh.boundary_node_sets
        assert "inner" in mesh.boundary_node_sets

    def test_outer_nodes_at_outer_radius(self, simple_stator):
        mesh, _ = load_stator_geometry(simple_stator)
        outer_nodes = mesh.boundary_node_sets["outer"]
        r_outer = simple_stator.outer_diameter / 2.0
        r = np.linalg.norm(mesh.nodes[outer_nodes], axis=1)
        assert np.allclose(r, r_outer, rtol=1e-3)

    def test_unsupported_format_raises(self):
        from fea_pipeline.io.schema import StatorMeshInput
        inp = StatorMeshInput(
            stator_id="x",
            outer_diameter=0.2, inner_diameter=0.1, axial_length=0.08,
            num_slots=6, num_poles=4, slot_opening=0.005, tooth_width=0.01,
            yoke_height=0.015, slot_depth=0.02,
            mesh_format="unknown_format",
            mesh_file_path="/nonexistent.xyz",
        )
        with pytest.raises(ValueError, match="Unsupported"):
            load_stator_geometry(inp)

    def test_missing_file_raises(self):
        from fea_pipeline.io.schema import StatorMeshInput
        inp = StatorMeshInput(
            stator_id="x",
            outer_diameter=0.2, inner_diameter=0.1, axial_length=0.08,
            num_slots=6, num_poles=4, slot_opening=0.005, tooth_width=0.01,
            yoke_height=0.015, slot_depth=0.02,
            mesh_format="vtk",
            mesh_file_path="/nonexistent_file.vtk",
        )
        with pytest.raises((FileNotFoundError, ImportError)):
            load_stator_geometry(inp)


class TestFEAMeshHelpers:
    def test_element_centroids_shape(self, simple_mesh):
        c = simple_mesh.element_centroids()
        assert c.shape == (simple_mesh.n_elements, 2)

    def test_gradient_operators_shapes(self, simple_mesh):
        b, c, area = simple_mesh.gradient_operators()
        n = simple_mesh.n_elements
        assert b.shape == (n, 3)
        assert c.shape == (n, 3)
        assert area.shape == (n,)

    def test_gradient_operators_areas_positive(self, simple_mesh):
        _, _, area = simple_mesh.gradient_operators()
        assert np.all(area > 0)

    def test_total_area_close_to_analytical(self, simple_mesh):
        import math
        # Annular area = pi * (r_outer² - r_inner²)
        areas = simple_mesh.element_areas()
        total = float(areas.sum())
        r_o = np.linalg.norm(simple_mesh.nodes, axis=1).max()
        r_i = np.linalg.norm(simple_mesh.nodes, axis=1).min()
        expected = math.pi * (r_o**2 - r_i**2)
        assert abs(total - expected) / expected < 0.02, (
            f"Total area {total:.6f} vs expected {expected:.6f}"
        )
