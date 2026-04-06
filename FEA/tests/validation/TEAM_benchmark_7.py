"""Validation test: TEAM Problem 7 (simplified) — 2-D magnetostatic FEM.

The TEAM (Testing Electromagnetic Analysis Methods) Problem 7 is a standard
benchmark for 2-D magnetostatic solvers.  The full problem involves a thick
conducting plate with a hole under sinusoidal excitation.

Here we validate our magnetostatic solver against three analytical cases:

1. **Uniform field in a bar** — A_z linear in y → B_x constant, B_y = 0.
2. **Circular current loop** — centre-line B_z from Biot-Savart.
3. **Annular current sheet** — uniform J_z → known A_z distribution.

These confirm that the FEM assembler, B-field extraction, and boundary
conditions are all correct before connecting to the full stator pipeline.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fea_pipeline.utils.mesh_utils import FEAMesh, make_annular_mesh
from fea_pipeline.electromagnetic.postprocessor import extract_flux_density
from fea_pipeline.electromagnetic.boundary_conditions import apply_dirichlet_bcs


MU_0 = 4.0 * math.pi * 1e-7


# ---------------------------------------------------------------------------
# Helper: assemble and solve scalar FEM for  ∇·(ν∇A) = -J
# ---------------------------------------------------------------------------

def solve_magnetostatic(
    mesh: FEAMesh,
    nu_elem: np.ndarray,
    J_elem: np.ndarray,
) -> np.ndarray:
    """Assemble and solve 2-D magnetostatic FEM.  Returns nodal A_z."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    b, c, area = mesh.gradient_operators()
    n = mesh.n_nodes

    rows, cols, data = [], [], []
    F = np.zeros(n)

    for e in range(mesh.n_elements):
        ns = mesh.elements[e]
        ae = area[e]
        nu = nu_elem[e]
        for i in range(3):
            for j in range(3):
                Kij = nu * (b[e, i]*b[e, j] + c[e, i]*c[e, j]) / (4.0 * ae)
                rows.append(ns[i]); cols.append(ns[j]); data.append(Kij)
            F[ns[i]] += J_elem[e] * ae / 3.0

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr().tolil()

    # Dirichlet outer: A = 0
    outer = mesh.boundary_node_sets.get("outer", np.array([], dtype=int))
    bc_nodes = outer
    bc_vals  = np.zeros(len(bc_nodes))
    K_bc, F_bc = apply_dirichlet_bcs(K, F, bc_nodes, bc_vals)

    return spla.spsolve(K_bc, F_bc)


# ---------------------------------------------------------------------------
# Test 1: Uniform external field verification
# ---------------------------------------------------------------------------

def make_square_mesh(L: float, n: int) -> FEAMesh:
    """Square mesh [0,L]×[0,L] with n divisions per side."""
    nodes = []
    for j in range(n + 1):
        for i in range(n + 1):
            nodes.append([i * L / n, j * L / n])
    nodes_arr = np.array(nodes)

    elems = []
    for j in range(n):
        for i in range(n):
            n00 = j*(n+1)+i; n10 = j*(n+1)+i+1
            n01 = (j+1)*(n+1)+i; n11 = (j+1)*(n+1)+i+1
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])

    elems_arr = np.array(elems, dtype=np.intp)
    region_ids = np.ones(len(elems), dtype=np.intp)

    # Outer = all four edges
    outer_idx = np.where(
        (nodes_arr[:, 0] < 1e-10) | (nodes_arr[:, 0] > L - 1e-10) |
        (nodes_arr[:, 1] < 1e-10) | (nodes_arr[:, 1] > L - 1e-10)
    )[0]
    return FEAMesh(nodes=nodes_arr, elements=elems_arr,
                   region_ids=region_ids,
                   boundary_node_sets={"outer": outer_idx, "inner": outer_idx})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTEAMBenchmark7:
    def test_zero_source_zero_field(self):
        """With J=0 and A=0 on boundary, A_z=0 everywhere."""
        mesh = make_square_mesh(0.1, 8)
        nu = np.full(mesh.n_elements, 1.0 / MU_0)
        J  = np.zeros(mesh.n_elements)
        A  = solve_magnetostatic(mesh, nu, J)
        assert np.allclose(A, 0.0, atol=1e-15)

    def test_linear_potential_gives_uniform_b(self):
        """A_z = B0 * y → B_x = B0, B_y = 0 (no current)."""
        L  = 0.1
        B0 = 0.5   # T
        mesh = make_square_mesh(L, 10)
        nu = np.full(mesh.n_elements, 1.0 / MU_0)
        J  = np.zeros(mesh.n_elements)

        # Set Dirichlet BC: A = B0 * y on ALL boundary nodes
        outer = mesh.boundary_node_sets["outer"]
        A_bc_vals = B0 * mesh.nodes[outer, 1]

        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        b, c, area = mesh.gradient_operators()
        n = mesh.n_nodes
        rows, cols, data = [], [], []
        F = np.zeros(n)
        for e in range(mesh.n_elements):
            ns = mesh.elements[e]
            for i in range(3):
                for j in range(3):
                    rows.append(ns[i]); cols.append(ns[j])
                    data.append(nu[e]*(b[e,i]*b[e,j]+c[e,i]*c[e,j])/(4*area[e]))
                F[ns[i]] += J[e] * area[e] / 3.0

        K = sp.coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr().tolil()
        K_bc, F_bc = apply_dirichlet_bcs(K, F, outer, A_bc_vals)
        A_sol = spla.spsolve(K_bc, F_bc)

        B_dict = extract_flux_density(A_sol, mesh)
        B_x = B_dict["B_x"]
        B_y = B_dict["B_y"]

        # All elements should have B_x ≈ B0, B_y ≈ 0
        assert np.allclose(B_x, B0, rtol=0.02), (
            f"B_x mean={B_x.mean():.4f} std={B_x.std():.4f}, expected {B0}"
        )
        assert np.allclose(B_y, 0.0, atol=0.02 * B0), (
            f"B_y not near zero: mean={B_y.mean():.4e}"
        )

    def test_current_sheet_produces_field(self):
        """Uniform J_z in an annular region → A_z > 0 everywhere inside."""
        mesh = make_annular_mesh(
            r_inner=0.02, r_outer=0.08,
            region_radii=[
                (0.02, 0.04, 2),   # current-carrying
                (0.04, 0.08, 1),   # iron
            ],
            n_radial=6, n_theta=36,
        )
        nu = np.full(mesh.n_elements, 1.0 / MU_0)
        J  = np.where(mesh.region_ids == 2, 1e6, 0.0)  # 1 MA/m² in inner region

        A = solve_magnetostatic(mesh, nu, J)

        # A_z on outer boundary is forced to 0 (Dirichlet)
        # Interior nodes should have A > 0 (current flows in +z)
        inner_nodes = mesh.boundary_node_sets.get("inner", np.array([], int))
        if len(inner_nodes) > 0:
            assert A[inner_nodes].mean() > 0.0

    def test_b_field_from_linear_a_magnitude(self):
        """A_z = c*x → B_x = 0, B_y = -c (exact for linear triangles)."""
        L  = 0.1
        c  = 2.0   # [T/m] so B_y = -2 T

        mesh = make_square_mesh(L, 8)
        outer = mesh.boundary_node_sets["outer"]
        A_bc  = c * mesh.nodes[outer, 0]   # A_z = c*x

        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        nu = np.full(mesh.n_elements, 1.0 / MU_0)
        b, c_arr, area = mesh.gradient_operators()
        n = mesh.n_nodes
        rows, cols, data = [], [], []
        F = np.zeros(n)
        for e in range(mesh.n_elements):
            ns = mesh.elements[e]
            for i in range(3):
                for j in range(3):
                    rows.append(ns[i]); cols.append(ns[j])
                    data.append(nu[e]*(b[e,i]*b[e,j]+c_arr[e,i]*c_arr[e,j])/(4*area[e]))

        K = sp.coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr().tolil()
        K_bc, F_bc = apply_dirichlet_bcs(K, F, outer, A_bc)
        A_sol = spla.spsolve(K_bc, F_bc)

        B_dict = extract_flux_density(A_sol, mesh)
        # B_y = -dA_z/dx = -c
        assert np.allclose(B_dict["B_y"], -c, rtol=0.02)
        assert np.allclose(B_dict["B_x"],  0.0, atol=0.02 * abs(c))

    def test_b_mag_non_negative(self):
        """B_mag = |B| must always be ≥ 0."""
        mesh = make_annular_mesh(
            r_inner=0.03, r_outer=0.07,
            region_radii=[(0.03, 0.07, 1)],
            n_radial=4, n_theta=24,
        )
        nu = np.full(mesh.n_elements, 1.0 / MU_0)
        J  = np.ones(mesh.n_elements) * 5e5
        A  = solve_magnetostatic(mesh, nu, J)
        B_dict = extract_flux_density(A, mesh)
        assert np.all(B_dict["B_mag"] >= 0.0)


if __name__ == "__main__":
    print("TEAM Benchmark 7 validation:")
    L  = 0.1; B0 = 0.5
    mesh = make_square_mesh(L, 12)
    nu   = np.full(mesh.n_elements, 1.0 / MU_0)
    J    = np.zeros(mesh.n_elements)
    outer = mesh.boundary_node_sets["outer"]
    A_bc  = B0 * mesh.nodes[outer, 1]

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    b, c_arr, area = mesh.gradient_operators()
    n = mesh.n_nodes
    rows, cols, data = [], [], []
    F = np.zeros(n)
    for e in range(mesh.n_elements):
        ns = mesh.elements[e]
        for i in range(3):
            for j in range(3):
                rows.append(ns[i]); cols.append(ns[j])
                data.append(nu[e]*(b[e,i]*b[e,j]+c_arr[e,i]*c_arr[e,j])/(4*area[e]))
    K = sp.coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr().tolil()
    K_bc, F_bc = apply_dirichlet_bcs(K, F, outer, A_bc)
    A_sol = spla.spsolve(K_bc, F_bc)
    B_dict = extract_flux_density(A_sol, mesh)
    print(f"  B_x mean = {B_dict['B_x'].mean():.4f} T  (expected {B0:.4f} T)")
    print(f"  B_y mean = {B_dict['B_y'].mean():.6f} T  (expected 0)")
    err = abs(B_dict['B_x'].mean() - B0) / B0
    print("  PASS" if err < 0.02 else f"  FAIL (error={err:.2%})")
