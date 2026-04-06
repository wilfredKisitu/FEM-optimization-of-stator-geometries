"""Validation test: 1-D thermal slab (NIST analytical reference).

Analytical solution for 1-D steady-state heat conduction with source:
    -k d²T/dx² = q    on [0, L]
    T(0) = T_L,  T(L) = T_R

Exact solution:
    T(x) = T_L + (T_R - T_L)*x/L + q/(2k) * x*(L - x)

We discretise this on a 1-D strip (thin rectangle) and compare nodal
temperatures to the analytical solution.  Expected: max error < 0.1 %.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fea_pipeline.utils.mesh_utils import FEAMesh


# ---------------------------------------------------------------------------
# Build a 1-D strip mesh (thin rectangle)
# ---------------------------------------------------------------------------

def make_strip_mesh(L: float, n: int) -> FEAMesh:
    """Create a thin rectangular mesh of length L with n intervals.

    The strip has height dy << L so it behaves as a 1-D problem.
    """
    dy = L / (10 * n)   # thin strip
    nodes: list[list[float]] = []
    for i in range(n + 1):
        x = i * L / n
        nodes.append([x, 0.0])
        nodes.append([x, dy])

    # Two triangles per quad
    elements: list[list[int]] = []
    for i in range(n):
        n00 = 2 * i
        n01 = 2 * i + 1
        n10 = 2 * (i + 1)
        n11 = 2 * (i + 1) + 1
        elements.append([n00, n10, n11])
        elements.append([n00, n11, n01])

    nodes_arr = np.array(nodes)
    elems_arr = np.array(elements, dtype=np.intp)
    region_ids = np.ones(len(elements), dtype=np.intp)

    left_nodes  = np.array([0, 1], dtype=np.intp)
    right_nodes = np.array([2 * n, 2 * n + 1], dtype=np.intp)

    return FEAMesh(
        nodes=nodes_arr,
        elements=elems_arr,
        region_ids=region_ids,
        boundary_node_sets={"left": left_nodes, "right": right_nodes,
                            "outer": right_nodes, "inner": left_nodes},
    )


def analytical_T(x: np.ndarray, T_L: float, T_R: float,
                 L: float, k: float, q: float) -> np.ndarray:
    return T_L + (T_R - T_L) * x / L + q / (2 * k) * x * (L - x)


def run_thermal_fem_1d(
    L: float = 0.1,
    n: int = 20,
    k: float = 28.0,
    q: float = 1e6,
    T_L: float = 300.0,
    T_R: float = 320.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve 1-D heat conduction FEM and return (x_nodal, T_nodal)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    mesh = make_strip_mesh(L, n)
    b, c, area = mesh.gradient_operators()

    n_nodes = mesh.n_nodes
    rows, cols, data = [], [], []
    F = np.zeros(n_nodes)

    for e in range(mesh.n_elements):
        ns = mesh.elements[e]
        for i in range(3):
            for j in range(3):
                Kij = k * (b[e, i] * b[e, j] + c[e, i] * c[e, j]) / (4.0 * area[e])
                rows.append(ns[i]); cols.append(ns[j]); data.append(Kij)
            F[ns[i]] += q * area[e] / 3.0

    K = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr().tolil()

    # Dirichlet BCs
    left_nodes  = mesh.boundary_node_sets["left"]
    right_nodes = mesh.boundary_node_sets["right"]
    for node in left_nodes:
        K[node, :] = 0; K[node, node] = 1; F[node] = T_L
    for node in right_nodes:
        K[node, :] = 0; K[node, node] = 1; F[node] = T_R

    T_sol = spla.spsolve(K.tocsr(), F)
    x_nodes = mesh.nodes[:, 0]
    return x_nodes, T_sol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNISTThermalBlock:
    L, k, q, T_L, T_R = 0.1, 28.0, 1e6, 300.0, 320.0

    def test_max_error_below_0_1_percent(self):
        x, T_num = run_thermal_fem_1d(
            L=self.L, n=30, k=self.k, q=self.q, T_L=self.T_L, T_R=self.T_R
        )
        T_exact = analytical_T(x, self.T_L, self.T_R, self.L, self.k, self.q)
        max_T = T_exact.max()
        rel_err = np.abs(T_num - T_exact).max() / max_T
        assert rel_err < 0.001, f"Max relative error {rel_err:.4%} exceeds 0.1%"

    def test_boundary_values_exact(self):
        x, T_num = run_thermal_fem_1d(
            L=self.L, n=20, k=self.k, q=self.q, T_L=self.T_L, T_R=self.T_R
        )
        left_mask  = x < 1e-10
        right_mask = x > self.L - 1e-10
        assert np.allclose(T_num[left_mask],  self.T_L, atol=1e-8)
        assert np.allclose(T_num[right_mask], self.T_R, atol=1e-8)

    def test_peak_temperature_at_correct_location(self):
        x, T_num = run_thermal_fem_1d(
            L=self.L, n=40, k=self.k, q=self.q, T_L=self.T_L, T_R=self.T_R
        )
        # Analytical peak at x* = L/2 + k*(T_R-T_L)/(q*L)
        x_peak_exact = self.L / 2 + self.k * (self.T_R - self.T_L) / (self.q * self.L)
        x_peak_num = x[np.argmax(T_num)]
        assert abs(x_peak_num - x_peak_exact) < 0.005 * self.L  # within 0.5% of L

    def test_solution_everywhere_above_min_bc(self):
        x, T_num = run_thermal_fem_1d(L=self.L, n=20, k=self.k, q=self.q,
                                       T_L=self.T_L, T_R=self.T_R)
        assert np.all(T_num >= min(self.T_L, self.T_R) - 1e-6)

    def test_refining_mesh_reduces_error(self):
        errs = []
        for n in (5, 10, 20):
            x, T_num = run_thermal_fem_1d(L=self.L, n=n, k=self.k, q=self.q,
                                           T_L=self.T_L, T_R=self.T_R)
            T_exact = analytical_T(x, self.T_L, self.T_R, self.L, self.k, self.q)
            errs.append(np.abs(T_num - T_exact).max())
        # Error should decrease with refinement
        assert errs[1] <= errs[0]
        assert errs[2] <= errs[1]


if __name__ == "__main__":
    x, T_num = run_thermal_fem_1d()
    T_exact = analytical_T(x, 300.0, 320.0, 0.1, 28.0, 1e6)
    rel_err = np.abs(T_num - T_exact).max() / T_exact.max()
    print(f"NIST 1-D thermal validation: max relative error = {rel_err:.4%}")
    print("PASS" if rel_err < 0.001 else "FAIL")
