"""Unit tests for hypervolume and IGD metrics."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ga_optimizer.utils.metrics import (
    compute_hypervolume, compute_igd, compute_gd,
    _hv_2d, _wfg_hypervolume,
)
from ga_optimizer.population import Individual
from ga_optimizer.objectives import ObjectiveVector


def _ind_obj(obj: list[float]) -> Individual:
    ov = ObjectiveVector(
        neg_efficiency=obj[0],
        total_loss_W=obj[1],
        neg_power_density=obj[2],
        temperature_violation_K=0.0,
        safety_factor_violation=0.0,
    )
    return Individual(genes=np.zeros(12), objectives=ov, evaluated=True)


# ---------------------------------------------------------------------------
# 2-D hypervolume (exact reference values)
# ---------------------------------------------------------------------------

class TestHV2D:
    def test_single_point(self):
        """HV of a single point [0, 0] with reference [2, 2] = 4."""
        pts = np.array([[0.0, 0.0]])
        ref = np.array([2.0, 2.0])
        hv = _hv_2d(pts, ref)
        assert math.isclose(hv, 4.0, rel_tol=1e-6)

    def test_two_points_no_overlap(self):
        """[0,1] and [1,0] with ref [2,2]: two non-overlapping rectangles."""
        pts = np.array([[0.0, 1.0], [1.0, 0.0]])
        ref = np.array([2.0, 2.0])
        # HV = area covered by both rectangles:
        # From [0,1] to ref = 2×1 = 2; from [1,0] to ref = 1×2 = 2; overlap = 1×1 = 1
        # Total = 2 + 2 - 1 = 3
        hv = _hv_2d(pts, ref)
        assert math.isclose(hv, 3.0, rel_tol=1e-6)

    def test_dominated_point_not_counted(self):
        """A dominated point should not increase HV."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])  # second point is dominated
        ref = np.array([2.0, 2.0])
        hv1 = _hv_2d(np.array([[0.0, 0.0]]), ref)
        hv2 = _hv_2d(pts, ref)
        assert math.isclose(hv1, hv2, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# compute_hypervolume with Individual list
# ---------------------------------------------------------------------------

class TestComputeHypervolume:
    def test_empty_archive_returns_zero(self):
        hv = compute_hypervolume([], [0.0, 5000.0, 0.0])
        assert hv == 0.0

    def test_single_feasible_individual(self):
        ind = _ind_obj([-0.9, 100.0, -1e6])
        # reference is [0, 5000, 0]; all objectives of ind are < ref
        ref = [0.0, 5000.0, 0.0]
        hv = compute_hypervolume([ind], ref)
        assert hv > 0.0

    def test_infeasible_individual_not_counted(self):
        from ga_optimizer.objectives import ObjectiveVector
        ov = ObjectiveVector(
            neg_efficiency=-0.9, total_loss_W=100.0, neg_power_density=-1e6,
            temperature_violation_K=100.0,   # infeasible
            safety_factor_violation=0.0,
        )
        ind = Individual(genes=np.zeros(12), objectives=ov, evaluated=True)
        hv = compute_hypervolume([ind], [0.0, 5000.0, 0.0])
        assert hv == 0.0

    def test_hypervolume_increases_with_better_solutions(self):
        """Adding a better solution should increase hypervolume."""
        ref = [0.0, 5000.0, 0.0]
        ind1 = _ind_obj([-0.85, 500.0, -5e5])
        hv1 = compute_hypervolume([ind1], ref)

        ind2 = _ind_obj([-0.92, 200.0, -1.5e6])  # Pareto-better
        hv2 = compute_hypervolume([ind1, ind2], ref)
        assert hv2 >= hv1

    def test_two_non_dominated_solutions(self):
        """Two non-dominated solutions should give larger HV than either alone."""
        ref = [0.0, 5000.0, 0.0]
        ind_a = _ind_obj([-0.90, 500.0, -8e5])
        ind_b = _ind_obj([-0.80, 200.0, -1e6])
        hv_a   = compute_hypervolume([ind_a], ref)
        hv_b   = compute_hypervolume([ind_b], ref)
        hv_both = compute_hypervolume([ind_a, ind_b], ref)
        assert hv_both > hv_a
        assert hv_both > hv_b


# ---------------------------------------------------------------------------
# IGD
# ---------------------------------------------------------------------------

class TestIGD:
    def test_perfect_approximation_gives_zero(self):
        """When approximation == reference, IGD = 0."""
        pts = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 1.5]])
        igd = compute_igd(pts, pts)
        assert math.isclose(igd, 0.0, abs_tol=1e-10)

    def test_shifted_approximation(self):
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        approx = ref + 0.1
        igd = compute_igd(approx, ref)
        # Each reference point is ~0.141 from the nearest approx point
        assert igd > 0.0

    def test_empty_inputs_give_inf(self):
        assert compute_igd(np.empty((0, 2)), np.array([[1.0, 2.0]])) == float("inf")
        assert compute_igd(np.array([[1.0, 2.0]]), np.empty((0, 2))) == float("inf")

    def test_gd_and_igd_order(self):
        """GD measures distance from approx → ref; IGD from ref → approx."""
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        approx = np.array([[0.5, 0.5]])
        gd  = compute_gd(approx, ref)
        igd = compute_igd(approx, ref)
        # GD: distance from [0.5,0.5] to nearest ref point
        # IGD: mean over ref points of distance to nearest approx point
        assert gd > 0.0
        assert igd > 0.0
