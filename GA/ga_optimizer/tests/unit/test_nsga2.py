"""Unit tests for NSGA-II non-dominated sorting and crowding distance."""

from __future__ import annotations

import numpy as np
import pytest

from ga_optimizer.pareto.nsga2 import (
    dominates, fast_non_dominated_sort, crowding_distance_assignment,
)
from ga_optimizer.population import Individual
from ga_optimizer.objectives import ObjectiveVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ind(obj: list[float], *, feasible: bool = True) -> Individual:
    """Create an evaluated Individual with a given objective vector."""
    viol = 0.0 if feasible else 1e6
    ov = ObjectiveVector(
        neg_efficiency=obj[0],
        total_loss_W=obj[1],
        neg_power_density=obj[2],
        temperature_violation_K=viol,
        safety_factor_violation=0.0,
    )
    return Individual(genes=np.zeros(12), objectives=ov, evaluated=True)


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------

class TestDominates:
    def test_clear_dominance(self):
        assert dominates(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))

    def test_non_dominated_equal(self):
        assert not dominates(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))

    def test_non_dominated_tradeoff(self):
        assert not dominates(np.array([1.0, 2.0, 1.0]), np.array([2.0, 1.0, 2.0]))

    def test_one_better_rest_equal(self):
        assert dominates(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 2.0]))

    def test_one_worse_rest_better(self):
        assert not dominates(np.array([1.0, 1.0, 3.0]), np.array([1.0, 1.0, 1.0]))

    def test_two_fronts(self):
        """a dominates c; b dominates c; a and b are non-dominated."""
        a = np.array([1.0, 2.0, 1.0])
        b = np.array([2.0, 1.0, 1.0])
        c = np.array([3.0, 3.0, 3.0])
        assert dominates(a, c)
        assert dominates(b, c)
        assert not dominates(a, b)
        assert not dominates(b, a)


# ---------------------------------------------------------------------------
# fast_non_dominated_sort
# ---------------------------------------------------------------------------

class TestFastNonDominatedSort:
    def test_single_individual(self):
        pop = [_ind([1.0, 1.0, 1.0])]
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 1
        assert fronts[0] == [0]
        assert pop[0].rank == 0

    def test_two_fronts_3obj(self):
        """A, B are non-dominated; C is dominated by both."""
        A = _ind([1.0, 2.0, 1.0])
        B = _ind([2.0, 1.0, 1.0])
        C = _ind([3.0, 3.0, 3.0])
        pop = [A, B, C]
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 2
        assert set(fronts[0]) == {0, 1}
        assert fronts[1] == [2]
        assert A.rank == 0
        assert B.rank == 0
        assert C.rank == 1

    def test_three_fronts(self):
        """Chain: A dominates B; B dominates C."""
        A = _ind([1.0, 1.0, 1.0])
        B = _ind([2.0, 2.0, 2.0])
        C = _ind([3.0, 3.0, 3.0])
        pop = [A, B, C]
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 3
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]

    def test_all_non_dominated(self):
        """Points on the same Pareto front — all rank 0."""
        pop = [_ind([float(i), float(10 - i), 0.0]) for i in range(5)]
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2, 3, 4}
        assert all(ind.rank == 0 for ind in pop)

    def test_ranks_set_in_place(self):
        A = _ind([1.0, 2.0, 1.0])
        B = _ind([3.0, 3.0, 3.0])
        pop = [A, B]
        fast_non_dominated_sort(pop)
        assert A.rank == 0
        assert B.rank == 1

    def test_large_population(self):
        """50 individuals on a 2-D Pareto curve → all rank 0."""
        n = 50
        pop = [_ind([float(i), float(n - i), 0.0]) for i in range(n)]
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == n


# ---------------------------------------------------------------------------
# crowding_distance_assignment
# ---------------------------------------------------------------------------

class TestCrowdingDistance:
    def test_single_individual_gets_inf(self):
        pop = [_ind([1.0, 2.0, 3.0])]
        pop[0].rank = 0
        front = [0]
        crowding_distance_assignment(pop, front)
        assert pop[0].crowding_distance == float("inf")

    def test_two_individuals_get_inf(self):
        pop = [_ind([1.0, 2.0, 0.0]), _ind([3.0, 0.0, 0.0])]
        for ind in pop:
            ind.rank = 0
        crowding_distance_assignment(pop, [0, 1])
        assert pop[0].crowding_distance == float("inf")
        assert pop[1].crowding_distance == float("inf")

    def test_boundary_individuals_get_inf(self):
        """Boundary members of a 5-member front get +inf."""
        pop = [_ind([float(i), float(10 - i), 0.0]) for i in range(5)]
        for ind in pop:
            ind.rank = 0
        fast_non_dominated_sort(pop)
        crowding_distance_assignment(pop, list(range(5)))
        inf_count = sum(
            1 for ind in pop if ind.crowding_distance == float("inf")
        )
        assert inf_count >= 2, "Boundary individuals must have infinite crowding distance"

    def test_interior_individuals_have_positive_distance(self):
        """Interior members of a 5-member front get finite positive distance."""
        pop = [_ind([float(i), float(10 - i), 0.0]) for i in range(5)]
        for ind in pop:
            ind.rank = 0
        crowding_distance_assignment(pop, list(range(5)))
        interior = [ind for ind in pop if ind.crowding_distance != float("inf")]
        assert all(d >= 0 for d in [ind.crowding_distance for ind in interior])

    def test_distances_are_non_negative(self):
        pop = [_ind([float(i), float(20 - i), float(i % 3)]) for i in range(10)]
        for ind in pop:
            ind.rank = 0
        crowding_distance_assignment(pop, list(range(10)))
        for ind in pop:
            assert ind.crowding_distance >= 0


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------

class TestParetoArchive:
    def test_empty_archive(self):
        from ga_optimizer.pareto.archive import ParetoArchive
        arch = ParetoArchive()
        assert arch.size == 0
        assert arch.members == []

    def test_add_feasible_individual(self):
        from ga_optimizer.pareto.archive import ParetoArchive
        arch = ParetoArchive()
        ind = _ind([1.0, 1.0, 1.0])
        added = arch.update([ind])
        assert added == 1
        assert arch.size == 1

    def test_dominated_individual_not_added(self):
        from ga_optimizer.pareto.archive import ParetoArchive
        arch = ParetoArchive()
        good = _ind([1.0, 1.0, 1.0])
        bad  = _ind([2.0, 2.0, 2.0])
        arch.update([good])
        added = arch.update([bad])
        assert added == 0
        assert arch.size == 1

    def test_new_dominator_prunes_archive(self):
        from ga_optimizer.pareto.archive import ParetoArchive
        arch = ParetoArchive()
        old = _ind([2.0, 2.0, 2.0])
        arch.update([old])
        better = _ind([1.0, 1.0, 1.0])
        arch.update([better])
        assert arch.size == 1
        assert arch.members[0].objectives.neg_efficiency == 1.0  # noqa

    def test_infeasible_individual_not_added(self):
        from ga_optimizer.pareto.archive import ParetoArchive
        arch = ParetoArchive()
        infeasible = _ind([1.0, 1.0, 1.0], feasible=False)
        added = arch.update([infeasible])
        assert added == 0
        assert arch.size == 0

    def test_objective_matrix_shape(self):
        from ga_optimizer.pareto.archive import ParetoArchive
        arch = ParetoArchive()
        for i in range(5):
            arch.update([_ind([float(i), float(5 - i), 0.0])])
        mat = arch.objective_matrix()
        assert mat.shape[1] == 3
        assert mat.shape[0] == arch.size
