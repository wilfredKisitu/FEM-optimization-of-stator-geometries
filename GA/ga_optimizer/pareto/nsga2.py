"""pareto/nsga2.py — NSGA-II fast non-dominated sorting and crowding distance.

Two functions implement the core of NSGA-II (Deb et al., 2002):

1. ``fast_non_dominated_sort`` — assigns a non-domination rank (0 = Pareto
   front) to every individual in O(M N²) time where M is the number of
   objectives.

2. ``crowding_distance_assignment`` — assigns a crowding-distance score to
   individuals within each front.  Boundary individuals receive +∞;
   interior individuals receive the sum of normalised gaps to their
   neighbours on each objective axis.

Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.  IEEE
Transactions on Evolutionary Computation, 6(2), 182–197.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..population import Individual, Population


# ---------------------------------------------------------------------------
# Dominance predicate
# ---------------------------------------------------------------------------

def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Return True iff *obj_a* Pareto-dominates *obj_b* (minimisation).

    Individual *a* dominates *b* iff it is no worse on any objective and
    strictly better on at least one objective.

    Parameters
    ----------
    obj_a, obj_b:
        Objective arrays to compare — same length, all values to be minimised.

    Returns
    -------
    bool
    """
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


# ---------------------------------------------------------------------------
# Fast non-dominated sort
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(population: "Population") -> list[list[int]]:
    """Partition *population* into non-domination fronts.

    Parameters
    ----------
    population:
        List of :class:`Individual` objects, each with a non-``None``
        ``objectives`` attribute.

    Returns
    -------
    list[list[int]]
        ``fronts[k]`` is the list of indices into *population* that belong
        to rank *k* (0 = current Pareto front, 1 = next, …).  Empty lists
        at the end are excluded.  The ``rank`` attribute of each individual
        is updated in-place.
    """
    n = len(population)
    dom_count  = [0] * n           # number of individuals that dominate i
    dom_set    = [[] for _ in range(n)]  # set of individuals that i dominates
    fronts: list[list[int]] = [[]]

    # Build objective matrix once to avoid repeated attribute access
    objs = np.array([
        ind.objectives.objective_array for ind in population
    ])

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objs[i], objs[j]):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif dominates(objs[j], objs[i]):
                dom_set[j].append(i)
                dom_count[i] += 1

        if dom_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    population[j].rank = current + 1
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------

def crowding_distance_assignment(
    population: "Population",
    front: list[int],
) -> None:
    """Assign crowding distance to all individuals in *front* (in-place).

    Boundary individuals (best and worst on any objective axis) receive
    ``+inf``.  Interior individuals accumulate the normalised gap to their
    two neighbours for each objective.

    Parameters
    ----------
    population:
        Full population list — only indices in *front* are modified.
    front:
        List of population indices belonging to one non-domination front.
    """
    n = len(front)
    if n == 0:
        return

    # Reset crowding distances for this front
    for idx in front:
        population[idx].crowding_distance = 0.0

    n_obj = len(population[front[0]].objectives.objective_array)

    for m in range(n_obj):
        # Sort front by objective m
        sorted_front = sorted(
            front,
            key=lambda i: population[i].objectives.objective_array[m]
        )

        # Boundary individuals get infinite distance
        population[sorted_front[0]].crowding_distance  = float("inf")
        population[sorted_front[-1]].crowding_distance = float("inf")

        f_min = population[sorted_front[0]].objectives.objective_array[m]
        f_max = population[sorted_front[-1]].objectives.objective_array[m]
        f_range = f_max - f_min if f_max > f_min else 1e-10

        for k in range(1, n - 1):
            left  = population[sorted_front[k - 1]].objectives.objective_array[m]
            right = population[sorted_front[k + 1]].objectives.objective_array[m]
            population[sorted_front[k]].crowding_distance += (right - left) / f_range
