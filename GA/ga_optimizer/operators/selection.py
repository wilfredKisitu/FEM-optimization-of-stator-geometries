"""selection.py — Binary tournament selection using the NSGA-II crowded
comparison operator."""

from __future__ import annotations

import numpy as np

from ..population import Individual, Population


def crowded_tournament(population: Population, rng: np.random.Generator) -> Individual:
    """Select one individual via binary tournament with crowded-comparison.

    Individual *a* is preferred over *b* when:
    - ``a.rank < b.rank``  (lower rank = closer to Pareto front), or
    - ``a.rank == b.rank`` and ``a.crowding_distance > b.crowding_distance``
      (higher crowding distance = less crowded, more diverse region).

    Parameters
    ----------
    population:
        Current population (all individuals must have ``rank`` and
        ``crowding_distance`` set by NSGA-II sorting).
    rng:
        NumPy random Generator.

    Returns
    -------
    Individual
        The winning individual (a reference, not a copy).
    """
    n = len(population)
    idx_a, idx_b = rng.choice(n, size=2, replace=False)
    a = population[idx_a]
    b = population[idx_b]

    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # Same rank — prefer larger crowding distance
    if a.crowding_distance >= b.crowding_distance:
        return a
    return b
