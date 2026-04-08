"""population.py — Individual dataclass and population initialisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .chromosome import random_individual, N_GENES
from .objectives import ObjectiveVector


@dataclass
class Individual:
    """One candidate stator design — a gene vector plus evaluation metadata."""

    genes: np.ndarray                         # shape (N_GENES,)
    objectives: Optional[ObjectiveVector] = None
    rank: int = -1                            # NSGA-II non-domination rank (0 = Pareto front)
    crowding_distance: float = 0.0
    stator_id: Optional[str] = None          # set by evaluator after FEA
    evaluated: bool = False

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def copy(self) -> "Individual":
        return Individual(
            genes=self.genes.copy(),
            objectives=self.objectives,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            stator_id=self.stator_id,
            evaluated=self.evaluated,
        )


# Type alias — a list of Individuals
Population = list[Individual]


def initialise_population(
    pop_size: int,
    rng: np.random.Generator,
    seed_designs: Optional[list[np.ndarray]] = None,
) -> Population:
    """Create the initial population.

    Parameters
    ----------
    pop_size:
        Target population size.
    rng:
        NumPy random Generator (for reproducibility, pass one seeded instance).
    seed_designs:
        Optional list of hand-crafted gene vectors (``np.ndarray`` of shape
        ``(N_GENES,)``).  These are inserted first; the remainder of the
        population is filled with random individuals.

    Returns
    -------
    Population
        List of ``pop_size`` un-evaluated :class:`Individual` objects.
    """
    pop: Population = []

    if seed_designs:
        for genes in seed_designs[:pop_size]:
            arr = np.asarray(genes, dtype=float)
            if arr.shape != (N_GENES,):
                raise ValueError(
                    f"Seed design has shape {arr.shape}; expected ({N_GENES},)"
                )
            pop.append(Individual(genes=arr.copy()))

    n_random = pop_size - len(pop)
    for _ in range(n_random):
        pop.append(Individual(genes=random_individual(rng)))

    return pop
