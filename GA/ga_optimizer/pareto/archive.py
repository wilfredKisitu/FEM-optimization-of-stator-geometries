"""pareto/archive.py — Persistent Pareto archive maintained across generations.

The archive stores every non-dominated feasible individual encountered during
the entire run — not just the current generation's Pareto front.  It is the
authoritative source for the final output and for hypervolume computation.

Update rule:
  A candidate is added if no current archive member dominates it.
  Whenever a new member is added, all archive members that it dominates
  are pruned.
"""

from __future__ import annotations

import numpy as np

from ..population import Individual
from .nsga2 import dominates


class ParetoArchive:
    """Maintains the all-time non-dominated feasible front.

    Thread-safety note: the archive is **not** thread-safe.  It should only
    be updated from the main GA loop (not from parallel worker processes).
    """

    def __init__(self) -> None:
        self._members: list[Individual] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, candidates: list[Individual]) -> int:
        """Add non-dominated feasible candidates; prune dominated members.

        Parameters
        ----------
        candidates:
            List of evaluated individuals (typically the current Pareto
            front of the combined population).

        Returns
        -------
        int
            Number of new entries added to the archive.
        """
        added = 0
        for candidate in candidates:
            if candidate.objectives is None or not candidate.objectives.is_feasible:
                continue
            if self._is_non_dominated(candidate):
                # Prune any existing archive members dominated by the candidate
                c_obj = candidate.objectives.objective_array
                self._members = [
                    m for m in self._members
                    if not dominates(c_obj, m.objectives.objective_array)
                ]
                self._members.append(candidate)
                added += 1
        return added

    def _is_non_dominated(self, candidate: Individual) -> bool:
        """Return True iff no archive member dominates *candidate*."""
        c_obj = candidate.objectives.objective_array
        return not any(
            dominates(m.objectives.objective_array, c_obj)
            for m in self._members
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def members(self) -> list[Individual]:
        """Read-only view of the current archive members."""
        return list(self._members)

    @property
    def size(self) -> int:
        """Number of solutions currently in the archive."""
        return len(self._members)

    def objective_matrix(self) -> np.ndarray:
        """Return (archive_size, n_objectives) array of objective values.

        Returns an empty (0, 3) array when the archive is empty.
        """
        if not self._members:
            return np.empty((0, 3))
        return np.array([m.objectives.objective_array for m in self._members])
