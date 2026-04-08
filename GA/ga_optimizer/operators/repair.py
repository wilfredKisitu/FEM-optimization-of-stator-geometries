"""repair.py — Hard-clamp genes back inside their bounds after variation."""

from __future__ import annotations

import numpy as np

from ..chromosome import LOWER_BOUNDS, UPPER_BOUNDS


def clamp(genes: np.ndarray) -> np.ndarray:
    """Clamp every gene to [LOWER_BOUNDS[i], UPPER_BOUNDS[i]].

    Always call after SBX crossover and polynomial mutation to guarantee
    that offspring stay within the search space.

    Parameters
    ----------
    genes:
        (N_GENES,) float array — possibly out-of-bounds after variation.

    Returns
    -------
    np.ndarray
        (N_GENES,) float array with all values within bounds.
    """
    return np.clip(genes, LOWER_BOUNDS, UPPER_BOUNDS)
