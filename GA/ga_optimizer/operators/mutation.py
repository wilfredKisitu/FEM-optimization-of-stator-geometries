"""mutation.py — Polynomial mutation for real-valued chromosomes.

Polynomial mutation perturbs each gene with probability p_mutation (default
1/N_GENES).  The perturbation distribution is polynomial and boundary-aware:
genes close to a bound receive smaller perturbations on that side.

Reference: Deb, K. (2001). Multi-Objective Optimization using Evolutionary
Algorithms. Wiley, pp. 114–118.
"""

from __future__ import annotations

import numpy as np

from ..chromosome import LOWER_BOUNDS, UPPER_BOUNDS, N_GENES
from .repair import clamp


def polynomial_mutation(
    genes: np.ndarray,
    eta_m: float,
    rng: np.random.Generator,
    p_mutation: float | None = None,
) -> np.ndarray:
    """Apply polynomial mutation to a gene vector.

    Parameters
    ----------
    genes:
        (N_GENES,) float array — original gene vector.
    eta_m:
        Distribution index (typically 20).  Higher values produce smaller
        perturbations (more exploitative behaviour).
    rng:
        NumPy random Generator.
    p_mutation:
        Per-gene mutation probability.  Default (``None``) uses the NSGA-II
        standard of ``1/N_GENES``.

    Returns
    -------
    np.ndarray
        (N_GENES,) float array — mutated gene vector, clamped to bounds.
    """
    if p_mutation is None:
        p_mutation = 1.0 / N_GENES

    mutant = genes.copy()

    for i in range(N_GENES):
        if rng.random() > p_mutation:
            continue

        lb = LOWER_BOUNDS[i]
        ub = UPPER_BOUNDS[i]
        x  = genes[i]

        span = ub - lb
        if span < 1e-14:
            continue

        delta1 = (x - lb) / span   # normalised distance to lower bound
        delta2 = (ub - x) / span   # normalised distance to upper bound
        u      = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if u < 0.5:
            xy      = 1.0 - delta1
            val     = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
            delta_q = val ** mut_pow - 1.0
        else:
            xy      = 1.0 - delta2
            val     = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
            delta_q = 1.0 - val ** mut_pow

        mutant[i] = x + delta_q * span

    return clamp(mutant)
