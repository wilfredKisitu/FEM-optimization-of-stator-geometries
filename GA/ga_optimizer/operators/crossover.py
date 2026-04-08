"""crossover.py — Simulated Binary Crossover (SBX) for real-valued chromosomes.

SBX mimics the behaviour of single-point crossover on binary strings while
operating on real-valued genes.  The spread of offspring relative to parents
is controlled by the distribution index ``eta_c`` (higher → offspring closer
to parents).

Reference: Deb, K. & Agrawal, R. B. (1995). Simulated binary crossover for
continuous search space. Complex Systems, 9(2), 115–148.
"""

from __future__ import annotations

import numpy as np

from ..chromosome import LOWER_BOUNDS, UPPER_BOUNDS, N_GENES
from .repair import clamp


def sbx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    eta_c: float,
    rng: np.random.Generator,
    p_crossover: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SBX to two parents and return two offspring.

    Each gene is crossed independently with probability 0.5.  Genes where
    parents are identical (|x2 − x1| < 1e-14) are copied unchanged.

    Parameters
    ----------
    parent_a, parent_b:
        (N_GENES,) float arrays — gene vectors of two parents.
    eta_c:
        Distribution index (typically 2–20).  Higher values keep offspring
        closer to the parents (lower exploration).
    rng:
        NumPy random Generator.
    p_crossover:
        Probability that crossover is applied at all.  Default is 1.0
        (always cross).  Set < 1 to allow some parent copies to pass through.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two offspring gene vectors, clamped to [LOWER_BOUNDS, UPPER_BOUNDS].
    """
    if rng.random() > p_crossover:
        return parent_a.copy(), parent_b.copy()

    child_a = parent_a.copy()
    child_b = parent_b.copy()

    for i in range(N_GENES):
        # Each gene crossed independently with 50% probability
        if rng.random() > 0.5:
            continue

        x1 = min(parent_a[i], parent_b[i])
        x2 = max(parent_a[i], parent_b[i])

        if abs(x2 - x1) < 1e-14:
            continue   # parents identical on this gene — no crossover effect

        lb = LOWER_BOUNDS[i]
        ub = UPPER_BOUNDS[i]
        u  = rng.random()

        beta_q = _beta_q(u, x1, x2, lb, ub, eta_c)

        child_a[i] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
        child_b[i] = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

    return clamp(child_a), clamp(child_b)


# ---------------------------------------------------------------------------
# Internal SBX helper
# ---------------------------------------------------------------------------

def _beta_q(u: float, x1: float, x2: float,
            lb: float, ub: float, eta_c: float) -> float:
    """Compute the spread factor β_q for SBX.

    The calculation is boundary-aware: it accounts for how close x1 is to
    the lower bound (left side) or x2 is to the upper bound (right side).
    """
    # alpha from the left-side boundary constraint
    beta  = 1.0 + (2.0 * (x1 - lb) / max(x2 - x1, 1e-14))
    alpha = 2.0 - beta ** (-(eta_c + 1.0))

    if u <= 1.0 / alpha:
        return (u * alpha) ** (1.0 / (eta_c + 1.0))
    else:
        return (1.0 / max(2.0 - u * alpha, 1e-300)) ** (1.0 / (eta_c + 1.0))
