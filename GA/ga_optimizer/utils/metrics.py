"""utils/metrics.py — Multi-objective quality indicators.

Implements:
- Hypervolume (HV) indicator using a pure-Python WFG sweep algorithm
- Inverted Generational Distance (IGD) for ZDT benchmark validation
- Generational Distance (GD)
- Spread / Δ metric

All functions accept (n_solutions, n_objectives) numpy arrays.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..population import Individual


# ---------------------------------------------------------------------------
# Hypervolume (WFG sweep — exact, works for 2 and 3 objectives)
# ---------------------------------------------------------------------------

def compute_hypervolume(
    members: "list[Individual]",
    reference_point: list[float],
) -> float:
    """Compute the hypervolume indicator of the Pareto archive.

    Hypervolume measures the volume of objective-space dominated by the
    archive relative to a reference point that is worse than all feasible
    solutions.  A larger HV indicates better coverage of the Pareto front.

    Uses the WFG (Walking Fish Group) recursive algorithm (exact for any
    number of objectives, efficient for 2–3).

    Parameters
    ----------
    members:
        List of archive :class:`Individual` objects (all must be feasible).
    reference_point:
        Reference point in objective space — must be dominated by every
        Pareto-optimal solution.  Length must equal the number of objectives.

    Returns
    -------
    float
        Hypervolume value.  Returns 0.0 if the archive is empty or all
        members are infeasible.
    """
    if not members:
        return 0.0

    pts = np.array([m.objectives.objective_array for m in members
                    if m.objectives is not None and m.objectives.is_feasible])
    if pts.shape[0] == 0:
        return 0.0

    ref = np.asarray(reference_point, dtype=float)
    return _wfg_hypervolume(pts, ref)


def _wfg_hypervolume(pts: np.ndarray, ref: np.ndarray) -> float:
    """WFG recursive hypervolume.  pts shape: (n, m), ref shape: (m,)."""
    # Remove points that don't dominate the reference point
    dominated = np.all(pts < ref, axis=1)
    pts = pts[dominated]
    if pts.shape[0] == 0:
        return 0.0

    m = pts.shape[1]

    if m == 1:
        return float(ref[0] - pts[:, 0].min())

    if m == 2:
        return _hv_2d(pts, ref)

    # General case: recursive WFG
    return _hv_nd(pts, ref)


def _hv_2d(pts: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume for 2-D objective space (left-to-right sweep)."""
    # Sort by first objective ascending
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    hv = 0.0
    y_min = ref[1]   # running minimum y seen so far (best y frontier)
    n = len(pts)
    for i in range(n):
        y_min = min(y_min, pts[i, 1])
        x_next = pts[i + 1, 0] if i < n - 1 else ref[0]
        hv += (x_next - pts[i, 0]) * (ref[1] - y_min)
    return hv


def _hv_nd(pts: np.ndarray, ref: np.ndarray) -> float:
    """WFG recursive algorithm for n ≥ 3 dimensions.

    Sweep from the reference point inward (worst z to best z).  Sort by last
    objective DESCENDING so the first slice spans from the worst z value to
    ref[-1], and subsequent slices fill the gap between consecutive z-levels.
    """
    # Sort by last objective DESCENDING (worst/largest first → closest to ref)
    order = np.argsort(-pts[:, -1])
    pts = pts[order]

    hv = 0.0
    for i in range(len(pts)):
        # Extent of this z-slice
        if i == 0:
            extent = ref[-1] - pts[i, -1]
        else:
            extent = pts[i - 1, -1] - pts[i, -1]

        if extent <= 0:
            continue

        # Project the first (i+1) points onto (n-1) dimensions, limited by pts[i]
        sub_pts = _limit(pts[:i + 1], pts[i])
        sub_hv = _wfg_hypervolume(sub_pts[:, :-1], ref[:-1])
        hv += sub_hv * extent

    return hv


def _limit(pts: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Component-wise worst (max) of pts and limit."""
    return np.maximum(pts, limit)


# ---------------------------------------------------------------------------
# IGD — Inverted Generational Distance
# ---------------------------------------------------------------------------

def compute_igd(
    approx_front: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Inverted Generational Distance.

    IGD is the average minimum distance from each point on the reference
    Pareto front to the nearest point in the approximation front.  Lower
    is better; IGD = 0 means the approximation front is a superset of the
    reference.

    Parameters
    ----------
    approx_front:
        (n, m) array — the approximated Pareto front.
    reference_front:
        (r, m) array — the true/reference Pareto front.

    Returns
    -------
    float
        IGD value.  Returns ``inf`` if either input is empty.
    """
    if approx_front.shape[0] == 0 or reference_front.shape[0] == 0:
        return float("inf")

    # For each point in the reference front, find the minimum distance
    # to any point in the approx front
    distances = []
    for ref_pt in reference_front:
        dists = np.linalg.norm(approx_front - ref_pt, axis=1)
        distances.append(dists.min())

    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# GD — Generational Distance
# ---------------------------------------------------------------------------

def compute_gd(
    approx_front: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Generational Distance (average distance from approx to reference)."""
    if approx_front.shape[0] == 0 or reference_front.shape[0] == 0:
        return float("inf")

    distances = []
    for pt in approx_front:
        dists = np.linalg.norm(reference_front - pt, axis=1)
        distances.append(dists.min())

    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

def compute_spread(
    approx_front: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Δ (Spread) metric — measures diversity of the approximate Pareto front.

    Returns a value in [0, ∞).  Closer to 0 means better spread.

    Parameters
    ----------
    approx_front:
        (n, m) sorted approximate Pareto front.
    reference_front:
        (r, m) reference Pareto front (used to find extreme distances).
    """
    if approx_front.shape[0] < 2:
        return float("inf")

    n = approx_front.shape[0]

    # Consecutive distances in the approximate front (sorted by first objective)
    order = np.argsort(approx_front[:, 0])
    sorted_pts = approx_front[order]
    d_i = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)   # (n-1,)
    d_mean = d_i.mean()

    # Extreme distances to reference endpoints
    d_f = np.linalg.norm(reference_front[0] - sorted_pts[0])
    d_l = np.linalg.norm(reference_front[-1] - sorted_pts[-1])

    numerator   = d_f + d_l + np.abs(d_i - d_mean).sum()
    denominator = d_f + d_l + (n - 1) * d_mean

    return float(numerator / max(denominator, 1e-12))
