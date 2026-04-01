"""Map electromagnetic loss density maps to volumetric heat sources on the FEA mesh.

The EM stage produces two loss maps (both in W/m³):

  loss_density_map         — iron (core) losses, shape (n_elems,) or scalar
  copper_loss_density_map  — copper (winding) losses; may be:
                               * np.ndarray  of shape (n_elems,)
                               * dict with key "spatial_W_per_m3" → float or ndarray
                               * float / scalar

The mapping rules are:
  - region tag == stator_core → iron loss density
  - region tag == winding     → copper loss density
  - everything else           → 0 W/m³

The axial_length parameter is accepted for API consistency with callers that
compute total power; it is not applied here because the solver works in per-
unit-volume quantities and accounts for axial_length when computing nodal RHS
contributions (area × axial_length × q_vol / 3).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def map_em_losses_to_heat_sources(
    mesh,
    em_results: dict,
    stator,
    axial_length: float,
) -> np.ndarray:
    """Return (n_elems,) volumetric heat source array [W/m³] for the thermal solve.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance — used for ``n_elements`` and
        ``region_ids``.
    em_results:
        Result dict from the EM stage.  Recognised keys:

        ``"loss_density_map"``
            Iron loss density [W/m³], shape (n_elems,) or broadcastable scalar.
        ``"copper_loss_density_map"``
            Copper loss density [W/m³]; may be ndarray, dict with key
            ``"spatial_W_per_m3"``, float, or ``None``.
        ``"domain"``
            :class:`FEAMesh` — ignored (caller passes *mesh* directly).

    stator:
        :class:`StatorMeshInput` instance — provides ``region_tags``.
    axial_length:
        Axial stack length [m].  Accepted for API consistency; not applied
        here (volumetric quantities are independent of axial_length).

    Returns
    -------
    np.ndarray
        (n_elems,) float array of volumetric heat source densities [W/m³].
    """
    n_elems = mesh.n_elements
    region_ids = mesh.region_ids                 # (n_elems,)
    q_vol = np.zeros(n_elems, dtype=float)

    # -----------------------------------------------------------------
    # Region tags
    # -----------------------------------------------------------------
    core_tag = stator.region_tags.get("stator_core", 1)
    winding_tag = stator.region_tags.get("winding", 2)

    core_mask = region_ids == core_tag           # (n_elems,) bool
    winding_mask = region_ids == winding_tag

    # -----------------------------------------------------------------
    # Iron (core) loss density
    # -----------------------------------------------------------------
    raw_iron = em_results.get("loss_density_map", 0.0)
    iron_density = _to_elem_array(raw_iron, n_elems)
    if np.any(core_mask):
        q_vol[core_mask] = iron_density[core_mask] if iron_density.shape == (n_elems,) \
            else iron_density
        # Clamp negatives (can arise from numerical noise in the EM stage)
        q_vol[core_mask] = np.maximum(q_vol[core_mask], 0.0)

    logger.debug(
        "Iron heat source: %.3e W/m³ (mean over %d core elements)",
        float(np.mean(q_vol[core_mask])) if np.any(core_mask) else 0.0,
        int(np.sum(core_mask)),
    )

    # -----------------------------------------------------------------
    # Copper (winding) loss density
    # -----------------------------------------------------------------
    raw_cu = em_results.get("copper_loss_density_map", 0.0)
    cu_density = _extract_copper_density(raw_cu, n_elems)
    if np.any(winding_mask):
        if cu_density.shape == (n_elems,):
            q_vol[winding_mask] = cu_density[winding_mask]
        else:
            # Scalar broadcast
            q_vol[winding_mask] = cu_density.flat[0] if cu_density.size > 0 else 0.0
        q_vol[winding_mask] = np.maximum(q_vol[winding_mask], 0.0)

    logger.debug(
        "Copper heat source: %.3e W/m³ (mean over %d winding elements)",
        float(np.mean(q_vol[winding_mask])) if np.any(winding_mask) else 0.0,
        int(np.sum(winding_mask)),
    )

    return q_vol


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_elem_array(value, n_elems: int) -> np.ndarray:
    """Convert a scalar, list, or ndarray to a float array of length n_elems.

    If *value* is already an ndarray of the correct length it is returned as-is
    (no copy).  Scalars are broadcast to a constant array.
    """
    if isinstance(value, np.ndarray):
        if value.shape == (n_elems,):
            return value.astype(float, copy=False)
        # Single-element array or other shape — treat as scalar
        return np.full(n_elems, float(value.flat[0]), dtype=float)
    # Python scalar or list
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        logger.warning("Cannot convert loss map value '%s' to float — using 0.", value)
        scalar = 0.0
    return np.full(n_elems, scalar, dtype=float)


def _extract_copper_density(raw, n_elems: int) -> np.ndarray:
    """Normalise the copper loss density from various formats.

    Accepted formats
    ----------------
    * ``np.ndarray`` of shape (n_elems,) — used directly.
    * ``dict`` with key ``"spatial_W_per_m3"`` → scalar or ndarray.
    * ``float`` / ``int`` — broadcast to all winding elements.
    * ``None`` — returns zeros.
    """
    if raw is None:
        return np.zeros(n_elems, dtype=float)

    if isinstance(raw, dict):
        inner = raw.get("spatial_W_per_m3", 0.0)
        return _to_elem_array(inner, n_elems)

    return _to_elem_array(raw, n_elems)
