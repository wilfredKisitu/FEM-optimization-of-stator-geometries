"""Post-process the thermal FEA solution.

Provides scalar metrics, hot-spot identification, and area-weighted averaging
over user-specified mesh regions.  All functions operate on nodal temperature
arrays and the :class:`FEAMesh` object — no I/O is performed here.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity extractor (API consistency with EM post-processor pattern)
# ---------------------------------------------------------------------------

def extract_temperature_field(T_nodal: np.ndarray) -> np.ndarray:
    """Return the nodal temperature field unchanged.

    This function exists for API symmetry with the EM post-processor and
    allows downstream stages to call ``extract_temperature_field`` without
    needing to know whether additional processing is involved.

    Parameters
    ----------
    T_nodal:
        (n_nodes,) float array of nodal temperatures [K].

    Returns
    -------
    np.ndarray
        The input array unchanged.
    """
    return T_nodal


# ---------------------------------------------------------------------------
# Hot-spot identification
# ---------------------------------------------------------------------------

def identify_hot_spots(
    T_nodal: np.ndarray,
    threshold_fraction: float = 0.95,
) -> dict:
    """Find nodes whose temperature exceeds a fraction of the peak temperature.

    Parameters
    ----------
    T_nodal:
        (n_nodes,) float array of nodal temperatures [K].
    threshold_fraction:
        Fraction of the peak temperature used as the hot-spot threshold.
        Default is 0.95, so nodes above 95 % of peak are flagged.

    Returns
    -------
    dict with keys:
        ``"n_hotspot_nodes"``    – int, number of nodes above threshold.
        ``"peak_T_K"``           – float, maximum nodal temperature [K].
        ``"threshold_T_K"``      – float, threshold temperature [K].
        ``"hotspot_node_indices"`` – list[int], sorted indices of hot-spot nodes.
    """
    if T_nodal.size == 0:
        return {
            "n_hotspot_nodes": 0,
            "peak_T_K": 0.0,
            "threshold_T_K": 0.0,
            "hotspot_node_indices": [],
        }

    peak_T = float(np.max(T_nodal))
    threshold_T = threshold_fraction * peak_T
    hotspot_mask = T_nodal >= threshold_T
    hotspot_indices = np.where(hotspot_mask)[0].tolist()

    logger.debug(
        "Hot-spot analysis: peak = %.2f K, threshold = %.2f K, "
        "hot-spot nodes = %d / %d",
        peak_T, threshold_T, len(hotspot_indices), len(T_nodal),
    )

    return {
        "n_hotspot_nodes": len(hotspot_indices),
        "peak_T_K": peak_T,
        "threshold_T_K": threshold_T,
        "hotspot_node_indices": hotspot_indices,
    }


# ---------------------------------------------------------------------------
# Winding average temperature
# ---------------------------------------------------------------------------

def compute_winding_average_temperature(
    T_nodal: np.ndarray,
    mesh,
    winding_region_id: int,
) -> float:
    """Compute the area-weighted average temperature in the winding region.

    Each element in the winding region contributes a weighted average of its
    three nodal temperatures, with the element area as the weight.  This is
    the standard FEM area-weighted average:

        T_avg = Σ_e ( area_e * mean(T_nodes_e) ) / Σ_e area_e

    Parameters
    ----------
    T_nodal:
        (n_nodes,) float array of nodal temperatures [K].
    mesh:
        :class:`FEAMesh` instance.
    winding_region_id:
        Integer region tag identifying winding elements.

    Returns
    -------
    float
        Area-weighted average winding temperature [K].  Returns 0.0 if no
        winding elements are found.
    """
    winding_mask = mesh.region_ids == winding_region_id
    if not np.any(winding_mask):
        logger.warning(
            "No winding elements with region_id = %d — returning 0 K.",
            winding_region_id,
        )
        return 0.0

    _, _, areas = mesh.gradient_operators()          # (n_elems,)
    winding_elems = mesh.elements[winding_mask]      # (n_wind, 3)
    winding_areas = areas[winding_mask]              # (n_wind,)

    # Mean nodal temperature per element: shape (n_wind,)
    T_elem_mean = T_nodal[winding_elems].mean(axis=1)

    total_weight = float(np.sum(winding_areas))
    if total_weight == 0.0:
        return 0.0

    T_avg = float(np.dot(T_elem_mean, winding_areas) / total_weight)

    logger.debug(
        "Winding average temperature: %.2f K (over %d elements, total area %.4e m²)",
        T_avg, int(np.sum(winding_mask)), total_weight,
    )

    return T_avg


# ---------------------------------------------------------------------------
# Temperature uniformity
# ---------------------------------------------------------------------------

def compute_temperature_uniformity(
    T_nodal: np.ndarray,
    mesh,
    region_id: int,
) -> float:
    """Return the area-weighted standard deviation of temperature in a region.

    A low value indicates a uniform temperature distribution; a high value
    indicates significant thermal gradients within the region.

    The area-weighted standard deviation is computed as:

        sigma = sqrt( Σ_e area_e * (T_e - T_avg)² / Σ_e area_e )

    where T_e is the mean nodal temperature of element e and T_avg is the
    area-weighted mean.

    Parameters
    ----------
    T_nodal:
        (n_nodes,) float array of nodal temperatures [K].
    mesh:
        :class:`FEAMesh` instance.
    region_id:
        Integer region tag of the region of interest.

    Returns
    -------
    float
        Area-weighted standard deviation of temperature [K].  Returns 0.0 if
        the region is empty or consists of a single element.
    """
    region_mask = mesh.region_ids == region_id
    if not np.any(region_mask):
        logger.debug(
            "No elements with region_id = %d — uniformity = 0 K.", region_id
        )
        return 0.0

    _, _, areas = mesh.gradient_operators()
    region_elems = mesh.elements[region_mask]        # (n_reg, 3)
    region_areas = areas[region_mask]                # (n_reg,)

    T_elem_mean = T_nodal[region_elems].mean(axis=1) # (n_reg,)

    total_weight = float(np.sum(region_areas))
    if total_weight == 0.0:
        return 0.0

    T_avg = float(np.dot(T_elem_mean, region_areas) / total_weight)
    variance = float(
        np.dot((T_elem_mean - T_avg) ** 2, region_areas) / total_weight
    )
    std_dev = float(np.sqrt(max(variance, 0.0)))

    logger.debug(
        "Temperature uniformity (region %d): std = %.4f K (mean = %.2f K)",
        region_id, std_dev, T_avg,
    )

    return std_dev
