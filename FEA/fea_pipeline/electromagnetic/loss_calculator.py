"""Iron and copper loss calculators for the 2-D magnetostatic FEM solver.

Iron losses use the classical Steinmetz model:
    P [W/m³] = kh * f * B^alpha  +  ke * f² * B²

Copper losses use the J²/σ resistive model with optional temperature correction.
"""

from __future__ import annotations

import numpy as np

# Copper conductivity at 20 °C (293.15 K)
_SIGMA_CU_REF: float = 5.8e7        # S/m
_T_REF_K: float = 293.15            # K
_ALPHA_CU: float = 0.00393          # temperature coefficient K⁻¹


# ---------------------------------------------------------------------------
# Steinmetz iron-loss model
# ---------------------------------------------------------------------------

def steinmetz_iron_loss(
    B_peak: float,
    freq_Hz: float,
    kh: float,
    ke: float,
    alpha: float,
) -> float:
    """Per-volume iron loss density [W/m³] using the Steinmetz model.

    Parameters
    ----------
    B_peak:
        Peak flux density [T] in the element.
    freq_Hz:
        Electrical frequency [Hz].
    kh:
        Hysteresis loss coefficient [W·s/m³/T^alpha].
    ke:
        Eddy-current loss coefficient [W·s²/m³/T²].
    alpha:
        Steinmetz flux-density exponent (typically 1.6 – 2.0).

    Returns
    -------
    float
        Total iron loss density  P_total = P_hyst + P_eddy  [W/m³].
    """
    if freq_Hz == 0.0 or B_peak == 0.0:
        return 0.0
    p_hyst = kh * freq_Hz * (B_peak ** alpha)
    p_eddy = ke * (freq_Hz ** 2) * (B_peak ** 2)
    return float(p_hyst + p_eddy)


# ---------------------------------------------------------------------------
# Element-wise iron loss
# ---------------------------------------------------------------------------

def compute_iron_losses(
    B_elem: np.ndarray,
    region_ids: np.ndarray,
    areas: np.ndarray,
    axial_length: float,
    freq_Hz: float,
    material_id: str,
) -> dict:
    """Compute iron losses for all iron-region elements.

    Parameters
    ----------
    B_elem:
        (n_elems,) array of per-element |B| [T].
    region_ids:
        (n_elems,) integer region tag for each element.
    areas:
        (n_elems,) unsigned element areas [m²].
    axial_length:
        Stack length [m].
    freq_Hz:
        Electrical frequency [Hz].
    material_id:
        Material key used to look up Steinmetz coefficients from
        :data:`material_library.MATERIAL_DB`.  If the material has no
        Steinmetz coefficients (e.g. "air", "copper_class_F") the
        function returns zeros for all elements.

    Returns
    -------
    dict with keys:
        "total"            – total iron loss [W]
        "eddy"             – total eddy-current contribution [W]
        "hysteresis"       – total hysteresis contribution [W]
        "spatial_W_per_m3" – (n_elems,) per-element loss density [W/m³]
    """
    # Import here to avoid circular imports at module level
    from .material_library import MATERIAL_DB

    props = MATERIAL_DB.get(material_id, {})
    kh = props.get("steinmetz_kh", None)
    ke = props.get("steinmetz_ke", None)
    alpha = props.get("steinmetz_alpha", 2.0)

    n_elems = len(B_elem)
    spatial = np.zeros(n_elems, dtype=float)
    eddy_spatial = np.zeros(n_elems, dtype=float)
    hyst_spatial = np.zeros(n_elems, dtype=float)

    if kh is None or ke is None or freq_Hz == 0.0:
        return {
            "total": 0.0,
            "eddy": 0.0,
            "hysteresis": 0.0,
            "spatial_W_per_m3": spatial,
        }

    # Only accumulate losses in iron elements (those tagged with this material).
    # The caller is responsible for passing region_ids that match the iron region
    # tag.  Here we treat ALL elements as iron (the caller filters by material);
    # but to be safe we skip elements with B ≈ 0.
    for e in range(n_elems):
        B = float(B_elem[e])
        if B == 0.0:
            continue
        p_hyst_e = kh * freq_Hz * (B ** alpha)
        p_eddy_e = ke * (freq_Hz ** 2) * (B ** 2)
        hyst_spatial[e] = p_hyst_e
        eddy_spatial[e] = p_eddy_e
        spatial[e] = p_hyst_e + p_eddy_e

    volume_elem = areas * axial_length          # (n_elems,) [m³]
    total_eddy = float(np.dot(eddy_spatial, volume_elem))
    total_hyst = float(np.dot(hyst_spatial, volume_elem))

    return {
        "total": total_eddy + total_hyst,
        "eddy": total_eddy,
        "hysteresis": total_hyst,
        "spatial_W_per_m3": spatial,
    }


# ---------------------------------------------------------------------------
# Copper losses
# ---------------------------------------------------------------------------

def compute_copper_losses(
    stator,
    B_avg_winding: float,
    config: dict,
) -> dict:
    """Estimate total copper (I²R) losses and uniform winding loss density.

    Parameters
    ----------
    stator:
        :class:`StatorMeshInput` instance providing geometry and operating
        point.
    B_avg_winding:
        Average flux density in the winding region [T].  Not used directly in
        the resistive model but kept for API completeness (could be used for AC
        resistance correction).
    config:
        Pipeline config dict.  If ``"copper_temperature_K"`` is present the
        conductivity is corrected for temperature.

    Returns
    -------
    dict with keys:
        "total"            – total copper loss [W]
        "spatial_W_per_m3" – uniform J²/σ loss density in the winding [W/m³]
    """
    # --- temperature-corrected conductivity -----------------------------------
    T_K = config.get("copper_temperature_K", _T_REF_K)
    sigma_cu = _SIGMA_CU_REF / (1.0 + _ALPHA_CU * (T_K - _T_REF_K))

    # --- geometry -----------------------------------------------------------
    r_outer = stator.outer_diameter / 2.0   # [m]
    r_inner = stator.inner_diameter / 2.0   # [m]
    n_slots = stator.num_slots
    slot_depth = stator.slot_depth          # [m]
    tooth_width = stator.tooth_width        # [m]
    fill_factor = stator.fill_factor
    axial_length = stator.axial_length      # [m]
    I_rms = stator.rated_current_rms        # [A]
    n_cond = stator.conductors_per_slot     # conductors per slot

    # Slot cross-sectional area estimate:  slot_depth × tooth_width
    # (tooth_width is a reasonable proxy for slot width in many designs).
    slot_area = slot_depth * tooth_width    # [m²]

    # Effective conductor area per slot (only fill_factor fraction is copper)
    conductor_area = fill_factor * slot_area  # [m²]
    if conductor_area <= 0.0:
        return {"total": 0.0, "spatial_W_per_m3": 0.0}

    # RMS current density in the conductors
    J_rms = (I_rms * n_cond) / conductor_area  # [A/m²]

    # Total winding volume (all slots, only copper fraction)
    winding_volume = fill_factor * slot_area * n_slots * axial_length  # [m³]

    # P_cu = J² / σ · V_winding
    spatial_density = (J_rms ** 2) / sigma_cu          # [W/m³]
    total_cu = spatial_density * winding_volume         # [W]

    return {
        "total": float(total_cu),
        "spatial_W_per_m3": float(spatial_density),
    }
