"""Material property database and field-dependent reluctivity for the EM solver.

All permeability values use SI units (H/m).  BH curves are stored as lists of
(H [A/m], B [T]) pairs; the curve must start at (0, 0) and be monotonically
increasing in both columns.

Reluctivity ν = H/B  [A·m/Wb = m/H].
"""

from __future__ import annotations

import numpy as np

MU_0: float = 4.0 * np.pi * 1e-7  # H/m


# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------

MATERIAL_DB: dict[str, dict] = {
    # ------------------------------------------------------------------
    # M250-35A — Cold-rolled grain-oriented silicon steel, 0.35 mm lamination
    # IEC 60404-8-4 typical values.  BH data representative of M250-35A.
    # ------------------------------------------------------------------
    "M250-35A": {
        "density_kg_m3": 7650.0,
        "electrical_conductivity_S_m": 2.0e6,       # 0.5 µΩ·m
        "specific_heat_J_kgK": 490.0,
        "thermal_conductivity_W_mK": 30.0,
        "steinmetz_kh": 143.0,                       # hysteresis loss coefficient
        "steinmetz_ke": 0.53,                        # eddy-current loss coefficient
        "steinmetz_alpha": 2.0,                      # Steinmetz flux-density exponent
        "relative_permeability": 8000.0,             # nominal (linear regime)
        "mu_r": 1.0,                                 # reserved — nonlinear via BH
        # BH curve: (H [A/m], B [T]) — 14 points, CCW from origin
        "BH_curve": [
            (0.0,    0.0),
            (50.0,   0.3),
            (100.0,  0.7),
            (150.0,  1.0),
            (200.0,  1.2),
            (300.0,  1.35),
            (400.0,  1.45),
            (600.0,  1.55),
            (800.0,  1.62),
            (1200.0, 1.72),
            (2000.0, 1.82),
            (4000.0, 1.95),
            (8000.0, 2.05),
            (16000.0, 2.15),
        ],
    },

    # ------------------------------------------------------------------
    # M330-50A — Non-oriented silicon steel, 0.50 mm lamination
    # Slightly lower peak B, higher losses — typical for larger machines.
    # ------------------------------------------------------------------
    "M330-50A": {
        "density_kg_m3": 7700.0,
        "electrical_conductivity_S_m": 2.04e6,      # ~0.49 µΩ·m
        "specific_heat_J_kgK": 490.0,
        "thermal_conductivity_W_mK": 28.0,
        "steinmetz_kh": 190.0,
        "steinmetz_ke": 0.75,
        "steinmetz_alpha": 2.0,
        "relative_permeability": 6000.0,
        "mu_r": 1.0,
        "BH_curve": [
            (0.0,    0.0),
            (60.0,   0.28),
            (120.0,  0.65),
            (200.0,  0.95),
            (300.0,  1.15),
            (450.0,  1.30),
            (700.0,  1.42),
            (1000.0, 1.52),
            (1500.0, 1.62),
            (2500.0, 1.72),
            (4000.0, 1.82),
            (7000.0, 1.93),
            (12000.0, 2.02),
            (20000.0, 2.10),
        ],
    },

    # ------------------------------------------------------------------
    # copper_class_F — Winding conductor, Class F insulation (155 °C / 428 K)
    # ------------------------------------------------------------------
    "copper_class_F": {
        "density_kg_m3": 8960.0,
        "electrical_conductivity_S_m": 5.8e7,       # at 20 °C (293.15 K)
        "specific_heat_J_kgK": 385.0,
        "thermal_conductivity_W_mK": 400.0,
        "resistivity_temperature_coefficient": 0.00393,  # K⁻¹
        "max_operating_temperature_K": 428.15,       # 155 °C
        "relative_permeability": 1.0,
        "mu_r": 1.0,
        # No ferromagnetic BH curve — reluctivity == 1/μ₀
        "BH_curve": None,
    },

    # ------------------------------------------------------------------
    # air — Air gap and all non-conducting / non-magnetic regions
    # ------------------------------------------------------------------
    "air": {
        "density_kg_m3": 1.204,
        "electrical_conductivity_S_m": 0.0,
        "specific_heat_J_kgK": 1005.0,
        "thermal_conductivity_W_mK": 0.025,
        "relative_permeability": 1.0,
        "mu_r": 1.0,
        "BH_curve": None,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_material_properties(material_id: str, config: dict | None = None) -> dict:
    """Return the property dict for *material_id*.

    Parameters
    ----------
    material_id:
        Key into :data:`MATERIAL_DB`.
    config:
        Optional pipeline config dict (reserved for future overrides; not used
        currently).

    Raises
    ------
    KeyError
        If *material_id* is not present in :data:`MATERIAL_DB`.
    """
    if material_id not in MATERIAL_DB:
        available = ", ".join(sorted(MATERIAL_DB.keys()))
        raise KeyError(
            f"Material '{material_id}' not found in MATERIAL_DB. "
            f"Available materials: {available}"
        )
    return MATERIAL_DB[material_id]


def interpolate_reluctivity(
    B_magnitude: float | np.ndarray,
    material_id: str,
) -> float | np.ndarray:
    """Compute reluctivity ν = H/B  [A·m/Wb] at the given flux density.

    For non-magnetic materials (no BH curve, or μ_r = 1 only), returns 1/μ₀.
    For iron materials, performs linear interpolation on the stored BH curve.

    Special case B = 0: use the initial differential permeability derived from
    the first linear segment of the BH curve to avoid division by zero.

    Parameters
    ----------
    B_magnitude:
        Scalar or array of |B| values [T].  Must be ≥ 0.
    material_id:
        Key into :data:`MATERIAL_DB`.

    Returns
    -------
    float or np.ndarray
        Reluctivity [A·m/Wb], same shape as *B_magnitude*.
    """
    props = get_material_properties(material_id)
    nu_free_space = 1.0 / MU_0  # ≈ 795 774 A·m/Wb

    bh = props.get("BH_curve")
    if bh is None:
        # Non-magnetic: air or copper
        scalar_in = np.ndim(B_magnitude) == 0
        result = np.full(np.shape(B_magnitude), nu_free_space)
        return float(result) if scalar_in else result

    # Build H and B arrays (skip the (0,0) sentinel — handled separately)
    H_arr = np.array([pt[0] for pt in bh], dtype=float)
    B_arr = np.array([pt[1] for pt in bh], dtype=float)

    # Initial linear permeability from the first non-zero segment
    # mu_initial = dB/dH at B→0
    idx0 = np.searchsorted(B_arr, 0.0, side="right")
    if idx0 < len(B_arr):
        mu_initial = B_arr[idx0] / H_arr[idx0] if H_arr[idx0] > 0 else MU_0
    else:
        mu_initial = MU_0
    nu_initial = 1.0 / mu_initial

    scalar_in = np.ndim(B_magnitude) == 0
    B_mag = np.atleast_1d(np.asarray(B_magnitude, dtype=float))
    nu_out = np.empty_like(B_mag)

    # Mask: B == 0 → use initial reluctivity; B > 0 → interpolate H/B
    zero_mask = B_mag <= 0.0
    nonzero_mask = ~zero_mask

    nu_out[zero_mask] = nu_initial

    if np.any(nonzero_mask):
        B_nz = B_mag[nonzero_mask]
        # Interpolate H from the BH table at each B value (extrapolate flat beyond table)
        H_interp = np.interp(B_nz, B_arr, H_arr)
        nu_out[nonzero_mask] = H_interp / B_nz

    if scalar_in:
        return float(nu_out[0])
    return nu_out
