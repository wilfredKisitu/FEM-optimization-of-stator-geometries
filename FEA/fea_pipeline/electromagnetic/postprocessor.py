"""Post-processing functions for the 2-D magnetostatic FEM solution.

Provides:
  - Nodal-to-element B-field extraction via shape-function gradients.
  - Torque computation via Maxwell stress tensor on the air-gap elements.
  - Cogging torque estimate.
  - Efficiency from torque, speed, and total losses.
"""

from __future__ import annotations

import numpy as np

MU_0: float = 4.0 * np.pi * 1e-7   # H/m


# ---------------------------------------------------------------------------
# Flux density extraction
# ---------------------------------------------------------------------------

def extract_flux_density(
    A_z_nodal: np.ndarray,
    mesh,
) -> dict:
    """Compute per-element B = curl(A_z ẑ) from nodal vector potential values.

    For a 2-D linear triangular element with nodes (i=0,1,2):

        grad(A_z)_x = Σ_i  A_i * b[e,i] / (2 * area[e])
        grad(A_z)_y = Σ_i  A_i * c[e,i] / (2 * area[e])

    The magnetic flux density follows from  B = ∇ × (A_z ẑ):

        B_x =  ∂A_z/∂y  =  grad(A_z)_y
        B_y = -∂A_z/∂x  = -grad(A_z)_x

    Parameters
    ----------
    A_z_nodal:
        (n_nodes,) nodal values of the magnetic vector potential [Wb/m].
    mesh:
        :class:`FEAMesh` instance.

    Returns
    -------
    dict with keys:
        "B_x"    – (n_elems,) x-component [T]
        "B_y"    – (n_elems,) y-component [T]
        "B_mag"  – (n_elems,) magnitude   [T]
    """
    b, c, area = mesh.gradient_operators()   # (n_elems, 3), (n_elems, 3), (n_elems,)

    # Gather nodal A_z for every element: shape (n_elems, 3)
    A_nodes = A_z_nodal[mesh.elements]       # (n_elems, 3)

    inv_2area = 1.0 / (2.0 * area)           # (n_elems,)

    # grad(A_z)_x = Σ_i A_i * b[e,i] / (2*area)
    grad_x = np.sum(A_nodes * b, axis=1) * inv_2area  # (n_elems,)

    # grad(A_z)_y = Σ_i A_i * c[e,i] / (2*area)
    grad_y = np.sum(A_nodes * c, axis=1) * inv_2area  # (n_elems,)

    B_x = grad_y          # ∂A_z/∂y
    B_y = -grad_x         # -∂A_z/∂x
    B_mag = np.sqrt(B_x ** 2 + B_y ** 2)

    return {"B_x": B_x, "B_y": B_y, "B_mag": B_mag}


# ---------------------------------------------------------------------------
# Torque via Maxwell stress tensor
# ---------------------------------------------------------------------------

def compute_torque(
    B_dict: dict,
    mesh,
    stator,
    air_gap_region_id: int,
) -> float:
    """Compute electromagnetic torque [N·m] using the Maxwell stress tensor.

    The integration is performed over air-gap elements.  For each element
    the centroid position (r, θ) is used to project B_x, B_y into radial
    and tangential components:

        B_r     =  B_x cos θ + B_y sin θ
        B_theta = -B_x sin θ + B_y cos θ

    The Maxwell stress contribution to torque is:

        dT = (1/μ₀) * B_r * B_theta * dA * L_axial * r_centroid

    Summing over all air-gap elements gives the net torque.

    Parameters
    ----------
    B_dict:
        Output of :func:`extract_flux_density`.
    mesh:
        :class:`FEAMesh` instance.
    stator:
        :class:`StatorMeshInput` instance (provides *axial_length*).
    air_gap_region_id:
        Integer region tag identifying air-gap elements.

    Returns
    -------
    float
        Net torque [N·m].
    """
    B_x = B_dict["B_x"]
    B_y = B_dict["B_y"]
    _, _, area = mesh.gradient_operators()

    air_gap_mask = mesh.region_ids == air_gap_region_id
    if not np.any(air_gap_mask):
        return 0.0

    # Element centroids
    centroids = mesh.element_centroids()    # (n_elems, 2)
    cx = centroids[air_gap_mask, 0]
    cy = centroids[air_gap_mask, 1]
    r_cent = np.sqrt(cx ** 2 + cy ** 2)
    theta = np.arctan2(cy, cx)

    Bx_ag = B_x[air_gap_mask]
    By_ag = B_y[air_gap_mask]
    area_ag = area[air_gap_mask]

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    B_r = Bx_ag * cos_t + By_ag * sin_t
    B_t = -Bx_ag * sin_t + By_ag * cos_t

    # dT = r * (1/μ₀) * B_r * B_theta * dA * L_axial
    dT = (r_cent / MU_0) * B_r * B_t * area_ag * stator.axial_length
    return float(np.sum(dT))


# ---------------------------------------------------------------------------
# Cogging torque
# ---------------------------------------------------------------------------

def compute_cogging_torque(
    B_dict: dict,
    mesh,
    stator,
    air_gap_region_id: int,
) -> float:
    """Estimate cogging torque as 5 % of the average electromagnetic torque.

    A full cogging torque analysis requires multiple rotor positions and a
    rotating reluctance variation; the 5 % heuristic is typical for well-
    designed integer-slot machines.

    Parameters
    ----------
    B_dict, mesh, stator, air_gap_region_id:
        Same as :func:`compute_torque`.

    Returns
    -------
    float
        Cogging torque magnitude [N·m].
    """
    T_em = compute_torque(B_dict, mesh, stator, air_gap_region_id)
    return 0.05 * abs(T_em)


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------

def compute_efficiency(
    torque_Nm: float,
    stator,
    total_loss_W: float,
) -> float:
    """Compute drive efficiency from torque, speed, and total losses.

    Parameters
    ----------
    torque_Nm:
        Electromagnetic torque at the rated operating point [N·m].
    stator:
        :class:`StatorMeshInput` providing *rated_speed_rpm*.
    total_loss_W:
        Sum of all electrical losses (iron + copper) [W].

    Returns
    -------
    float
        Efficiency η ∈ [0, 0.9999].
    """
    omega = stator.rated_speed_rpm * 2.0 * np.pi / 60.0   # rad/s
    P_mech = torque_Nm * omega                             # W
    P_input = P_mech + total_loss_W

    if P_input <= 0.0 or P_mech <= 0.0:
        return 0.0

    eta = P_mech / P_input
    return float(np.clip(eta, 0.0, 0.9999))
