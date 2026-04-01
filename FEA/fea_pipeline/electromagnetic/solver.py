"""2-D magnetostatic FEM solver (numpy / scipy, no HPC libraries).

Solves the weak form of the curl-curl equation for the magnetic vector
potential A_z:

    ∫_Ω ν(B) (∇A_z · ∇v) dΩ  =  ∫_Ω J_z v dΩ    ∀v ∈ H₀¹(Ω)

Assembly uses vectorised COO accumulation for the stiffness matrix, then
converts to CSR for the solve.  A Newton-Raphson (Picard) fixed-point
iteration is used for the nonlinear permeability update.

The public entry point is :func:`run_electromagnetic_analysis`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..utils.mesh_utils import FEAMesh
from ..utils.units import MU_0, electrical_frequency, rpm_to_rad_s
from .material_library import get_material_properties, interpolate_reluctivity, MATERIAL_DB
from .loss_calculator import compute_iron_losses, compute_copper_losses
from .boundary_conditions import (
    apply_dirichlet_bcs,
    get_em_boundary_nodes,
    build_current_density,
)
from .postprocessor import (
    extract_flux_density,
    compute_torque,
    compute_cogging_torque,
    compute_efficiency,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _initial_reluctivity_per_element(
    region_ids: np.ndarray,
    stator,
) -> np.ndarray:
    """Return the linear (low-B) reluctivity for every element.

    For iron regions the initial permeability is derived from the first segment
    of the BH curve (μ_initial = B₁/H₁).  For air and copper it is 1/μ₀.
    """
    # Build a reverse map: region_id → material_id
    tag_to_material: dict[int, str] = {}
    for region_name, tag in stator.region_tags.items():
        mat_id = stator.material_map.get(region_name, "air")
        tag_to_material[tag] = mat_id

    nu_elem = np.empty(len(region_ids), dtype=float)
    for tag, mat_id in tag_to_material.items():
        mask = region_ids == tag
        if not np.any(mask):
            continue
        props = MATERIAL_DB.get(mat_id, {})
        bh = props.get("BH_curve")
        if bh and len(bh) > 1:
            # Initial linear permeability from first non-zero BH segment
            H1, B1 = bh[1]
            mu_init = B1 / H1 if H1 > 0 else MU_0
            nu_elem[mask] = 1.0 / mu_init
        else:
            # Air, copper, or any non-magnetic material
            nu_elem[mask] = 1.0 / MU_0

    # Fill any remaining elements (unrecognised tags) with free-space value
    unset_mask = ~np.isfinite(nu_elem)
    nu_elem[unset_mask] = 1.0 / MU_0

    return nu_elem


def _update_reluctivity(
    A_z: np.ndarray,
    mesh: FEAMesh,
    region_ids: np.ndarray,
    stator,
) -> np.ndarray:
    """Recompute reluctivity per element from the current A_z solution."""
    B_dict = extract_flux_density(A_z, mesh)
    B_mag = B_dict["B_mag"]          # (n_elems,)

    tag_to_material: dict[int, str] = {}
    for region_name, tag in stator.region_tags.items():
        mat_id = stator.material_map.get(region_name, "air")
        tag_to_material[tag] = mat_id

    nu_elem = np.empty(len(region_ids), dtype=float)
    nu_elem[:] = 1.0 / MU_0   # default

    for tag, mat_id in tag_to_material.items():
        mask = region_ids == tag
        if not np.any(mask):
            continue
        nu_elem[mask] = interpolate_reluctivity(B_mag[mask], mat_id)

    return nu_elem


def _assemble_system(
    mesh: FEAMesh,
    nu_elem: np.ndarray,
    J_z: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Vectorised COO assembly of the global stiffness matrix and load vector.

    For a linear triangle element *e* with local nodes (0,1,2) mapped to
    global nodes (n₀, n₁, n₂):

        K[nᵢ, nⱼ] += ν_e * (b[e,i]·b[e,j] + c[e,i]·c[e,j]) / (4·A_e)

    and

        F[nᵢ] += J_e * A_e / 3

    Parameters
    ----------
    mesh:
        :class:`FEAMesh`.
    nu_elem:
        (n_elems,) reluctivity [A·m/Wb].
    J_z:
        (n_elems,) current density [A/m²].

    Returns
    -------
    K : csr_matrix  (n_nodes × n_nodes)
    F : ndarray     (n_nodes,)
    """
    b, c, area = mesh.gradient_operators()          # (n_elems,3), (n_elems,3), (n_elems,)
    n_nodes = mesh.n_nodes
    n_elems = mesh.n_elements
    elems = mesh.elements                           # (n_elems, 3)

    # Stiffness — build COO arrays
    # For each pair (i, j) in {0,1,2}² and each element we produce one entry.
    row_idx_list = []
    col_idx_list = []
    data_list = []

    # Pre-compute: ν_e / (4 * A_e)
    nu_over_4A = nu_elem / (4.0 * area)             # (n_elems,)

    for i in range(3):
        for j in range(3):
            val = nu_over_4A * (b[:, i] * b[:, j] + c[:, i] * c[:, j])  # (n_elems,)
            row_idx_list.append(elems[:, i])
            col_idx_list.append(elems[:, j])
            data_list.append(val)

    rows = np.concatenate(row_idx_list)
    cols = np.concatenate(col_idx_list)
    data = np.concatenate(data_list)

    K = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    # Load vector
    F = np.zeros(n_nodes, dtype=float)
    contrib = J_z * area / 3.0                      # (n_elems,)
    np.add.at(F, elems[:, 0], contrib)
    np.add.at(F, elems[:, 1], contrib)
    np.add.at(F, elems[:, 2], contrib)

    return K, F


# ---------------------------------------------------------------------------
# Main solver entry point
# ---------------------------------------------------------------------------

def run_electromagnetic_analysis(
    mesh: FEAMesh,
    regions: dict,
    stator: Any,
    config: dict,
) -> dict:
    """Assemble and solve the 2-D magnetostatic FEM problem.

    The function performs:

    1. Initial (linear) reluctivity assignment from the BH curve.
    2. Current-density vector assembly.
    3. Global stiffness matrix and load vector assembly (COO → CSR).
    4. Dirichlet BC application (A_z = 0 on outer boundary).
    5. Direct sparse solve: A_z = K⁻¹ F.
    6. Optional Picard (fixed-point) nonlinear iteration for B-dependent ν.
    7. Post-processing: B-field, torque, losses, efficiency.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    regions:
        Dict mapping region name → region tag (may overlap with
        ``stator.region_tags``; unused directly but available for extension).
    stator:
        :class:`StatorMeshInput` instance.
    config:
        Pipeline configuration dict.  Relevant sub-keys:

        ``config["nonlinear"]["enabled"]``    – bool, default True
        ``config["nonlinear"]["max_iterations"]`` – int, default 20
        ``config["nonlinear"]["tolerance"]``  – float, default 1e-5
        ``config["copper_temperature_K"]``    – float, default 293.15

    Returns
    -------
    dict
        EM analysis results with keys:

        ``torque_Nm``, ``cogging_torque_Nm``,
        ``iron_loss_W``, ``eddy_current_loss_W``, ``hysteresis_loss_W``,
        ``copper_loss_W``, ``total_loss_W``, ``efficiency``,
        ``A_field``, ``B_field``, ``loss_density_map``,
        ``copper_loss_density_map``, ``domain``.
    """
    region_ids = mesh.region_ids         # (n_elems,)

    # ------------------------------------------------------------------
    # Step 1: initial (linear) reluctivity
    # ------------------------------------------------------------------
    nu_elem = _initial_reluctivity_per_element(region_ids, stator)
    logger.debug("Initial ν: min=%.3e  max=%.3e", nu_elem.min(), nu_elem.max())

    # ------------------------------------------------------------------
    # Step 2: current density
    # ------------------------------------------------------------------
    J_z = build_current_density(mesh, stator, config)
    logger.debug("J_z: non-zero elements = %d  max=%.3e A/m²",
                 np.count_nonzero(J_z), np.abs(J_z).max() if J_z.size else 0.0)

    # ------------------------------------------------------------------
    # Step 3 & 4: assemble + apply BCs
    # ------------------------------------------------------------------
    def _solve_linear(nu: np.ndarray) -> np.ndarray:
        K, F = _assemble_system(mesh, nu, J_z)
        bc_nodes, bc_vals = get_em_boundary_nodes(mesh, config)
        K_bc, F_bc = apply_dirichlet_bcs(K, F, bc_nodes, bc_vals)
        A_z = spla.spsolve(K_bc, F_bc)
        return A_z

    # ------------------------------------------------------------------
    # Step 5: initial solve
    # ------------------------------------------------------------------
    A_z = _solve_linear(nu_elem)

    # ------------------------------------------------------------------
    # Step 6: nonlinear (Picard) iteration
    # ------------------------------------------------------------------
    nl_config = config.get("nonlinear", {})
    nonlinear_enabled = nl_config.get("enabled", True)
    max_iter = int(nl_config.get("max_iterations", 20))
    tol = float(nl_config.get("tolerance", 1e-5))

    if nonlinear_enabled:
        for iteration in range(max_iter):
            nu_new = _update_reluctivity(A_z, mesh, region_ids, stator)
            A_z_new = _solve_linear(nu_new)

            # Convergence check: relative change in A_z field
            delta = np.linalg.norm(A_z_new - A_z)
            ref = max(np.linalg.norm(A_z_new), 1e-30)
            rel_err = delta / ref
            logger.debug("NL iter %d: rel_err = %.3e", iteration + 1, rel_err)

            A_z = A_z_new
            nu_elem = nu_new

            if rel_err < tol:
                logger.info("Nonlinear solver converged in %d iterations "
                            "(rel_err=%.3e).", iteration + 1, rel_err)
                break
        else:
            logger.warning("Nonlinear solver reached max iterations (%d) "
                           "without full convergence.", max_iter)

    # ------------------------------------------------------------------
    # Step 7: post-processing
    # ------------------------------------------------------------------
    B_dict = extract_flux_density(A_z, mesh)
    B_mag = B_dict["B_mag"]                   # (n_elems,)

    # --- Electrical frequency ---
    freq_Hz = electrical_frequency(stator.rated_speed_rpm, stator.num_poles)

    # --- Iron losses (stator core only) ---
    iron_tag = stator.region_tags.get("stator_core", 1)
    iron_material = stator.material_map.get("stator_core", "M250-35A")
    iron_mask = region_ids == iron_tag

    # Areas needed for volume computation
    _, _, area_elems = mesh.gradient_operators()

    # Per-element iron losses (only iron elements contribute)
    iron_result = compute_iron_losses(
        B_elem=B_mag * iron_mask.astype(float),  # zero out non-iron
        region_ids=region_ids,
        areas=area_elems,
        axial_length=stator.axial_length,
        freq_Hz=freq_Hz,
        material_id=iron_material,
    )

    # --- Copper losses ---
    winding_tag = stator.region_tags.get("winding", 2)
    winding_mask = region_ids == winding_tag
    B_avg_winding = float(np.mean(B_mag[winding_mask])) if np.any(winding_mask) else 0.0

    cu_result = compute_copper_losses(stator, B_avg_winding, config)

    # Build per-element copper loss density map
    copper_density_map = np.zeros(mesh.n_elements, dtype=float)
    copper_density_map[winding_mask] = cu_result["spatial_W_per_m3"]

    # --- Torque ---
    air_gap_tag = stator.region_tags.get("air_gap", 3)
    torque_Nm = compute_torque(B_dict, mesh, stator, air_gap_tag)
    cogging_Nm = compute_cogging_torque(B_dict, mesh, stator, air_gap_tag)

    # --- Efficiency ---
    total_loss = iron_result["total"] + cu_result["total"]
    eta = compute_efficiency(torque_Nm, stator, total_loss)

    return {
        "torque_Nm": float(torque_Nm),
        "cogging_torque_Nm": float(cogging_Nm),
        "iron_loss_W": float(iron_result["total"]),
        "eddy_current_loss_W": float(iron_result["eddy"]),
        "hysteresis_loss_W": float(iron_result["hysteresis"]),
        "copper_loss_W": float(cu_result["total"]),
        "total_loss_W": float(total_loss),
        "efficiency": float(eta),
        "A_field": A_z,
        "B_field": B_dict,
        "loss_density_map": iron_result["spatial_W_per_m3"],
        "copper_loss_density_map": copper_density_map,
        "domain": mesh,
    }
