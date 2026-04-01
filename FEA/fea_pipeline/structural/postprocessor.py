"""Post-processing for the 2-D plane-stress structural FEM solver.

Functions
---------
compute_von_mises
    Element-wise von Mises stress from the displacement field.
compute_principal_stresses
    Element-wise principal stresses (eigenvalues of the stress tensor).
compute_fatigue_life
    Goodman–Basquin fatigue life estimate from the von Mises stress field.
compute_natural_frequencies
    Modal analysis via sparse eigensolver (shift-invert ARPACK).

Stress computation (plane stress, CST elements)
------------------------------------------------
For each element *e*:

1. Extract the 6 displacement DOFs:
       u_e = [u_x0, u_y0, u_x1, u_y1, u_x2, u_y2]

2. Compute mechanical strain:
       eps = B_e @ u_e      →  [eps_xx, eps_yy, gamma_xy]

   where  B_e = 1/(2A) * [[b0, 0, b1, 0, b2, 0],
                            [0, c0, 0, c1, 0, c2],
                            [c0, b0, c1, b1, c2, b2]]

3. Compute thermal strain:
       T_avg = mean(T_nodal[nodes_e])
       eps_th = alpha_e * (T_avg - T_ref) * [1, 1, 0]

4. Effective mechanical strain:  eps_mech = eps - eps_th

5. Plane-stress constitutive law:
       sigma = D_e @ eps_mech
       D_e = E/(1-nu²) * [[1, nu, 0],
                           [nu, 1, 0],
                           [0, 0, (1-nu)/2]]

6. Von Mises:
       sigma_vm = sqrt(sx²  + sy² - sx*sy + 3*tau²)
       where sx = sigma[0], sy = sigma[1], tau = sigma[2]

7. Principal stresses:
       s1, s2 = (sx+sy)/2 ± sqrt(((sx-sy)/2)² + tau²)
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..utils.mesh_utils import FEAMesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: per-element stress tensor
# ---------------------------------------------------------------------------

def _compute_element_stress(
    u_nodal: np.ndarray,
    mesh: FEAMesh,
    E_elem: np.ndarray,
    nu_elem: np.ndarray,
    alpha_elem: np.ndarray,
    T_nodal: np.ndarray,
    T_ref_K: float,
) -> np.ndarray:
    """Return (n_elems, 3) stress array [sigma_x, sigma_y, tau_xy].

    This is the shared kernel used by both :func:`compute_von_mises` and
    :func:`compute_principal_stresses`.
    """
    b, c, area = mesh.gradient_operators()   # (n_elems, 3) each; area (n_elems,)
    elems = mesh.elements                    # (n_elems, 3)
    n_elems = mesh.n_elements

    # --- Mechanical strain via B @ u_e ---
    # u_e = [u_x0, u_y0, u_x1, u_y1, u_x2, u_y2]  shape (n_elems, 6)
    u_x = u_nodal[0::2]   # (n_nodes,)
    u_y = u_nodal[1::2]   # (n_nodes,)

    # Gather per-element nodal displacements
    ux_e = u_x[elems]     # (n_elems, 3)
    uy_e = u_y[elems]     # (n_elems, 3)

    inv2A = 1.0 / (2.0 * area)   # (n_elems,)

    # Strain components: eps = B @ u_e
    # eps_xx = sum_i b_i * u_xi / (2A)
    # eps_yy = sum_i c_i * u_yi / (2A)
    # gamma_xy = sum_i (c_i * u_xi + b_i * u_yi) / (2A)
    eps_xx = inv2A * np.einsum("ei,ei->e", b, ux_e)
    eps_yy = inv2A * np.einsum("ei,ei->e", c, uy_e)
    gam_xy = inv2A * (np.einsum("ei,ei->e", c, ux_e) + np.einsum("ei,ei->e", b, uy_e))

    # --- Thermal strain ---
    T_avg = T_nodal[elems].mean(axis=1)    # (n_elems,)
    dT = T_avg - T_ref_K
    eps_th = alpha_elem * dT               # (n_elems,)

    # Effective strain (mechanical minus free thermal)
    eps_xx_m = eps_xx - eps_th
    eps_yy_m = eps_yy - eps_th
    gam_xy_m = gam_xy                      # thermal has no shear contribution

    # --- Plane-stress constitutive law: sigma = D @ eps_mech ---
    fac = E_elem / (1.0 - nu_elem ** 2)   # (n_elems,)
    sigma_x = fac * (eps_xx_m + nu_elem * eps_yy_m)
    sigma_y = fac * (nu_elem * eps_xx_m + eps_yy_m)
    tau_xy  = fac * (1.0 - nu_elem) / 2.0 * gam_xy_m

    return np.column_stack([sigma_x, sigma_y, tau_xy])   # (n_elems, 3)


# ---------------------------------------------------------------------------
# Public: von Mises stress
# ---------------------------------------------------------------------------

def compute_von_mises(
    u_nodal: np.ndarray,
    mesh: FEAMesh,
    E_elem: np.ndarray,
    nu_elem: np.ndarray,
    alpha_elem: np.ndarray,
    T_nodal: np.ndarray,
    T_ref_K: float,
) -> np.ndarray:
    """Compute element-wise von Mises stress [Pa].

    For each element:
      1. Compute strain: eps = B_e @ u_e  (mechanical strain)
      2. Compute thermal strain: eps_th = alpha * (T_avg - T_ref) * [1,1,0]
      3. Compute stress: sigma = D @ (eps - eps_th)  → [sigma_x, sigma_y, tau_xy]
      4. Von Mises: sigma_vm = sqrt(sx²+sy²-sx*sy+3*tau²)

    Parameters
    ----------
    u_nodal:
        (2*n_nodes,) displacement DOF vector.
    mesh:
        :class:`FEAMesh` instance.
    E_elem, nu_elem, alpha_elem:
        (n_elems,) material property arrays.
    T_nodal:
        (n_nodes,) nodal temperature [K].
    T_ref_K:
        Stress-free reference temperature [K].

    Returns
    -------
    np.ndarray
        (n_elems,) von Mises stress [Pa].
    """
    stress = _compute_element_stress(
        u_nodal, mesh, E_elem, nu_elem, alpha_elem, T_nodal, T_ref_K
    )
    sx  = stress[:, 0]
    sy  = stress[:, 1]
    tau = stress[:, 2]
    vm = np.sqrt(np.maximum(sx ** 2 + sy ** 2 - sx * sy + 3.0 * tau ** 2, 0.0))
    return vm


# ---------------------------------------------------------------------------
# Public: principal stresses
# ---------------------------------------------------------------------------

def compute_principal_stresses(
    u_nodal: np.ndarray,
    mesh: FEAMesh,
    E_elem: np.ndarray,
    nu_elem: np.ndarray,
    alpha_elem: np.ndarray,
    T_nodal: np.ndarray,
    T_ref_K: float,
) -> np.ndarray:
    """Return (n_elems, 2) principal stresses [sigma1, sigma2] per element.

    sigma1 >= sigma2 by convention (algebraic ordering).

    Parameters
    ----------
    u_nodal:
        (2*n_nodes,) displacement DOF vector.
    mesh:
        :class:`FEAMesh` instance.
    E_elem, nu_elem, alpha_elem:
        (n_elems,) material property arrays.
    T_nodal:
        (n_nodes,) nodal temperature [K].
    T_ref_K:
        Stress-free reference temperature [K].

    Returns
    -------
    np.ndarray
        (n_elems, 2) — columns are [sigma_1, sigma_2].
    """
    stress = _compute_element_stress(
        u_nodal, mesh, E_elem, nu_elem, alpha_elem, T_nodal, T_ref_K
    )
    sx  = stress[:, 0]
    sy  = stress[:, 1]
    tau = stress[:, 2]

    # Mohr's circle:  s1, s2 = (sx+sy)/2  ±  sqrt(((sx-sy)/2)² + tau²)
    s_avg = 0.5 * (sx + sy)
    radius = np.sqrt(np.maximum((0.5 * (sx - sy)) ** 2 + tau ** 2, 0.0))

    sigma1 = s_avg + radius
    sigma2 = s_avg - radius
    return np.column_stack([sigma1, sigma2])


# ---------------------------------------------------------------------------
# Public: fatigue life
# ---------------------------------------------------------------------------

def compute_fatigue_life(
    von_mises: np.ndarray,
    config: dict,
) -> float:
    """Estimate minimum fatigue life [cycles] using modified Goodman criterion.

    Approach
    --------
    1. Extract peak von Mises stress as the critical amplitude.
       Assume fully-reversed loading (R = -1) with worst-case mean stress:
         sigma_a = 0.5 * sigma_max   (alternating component)
         sigma_m = 0.5 * sigma_max   (mean component)

    2. Build the modified endurance limit:
         S_e = S_e_base * K_sf * K_rel / K_t
       where the three knockdown factors are surface finish, reliability, and
       stress concentration.

    3. Modified Goodman criterion:
         sigma_a / S_e + sigma_m / S_u = 1/N  (at failure, N=1)

       The *safe* amplitude at infinite life satisfies the LHS = 1.  If the
       applied sigma_a < corrected endurance limit, clamp N to 1e12.

    4. Basquin power law for finite life:
         N = (S_e / sigma_a)^(1/b)   where b ≈ 0.085 (steel)

    Parameters
    ----------
    von_mises:
        (n_elems,) von Mises stress field [Pa].
    config:
        Structural configuration dict.  Uses the ``"fatigue"`` and
        ``"materials"`` sub-dicts.

    Returns
    -------
    float
        Estimated fatigue life [cycles].  Clamped to 1e12.
    """
    fatigue_cfg = config.get("fatigue", {})
    mat_cfg = config.get("materials", {}).get("stator_core", {})

    # Material fatigue properties
    S_u  = float(mat_cfg.get("ultimate_strength_Pa", 5.0e8))
    S_e0 = float(mat_cfg.get("fatigue_limit_Pa",    2.0e8))

    # Knockdown factors
    K_sf  = float(fatigue_cfg.get("surface_finish_factor",    0.85))
    K_rel = float(fatigue_cfg.get("reliability_factor",       0.897))
    K_t   = float(fatigue_cfg.get("stress_concentration_factor", 1.5))

    # Corrected endurance limit
    S_e = S_e0 * K_sf * K_rel / K_t

    # Basquin slope exponent for steel
    b_exp = 0.085

    sigma_max = float(np.max(von_mises)) if von_mises.size else 0.0

    if sigma_max < 1.0:
        # Essentially zero stress — return infinite life
        logger.debug("Near-zero stress field — returning infinite fatigue life.")
        return 1.0e12

    # Worst-case split
    sigma_a = 0.5 * sigma_max
    sigma_m = 0.5 * sigma_max

    # Goodman equivalent fully-reversed stress amplitude at N=1:
    #   sigma_a / S_e + sigma_m / S_u < 1  → safe for infinite life
    goodman_lhs = sigma_a / S_e + sigma_m / S_u

    logger.debug(
        "Fatigue: sigma_max=%.3e Pa, S_e=%.3e Pa, S_u=%.3e Pa, "
        "Goodman LHS=%.4f",
        sigma_max, S_e, S_u, goodman_lhs,
    )

    if goodman_lhs <= 1.0:
        # Stress is below the endurance limit — effectively infinite life
        return 1.0e12

    # Finite life via Basquin:  N = (S_e / sigma_a)^(1/b)
    # Goodman reduces the effective endurance limit for combined loading:
    #   sigma_a_eff = sigma_a / (1 - sigma_m / S_u)
    # Alternatively, compute the Goodman-equivalent stress amplitude:
    sigma_a_goodman = sigma_a / (1.0 - sigma_m / S_u)

    if sigma_a_goodman <= 0.0 or S_e <= 0.0:
        return 1.0

    N = (S_e / sigma_a_goodman) ** (1.0 / b_exp)
    N = float(np.clip(N, 1.0, 1.0e12))

    logger.debug("Fatigue life estimate: N = %.4e cycles", N)
    return N


# ---------------------------------------------------------------------------
# Public: natural frequencies
# ---------------------------------------------------------------------------

def compute_natural_frequencies(
    mesh: FEAMesh,
    E_elem: np.ndarray,
    nu_elem: np.ndarray,
    rho_elem: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Compute natural frequencies [Hz] by solving generalised eigenvalue problem.

    Assembles the global stiffness matrix K and consistent mass matrix M for
    the 2-D plane-stress problem, then solves:

        K φ = ω² M φ

    using ``scipy.sparse.linalg.eigsh`` with shift-invert (sigma=0) to
    efficiently extract the lowest modes.

    Consistent mass matrix for a CST element (thickness t=1 plane, rho*A/12):

        M_e = rho_e * A_e / 12 *
              [[2,0,1,0,1,0],
               [0,2,0,1,0,1],
               [1,0,2,0,1,0],
               [0,1,0,2,0,1],
               [1,0,1,0,2,0],
               [0,1,0,1,0,2]]

    Structural BCs (outer boundary fixed) are applied before the solve by
    removing the constrained DOFs from the system (elimination method).

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    E_elem:
        (n_elems,) Young's modulus [Pa].
    nu_elem:
        (n_elems,) Poisson's ratio.
    rho_elem:
        (n_elems,) mass density [kg/m³].
    config:
        Structural configuration dict.  Uses ``config["modal"]``:
          ``enabled`` (bool), ``num_modes`` (int).

    Returns
    -------
    np.ndarray
        Sorted array of natural frequencies [Hz].  Empty array if modal
        analysis is disabled.
    """
    modal_cfg = config.get("modal", {})
    if not modal_cfg.get("enabled", True):
        logger.debug("Modal analysis disabled by config.")
        return np.array([], dtype=float)

    n_modes = int(modal_cfg.get("num_modes", 6))
    n_nodes = mesh.n_nodes
    n_dofs = 2 * n_nodes

    b, c, area = mesh.gradient_operators()   # (n_elems, 3), (n_elems, 3), (n_elems,)
    elems = mesh.elements                    # (n_elems, 3)
    n_elems = mesh.n_elements

    # --- Stiffness matrix (same as main solver) ---
    fac = E_elem / (1.0 - nu_elem ** 2)     # (n_elems,)
    inv4A = 1.0 / (4.0 * area)              # (n_elems,)

    # B^T D B expansion for the 6×6 element stiffness matrix.
    # DOF ordering per element: [u_x0, u_y0, u_x1, u_y1, u_x2, u_y2]
    # Build row / column indices for the 6×6 blocks
    dof_map = np.zeros((n_elems, 6), dtype=np.intp)
    for i in range(3):
        dof_map[:, 2 * i]     = 2 * elems[:, i]       # u_x DOF
        dof_map[:, 2 * i + 1] = 2 * elems[:, i] + 1   # u_y DOF

    k_rows: list[np.ndarray] = []
    k_cols: list[np.ndarray] = []
    k_data: list[np.ndarray] = []

    for i in range(3):
        for j in range(3):
            # Sub-block (2i, 2j) — 2×2 local stiffness contribution
            # k_uu = (D11*b_i*b_j + D33*c_i*c_j) / (4A)
            # k_uv = (D12*b_i*c_j + D33*c_i*b_j) / (4A)
            # k_vu = (D12*c_i*b_j + D33*b_i*c_j) / (4A)
            # k_vv = (D11*c_i*c_j + D33*b_i*b_j) / (4A)
            D11 = fac
            D12 = fac * nu_elem
            D33 = fac * (1.0 - nu_elem) / 2.0

            k_uu = inv4A * (D11 * b[:, i] * b[:, j] + D33 * c[:, i] * c[:, j])
            k_uv = inv4A * (D12 * b[:, i] * c[:, j] + D33 * c[:, i] * b[:, j])
            k_vu = inv4A * (D12 * c[:, i] * b[:, j] + D33 * b[:, i] * c[:, j])
            k_vv = inv4A * (D11 * c[:, i] * c[:, j] + D33 * b[:, i] * b[:, j])

            r_ux = dof_map[:, 2 * i]
            r_uy = dof_map[:, 2 * i + 1]
            c_ux = dof_map[:, 2 * j]
            c_uy = dof_map[:, 2 * j + 1]

            k_rows.extend([r_ux, r_ux, r_uy, r_uy])
            k_cols.extend([c_ux, c_uy, c_ux, c_uy])
            k_data.extend([k_uu, k_uv, k_vu, k_vv])

    K_rows = np.concatenate(k_rows)
    K_cols = np.concatenate(k_cols)
    K_data = np.concatenate(k_data)
    K = sp.coo_matrix((K_data, (K_rows, K_cols)), shape=(n_dofs, n_dofs)).tocsr()

    # --- Consistent mass matrix ---
    # M_e = rho*A/12 * L   where L is the 6×6 matrix with 2 on diagonal, 1 off
    # The 2×2 sub-blocks of L are:
    #   L[2i, 2j]   = L[2i+1, 2j+1] = 2 if i==j else 1
    #   L[2i, 2j+1] = L[2i+1, 2j]   = 0
    m_rows: list[np.ndarray] = []
    m_cols: list[np.ndarray] = []
    m_data: list[np.ndarray] = []

    rho_A_over12 = rho_elem * area / 12.0   # (n_elems,)

    for i in range(3):
        for j in range(3):
            coupling = 2.0 if i == j else 1.0
            m_val = coupling * rho_A_over12

            r_ux = dof_map[:, 2 * i]
            r_uy = dof_map[:, 2 * i + 1]
            c_ux = dof_map[:, 2 * j]
            c_uy = dof_map[:, 2 * j + 1]

            # Diagonal 2×2 coupling (x with x, y with y)
            m_rows.extend([r_ux, r_uy])
            m_cols.extend([c_ux, c_uy])
            m_data.extend([m_val, m_val])

    M_rows = np.concatenate(m_rows)
    M_cols = np.concatenate(m_cols)
    M_data = np.concatenate(m_data)
    M = sp.coo_matrix((M_data, (M_rows, M_cols)), shape=(n_dofs, n_dofs)).tocsr()

    # --- Apply structural BCs (elimination of constrained DOFs) ---
    outer_nodes = mesh.boundary_node_sets.get("outer", np.array([], dtype=np.intp))
    constrained_dofs: set[int] = set()
    for node in outer_nodes:
        constrained_dofs.add(2 * int(node))
        constrained_dofs.add(2 * int(node) + 1)

    free_dofs = np.array(
        [d for d in range(n_dofs) if d not in constrained_dofs], dtype=np.intp
    )
    n_free = len(free_dofs)

    if n_free < 2 * n_modes:
        logger.warning(
            "Too few free DOFs (%d) for %d modes — returning zeros.", n_free, n_modes
        )
        return np.zeros(n_modes, dtype=float)

    K_free = K[np.ix_(free_dofs, free_dofs)].tocsr()
    M_free = M[np.ix_(free_dofs, free_dofs)].tocsr()

    # Regularise for shift-invert: add small diagonal to K to avoid exact zero
    K_reg = K_free + sp.eye(n_free, format="csr") * 1.0

    logger.debug(
        "Modal analysis: %d free DOFs, requesting %d modes", n_free, n_modes
    )

    try:
        eigenvalues, _ = spla.eigsh(
            K_reg,
            k=n_modes,
            M=M_free,
            sigma=0.0,
            which="LM",
            tol=1e-8,
            maxiter=10 * n_free,
        )
        # Remove the artificial regularisation shift (shift = 1.0 was added to K)
        # eigsh with shift-invert on K_reg gives eigenvalues of (K + I)
        # The true eigenvalue is lambda_true = lambda_reg - 1
        eigenvalues = eigenvalues - 1.0
        # Clamp negative eigenvalues (numerical noise in rigid-body modes)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(eigenvalues)         # rad/s
        freqs_Hz = omega / (2.0 * np.pi)
        freqs_Hz.sort()
    except Exception as exc:
        logger.warning("eigsh failed (%s) — returning zero frequencies.", exc)
        freqs_Hz = np.zeros(n_modes, dtype=float)

    logger.info(
        "Natural frequencies [Hz]: %s",
        np.array2string(freqs_Hz, precision=2, separator=", "),
    )
    return freqs_Hz
