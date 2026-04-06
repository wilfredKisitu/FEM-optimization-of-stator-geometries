"""2-D plane-stress structural FEM solver.

Solves the linear elasticity problem with thermal and electromagnetic loads:

    K u = F_thermal + F_em

DOF layout: node i → global DOFs [2*i, 2*i+1] for [u_x, u_y].

Material properties are assigned per element from ``stator.region_tags``:
- stator_core : E = 200 GPa, ν = 0.28, ρ = 7650 kg/m³, α = 12e-6 /K
- winding     : E = 3 GPa,   ν = 0.35, ρ = 3500 kg/m³, α = 18e-6 /K
- air_gap     : E = 1 MPa (soft filler), ν = 0.3, ρ = 1.2 kg/m³, α = 0
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..utils.mesh_utils import FEAMesh
from ..io.schema import StatorMeshInput
from .boundary_conditions import apply_structural_boundary_conditions
from .load_mapper import compute_thermal_expansion_load, compute_maxwell_stress_load
from .postprocessor import (
    compute_von_mises,
    compute_principal_stresses,
    compute_fatigue_life,
    compute_natural_frequencies,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default material properties per region role
# ---------------------------------------------------------------------------

_CORE_PROPS = dict(E=2.0e11, nu=0.28, rho=7650.0, alpha=12.0e-6,
                   yield_Pa=3.5e8, S_u=5.0e8, S_e=2.0e8)
_WINDING_PROPS = dict(E=3.0e9, nu=0.35, rho=3500.0, alpha=18.0e-6,
                      yield_Pa=3.5e8, S_u=5.0e8, S_e=2.0e8)
_AIR_PROPS = dict(E=1.0e6, nu=0.30, rho=1.2, alpha=0.0,
                  yield_Pa=3.5e8, S_u=5.0e8, S_e=2.0e8)


def run_structural_analysis(
    mesh: FEAMesh,
    regions: dict,
    stator: StatorMeshInput,
    em_results: dict,
    thermal_results: dict,
    config: dict,
) -> dict:
    """Run plane-stress structural FEA with thermal and EM loads.

    Parameters
    ----------
    mesh:
        Full FEAMesh (all regions).
    regions:
        Dict of region sub-meshes (not used directly — provided for API
        consistency with other stages).
    stator:
        Stator input specification.
    em_results:
        Output dict from :func:`run_electromagnetic_analysis`.
    thermal_results:
        Output dict from :func:`run_thermal_analysis`.
    config:
        Structural section of the pipeline configuration YAML.

    Returns
    -------
    dict with scalar KPIs and spatial field arrays.
    """
    T_ref_K = config.get("reference_temperature_K", 293.15)
    T_nodal  = thermal_results.get("T_field", np.full(mesh.n_nodes, T_ref_K))

    # --- Per-element material arrays ---
    E_elem, nu_elem, rho_elem, alpha_elem = _build_material_arrays(
        mesh, stator, config
    )
    yield_strength_Pa = _dominant_yield_strength(mesh, stator, config)

    # --- Assemble global stiffness (vectorised CST) ---
    log.info("Assembling structural stiffness matrix (%d DOFs)…",
             2 * mesh.n_nodes)
    K = _assemble_stiffness(mesh, E_elem, nu_elem)

    # --- Compute load vectors ---
    F = np.zeros(2 * mesh.n_nodes)

    if config.get("thermal_loads", True):
        F_th = compute_thermal_expansion_load(
            mesh, T_nodal, E_elem, nu_elem, alpha_elem, T_ref_K
        )
        F += F_th

    if config.get("electromagnetic_loads", True):
        B_field = em_results.get("B_field", {})
        if B_field:
            F_em = compute_maxwell_stress_load(mesh, B_field, config)
            F += F_em

    # --- Apply boundary conditions (fix outer nodes) ---
    K_lil = K.tolil()
    K_bc, F_bc = apply_structural_boundary_conditions(mesh, K_lil, F, config)

    # --- Solve ---
    log.info("Solving structural system…")
    try:
        u_sol = spla.spsolve(K_bc, F_bc)
    except Exception as exc:
        log.warning("Direct solve failed (%s); falling back to MINRES.", exc)
        u_sol, _ = spla.minres(K_bc, F_bc, tol=1e-10, maxiter=10 * mesh.n_nodes)

    if not np.all(np.isfinite(u_sol)):
        log.warning("Non-finite displacements detected; clamping to zero.")
        u_sol = np.where(np.isfinite(u_sol), u_sol, 0.0)

    # --- Post-processing ---
    vm_field = compute_von_mises(
        u_sol, mesh, E_elem, nu_elem, alpha_elem, T_nodal, T_ref_K
    )
    principal_field = compute_principal_stresses(
        u_sol, mesh, E_elem, nu_elem, alpha_elem, T_nodal, T_ref_K
    )

    max_vm   = float(vm_field.max()) if len(vm_field) > 0 else 0.0
    max_disp = float(
        np.linalg.norm(
            u_sol.reshape(-1, 2), axis=1
        ).max()
    ) if mesh.n_nodes > 0 else 0.0

    safety_factor = (yield_strength_Pa / max_vm) if max_vm > 0 else float("inf")

    fatigue_life = compute_fatigue_life(vm_field, config)

    # Modal analysis
    nat_freqs = np.array([])
    if config.get("modal", {}).get("enabled", True):
        nat_freqs = compute_natural_frequencies(
            mesh, E_elem, nu_elem, rho_elem, config
        )

    critical_mode = int(np.argmin(nat_freqs)) + 1 if len(nat_freqs) > 0 else 1

    log.info(
        "Structural: max_vm=%.3e Pa  max_disp=%.3e m  SF=%.2f",
        max_vm, max_disp, safety_factor,
    )

    return {
        "max_von_mises_Pa":       max_vm,
        "max_displacement_m":     max_disp,
        "yield_strength_Pa":      yield_strength_Pa,
        "safety_factor":          safety_factor,
        "fatigue_life_cycles":    fatigue_life,
        "natural_frequencies_Hz": nat_freqs,
        "critical_mode":          critical_mode,
        "u_field":                u_sol,
        "von_mises_field":        vm_field,
        "principal_stress_field": principal_field,
        "domain":                 mesh,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_material_arrays(
    mesh: FEAMesh,
    stator: StatorMeshInput,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-element material property arrays from region tags."""
    tags = stator.region_tags
    core_id = tags.get("stator_core", 1)
    wnd_id  = tags.get("winding",    2)
    # all other IDs → air defaults

    mat_config = config.get("materials", {})
    core_cfg   = mat_config.get("stator_core", {})
    wnd_cfg    = mat_config.get("winding_equivalent", {})

    E_arr     = np.empty(mesh.n_elements)
    nu_arr    = np.empty(mesh.n_elements)
    rho_arr   = np.empty(mesh.n_elements)
    alpha_arr = np.empty(mesh.n_elements)

    for e_mask, props, cfg in [
        (mesh.region_ids == core_id, _CORE_PROPS, core_cfg),
        (mesh.region_ids == wnd_id,  _WINDING_PROPS, wnd_cfg),
    ]:
        if not e_mask.any():
            continue
        E_arr[e_mask]     = cfg.get("youngs_modulus_Pa",       props["E"])
        nu_arr[e_mask]    = cfg.get("poisson_ratio",           props["nu"])
        rho_arr[e_mask]   = cfg.get("density_kg_m3",           props["rho"])
        alpha_arr[e_mask] = cfg.get("thermal_expansion_1_K",   props["alpha"])

    # Everything else (air, unrecognised) → soft filler
    other = ~((mesh.region_ids == core_id) | (mesh.region_ids == wnd_id))
    E_arr[other]     = _AIR_PROPS["E"]
    nu_arr[other]    = _AIR_PROPS["nu"]
    rho_arr[other]   = _AIR_PROPS["rho"]
    alpha_arr[other] = _AIR_PROPS["alpha"]

    return E_arr, nu_arr, rho_arr, alpha_arr


def _dominant_yield_strength(
    mesh: FEAMesh, stator: StatorMeshInput, config: dict
) -> float:
    """Return the stator core yield strength (governing failure mode)."""
    core_cfg = config.get("materials", {}).get("stator_core", {})
    return float(core_cfg.get("yield_strength_Pa", _CORE_PROPS["yield_Pa"]))


def _assemble_stiffness(
    mesh: FEAMesh,
    E_elem: np.ndarray,
    nu_elem: np.ndarray,
) -> sp.csr_matrix:
    """Vectorised COO assembly of the global 2-D plane-stress stiffness matrix.

    The element stiffness is:

        K_e = A * B^T D B    (6 × 6)

    where B = 1/(2A) * [[b0,0,b1,0,b2,0],[0,c0,0,c1,0,c2],[c0,b0,c1,b1,c2,b2]]
    and   D = E/(1−ν²) * [[1,ν,0],[ν,1,0],[0,0,(1−ν)/2]].
    """
    b, c, area = mesh.gradient_operators()   # (n_elems, 3) each
    elems = mesh.elements                    # (n_elems, 3)
    n_dof = 2 * mesh.n_nodes
    n_e   = mesh.n_elements

    # Avoid degenerate elements
    good = area > 1e-20

    # D matrix components per element
    E  = E_elem[good]
    nu = nu_elem[good]
    denom = 1.0 - nu ** 2
    C11 = E / denom
    C12 = nu * E / denom
    C33 = 0.5 * E / (1.0 + nu)     # = E*(1-nu)/(2*(1-nu²)) = G

    b_g = b[good]    # (n_good, 3)
    c_g = c[good]    # (n_good, 3)
    a_g = area[good] # (n_good,)
    el_g = elems[good]

    # Build B matrix columns scaled by area:  B̃ = B * (2A) = [[b0,0,...], [0,c0,...], [c0,b0,...]]
    # K_e = (1/(4A)) * B̃^T D B̃   (scalar pre-factor combines Area and 1/(2A)² = 1/(4A))
    # Build COO entries for all 9 local (i,j) pairs

    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []

    # Global DOF indices for each element row
    # dofs_e[e, k] = [2*n_k, 2*n_k+1] for k=0,1,2
    dofs = np.zeros((el_g.shape[0], 6), dtype=np.intp)
    for k in range(3):
        dofs[:, 2*k]     = 2 * el_g[:, k]
        dofs[:, 2*k + 1] = 2 * el_g[:, k] + 1

    # B̃ columns  (each column is a length-3 vector of {b_i, c_i, 0} etc.)
    # Unrolled 6 columns of B̃:  col 0=[b0,0,c0], col1=[0,c0,b0], col2=[b1,0,c1], ...
    Btil = np.zeros((el_g.shape[0], 3, 6))
    for k in range(3):
        Btil[:, 0, 2*k]     = b_g[:, k]     # row 0 (eps_xx)
        Btil[:, 1, 2*k + 1] = c_g[:, k]     # row 1 (eps_yy)
        Btil[:, 2, 2*k]     = c_g[:, k]     # row 2 (gamma_xy)
        Btil[:, 2, 2*k + 1] = b_g[:, k]

    # D per element:  shape (n_good, 3, 3)
    D = np.zeros((el_g.shape[0], 3, 3))
    D[:, 0, 0] = C11;  D[:, 0, 1] = C12
    D[:, 1, 0] = C12;  D[:, 1, 1] = C11
    D[:, 2, 2] = C33

    # K̃ = Btil^T @ D @ Btil  shape (n_good, 6, 6)
    DB   = np.einsum("eij,ejk->eik", D, Btil)    # (n_good, 3, 6)
    Ke   = np.einsum("eji,ejk->eik", Btil, DB)   # (n_good, 6, 6)

    # Pre-factor: 1/(4*A)
    prefactor = 1.0 / (4.0 * a_g)               # (n_good,)
    Ke = Ke * prefactor[:, None, None]

    # Scatter into COO
    i_idx, j_idx = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")  # (6,6)
    i_flat = i_idx.ravel()   # 36
    j_flat = j_idx.ravel()

    global_rows = dofs[:, i_flat]  # (n_good, 36)
    global_cols = dofs[:, j_flat]
    global_vals = Ke[:, i_flat, j_flat]

    K_coo = sp.coo_matrix(
        (global_vals.ravel(), (global_rows.ravel(), global_cols.ravel())),
        shape=(n_dof, n_dof),
    )
    return K_coo.tocsr()
