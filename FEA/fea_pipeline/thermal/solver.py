"""2-D steady-state thermal FEM solver (numpy / scipy, no HPC libraries).

Solves the scalar heat-conduction equation:

    -∇·(k ∇T) = q_vol    in Ω
    q_n = h (T - T_inf)   on ∂Ω  (Robin / convection BC)

Assembly uses standard CST (Constant Strain Triangle) scalar FEM:

    K[nᵢ, nⱼ] += k_e * (b[e,i]*b[e,j] + c[e,i]*c[e,j]) / (4 * area[e])
    F[nᵢ]      += q_vol[e] * area[e] * L_axial / 3

The factor L_axial converts the 2-D element area to a 3-D volume so that the
load vector has units of Watts (matching the heat-flux boundary contributions).
After assembly the boundary conditions are applied, then the system is solved
with scipy.sparse.linalg.spsolve.

Material thermal conductivity is assigned per region:
  - stator_core  → k_in_plane from config["anisotropy"] (default 28 W/mK)
  - winding       → effective k: fill * k_cu + (1-fill) * k_ins
                     k_cu = 400 W/mK, k_ins = 0.2 W/mK (class-F insulation)
  - air_gap       → 0.025 W/mK
  - unknown tags  → 1.0 W/mK (safe fallback)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..utils.mesh_utils import FEAMesh
from ..io.schema import StatorMeshInput
from .boundary_conditions import apply_thermal_boundary_conditions
from .heat_sources import map_em_losses_to_heat_sources
from .postprocessor import (
    extract_temperature_field,
    identify_hot_spots,
    compute_winding_average_temperature,
    compute_temperature_uniformity,
)

logger = logging.getLogger(__name__)

# Physical constants / default material properties
_K_COPPER: float = 400.0    # W/mK — bulk copper
_K_INSULATION: float = 0.2  # W/mK — class-F winding insulation
_K_AIR: float = 0.025       # W/mK — still air
_K_FALLBACK: float = 1.0    # W/mK — unknown regions


# ---------------------------------------------------------------------------
# Material conductivity lookup
# ---------------------------------------------------------------------------

def _build_conductivity_map(stator: StatorMeshInput, config: dict) -> dict[int, float]:
    """Return a mapping from region_tag → thermal conductivity [W/mK].

    Parameters
    ----------
    stator:
        :class:`StatorMeshInput` providing ``region_tags`` and ``fill_factor``.
    config:
        Thermal config dict.  Uses ``config["anisotropy"]["k_in_plane"]`` for
        the stator core.

    Returns
    -------
    dict[int, float]
        Maps each region tag to its effective isotropic thermal conductivity.
    """
    aniso = config.get("anisotropy", {})
    k_core = float(aniso.get("k_in_plane", 28.0))

    fill = float(getattr(stator, "fill_factor", 0.45))
    k_winding = fill * _K_COPPER + (1.0 - fill) * _K_INSULATION

    region_tags = stator.region_tags   # dict[str, int]

    k_map: dict[int, float] = {}

    for region_name, tag in region_tags.items():
        if region_name == "stator_core":
            k_map[tag] = k_core
        elif region_name == "winding":
            k_map[tag] = k_winding
        elif region_name == "air_gap":
            k_map[tag] = _K_AIR
        else:
            k_map[tag] = _K_FALLBACK

    logger.debug("Thermal conductivity map (tag → k): %s", k_map)
    return k_map


def _conductivity_per_element(
    region_ids: np.ndarray,
    k_map: dict[int, float],
) -> np.ndarray:
    """Return (n_elems,) thermal conductivity array [W/mK].

    Elements with tags not present in *k_map* receive ``_K_FALLBACK``.
    """
    k_elem = np.full(len(region_ids), _K_FALLBACK, dtype=float)
    for tag, k_val in k_map.items():
        k_elem[region_ids == tag] = k_val
    return k_elem


# ---------------------------------------------------------------------------
# Global stiffness + load assembly
# ---------------------------------------------------------------------------

def _assemble_thermal_system(
    mesh: FEAMesh,
    k_elem: np.ndarray,
    q_vol: np.ndarray,
    axial_length: float,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Vectorised COO assembly of the thermal stiffness matrix and load vector.

    For a linear CST element *e* with nodes (0,1,2):

        K[nᵢ,nⱼ] += k_e * (b[e,i]*b[e,j] + c[e,i]*c[e,j]) / (4*area[e])

    and

        F[nᵢ] += q_vol[e] * area[e] * axial_length / 3

    The stiffness is assembled via COO accumulation (identical pattern to the
    EM solver) then converted to CSR.  The load vector uses ``np.add.at`` for
    correct scatter-accumulation.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    k_elem:
        (n_elems,) thermal conductivity [W/mK] per element.
    q_vol:
        (n_elems,) volumetric heat source [W/m³] per element.
    axial_length:
        Stack length [m] — converts 2-D area to 3-D volume.

    Returns
    -------
    K : csr_matrix  (n_nodes × n_nodes)
    F : ndarray     (n_nodes,)
    """
    b, c, area = mesh.gradient_operators()   # (n_elems, 3) each; area (n_elems,)
    n_nodes = mesh.n_nodes
    elems = mesh.elements                    # (n_elems, 3)

    # Pre-compute k_e / (4 * area_e) — shared factor for all (i,j) pairs
    k_over_4A = k_elem / (4.0 * area)       # (n_elems,)

    # Build COO data for the 3×3 local stiffness matrices
    row_list, col_list, data_list = [], [], []
    for i in range(3):
        for j in range(3):
            val = k_over_4A * (b[:, i] * b[:, j] + c[:, i] * c[:, j])
            row_list.append(elems[:, i])
            col_list.append(elems[:, j])
            data_list.append(val)

    rows = np.concatenate(row_list)
    cols = np.concatenate(col_list)
    data = np.concatenate(data_list)

    K = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    # Load vector: F[nᵢ] += q_vol[e] * area[e] * L_axial / 3
    F = np.zeros(n_nodes, dtype=float)
    contrib = q_vol * area * axial_length / 3.0   # (n_elems,)
    np.add.at(F, elems[:, 0], contrib)
    np.add.at(F, elems[:, 1], contrib)
    np.add.at(F, elems[:, 2], contrib)

    return K, F


# ---------------------------------------------------------------------------
# Main solver entry point
# ---------------------------------------------------------------------------

def run_thermal_analysis(
    mesh: FEAMesh,
    regions: dict,
    stator: StatorMeshInput,
    em_results: dict,
    config: dict,
) -> dict:
    """Assemble and solve the 2-D steady-state heat-conduction FEM problem.

    Steps
    -----
    1. Map EM loss densities to per-element volumetric heat sources.
    2. Assign thermal conductivity per element from material map + config.
    3. Assemble global conductance matrix ``K`` and load vector ``F``.
    4. Apply thermal boundary conditions (Robin convection on outer boundary).
    5. Check whether total heat input is negligible; if so, return ambient T.
    6. Solve sparse linear system: K·T = F.
    7. Post-process: peak temperature, hot spots, winding average, uniformity.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance (shared with EM stage).
    regions:
        Dict mapping region name → region tag.  Accepted for API consistency
        with the EM solver; ``stator.region_tags`` is used internally.
    stator:
        :class:`StatorMeshInput` instance providing geometry, fill factor, and
        region tag mapping.
    em_results:
        Result dict from :func:`run_electromagnetic_analysis`.  Required keys:
        ``"loss_density_map"``, ``"copper_loss_density_map"``, ``"domain"``.
    config:
        Thermal configuration dict (see module docstring for structure).

    Returns
    -------
    dict with keys:
        ``"peak_temperature_K"``         – float, maximum nodal temperature [K]
        ``"peak_temperature_C"``         – float, same in Celsius
        ``"winding_average_temperature_K"`` – float, area-weighted winding T [K]
        ``"hot_spot_locations"``         – dict from :func:`identify_hot_spots`
        ``"thermal_margin_K"``           – float, headroom to insulation limit [K]
        ``"T_field"``                    – np.ndarray (n_nodes,) nodal T [K]
        ``"temperature_uniformity_K"``   – float, std-dev of winding T [K]
        ``"domain"``                     – :class:`FEAMesh`
    """
    axial_length = float(stator.axial_length)

    # Ambient / coolant temperature for the near-zero-load fallback
    cooling = config.get("cooling", {})
    T_amb = float(cooling.get("coolant_temperature_K", 313.15))

    # Insulation class temperature limit
    insulation = config.get("insulation", {})
    T_max_insulation = float(insulation.get("max_temperature_K", 428.15))

    # ------------------------------------------------------------------
    # Step 1: volumetric heat sources from EM results
    # ------------------------------------------------------------------
    q_vol = map_em_losses_to_heat_sources(mesh, em_results, stator, axial_length)
    total_heat_W = float(np.sum(q_vol * mesh.element_areas() * axial_length))

    logger.info(
        "Thermal solver: total heat input = %.4f W (%.3e W/m³ mean)",
        total_heat_W,
        float(np.mean(q_vol)) if q_vol.size else 0.0,
    )

    # ------------------------------------------------------------------
    # Step 2: thermal conductivity per element
    # ------------------------------------------------------------------
    k_map = _build_conductivity_map(stator, config)
    k_elem = _conductivity_per_element(mesh.region_ids, k_map)

    # ------------------------------------------------------------------
    # Step 3: assemble global system (K as lil for BC application)
    # ------------------------------------------------------------------
    K_csr, F = _assemble_thermal_system(mesh, k_elem, q_vol, axial_length)
    K_lil = K_csr.tolil()

    # ------------------------------------------------------------------
    # Step 4: apply boundary conditions
    # ------------------------------------------------------------------
    K_bc, F_bc = apply_thermal_boundary_conditions(
        mesh, K_lil, F, stator, config
    )

    # ------------------------------------------------------------------
    # Step 5: near-zero heat-source guard
    # ------------------------------------------------------------------
    tol = float(config.get("convergence_tolerance", 1e-8))
    if abs(total_heat_W) < tol:
        logger.info(
            "Total heat source (%.3e W) is below tolerance — returning ambient "
            "temperature %.2f K everywhere.",
            total_heat_W, T_amb,
        )
        T_nodal = np.full(mesh.n_nodes, T_amb, dtype=float)
        winding_tag = stator.region_tags.get("winding", 2)
        return _build_result_dict(
            T_nodal, mesh, stator, winding_tag, T_max_insulation
        )

    # ------------------------------------------------------------------
    # Step 6: sparse direct solve
    # ------------------------------------------------------------------
    logger.debug(
        "Solving thermal system: %d DOFs, %d non-zeros",
        mesh.n_nodes, K_bc.nnz,
    )
    T_nodal = spla.spsolve(K_bc, F_bc)

    # Guard against NaN/Inf from a singular or near-singular system
    if not np.all(np.isfinite(T_nodal)):
        n_bad = int(np.sum(~np.isfinite(T_nodal)))
        logger.warning(
            "Thermal solve produced %d non-finite node values — "
            "replacing with ambient temperature %.2f K.", n_bad, T_amb
        )
        T_nodal = np.where(np.isfinite(T_nodal), T_nodal, T_amb)

    # ------------------------------------------------------------------
    # Step 7: post-processing
    # ------------------------------------------------------------------
    winding_tag = stator.region_tags.get("winding", 2)
    return _build_result_dict(T_nodal, mesh, stator, winding_tag, T_max_insulation)


# ---------------------------------------------------------------------------
# Internal: result dict construction
# ---------------------------------------------------------------------------

def _build_result_dict(
    T_nodal: np.ndarray,
    mesh: FEAMesh,
    stator: StatorMeshInput,
    winding_tag: int,
    T_max_insulation: float,
) -> dict:
    """Compute post-processed metrics and package them into the result dict."""
    T_field = extract_temperature_field(T_nodal)

    peak_T_K = float(np.max(T_field))
    peak_T_C = peak_T_K - 273.15
    thermal_margin_K = T_max_insulation - peak_T_K

    hot_spots = identify_hot_spots(T_field, threshold_fraction=0.95)

    winding_avg_T = compute_winding_average_temperature(T_field, mesh, winding_tag)

    # Temperature uniformity is assessed in the winding region
    uniformity_K = compute_temperature_uniformity(T_field, mesh, winding_tag)

    logger.info(
        "Thermal results: peak = %.2f K (%.2f °C), winding avg = %.2f K, "
        "margin = %.2f K, uniformity std = %.4f K",
        peak_T_K, peak_T_C, winding_avg_T, thermal_margin_K, uniformity_K,
    )

    return {
        "peak_temperature_K": peak_T_K,
        "peak_temperature_C": peak_T_C,
        "winding_average_temperature_K": winding_avg_T,
        "hot_spot_locations": hot_spots,
        "thermal_margin_K": thermal_margin_K,
        "T_field": T_field,
        "temperature_uniformity_K": uniformity_K,
        "domain": mesh,
    }
