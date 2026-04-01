"""Thermal boundary conditions for the 2-D thermal FEM solver.

Supports:
  - Robin (convective) BCs for water-jacket and natural-convection cooling.
  - Dirichlet (fixed temperature) BCs for prescribed outer-boundary temperature.

The outer boundary perimeter is treated as a uniform heat-transfer surface.
Each outer-boundary node receives a share of the total perimeter equal to:

    L_per_node = 2 * pi * r_outer / n_outer_nodes

This is a valid lumped approximation for the structured annular mesh where all
outer-ring nodes are equally spaced in angle.

Robin BC contribution for node i (per-node perimeter segment L_i):
    K[i, i] += h * L_i * L_axial      (adds to diagonal)
    F[i]    += h * T_inf * L_i * L_axial

The axial_length factor converts the 2-D perimeter segment length to a 3-D
surface area, consistent with how the volume heat sources are scaled.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helper: arc-length per boundary node
# ---------------------------------------------------------------------------

def get_boundary_segment_lengths(mesh, boundary_name: str) -> np.ndarray:
    """Compute the arc length [m] associated with each node on *boundary_name*.

    For the outer or inner rings of the structured annular mesh the nodes are
    equally spaced in angle, so every node receives the same segment length:

        L_i = 2 * pi * r / n_nodes_on_boundary

    where *r* is estimated as the mean radial distance of the boundary nodes.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    boundary_name:
        Key into ``mesh.boundary_node_sets`` — typically ``"outer"`` or
        ``"inner"``.

    Returns
    -------
    np.ndarray
        (n_boundary_nodes,) float array of arc lengths [m].
    """
    node_indices = mesh.boundary_node_sets.get(boundary_name, np.array([], dtype=np.intp))
    n = len(node_indices)
    if n == 0:
        return np.array([], dtype=float)

    xy = mesh.nodes[node_indices]                        # (n, 2)
    r_mean = float(np.mean(np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)))
    perimeter = 2.0 * np.pi * r_mean
    L_per_node = perimeter / n
    return np.full(n, L_per_node, dtype=float)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_thermal_boundary_conditions(
    mesh,
    K: lil_matrix,
    F: np.ndarray,
    stator,
    config: dict,
) -> tuple[csr_matrix, np.ndarray]:
    """Apply thermal boundary conditions to the global stiffness system.

    For **water_jacket** cooling the outer boundary receives a Robin
    (convective) BC:

        q_n = h * (T - T_inf)

    which contributes:
        K[i, i] += h * L_i * L_axial
        F[i]    += h * T_inf * L_i * L_axial

    For **fixed_temperature** cooling the outer boundary receives a Dirichlet
    BC:
        T[i] = T_inf  (prescribed temperature)

    Any other cooling type falls back to natural convection with h = 50 W/m²K.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    K:
        Global stiffness matrix as a ``lil_matrix`` (will be modified in-place
        then converted to CSR).
    F:
        (n_nodes,) RHS load vector (modified in-place).
    stator:
        :class:`StatorMeshInput` instance — provides *axial_length*.
    config:
        Thermal configuration dict.  Must contain the ``"cooling"`` sub-dict.

    Returns
    -------
    K_csr : csr_matrix
    F : np.ndarray

    Raises
    ------
    KeyError
        If ``config["cooling"]`` is absent.
    """
    cooling = config["cooling"]           # KeyError propagates intentionally
    cooling_type = cooling.get("type", "water_jacket")
    T_inf = float(cooling.get("coolant_temperature_K", 313.15))
    axial_length = float(stator.axial_length)

    outer_nodes = mesh.boundary_node_sets.get("outer", np.array([], dtype=np.intp))
    n_outer = len(outer_nodes)

    if n_outer == 0:
        logger.warning("No outer boundary nodes found — thermal BCs not applied.")
        return K.tocsr(), F

    seg_lengths = get_boundary_segment_lengths(mesh, "outer")   # (n_outer,)

    if cooling_type == "fixed_temperature":
        # Dirichlet BC: set T = T_inf on outer boundary
        logger.debug(
            "Applying Dirichlet thermal BC: T_outer = %.2f K on %d nodes",
            T_inf, n_outer,
        )
        _apply_dirichlet_thermal(K, F, outer_nodes, T_inf)

    else:
        # Robin (convective) BC
        if cooling_type == "water_jacket":
            h = float(cooling.get("h_outer", 500.0))
        else:
            # Natural convection fallback
            h = 50.0
            logger.debug(
                "Unknown cooling type '%s' — falling back to natural convection "
                "(h = 50 W/m²K).", cooling_type,
            )

        logger.debug(
            "Applying Robin thermal BC: h = %.1f W/m²K, T_inf = %.2f K on %d nodes",
            h, T_inf, n_outer,
        )

        for idx, node_i in enumerate(outer_nodes):
            L_i = float(seg_lengths[idx])
            area_contrib = L_i * axial_length          # surface area attributed to node_i [m²]
            K[node_i, node_i] += h * area_contrib
            F[node_i] += h * T_inf * area_contrib

    return K.tocsr(), F


# ---------------------------------------------------------------------------
# Internal: Dirichlet application for thermal DOFs
# ---------------------------------------------------------------------------

def _apply_dirichlet_thermal(
    K: lil_matrix,
    F: np.ndarray,
    bc_nodes: np.ndarray,
    bc_value: float,
) -> None:
    """Apply Dirichlet BC T = bc_value on bc_nodes (modifies K and F in-place).

    Uses the direct-substitution (penalty-free) approach:
      1. Adjust F for all non-BC rows using the off-diagonal column values.
      2. Zero the row and set the diagonal to 1.
      3. Set F[node] = bc_value.

    The matrix is expected to be a ``lil_matrix`` for efficient row/column
    access.
    """
    bc_set = set(int(n) for n in bc_nodes)

    for node in bc_nodes:
        node = int(node)
        # Adjust coupled non-BC nodes: F[r] -= K[r, node] * bc_value
        col = K.getcol(node).tolil()
        for r, row_data in enumerate(col.data):
            if row_data and r not in bc_set:
                F[r] -= row_data[0] * bc_value

        # Enforce Dirichlet on this row
        K[node, :] = 0.0
        K[node, node] = 1.0
        F[node] = bc_value
