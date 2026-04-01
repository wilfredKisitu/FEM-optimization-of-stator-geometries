"""Boundary condition assembly for the 2-D magnetostatic FEM solver.

Provides:
  - Dirichlet BC application (direct substitution, operates on copies).
  - Outer-boundary flux-confinement BC (A_z = 0 on outer ring).
  - Current-density vector assembly for the winding region.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Dirichlet boundary conditions — direct substitution
# ---------------------------------------------------------------------------

def apply_dirichlet_bcs(
    K,
    F: np.ndarray,
    bc_nodes: np.ndarray,
    bc_values: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Apply Dirichlet BCs to the global stiffness system by direct substitution.

    Zeros the affected row, sets the diagonal to 1, and adjusts the RHS so
    that ``A_z[bc_nodes] == bc_values`` in the solution.

    Parameters
    ----------
    K:
        Global stiffness matrix (any scipy sparse format).  A *copy* is made;
        the input is not modified.
    F:
        (n_nodes,) RHS load vector.  A copy is made.
    bc_nodes:
        1-D integer array of 0-based node indices where Dirichlet BCs apply.
    bc_values:
        1-D float array of prescribed values, same length as *bc_nodes*.

    Returns
    -------
    K_mod : csr_matrix
    F_mod : np.ndarray
    """
    K_mod = K.tolil()           # lil_matrix supports efficient row slicing
    F_mod = F.copy()

    for node, val in zip(bc_nodes, bc_values):
        # Adjust load vector for coupled DOFs before zeroing the row so that
        # non-BC nodes see the correct contribution:
        #   F_mod[other] -= K[other, node] * val
        # Retrieve the column as lil (already lil_matrix), then iterate.
        col_lil = K_mod.getcol(node).tolil()
        # col_lil is (n_nodes × 1); iterate over its row data
        for r, row_data in enumerate(col_lil.data):
            if row_data and r != node:
                F_mod[r] -= row_data[0] * val

        # Zero out the row and set diagonal to 1
        K_mod[node, :] = 0.0
        K_mod[node, node] = 1.0
        F_mod[node] = val

    return K_mod.tocsr(), F_mod


# ---------------------------------------------------------------------------
# Electromagnetic boundary nodes
# ---------------------------------------------------------------------------

def get_em_boundary_nodes(
    mesh,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bc_nodes, bc_values) for the outer flux-confinement BC.

    The magnetic vector potential A_z is set to zero on the outer boundary,
    which forces all flux to remain within the domain (Dirichlet zero).

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.  Must have ``boundary_node_sets["outer"]``.
    config:
        Pipeline config dict (reserved for future per-boundary overrides).

    Returns
    -------
    bc_nodes : np.ndarray
        1-D integer array of outer-boundary node indices.
    bc_values : np.ndarray
        1-D float array of zeros, same length as *bc_nodes*.
    """
    outer_nodes = mesh.boundary_node_sets.get("outer", np.array([], dtype=np.intp))
    bc_nodes = np.asarray(outer_nodes, dtype=np.intp)
    bc_values = np.zeros(len(bc_nodes), dtype=float)
    return bc_nodes, bc_values


# ---------------------------------------------------------------------------
# Current density vector
# ---------------------------------------------------------------------------

def build_current_density(
    mesh,
    stator,
    config: dict,
) -> np.ndarray:
    """Build the per-element current density J_z [A/m²] array.

    Assignment strategy (simplified 3-phase, single-phase excitation):
        Winding elements are visited in order and assigned a repeating pattern
        of  [+J, +J, -J, -J, 0, 0, ...]  so that approximately one-third of
        slots carry phase-A+ current, one-third phase-A-, and one-third are
        either phase B or C (zero in the static snapshot).

    The peak current density is:
        J_peak = I_rms * sqrt(2) * conductors_per_slot / slot_area

    where ``slot_area`` is estimated from the stator annular geometry.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    stator:
        :class:`StatorMeshInput` instance.
    config:
        Pipeline config dict (unused currently; reserved for winding map
        injection).

    Returns
    -------
    np.ndarray
        (n_elems,) float array of J_z values [A/m²].
        Non-winding elements are zero.
    """
    # Identify the winding region tag
    winding_tag = stator.region_tags.get("winding", 2)

    # Winding slot area estimate from stator annular geometry
    r_outer = stator.outer_diameter / 2.0
    r_inner = stator.inner_diameter / 2.0
    n_slots = stator.num_slots
    fill_factor = stator.fill_factor

    # Annular area occupied by all winding slots (yoke not included)
    # Use a geometric estimate: pi*(r_slot_outer² - r_slot_inner²)/n_slots
    # r_slot_inner = r_inner (bore), r_slot_outer = r_inner + slot_depth
    r_slot_inner = r_inner
    r_slot_outer = r_inner + stator.slot_depth
    slot_area_annular = np.pi * (r_slot_outer ** 2 - r_slot_inner ** 2) / n_slots

    # Fall back to depth × tooth_width if annular estimate is too small
    slot_area_rect = stator.slot_depth * stator.tooth_width
    slot_area = max(slot_area_annular, slot_area_rect)

    if slot_area <= 0.0:
        return np.zeros(mesh.n_elements, dtype=float)

    # Effective (conductor-only) area inside the slot
    conductor_area = fill_factor * slot_area

    I_rms = stator.rated_current_rms
    n_cond = stator.conductors_per_slot

    # Peak (√2) current density in A/m²
    J_peak = I_rms * np.sqrt(2.0) * n_cond / conductor_area

    J_z = np.zeros(mesh.n_elements, dtype=float)
    winding_mask = mesh.region_ids == winding_tag
    winding_indices = np.where(winding_mask)[0]

    # Pattern repeats every 6 elements (2 per phase × 3 phases):
    #   slot 0 → +J  (A+)
    #   slot 1 → +J  (A+)
    #   slot 2 → -J  (A-)
    #   slot 3 → -J  (A-)
    #   slot 4 → 0   (B — not excited in this static snapshot)
    #   slot 5 → 0   (C — not excited in this static snapshot)
    pattern = np.array([+1.0, +1.0, -1.0, -1.0, 0.0, 0.0])
    n_wind = len(winding_indices)
    if n_wind > 0:
        # Tile pattern to cover all winding elements
        phase_factors = np.tile(pattern, (n_wind // len(pattern)) + 1)[:n_wind]
        J_z[winding_indices] = J_peak * phase_factors

    return J_z
