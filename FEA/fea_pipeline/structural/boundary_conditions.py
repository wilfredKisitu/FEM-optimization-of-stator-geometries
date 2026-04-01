"""Structural boundary conditions for the 2-D plane-stress FEM solver.

Applies Dirichlet displacement BCs on the outer boundary to model the stator
being fixed inside the motor housing.  Both translational DOFs (u_x and u_y)
are fixed to zero at every outer-boundary node.

DOF layout:  node i  →  DOFs [2*i, 2*i+1]  for [u_x, u_y]

The direct-substitution (penalty-free) method is used:
  1. For each constrained DOF d:
       a. Adjust RHS of all other rows:  F[r] -= K[r, d] * 0  (= 0 since u=0)
       b. Zero the entire row and column of K corresponding to d.
       c. Set the diagonal K[d, d] = 1.
       d. Set F[d] = 0.
  Since the prescribed displacement is zero, steps (a) is trivially zero and is
  skipped for efficiency, but the column zeroing is still essential for symmetry.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from ..utils.mesh_utils import FEAMesh

logger = logging.getLogger(__name__)


def apply_structural_boundary_conditions(
    mesh: FEAMesh,
    K: lil_matrix,
    F: np.ndarray,
    config: dict,
) -> tuple[csr_matrix, np.ndarray]:
    """Fix outer boundary nodes in both DOFs (fixed frame contact).

    DOF layout: node i → DOFs [2*i, 2*i+1] for [u_x, u_y].
    Apply Dirichlet u=0 on outer boundary nodes via direct substitution:
    zero row and column, set diagonal = 1, set F[dof] = 0.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.  Must have an ``"outer"`` entry in
        ``boundary_node_sets``.
    K:
        Global stiffness matrix as a ``lil_matrix`` (2*n_nodes × 2*n_nodes).
        Modified in-place then returned as CSR.
    F:
        Global force vector (2*n_nodes,).  Modified in-place.
    config:
        Structural configuration dict (not used directly; accepted for API
        consistency with the other solver stages).

    Returns
    -------
    K_csr : csr_matrix
        Stiffness matrix with boundary conditions enforced.
    F : np.ndarray
        Modified RHS with Dirichlet values applied.
    """
    outer_nodes = mesh.boundary_node_sets.get("outer", np.array([], dtype=np.intp))
    n_outer = len(outer_nodes)

    if n_outer == 0:
        logger.warning(
            "No outer boundary nodes found — structural Dirichlet BCs not applied."
        )
        return K.tocsr(), F

    # Collect all constrained DOFs: both u_x and u_y for each outer node
    constrained_dofs: list[int] = []
    for node in outer_nodes:
        node = int(node)
        constrained_dofs.append(2 * node)       # u_x
        constrained_dofs.append(2 * node + 1)   # u_y

    constrained_set = set(constrained_dofs)

    logger.debug(
        "Applying structural Dirichlet BCs: %d outer nodes → %d constrained DOFs",
        n_outer, len(constrained_dofs),
    )

    # Direct substitution: u = 0 on all constrained DOFs.
    # Because the prescribed value is zero, the RHS adjustment
    #   F[r] -= K[r, d] * 0
    # is always zero. We still need to zero rows/columns and restore symmetry.
    for dof in constrained_dofs:
        # Zero the column to maintain symmetry (F adjustment is 0 * col = 0)
        col_data = K.getcol(dof).tocoo()
        for r in col_data.row:
            if r not in constrained_set:
                K[r, dof] = 0.0

        # Zero the row, set diagonal, fix RHS
        K[dof, :] = 0.0
        K[dof, dof] = 1.0
        F[dof] = 0.0

    return K.tocsr(), F
