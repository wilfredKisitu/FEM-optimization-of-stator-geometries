"""Load mappers for the 2-D plane-stress structural FEM solver.

Two load types are supported:

1. **Thermal expansion load** — equivalent nodal forces arising from
   constrained thermal strains (temperature difference from reference).

2. **Maxwell stress electromagnetic load** — radial pressure on the inner
   (air-gap) boundary from the magnetic flux density field.

Both functions return global nodal force vectors of length ``2 * n_nodes``
with the DOF ordering:

    DOF[2*i]     = u_x force on node i
    DOF[2*i + 1] = u_y force on node i

Thermal load derivation (plane stress)
---------------------------------------
For a CST element *e* the thermal strain vector is:

    eps_th = alpha_e * (T_avg_e - T_ref) * [1, 1, 0]^T

The equivalent nodal forces are:

    F_th_e = -A_e * B_e^T @ D_e @ eps_th_e

where the negative sign means the stiffness resists thermal expansion:
the external force needed to hold the body at zero displacement equals the
negative of what the stiffness would produce from the free thermal strain.

D_e (plane stress):
    E/(1-nu²) * [[1, nu, 0],
                 [nu, 1, 0],
                 [0,  0, (1-nu)/2]]

B_e:
    1/(2A) * [[b0, 0,  b1, 0,  b2, 0 ],
              [0,  c0, 0,  c1, 0,  c2],
              [c0, b0, c1, b1, c2, b2]]

Maxwell stress load derivation
-------------------------------
For linear CST elements the flux density B is constant within each element,
so element-interior body forces (which involve spatial gradients of B) are
identically zero.  The dominant EM forcing is therefore a surface traction on
the inner (air-gap) boundary.

For an air-gap boundary edge with unit outward normal n̂ = (n_r, n_θ) the
Maxwell traction is:

    T_i = (1/μ₀) * (B·n̂) B_i  -  (B·B)/(2μ₀) n_i

Projected onto the radial direction at node position (x, y):

    f_radial = (B_r² - B_t²) / (2μ₀)

where B_r and B_t are the radial and tangential components of B at the
element centroid nearest to the boundary node.  This is distributed as an
equivalent nodal force over the two nodes of each boundary edge.
"""

from __future__ import annotations

import logging

import numpy as np

from ..utils.mesh_utils import FEAMesh
from ..utils.units import MU_0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thermal load
# ---------------------------------------------------------------------------

def compute_thermal_expansion_load(
    mesh: FEAMesh,
    T_nodal: np.ndarray,
    E_elem: np.ndarray,
    nu_elem: np.ndarray,
    alpha_elem: np.ndarray,
    T_ref_K: float,
) -> np.ndarray:
    """Compute global thermal load vector from temperature field.

    For plane stress, the thermal strain is:
        eps_th = [1, 1, 0] * alpha * (T_avg_elem - T_ref)

    The thermal load contribution per element:
        F_th_e = -Area * B_e.T @ D_e @ eps_th_e

    where:
        D_e = E/(1-nu²) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]
        B_e = 1/(2A) * [[b0,0,b1,0,b2,0],[0,c0,0,c1,0,c2],[c0,b0,c1,b1,c2,b2]]

    The sign is negative: thermal expansion resisted by stiffness produces
    equivalent nodal forces opposite to the free thermal strain direction.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    T_nodal:
        (n_nodes,) nodal temperature [K].
    E_elem:
        (n_elems,) Young's modulus [Pa] per element.
    nu_elem:
        (n_elems,) Poisson's ratio per element.
    alpha_elem:
        (n_elems,) thermal expansion coefficient [1/K] per element.
    T_ref_K:
        Reference (stress-free) temperature [K].

    Returns
    -------
    F_thermal : np.ndarray
        Global load vector (2*n_nodes,).
    """
    b, c, area = mesh.gradient_operators()   # (n_elems, 3), (n_elems, 3), (n_elems,)
    elems = mesh.elements                    # (n_elems, 3)
    n_elems = mesh.n_elements
    n_nodes = mesh.n_nodes

    # Average temperature at each element centroid
    T_elem = T_nodal[elems].mean(axis=1)    # (n_elems,)
    dT = T_elem - T_ref_K                   # (n_elems,)

    # Plane-stress D matrix coefficients per element
    # D = E/(1-nu²) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
    fac = E_elem / (1.0 - nu_elem ** 2)     # (n_elems,)
    D11 = fac                                # E/(1-nu²)
    D12 = fac * nu_elem                      # E*nu/(1-nu²)
    D33 = fac * (1.0 - nu_elem) / 2.0       # E*(1-nu)/(2*(1-nu²))

    # Thermal strain vector: eps_th = alpha * dT * [1, 1, 0]
    eps_th_x = alpha_elem * dT              # (n_elems,)  component 0
    eps_th_y = alpha_elem * dT              # (n_elems,)  component 1
    # eps_th_xy = 0                          # component 2

    # D @ eps_th — result is a (n_elems, 3) stress-like vector
    # sigma_th[0] = D11*eps_th_x + D12*eps_th_y
    # sigma_th[1] = D12*eps_th_x + D11*eps_th_y
    # sigma_th[2] = D33 * 0 = 0
    sig_th_0 = D11 * eps_th_x + D12 * eps_th_y   # (n_elems,)
    sig_th_1 = D12 * eps_th_x + D11 * eps_th_y   # (n_elems,)
    # sig_th_2 = 0

    # B^T @ sigma_th gives a 6-vector per element.
    # B = 1/(2A) * [[b0, 0, b1, 0, b2, 0],
    #               [0, c0, 0, c1, 0, c2],
    #               [c0, b0, c1, b1, c2, b2]]
    # B^T has rows indexed by DOF (0..5) and cols by stress components (0..2).
    # Row 2*i  : [b_i/(2A), 0,        c_i/(2A)]
    # Row 2*i+1: [0,        c_i/(2A), b_i/(2A)]
    #
    # (B^T @ sigma_th)[2*i]   = b_i/(2A)*sig0 + c_i/(2A)*sig2
    #                         = b_i/(2A)*sig0  (since sig2=0)
    # (B^T @ sigma_th)[2*i+1] = c_i/(2A)*sig1 + b_i/(2A)*sig2
    #                         = c_i/(2A)*sig1
    #
    # F_th_e = -area * B^T @ sigma_th
    # F_th_e[2*i]   = -area * b_i/(2A) * sig0 = -(b_i/2) * sig0
    # F_th_e[2*i+1] = -area * c_i/(2A) * sig1 = -(c_i/2) * sig1

    F_thermal = np.zeros(2 * n_nodes, dtype=float)

    for i in range(3):
        # x-DOF for local node i
        fx = -(b[:, i] / 2.0) * sig_th_0   # (n_elems,)
        # y-DOF for local node i
        fy = -(c[:, i] / 2.0) * sig_th_1   # (n_elems,)

        dof_x = 2 * elems[:, i]
        dof_y = 2 * elems[:, i] + 1
        np.add.at(F_thermal, dof_x, fx)
        np.add.at(F_thermal, dof_y, fy)

    logger.debug(
        "Thermal load: max nodal force = %.4e N, total dT range [%.2f, %.2f] K",
        float(np.max(np.abs(F_thermal))),
        float(dT.min()), float(dT.max()),
    )

    return F_thermal


# ---------------------------------------------------------------------------
# Maxwell stress load
# ---------------------------------------------------------------------------

def compute_maxwell_stress_load(
    mesh: FEAMesh,
    B_field: dict,
    config: dict,
) -> np.ndarray:
    """Compute nodal force vector from Maxwell stress on the air-gap boundary.

    For piecewise-constant B on linear triangles the volumetric body force
    is zero (B has no spatial gradient within each element).  Instead, a
    surface traction is applied to the inner (air-gap) boundary nodes.

    For each inner-boundary node the nearest element centroid is used to
    obtain B_x, B_y.  The radial and tangential B components are then:

        B_r =  B_x * cos(θ) + B_y * sin(θ)
        B_t = -B_x * sin(θ) + B_y * cos(θ)

    The net radial Maxwell pressure is:

        p = (B_r² - B_t²) / (2μ₀)

    This pressure is distributed as a force along the arc-length segment
    attributed to each boundary node, applied in the outward radial direction.

    If ``config.get("electromagnetic_loads", True)`` is False the function
    returns a zero vector without performing any computation.

    Parameters
    ----------
    mesh:
        :class:`FEAMesh` instance.
    B_field:
        Dict with keys ``"B_x"``, ``"B_y"``, ``"B_mag"`` — each a (n_elems,)
        array of element-averaged flux density components [T].
    config:
        Structural configuration dict.

    Returns
    -------
    F_em : np.ndarray
        Global nodal force vector (2*n_nodes,).
    """
    n_nodes = mesh.n_nodes
    F_em = np.zeros(2 * n_nodes, dtype=float)

    if not config.get("electromagnetic_loads", True):
        logger.debug("EM loads disabled by config — returning zero EM force vector.")
        return F_em

    inner_nodes = mesh.boundary_node_sets.get("inner", np.array([], dtype=np.intp))
    n_inner = len(inner_nodes)

    if n_inner == 0:
        logger.warning("No inner boundary nodes — Maxwell stress load not applied.")
        return F_em

    B_x = B_field.get("B_x", np.zeros(mesh.n_elements))
    B_y = B_field.get("B_y", np.zeros(mesh.n_elements))

    # Element centroids — used to find nearest element for each boundary node
    centroids = mesh.element_centroids()    # (n_elems, 2)

    # Compute mean radius of the inner boundary for the arc-length estimate
    xy_inner = mesh.nodes[inner_nodes]      # (n_inner, 2)
    r_inner_mean = float(np.mean(np.sqrt(xy_inner[:, 0] ** 2 + xy_inner[:, 1] ** 2)))
    arc_per_node = 2.0 * np.pi * r_inner_mean / n_inner   # [m] arc-length per node

    for node_idx in inner_nodes:
        node_idx = int(node_idx)
        x_n = mesh.nodes[node_idx, 0]
        y_n = mesh.nodes[node_idx, 1]

        # Find nearest element centroid
        dist2 = (centroids[:, 0] - x_n) ** 2 + (centroids[:, 1] - y_n) ** 2
        e_near = int(np.argmin(dist2))

        bx = float(B_x[e_near])
        by = float(B_y[e_near])

        # Radial unit vector at this node
        r_node = np.sqrt(x_n ** 2 + y_n ** 2)
        if r_node < 1e-15:
            continue
        cos_t = x_n / r_node
        sin_t = y_n / r_node

        # Radial and tangential B components
        B_r = bx * cos_t + by * sin_t
        B_t = -bx * sin_t + by * cos_t

        # Maxwell radial pressure [Pa]
        pressure = (B_r ** 2 - B_t ** 2) / (2.0 * MU_0)

        # Force on this node = pressure * arc-length * (unit radial direction)
        # Positive pressure → outward (away from axis)
        force_mag = pressure * arc_per_node   # [N]
        F_em[2 * node_idx]     += force_mag * cos_t   # x-component
        F_em[2 * node_idx + 1] += force_mag * sin_t   # y-component

    logger.debug(
        "Maxwell stress load: max nodal force = %.4e N on %d inner nodes",
        float(np.max(np.abs(F_em))), n_inner,
    )

    return F_em
