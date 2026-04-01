"""Core mesh data structure and synthetic mesh generator.

The FEAMesh dataclass is the single mesh representation shared across all
three solver stages.  The make_annular_mesh factory creates structured
triangular meshes in polar coordinates — used by the test suite without
requiring any external mesh file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FEAMesh:
    """Triangular 2-D finite-element mesh.

    All coordinates are in SI units (metres).

    Attributes
    ----------
    nodes:
        Float array of shape ``(n_nodes, 2)`` — (x, y) positions.
    elements:
        Integer array of shape ``(n_elems, 3)`` — 0-based node indices per
        triangle (counter-clockwise convention).
    region_ids:
        Integer array of shape ``(n_elems,)`` — physical region tag for each
        triangle.  Tags must match the keys in ``region_tags`` from
        ``StatorMeshInput``.
    boundary_node_sets:
        Dict mapping boundary-name → array of 0-based node indices that lie on
        that boundary.  Standard keys: ``"outer"``, ``"inner"``.
    """

    nodes: np.ndarray
    elements: np.ndarray
    region_ids: np.ndarray
    boundary_node_sets: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def element_centroids(self) -> np.ndarray:
        """Return (n_elems, 2) array of element barycentres."""
        return self.nodes[self.elements].mean(axis=1)

    def element_areas(self) -> np.ndarray:
        """Return (n_elems,) array of signed element areas (positive CCW)."""
        n = self.elements
        x = self.nodes[n, 0]
        y = self.nodes[n, 1]
        return 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                      - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))

    def gradient_operators(self):
        """Compute per-element gradient shape-function coefficients.

        Returns
        -------
        b : ndarray (n_elems, 3)
            b[e, i] = y_{i+1} - y_{i+2}  (cyclic) — x-gradient factor.
        c : ndarray (n_elems, 3)
            c[e, i] = x_{i+2} - x_{i+1}  (cyclic) — y-gradient factor.
        area : ndarray (n_elems,)
            Unsigned triangle area.
        """
        n = self.elements
        x = self.nodes[n, 0]  # (n_elems, 3)
        y = self.nodes[n, 1]

        b = np.stack([y[:, 1] - y[:, 2],
                      y[:, 2] - y[:, 0],
                      y[:, 0] - y[:, 1]], axis=1)

        c = np.stack([x[:, 2] - x[:, 1],
                      x[:, 0] - x[:, 2],
                      x[:, 1] - x[:, 0]], axis=1)

        area = np.abs(self.element_areas())
        return b, c, area


def make_annular_mesh(
    r_inner: float,
    r_outer: float,
    region_radii: list[tuple[float, float, int]],
    n_radial: int = 8,
    n_theta: int = 48,
) -> FEAMesh:
    """Create a structured triangular mesh of an annulus.

    Parameters
    ----------
    r_inner:
        Inner radius of the annulus [m].
    r_outer:
        Outer radius of the annulus [m].
    region_radii:
        List of ``(r_min, r_max, region_id)`` tuples defining the radial extent
        of each physical region.  Radii must tile ``[r_inner, r_outer]``
        contiguously.
    n_radial:
        Number of radial node layers *per region*.
    n_theta:
        Number of angular divisions (must divide evenly into 3 for winding
        phase assignment).

    Returns
    -------
    FEAMesh
        Structured mesh with ``n_theta * (n_radial * n_regions + 1)`` nodes and
        ``2 * n_theta * n_radial * n_regions`` triangular elements.
    """
    # Build the radial node levels spanning all regions
    r_levels: list[float] = []
    for r_min, r_max, _ in region_radii:
        levels = np.linspace(r_min, r_max, n_radial + 1)
        if r_levels and abs(r_levels[-1] - levels[0]) < 1e-12:
            r_levels.extend(levels[1:].tolist())
        else:
            r_levels.extend(levels.tolist())

    r_arr = np.array(r_levels)           # n_r levels
    theta_arr = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)

    n_r = len(r_arr)
    n_t = n_theta

    # Node coordinates
    nodes = np.zeros((n_r * n_t, 2))
    for ri, r in enumerate(r_arr):
        for ti, th in enumerate(theta_arr):
            idx = ri * n_t + ti
            nodes[idx, 0] = r * math.cos(th)
            nodes[idx, 1] = r * math.sin(th)

    # Elements — two triangles per quad cell
    elements: list[list[int]] = []
    region_ids_list: list[int] = []

    # Build a lookup for which region_id each r_level pair belongs to
    def _region_for_r(r_lo: float, r_hi: float) -> int:
        r_mid = 0.5 * (r_lo + r_hi)
        for r_min, r_max, rid in region_radii:
            if r_min - 1e-12 <= r_mid <= r_max + 1e-12:
                return rid
        return -1

    for ri in range(n_r - 1):
        r_lo = r_arr[ri]
        r_hi = r_arr[ri + 1]
        rid = _region_for_r(r_lo, r_hi)
        for ti in range(n_t):
            ti_next = (ti + 1) % n_t
            # Quad corners (CCW)
            n00 = ri * n_t + ti
            n01 = ri * n_t + ti_next
            n10 = (ri + 1) * n_t + ti
            n11 = (ri + 1) * n_t + ti_next
            # Split quad into 2 triangles (CCW)
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])
            region_ids_list.extend([rid, rid])

    elements_arr = np.array(elements, dtype=np.intp)
    region_ids_arr = np.array(region_ids_list, dtype=np.intp)

    # Boundary node sets
    outer_ring_idx = (n_r - 1) * n_t
    inner_ring_idx = 0
    outer_nodes = np.arange(outer_ring_idx, outer_ring_idx + n_t, dtype=np.intp)
    inner_nodes = np.arange(inner_ring_idx, inner_ring_idx + n_t, dtype=np.intp)

    mesh = FEAMesh(
        nodes=nodes,
        elements=elements_arr,
        region_ids=region_ids_arr,
        boundary_node_sets={"outer": outer_nodes, "inner": inner_nodes},
    )
    return mesh


def node_to_element_average(mesh: FEAMesh, node_values: np.ndarray) -> np.ndarray:
    """Average a node-based field to element centroids (arithmetic mean)."""
    return node_values[mesh.elements].mean(axis=1)


def element_to_node_average(mesh: FEAMesh, elem_values: np.ndarray) -> np.ndarray:
    """Scatter-average element values to nodes (area-weighted)."""
    areas = np.abs(mesh.element_areas())
    node_vals = np.zeros(mesh.n_nodes)
    node_weights = np.zeros(mesh.n_nodes)
    for e in range(mesh.n_elements):
        for local_n in mesh.elements[e]:
            node_vals[local_n] += elem_values[e] * areas[e]
            node_weights[local_n] += areas[e]
    mask = node_weights > 0
    node_vals[mask] /= node_weights[mask]
    return node_vals
