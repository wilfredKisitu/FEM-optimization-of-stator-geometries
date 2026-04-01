"""Field interpolation utilities between meshes and coordinate systems."""

from __future__ import annotations

import numpy as np


def interpolate_to_points(
    mesh_nodes: np.ndarray,
    mesh_elements: np.ndarray,
    field_values: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of a node-based field to arbitrary query points.

    Uses a nearest-centroid fallback when a query point falls outside every
    element.

    Parameters
    ----------
    mesh_nodes:
        (N, 2) source mesh nodes.
    mesh_elements:
        (M, 3) triangle connectivity.
    field_values:
        (N,) or (N, k) node-based field values.
    query_points:
        (P, 2) coordinates to evaluate at.

    Returns
    -------
    ndarray (P,) or (P, k)
    """
    from scipy.spatial import cKDTree

    # Compute centroids for fast nearest lookup
    centroids = mesh_nodes[mesh_elements].mean(axis=1)  # (M, 2)
    tree = cKDTree(centroids)

    n_q = len(query_points)
    is_1d = field_values.ndim == 1
    if is_1d:
        result = np.zeros(n_q)
    else:
        result = np.zeros((n_q, field_values.shape[1]))

    for qi, pt in enumerate(query_points):
        # Find nearest element centroid
        _, elem_idx = tree.query(pt, k=1)
        nodes = mesh_elements[elem_idx]
        coords = mesh_nodes[nodes]  # (3, 2)
        lam = _barycentric(pt, coords)
        if lam is not None and np.all(lam >= -1e-10):
            lam = np.clip(lam, 0.0, None)
            lam /= lam.sum()
            if is_1d:
                result[qi] = (field_values[nodes] * lam).sum()
            else:
                result[qi] = (field_values[nodes] * lam[:, None]).sum(axis=0)
        else:
            # Fallback: use nearest node value
            _, nidx = cKDTree(mesh_nodes).query(pt)
            if is_1d:
                result[qi] = field_values[nidx]
            else:
                result[qi] = field_values[nidx]

    return result


def _barycentric(
    point: np.ndarray, triangle: np.ndarray
) -> np.ndarray | None:
    """Compute barycentric coordinates of *point* in *triangle*.

    Parameters
    ----------
    point:
        (2,) query point.
    triangle:
        (3, 2) vertex coordinates.

    Returns
    -------
    (3,) barycentric coordinates, or None if the triangle is degenerate.
    """
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


def radial_average(
    mesh: "FEAMesh",  # noqa: F821 — avoid circular import
    field: np.ndarray,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial average of an element-based field.

    Parameters
    ----------
    mesh:
        Source FEAMesh.
    field:
        (n_elems,) element-based field values.
    n_bins:
        Number of radial bins.

    Returns
    -------
    r_centres, avg_values — each of shape (n_bins,).
    """
    centroids = mesh.element_centroids()
    radii = np.linalg.norm(centroids, axis=1)

    r_min, r_max = radii.min(), radii.max()
    bins = np.linspace(r_min, r_max, n_bins + 1)
    r_centres = 0.5 * (bins[:-1] + bins[1:])
    avg_values = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (radii >= bins[b]) & (radii < bins[b + 1])
        if mask.any():
            avg_values[b] = field[mask].mean()

    return r_centres, avg_values
