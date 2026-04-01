"""Mesh reader — loads stator geometry from file or generates a synthetic mesh.

When ``mesh_format == "synthetic"`` (or no file path is given) the reader
builds a structured annular mesh directly from the geometric parameters in
``StatorMeshInput``.  This makes the whole pipeline runnable without any
external mesh file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..utils.mesh_utils import FEAMesh, make_annular_mesh
from .schema import StatorMeshInput


def load_stator_geometry(
    inp: StatorMeshInput,
) -> tuple[FEAMesh, dict[str, FEAMesh]]:
    """Load (or synthesise) the stator mesh described by *inp*.

    Parameters
    ----------
    inp:
        Stator mesh input specification.

    Returns
    -------
    mesh:
        Full ``FEAMesh`` containing all regions.
    region_submeshes:
        Dict mapping each region name to a sub-``FEAMesh`` containing only the
        elements of that region.

    Raises
    ------
    FileNotFoundError
        If a non-synthetic format is requested and the file does not exist.
    ValueError
        If the mesh format is unsupported.
    """
    if inp.mesh_format == "synthetic" or inp.mesh_file_path == "":
        mesh = _synthesise_mesh(inp)
    elif inp.mesh_format in ("gmsh4", "hdf5_xdmf", "vtk"):
        mesh = _load_from_file(inp)
    else:
        raise ValueError(f"Unsupported mesh format: {inp.mesh_format!r}")

    region_submeshes = _extract_region_submeshes(mesh, inp.region_tags)
    _validate_mesh_quality(mesh, inp)
    return mesh, region_submeshes


# ---------------------------------------------------------------------------
# Synthetic mesh
# ---------------------------------------------------------------------------

def _synthesise_mesh(inp: StatorMeshInput) -> FEAMesh:
    """Build a structured annular mesh from *inp* geometry parameters.

    Three radial regions are created:
    - ``air_gap``    : r_inner … r_inner + slot_depth * 0.15
    - ``winding``    : r_inner + slot_depth * 0.15 … r_inner + slot_depth
    - ``stator_core``: r_inner + slot_depth … r_outer

    Region integer IDs come from ``inp.region_tags``.
    """
    r_i = inp.inner_diameter / 2.0
    r_o = inp.outer_diameter / 2.0

    tags = inp.region_tags
    ag_id   = tags.get("air_gap",    3)
    wnd_id  = tags.get("winding",    2)
    core_id = tags.get("stator_core", 1)

    slot_d = inp.slot_depth
    r_ag   = r_i + slot_d * 0.15
    r_wnd  = r_i + slot_d

    # Clamp to valid range
    r_ag  = min(r_ag,  r_o * 0.95)
    r_wnd = min(r_wnd, r_o * 0.97)

    region_radii = [
        (r_i,   r_ag,  ag_id),
        (r_ag,  r_wnd, wnd_id),
        (r_wnd, r_o,   core_id),
    ]

    n_theta = max(inp.num_slots * 4, 48)
    # keep n_theta divisible by 6 (3 phases × 2 layers)
    n_theta = (n_theta // 6) * 6

    mesh = make_annular_mesh(
        r_inner=r_i,
        r_outer=r_o,
        region_radii=region_radii,
        n_radial=6,
        n_theta=n_theta,
    )
    return mesh


# ---------------------------------------------------------------------------
# File-based loading
# ---------------------------------------------------------------------------

def _load_from_file(inp: StatorMeshInput) -> FEAMesh:
    """Load mesh from a real file using meshio."""
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "meshio is required for file-based mesh loading.  "
            "Install it with: pip install meshio"
        ) from exc

    path = Path(inp.mesh_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    m = meshio.read(str(path))
    return _meshio_to_fea_mesh(m, inp.region_tags)


def _meshio_to_fea_mesh(m, region_tags: dict[str, int]) -> FEAMesh:
    """Convert a meshio.Mesh to FEAMesh (triangles only)."""
    nodes = np.asarray(m.points[:, :2], dtype=float)  # drop z

    triangles: list[np.ndarray] = []
    rids: list[int] = []

    # Collect physical group data
    phys_data = m.cell_data.get("gmsh:physical", [])

    for ci, cell_block in enumerate(m.cells):
        if cell_block.type not in ("triangle", "triangle6"):
            continue
        conn = cell_block.data
        if cell_block.type == "triangle6":
            conn = conn[:, :3]          # keep corner nodes only
        triangles.append(conn)

        tags_for_block = phys_data[ci] if ci < len(phys_data) else np.zeros(len(conn), dtype=int)
        rids.append(np.asarray(tags_for_block, dtype=np.intp))

    if not triangles:
        raise ValueError("No triangular elements found in mesh file.")

    elements = np.vstack(triangles).astype(np.intp)
    region_ids = np.concatenate(rids).astype(np.intp)

    return FEAMesh(nodes=nodes, elements=elements, region_ids=region_ids)


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _extract_region_submeshes(
    mesh: FEAMesh, region_tags: dict[str, int]
) -> dict[str, FEAMesh]:
    """Build one sub-FEAMesh per named region."""
    submeshes: dict[str, FEAMesh] = {}
    for name, tag_id in region_tags.items():
        mask = mesh.region_ids == tag_id
        if not mask.any():
            # Return empty mesh rather than raising — lets tests inspect
            submeshes[name] = FEAMesh(
                nodes=mesh.nodes,
                elements=np.zeros((0, 3), dtype=np.intp),
                region_ids=np.zeros(0, dtype=np.intp),
            )
            continue

        elem_sub = mesh.elements[mask]
        # Re-index nodes to a compact local set
        unique_nodes, inv = np.unique(elem_sub, return_inverse=True)
        nodes_sub = mesh.nodes[unique_nodes]
        elems_sub = inv.reshape(-1, 3).astype(np.intp)
        rids_sub  = mesh.region_ids[mask]

        submeshes[name] = FEAMesh(
            nodes=nodes_sub,
            elements=elems_sub,
            region_ids=rids_sub,
        )
    return submeshes


# ---------------------------------------------------------------------------
# Mesh quality validation
# ---------------------------------------------------------------------------

def _validate_mesh_quality(mesh: FEAMesh, inp: StatorMeshInput) -> None:
    areas = mesh.element_areas()
    if np.any(areas <= 0.0):
        n_bad = int((areas <= 0.0).sum())
        raise ValueError(
            f"Mesh contains {n_bad} element(s) with non-positive area "
            f"(possible node ordering issue)."
        )

    # Estimate minimum element quality as area / (max_edge² / 4√3)
    n = mesh.elements
    p = mesh.nodes
    d01 = np.linalg.norm(p[n[:, 1]] - p[n[:, 0]], axis=1)
    d12 = np.linalg.norm(p[n[:, 2]] - p[n[:, 1]], axis=1)
    d20 = np.linalg.norm(p[n[:, 0]] - p[n[:, 2]], axis=1)
    max_edge2 = np.maximum(d01, np.maximum(d12, d20)) ** 2
    ideal_area = max_edge2 * math.sqrt(3) / 4.0
    quality = areas / np.where(ideal_area > 0, ideal_area, 1.0)
    min_q = float(quality.min())

    if min_q < 0.05:
        raise ValueError(
            f"Minimum element quality {min_q:.3f} is below 0.05 — "
            "mesh is too distorted for reliable FEA."
        )
