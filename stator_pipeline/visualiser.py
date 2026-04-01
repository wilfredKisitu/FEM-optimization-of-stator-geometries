"""visualiser.py — Stator geometry and mesh visualisation helpers."""
from __future__ import annotations
from typing import Optional

_REGION_COLOURS = {
    "YOKE":          "#4a90d9",
    "TOOTH":         "#2c5f8a",
    "SLOT_AIR":      "#b0d4f1",
    "SLOT_INS":      "#f5e642",
    "COIL_A_POS":    "#e63946",
    "COIL_A_NEG":    "#ff8fa3",
    "COIL_B_POS":    "#2a9d8f",
    "COIL_B_NEG":    "#80cdc1",
    "COIL_C_POS":    "#f4a261",
    "COIL_C_NEG":    "#ffd6a5",
    "BORE_AIR":      "#e8f4f8",
    "BOUNDARY_BORE": "#264653",
    "BOUNDARY_OUTER":"#1a1a2e",
}


class StatorVisualiser:
    """Load and render stator VTK / mesh output files."""

    def __init__(self) -> None:
        self._vtk_available = False
        self._mpl_available = False
        try:
            import matplotlib  # noqa: F401
            self._mpl_available = True
        except ImportError:
            pass

    def plot_cross_section(self, vtk_path: str, output_png: Optional[str] = None) -> None:
        """Render the 2-D cross-section, coloured by RegionType.

        Tries vtk/pyvtk first; falls back to manual VTK ASCII parsing.
        Requires matplotlib.
        """
        if not self._mpl_available:
            raise ImportError("matplotlib is required for visualisation")

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import numpy as np

        points, cells, scalars = self._load_vtk(vtk_path)

        fig, ax = plt.subplots(figsize=(8, 8))
        patches = []
        colours = []
        for cell in cells:
            verts = points[cell]
            patches.append(Polygon(verts))
            colours.append(scalars[len(patches) - 1] if scalars else 0)

        collection = PatchCollection(patches, cmap="tab20", alpha=0.8)
        collection.set_array(np.array(colours))
        ax.add_collection(collection)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_title("Stator Cross-Section")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        if output_png:
            plt.savefig(output_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_mesh(self, vtk_path: str, show_quality: bool = True) -> None:
        """Colour mesh elements by element quality scalar."""
        if not self._mpl_available:
            raise ImportError("matplotlib is required for visualisation")

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import numpy as np

        points, cells, scalars = self._load_vtk(vtk_path)

        fig, ax = plt.subplots(figsize=(8, 8))
        patches = [Polygon(points[c]) for c in cells]
        colours = scalars if scalars else [0] * len(cells)

        col = PatchCollection(patches, cmap="RdYlGn", alpha=0.9)
        col.set_array(np.array(colours))
        ax.add_collection(col)
        plt.colorbar(col, ax=ax, label="Element quality" if show_quality else "Region")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_title("Stator Mesh Quality" if show_quality else "Stator Mesh")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.show()

    # ─── VTK loader ───────────────────────────────────────────────────────────

    def _load_vtk(self, vtk_path: str):
        """Returns (points, cells, scalars). Falls back to manual ASCII parse."""
        try:
            return self._load_vtk_via_library(vtk_path)
        except Exception:
            return self._load_vtk_ascii(vtk_path)

    def _load_vtk_via_library(self, vtk_path: str):
        import vtk  # type: ignore
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(vtk_path)
        reader.Update()
        grid = reader.GetOutput()
        pts = []
        for i in range(grid.GetNumberOfPoints()):
            p = grid.GetPoint(i)
            pts.append([p[0], p[1]])
        import numpy as np
        points = np.array(pts)
        cells = []
        scalars = []
        pd = grid.GetCellData().GetArray(0)
        for i in range(grid.GetNumberOfCells()):
            cell = grid.GetCell(i)
            ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            cells.append(ids)
            scalars.append(pd.GetValue(i) if pd else 0)
        return points, cells, scalars

    def _load_vtk_ascii(self, vtk_path: str):
        """Minimal VTK legacy ASCII parser for UNSTRUCTURED_GRID."""
        import numpy as np
        points = []
        cells  = []
        scalars = []
        section = None
        n_points = 0
        n_cells  = 0
        with open(vtk_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("POINTS"):
                    parts  = line.split()
                    n_points = int(parts[1])
                    section  = "POINTS"
                elif line.startswith("CELLS"):
                    parts   = line.split()
                    n_cells = int(parts[1])
                    section = "CELLS"
                elif line.startswith("CELL_TYPES"):
                    section = "SKIP"
                elif line.startswith("SCALARS"):
                    section = "SCALARS_HDR"
                elif line.startswith("LOOKUP_TABLE"):
                    section = "SCALARS"
                elif section == "POINTS" and len(points) < n_points:
                    vals = list(map(float, line.split()))
                    for i in range(0, len(vals), 3):
                        points.append([vals[i], vals[i + 1]])
                elif section == "CELLS" and len(cells) < n_cells:
                    vals = list(map(int, line.split()))
                    cells.append(vals[1:])
                elif section == "SCALARS" and len(scalars) < n_cells:
                    scalars.append(float(line))
        return np.array(points) if points else np.zeros((0, 2)), cells, scalars
