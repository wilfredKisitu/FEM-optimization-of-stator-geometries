# FEA Pipeline — API Reference & Integration Guide

This document is the authoritative reference for the `fea_pipeline` package. It covers every public function and class, their parameters, return values, and the data that flows between pipeline stages. It also explains how to connect the upstream mesh-generation module (`stator_pipeline`) to this FEA package.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Installation & Dependencies](#2-installation--dependencies)
3. [Data Structures](#3-data-structures)
   - 3.1 [StatorMeshInput](#31-statormeshinput)
   - 3.2 [FEAMesh](#32-feamesh)
   - 3.3 [PipelineResults](#33-pipelineresults)
4. [IO Layer](#4-io-layer)
   - 4.1 [load_stator_geometry](#41-load_stator_geometry)
   - 4.2 [write_results](#42-write_results)
5. [Electromagnetic Stage](#5-electromagnetic-stage)
   - 5.1 [run_electromagnetic_analysis](#51-run_electromagnetic_analysis)
   - 5.2 [Material Library](#52-material-library)
   - 5.3 [Loss Calculator](#53-loss-calculator)
   - 5.4 [Boundary Conditions](#54-boundary-conditions-em)
   - 5.5 [Post-processor](#55-post-processor-em)
6. [Thermal Stage](#6-thermal-stage)
   - 6.1 [run_thermal_analysis](#61-run_thermal_analysis)
   - 6.2 [Heat Source Mapper](#62-heat-source-mapper)
   - 6.3 [Thermal Boundary Conditions](#63-thermal-boundary-conditions)
   - 6.4 [Thermal Post-processor](#64-thermal-post-processor)
7. [Structural Stage](#7-structural-stage)
   - 7.1 [run_structural_analysis](#71-run_structural_analysis)
   - 7.2 [Load Mapper](#72-load-mapper)
   - 7.3 [Structural Post-processor](#73-structural-post-processor)
8. [Orchestrator](#8-orchestrator)
   - 8.1 [run_fea_pipeline](#81-run_fea_pipeline)
9. [Utility Functions](#9-utility-functions)
   - 9.1 [Mesh Utilities](#91-mesh-utilities)
   - 9.2 [Unit Conversions](#92-unit-conversions)
   - 9.3 [Interpolation Helpers](#93-interpolation-helpers)
10. [Configuration Reference](#10-configuration-reference)
11. [Chaining with the Mesh Generation Pipeline](#11-chaining-with-the-mesh-generation-pipeline)
12. [End-to-End Examples](#12-end-to-end-examples)

---

## 1. Architecture Overview

The FEA package implements a three-stage coupled multiphysics solver built on pure NumPy and SciPy — no FEA framework or compiled extensions are required.

```
stator_pipeline (upstream)
       │  StatorConfig → exported JSON / StatorMeshInput
       ▼
┌─────────────────────────────────────────────────────┐
│                  fea_pipeline                        │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐    │
│  │   IO     │   │  Utils   │   │  Orchestrator│    │
│  │ schema   │   │mesh_utils│   │  (top-level) │    │
│  │mesh_rdr  │   │  units   │   │              │    │
│  │result_wr │   │  interp  │   │              │    │
│  └──────────┘   └──────────┘   └──────┬───────┘    │
│                                        │            │
│          ┌─────────────────────────────┘            │
│          │ shared FEAMesh                           │
│          ▼                                          │
│  ┌──────────────┐                                   │
│  │Electromagnetic│  A_z, B, losses                  │
│  │    Stage     │──────────────────────────────┐   │
│  └──────────────┘                              │   │
│          │ loss_density_map                    │   │
│          ▼                                     │   │
│  ┌──────────────┐                              │   │
│  │   Thermal    │  T_field                     │   │
│  │    Stage     │───────────────────────┐      │   │
│  └──────────────┘                       │      │   │
│          │ T_field                      │      │   │
│          ▼                              ▼      │   │
│  ┌──────────────┐                              │   │
│  │  Structural  │← F_thermal + F_maxwell ──────┘   │
│  │    Stage     │  σ_vm, u, f_nat                  │
│  └──────────────┘                                   │
│          │                                          │
│          ▼                                          │
│    results/ (JSON)                                  │
└─────────────────────────────────────────────────────┘
```

Each stage uses the **same** `FEAMesh` object. No field interpolation between stages is needed because all three solvers share identical node and element numbering.

---

## 2. Installation & Dependencies

### Minimum requirements

```bash
# From the FEA directory
pip install -e .
```

`pyproject.toml` specifies:

| Package       | Version | Role                                 |
|---------------|---------|--------------------------------------|
| `numpy`       | ≥1.24   | Array operations, COO assembly       |
| `scipy`       | ≥1.10   | Sparse matrices, direct sparse solve |
| `pyyaml`      | ≥6.0    | Config file loading                  |

### Optional extras

```bash
pip install -e ".[full]"     # meshio, h5py, pydantic
```

| Package   | Role                                           |
|-----------|------------------------------------------------|
| `meshio`  | Reading external mesh files (.msh, .vtk, .xdmf)|
| `h5py`    | HDF5 result writing (future)                   |
| `pydantic`| Schema validation with field-level errors      |

Without `pydantic`, the package falls back to plain Python `dataclasses` — all functionality remains available but validation error messages are less descriptive.

### Running the test suite

```bash
cd FEA
pytest tests/ -v          # all 123 tests
pytest tests/unit/ -v     # unit tests only
pytest tests/validation/  # physics benchmarks
```

---

## 3. Data Structures

### 3.1 `StatorMeshInput`

**Module:** `fea_pipeline.io.schema`

The main input contract. Constructed once and passed unmodified to every pipeline stage. When Pydantic is installed, all fields are validated on construction. When it is not, a plain dataclass is used.

```python
from fea_pipeline.io.schema import StatorMeshInput

stator = StatorMeshInput(
    stator_id       = "gen_650",
    outer_diameter  = 1.300,       # metres
    inner_diameter  = 0.840,       # metres (bore diameter)
    axial_length    = 0.700,       # metres (stack length)
    num_slots       = 48,
    num_poles       = 4,
    slot_opening    = 0.008,       # metres
    tooth_width     = 0.022,       # metres
    yoke_height     = 0.085,       # metres
    slot_depth      = 0.115,       # metres
)
```

#### Required fields

| Field            | Type    | Unit | Description                                      |
|------------------|---------|------|--------------------------------------------------|
| `stator_id`      | `str`   | —    | Unique identifier, used as output sub-directory  |
| `outer_diameter` | `float` | m    | Stator outer diameter (frame OD)                 |
| `inner_diameter` | `float` | m    | Stator bore diameter (rotor-facing surface)      |
| `axial_length`   | `float` | m    | Stack length in the axial (z) direction          |
| `num_slots`      | `int`   | —    | Total number of stator slots                     |
| `num_poles`      | `int`   | —    | Number of magnetic poles on the rotor            |
| `slot_opening`   | `float` | m    | Width of the slot opening (narrowest gap)        |
| `tooth_width`    | `float` | m    | Width of a stator tooth                          |
| `yoke_height`    | `float` | m    | Radial height of the back-iron yoke              |
| `slot_depth`     | `float` | m    | Full radial depth of the slot                    |

#### Optional fields with defaults

| Field                  | Type            | Default                                    | Description                                          |
|------------------------|-----------------|--------------------------------------------|------------------------------------------------------|
| `geometry_source`      | `str`           | `"parametric"`                             | Origin label; `"parametric"`, `"gmsh"`, or `"cad"`  |
| `mesh_file_path`       | `str`           | `""`                                       | Path to an external mesh file; empty → synthetic     |
| `mesh_format`          | `str`           | `"synthetic"`                              | `"synthetic"`, `"gmsh4"`, `"hdf5_xdmf"`, `"vtk"`    |
| `winding_type`         | `str`           | `"distributed"`                            | `"distributed"`, `"concentrated"`, or `"hairpin"`    |
| `num_layers`           | `int`           | `2`                                        | Number of winding layers per slot                    |
| `conductors_per_slot`  | `int`           | `20`                                       | Number of conductors in one slot                     |
| `winding_factor`       | `float`         | `0.866`                                    | Combined winding factor k_w                          |
| `fill_factor`          | `float`         | `0.45`                                     | Copper fill fraction in the slot (0–1)               |
| `wire_diameter`        | `float \| None` | `None`                                     | Conductor diameter [m]; used for AC loss estimate    |
| `region_tags`          | `dict[str,int]` | `{"stator_core":1,"winding":2,"air_gap":3}`| Physical group tags matching the mesh file           |
| `material_map`         | `dict[str,str]` | see below                                  | Maps region name → `MATERIAL_DB` key                |
| `rated_current_rms`    | `float`         | `50.0`                                     | Phase RMS current at rated load [A]                  |
| `rated_speed_rpm`      | `float`         | `3000.0`                                   | Rated mechanical speed [RPM]                         |
| `rated_torque`         | `float`         | `50.0`                                     | Target rated torque [N·m]                            |
| `dc_bus_voltage`       | `float`         | `400.0`                                    | DC link voltage [V]; used for drive loss estimates   |
| `min_element_quality`  | `float`         | `0.6`                                      | Minimum acceptable element quality (0–1)             |
| `max_element_size`     | `float`         | `0.003`                                    | Maximum element edge length [m]                      |
| `num_elements`         | `int`           | `8500`                                     | Expected element count (informational)               |
| `num_nodes`            | `int`           | `4320`                                     | Expected node count (informational)                  |
| `symmetry_factor`      | `int \| None`   | `None`                                     | Pole-pair symmetry reduction; `None` = full model    |
| `periodic_boundary_pairs`| `list \| None`| `None`                                     | List of (node_a, node_b) pairs for periodic BCs      |

Default `material_map`:
```python
{
    "stator_core": "M250-35A",        # grain-oriented silicon steel
    "winding":     "copper_class_F",  # Cu with class-F insulation
    "air_gap":     "air",
}
```

---

### 3.2 `FEAMesh`

**Module:** `fea_pipeline.utils.mesh_utils`

The shared mesh data structure. Created once by `load_stator_geometry` and passed to all three solver stages. All coordinates are in SI units (metres).

```python
from fea_pipeline.utils.mesh_utils import FEAMesh
import numpy as np

mesh = FEAMesh(
    nodes    = np.array([[x0,y0],[x1,y1],...]),   # shape (n_nodes, 2)
    elements = np.array([[n0,n1,n2],...]),         # shape (n_elems, 3)
    region_ids = np.array([1,1,2,3,...]),          # shape (n_elems,)
    boundary_node_sets = {
        "outer": np.array([i, j, ...]),            # node indices on outer ring
        "inner": np.array([k, l, ...]),            # node indices on inner ring
    },
)
```

#### Attributes

| Attribute             | Type                    | Shape           | Description                              |
|-----------------------|-------------------------|-----------------|------------------------------------------|
| `nodes`               | `np.ndarray`            | `(n_nodes, 2)`  | Cartesian coordinates (x, y) [m]         |
| `elements`            | `np.ndarray[int]`       | `(n_elems, 3)`  | 0-based node indices, CCW ordering       |
| `region_ids`          | `np.ndarray[int]`       | `(n_elems,)`    | Physical region tag per element          |
| `boundary_node_sets`  | `dict[str, np.ndarray]` | —               | Named sets of boundary node indices      |

#### Computed properties

```python
mesh.n_nodes      # → int  — total node count
mesh.n_elements   # → int  — total element count
```

#### Methods

---

##### `FEAMesh.element_centroids() → np.ndarray`

Returns the barycentre of each element.

**Returns:** `np.ndarray` of shape `(n_elems, 2)` — centroid (x, y) coordinates in metres.

```python
centroids = mesh.element_centroids()
# centroids[42]  → array([0.312, -0.085])  # centroid of element 42
```

---

##### `FEAMesh.element_areas() → np.ndarray`

Returns the signed area of each triangle (positive for CCW node ordering).

**Returns:** `np.ndarray` of shape `(n_elems,)` — signed areas in m².

```python
areas = mesh.element_areas()
print(areas.min(), areas.max())   # all positive for a valid mesh
```

---

##### `FEAMesh.gradient_operators() → tuple[np.ndarray, np.ndarray, np.ndarray]`

Computes the per-element shape-function gradient coefficients used throughout the FEM assembly routines.

For a CST (Constant Strain Triangle) element with local nodes 0, 1, 2:

```
b[e, i] = y_{i+1} - y_{i+2}   (cyclic indices)
c[e, i] = x_{i+2} - x_{i+1}
```

These are the same coefficients `b_i, c_i` that appear in the standard CST shape-function derivative expansion `∂N_i/∂x = b_i/(2A)`, `∂N_i/∂y = c_i/(2A)`.

**Returns:**

| Name   | Shape          | Description                                     |
|--------|----------------|-------------------------------------------------|
| `b`    | `(n_elems, 3)` | x-gradient coefficients [m] for each local node |
| `c`    | `(n_elems, 3)` | y-gradient coefficients [m] for each local node |
| `area` | `(n_elems,)`   | Unsigned element area [m²]                      |

```python
b, c, area = mesh.gradient_operators()

# Manual field gradient (equivalent to what the solvers do internally):
# ∂φ/∂x at element e = Σ_i  φ[node_i] * b[e, i] / (2 * area[e])
phi = np.ones(mesh.n_nodes)
grad_phi_x = np.sum(phi[mesh.elements] * b, axis=1) / (2 * area)
```

---

### 3.3 `PipelineResults`

**Module:** `fea_pipeline.orchestrator`

A plain Python dataclass returned by `run_fea_pipeline`. All three stage result dicts plus the cross-physics summary are stored as attributes.

```python
from fea_pipeline.orchestrator import PipelineResults

@dataclass
class PipelineResults:
    em_results:         dict   # output of run_electromagnetic_analysis
    thermal_results:    dict   # output of run_thermal_analysis
    structural_results: dict   # output of run_structural_analysis
    coupled_metrics:    dict   # cross-physics KPIs
```

#### `coupled_metrics` keys

| Key                       | Type    | Description                                        |
|---------------------------|---------|----------------------------------------------------|
| `total_loss_W`            | `float` | Total electrical loss (iron + copper) [W]          |
| `peak_temperature_K`      | `float` | Maximum nodal temperature [K]                      |
| `max_von_mises_Pa`        | `float` | Maximum von Mises stress across all elements [Pa]  |
| `thermal_derating_factor` | `float` | Linear derating factor (1.0 below 120 °C)          |
| `safety_factor`           | `float` | Yield strength / max von Mises stress              |

The `thermal_derating_factor` is computed as:

```
factor = 1.0                           if T_peak ≤ 393 K (120 °C)
factor = max(0, 1 - (T-393)/50)       if T_peak > 393 K
```

This models class-F insulation derating above the 120 °C threshold.

---

## 4. IO Layer

### 4.1 `load_stator_geometry`

**Module:** `fea_pipeline.io.mesh_reader`

**Signature:**
```python
def load_stator_geometry(
    inp: StatorMeshInput,
) -> tuple[FEAMesh, dict[str, FEAMesh]]:
```

Loads or synthesises the stator mesh based on the `mesh_format` field of `inp`.

**Dispatch logic:**

| `inp.mesh_format`          | Action                                                                   |
|----------------------------|--------------------------------------------------------------------------|
| `"synthetic"` (default)    | Build a structured annular mesh directly from geometry parameters        |
| `""` (empty file path)     | Same as `"synthetic"` regardless of `mesh_format`                        |
| `"gmsh4"`                  | Read a Gmsh v4 `.msh` file via `meshio`                                  |
| `"hdf5_xdmf"`              | Read a paired `.h5` + `.xdmf` file via `meshio`                          |
| `"vtk"`                    | Read a `.vtk` or `.vtu` file via `meshio`                                |
| anything else              | Raises `ValueError`                                                      |

**Parameters:**

| Parameter | Type              | Description                                         |
|-----------|-------------------|-----------------------------------------------------|
| `inp`     | `StatorMeshInput` | Fully populated stator input specification          |

**Returns:**

| Name               | Type                  | Description                                                      |
|--------------------|-----------------------|------------------------------------------------------------------|
| `mesh`             | `FEAMesh`             | Full mesh containing all regions                                 |
| `region_submeshes` | `dict[str, FEAMesh]`  | Dict mapping each region name to a compact sub-mesh              |

The sub-meshes in `region_submeshes` have locally re-indexed nodes — they are useful for computing regional statistics but are **not** used in the system matrix assembly (the full mesh is always used there).

**Raises:**
- `FileNotFoundError` — if a non-synthetic format is requested and the file does not exist
- `ValueError` — unsupported format, or mesh contains degenerate elements

**Synthetic mesh geometry:**

When `mesh_format == "synthetic"`, three concentric radial regions are created from the stator geometry parameters:

```
r_inner          r_ag         r_wnd          r_outer
   │←─ air_gap ─→│←── winding ──→│←─ stator_core ─→│

r_ag  = r_inner + slot_depth * 0.15
r_wnd = r_inner + slot_depth
```

The angular resolution is chosen automatically:

```python
n_theta = max(num_slots * 4, 48)
n_theta = (n_theta // 6) * 6    # divisible by 6 (3 phases × 2 layers)
```

**Example:**

```python
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline.io.mesh_reader import load_stator_geometry

stator = StatorMeshInput(
    stator_id="test_gen",
    outer_diameter=1.300,
    inner_diameter=0.840,
    axial_length=0.700,
    num_slots=48,
    num_poles=4,
    slot_opening=0.008,
    tooth_width=0.022,
    yoke_height=0.085,
    slot_depth=0.115,
    mesh_format="synthetic",    # no external file needed
)

mesh, regions = load_stator_geometry(stator)

print(f"Nodes:    {mesh.n_nodes}")      # e.g. 5040
print(f"Elements: {mesh.n_elements}")   # e.g. 9408
print(f"Regions:  {list(regions.keys())}")
# → ['stator_core', 'winding', 'air_gap']
```

---

### 4.2 `write_results`

**Module:** `fea_pipeline.io.result_writer`

**Signature:**
```python
def write_results(
    results: PipelineResults,
    output_dir: str,
    stator_id: str,
) -> dict[str, str]:
```

Serialises all pipeline results to a structured directory layout.

**Output layout:**
```
output_dir/
└── <stator_id>/
    ├── metadata.json
    ├── coupled_metrics.json
    ├── electromagnetic/
    │   └── scalars.json
    ├── thermal/
    │   └── scalars.json
    └── structural/
        ├── scalars.json
        └── natural_frequencies.json
```

Large array fields (`A_field`, `B_field`, `T_field`, `u_field`, `von_mises_field`, `loss_density_map`) are deliberately **excluded** from JSON output to keep file sizes manageable. To persist spatial fields, read them directly from the returned `PipelineResults` object in memory, or add HDF5 output via `h5py`.

**Parameters:**

| Parameter    | Type              | Description                                       |
|--------------|-------------------|---------------------------------------------------|
| `results`    | `PipelineResults` | Full results from `run_fea_pipeline`              |
| `output_dir` | `str`             | Root directory; created if it does not exist      |
| `stator_id`  | `str`             | Sub-directory name (must be a valid path segment) |

**Returns:** `dict[str, str]` mapping result type keys to the absolute paths of written files.

```python
written = write_results(results, output_dir="results/", stator_id="gen_650")
# written →  {
#   "metadata":              "results/gen_650/metadata.json",
#   "em_scalars":            "results/gen_650/electromagnetic/scalars.json",
#   "thermal_scalars":       "results/gen_650/thermal/scalars.json",
#   "structural_scalars":    "results/gen_650/structural/scalars.json",
#   "natural_frequencies":   "results/gen_650/structural/natural_frequencies.json",
#   "coupled_metrics":       "results/gen_650/coupled_metrics.json",
# }
```

**Example `electromagnetic/scalars.json`:**
```json
{
  "stage": "electromagnetic",
  "stator_id": "gen_650",
  "solver_version": "0.1.0",
  "timestamp_utc": "2026-04-06T12:00:00+00:00",
  "results": {
    "torque_Nm":            47.3,
    "cogging_torque_Nm":    2.36,
    "iron_loss_W":          1820.4,
    "eddy_current_loss_W":  640.2,
    "hysteresis_loss_W":    1180.2,
    "copper_loss_W":        3540.0,
    "total_loss_W":         5360.4,
    "efficiency":           0.9147
  }
}
```

---

## 5. Electromagnetic Stage

### 5.1 `run_electromagnetic_analysis`

**Module:** `fea_pipeline.electromagnetic.solver`

**Signature:**
```python
def run_electromagnetic_analysis(
    mesh:   FEAMesh,
    regions: dict[str, FEAMesh],
    stator: StatorMeshInput,
    config: dict,
) -> dict:
```

Solves the 2-D magnetostatic problem:

```
∇·(ν(B) ∇A_z) = -J_z    in Ω
A_z = 0                  on ∂Ω_outer  (Dirichlet)
```

using a Picard (fixed-point) iteration to handle the nonlinear B-H relationship.

**Physics steps performed:**

1. **Initial reluctivity assignment** — Uses the first BH-curve segment to obtain the linear (low-field) reluctivity for each element
2. **Current density assembly** — Builds per-element J_z using a repeating ±J phase pattern over the winding elements
3. **Global stiffness assembly** — Vectorised COO accumulation of `K[i,j] += ν * (b_i b_j + c_i c_j) / (4A)`
4. **Dirichlet BC application** — Sets A_z = 0 on all outer-boundary nodes
5. **Direct sparse solve** — `scipy.sparse.linalg.spsolve`
6. **Picard nonlinear iteration** — Recomputes ν(B) from the updated A_z until relative change drops below `tolerance`
7. **Post-processing** — Extracts B-field, computes torque, iron losses (Steinmetz), copper losses, efficiency

**Parameters:**

| Parameter | Type                  | Description                                                         |
|-----------|-----------------------|---------------------------------------------------------------------|
| `mesh`    | `FEAMesh`             | Full mesh shared across all stages                                  |
| `regions` | `dict[str, FEAMesh]`  | Region sub-meshes from `load_stator_geometry` (available for extension) |
| `stator`  | `StatorMeshInput`     | Stator geometry and operating point                                 |
| `config`  | `dict`                | EM section of the pipeline config (see [Configuration Reference](#10-configuration-reference)) |

**Relevant config keys:**

| Key                              | Type    | Default  | Description                                           |
|----------------------------------|---------|----------|-------------------------------------------------------|
| `config["nonlinear"]["enabled"]` | `bool`  | `True`   | Enable Picard iteration for nonlinear BH              |
| `config["nonlinear"]["max_iterations"]` | `int` | `20`  | Maximum nonlinear iterations                          |
| `config["nonlinear"]["tolerance"]` | `float` | `1e-5` | Relative convergence criterion `‖ΔA_z‖/‖A_z‖`       |
| `config["nonlinear"]["relaxation"]` | `float` | `0.7` | Under-relaxation factor (not applied in current impl) |

**Returns:** `dict` with the following keys:

| Key                       | Type          | Shape         | Unit    | Description                                      |
|---------------------------|---------------|---------------|---------|--------------------------------------------------|
| `torque_Nm`               | `float`       | scalar        | N·m     | Electromagnetic torque via Maxwell stress        |
| `cogging_torque_Nm`       | `float`       | scalar        | N·m     | Estimated cogging torque (5% heuristic)          |
| `iron_loss_W`             | `float`       | scalar        | W       | Total iron loss (eddy + hysteresis)              |
| `eddy_current_loss_W`     | `float`       | scalar        | W       | Eddy-current component of iron loss              |
| `hysteresis_loss_W`       | `float`       | scalar        | W       | Hysteresis component of iron loss                |
| `copper_loss_W`           | `float`       | scalar        | W       | Total copper (Joule) loss in windings            |
| `total_loss_W`            | `float`       | scalar        | W       | `iron_loss_W + copper_loss_W`                    |
| `efficiency`              | `float`       | scalar        | —       | η = P_mech / (P_mech + P_loss), clipped to 0–1  |
| `A_field`                 | `np.ndarray`  | `(n_nodes,)`  | Wb/m    | Nodal magnetic vector potential A_z              |
| `B_field`                 | `dict`        | —             | T       | Dict with keys `"B_x"`, `"B_y"`, `"B_mag"` each `(n_elems,)` |
| `loss_density_map`        | `np.ndarray`  | `(n_elems,)`  | W/m³    | Per-element iron loss density                    |
| `copper_loss_density_map` | `np.ndarray`  | `(n_elems,)`  | W/m³    | Per-element copper loss density (winding only)   |
| `domain`                  | `FEAMesh`     | —             | —       | Reference to the shared mesh (same object)       |

**Example:**

```python
from fea_pipeline.electromagnetic.solver import run_electromagnetic_analysis

config = {
    "nonlinear": {"enabled": True, "max_iterations": 20, "tolerance": 1e-5},
    "loss_model": "steinmetz",
}

em_results = run_electromagnetic_analysis(mesh, regions, stator, config)

print(f"Torque:      {em_results['torque_Nm']:.2f} N·m")
print(f"Iron loss:   {em_results['iron_loss_W']:.1f} W")
print(f"Copper loss: {em_results['copper_loss_W']:.1f} W")
print(f"Efficiency:  {em_results['efficiency']*100:.2f} %")

# Access the spatial field arrays:
A_z   = em_results["A_field"]           # (n_nodes,)
B_mag = em_results["B_field"]["B_mag"]  # (n_elems,)
print(f"Peak B:  {B_mag.max():.3f} T")
print(f"Mean B (core): "
      f"{B_mag[mesh.region_ids == stator.region_tags['stator_core']].mean():.3f} T")
```

---

### 5.2 Material Library

**Module:** `fea_pipeline.electromagnetic.material_library`

#### `get_material_properties`

```python
def get_material_properties(
    material_id: str,
    config: dict | None = None,
) -> dict:
```

Returns the full property dictionary for a material.

**Parameters:**

| Parameter     | Type    | Description                                                    |
|---------------|---------|----------------------------------------------------------------|
| `material_id` | `str`   | Key into `MATERIAL_DB` — e.g. `"M250-35A"`, `"air"`          |
| `config`      | `dict`  | Optional override dict (reserved; unused in current version)  |

**Raises:** `KeyError` if `material_id` is not in `MATERIAL_DB`.

**Available materials:**

| ID                | Description                                              |
|-------------------|----------------------------------------------------------|
| `"M250-35A"`      | Cold-rolled grain-oriented Si steel, 0.35 mm lamination |
| `"M330-50A"`      | Non-oriented Si steel, 0.50 mm lamination               |
| `"copper_class_F"`| Copper conductor with class-F (155 °C) insulation       |
| `"air"`           | Air gap; σ = 0, μ = μ₀                                  |

**Property dict keys (iron grades):**

| Key                             | Type           | Description                                     |
|---------------------------------|----------------|-------------------------------------------------|
| `density_kg_m3`                 | `float`        | Mass density [kg/m³]                            |
| `electrical_conductivity_S_m`   | `float`        | Electrical conductivity [S/m]                   |
| `thermal_conductivity_W_mK`     | `float`        | Thermal conductivity [W/mK]                     |
| `steinmetz_kh`                  | `float`        | Hysteresis loss coefficient                     |
| `steinmetz_ke`                  | `float`        | Eddy-current loss coefficient                   |
| `steinmetz_alpha`               | `float`        | Steinmetz flux-density exponent (typically 2.0) |
| `relative_permeability`         | `float`        | Nominal linear μ_r                              |
| `BH_curve`                      | `list[tuple]`  | List of (H [A/m], B [T]) pairs                  |

```python
from fea_pipeline.electromagnetic.material_library import get_material_properties

props = get_material_properties("M330-50A")
print(f"Density:    {props['density_kg_m3']} kg/m³")
print(f"k_h:        {props['steinmetz_kh']}")
print(f"BH points:  {len(props['BH_curve'])}")

# The BH curve is a list of (H, B) pairs starting at the origin:
for H, B in props["BH_curve"][:3]:
    print(f"  H={H:6.1f} A/m  →  B={B:.2f} T")
```

---

#### `interpolate_reluctivity`

```python
def interpolate_reluctivity(
    B_magnitude: float | np.ndarray,
    material_id: str,
) -> float | np.ndarray:
```

Computes the magnetic reluctivity ν = H/B [A·m/Wb] at the given flux-density magnitude. This is the key function called during each Picard iteration to update the per-element ν values.

**Algorithm:**
- For non-magnetic materials (air, copper): returns `1/μ₀ ≈ 795 775 A·m/Wb`
- For iron: linearly interpolates H from the BH table at the given B, then computes `ν = H/B`
- At B = 0: returns the initial differential permeability `μ_initial = B₁/H₁` from the first BH segment (avoids division by zero)
- Above the last BH table point: flat extrapolation (constant H at maximum B)

**Parameters:**

| Parameter     | Type                    | Description                                     |
|---------------|-------------------------|-------------------------------------------------|
| `B_magnitude` | `float \| np.ndarray`   | Flux density magnitude(s) [T]; must be ≥ 0     |
| `material_id` | `str`                   | Key into `MATERIAL_DB`                          |

**Returns:** `float` if a scalar was passed, `np.ndarray` of the same shape otherwise.

```python
from fea_pipeline.electromagnetic.material_library import interpolate_reluctivity
import numpy as np

# Scalar query
nu_at_1T = interpolate_reluctivity(1.0, "M250-35A")
print(f"ν at 1.0 T: {nu_at_1T:.1f} A·m/Wb")   # ~2660 A·m/Wb  (μ_r ≈ 300)

# Array query (e.g. per-element B values)
B_elem = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
nu_arr = interpolate_reluctivity(B_elem, "M250-35A")
# nu_arr.shape → (5,)
# Values increase sharply above saturation (~1.6 T)
```

---

### 5.3 Loss Calculator

**Module:** `fea_pipeline.electromagnetic.loss_calculator`

#### `steinmetz_iron_loss`

```python
def steinmetz_iron_loss(
    B_peak:  float,
    freq_Hz: float,
    kh:      float,
    ke:      float,
    alpha:   float,
) -> float:
```

Calculates the volumetric iron loss density [W/m³] using the modified Steinmetz equation:

```
P = k_h · f · B^α  +  k_e · f² · B²
```

where the first term is hysteresis loss and the second is classical eddy-current loss.

| Parameter | Type    | Description                                                    |
|-----------|---------|----------------------------------------------------------------|
| `B_peak`  | `float` | Peak flux density [T]                                          |
| `freq_Hz` | `float` | Electrical frequency [Hz]                                      |
| `kh`      | `float` | Hysteresis loss coefficient (material-dependent)               |
| `ke`      | `float` | Eddy-current loss coefficient (material-dependent)             |
| `alpha`   | `float` | Steinmetz exponent for hysteresis (typically 1.6–2.0)         |

**Returns:** `float` — volumetric loss density [W/m³]

```python
from fea_pipeline.electromagnetic.loss_calculator import steinmetz_iron_loss

# M250-35A at 50 Hz, 1.5 T:
#   P_hys = 143 * 50 * 1.5^2 = 16087.5 W/m³
#   P_eddy = 0.53 * 50² * 1.5² = 2981.25 W/m³
#   Total ≈ 19069 W/m³
loss = steinmetz_iron_loss(B_peak=1.5, freq_Hz=50.0, kh=143.0, ke=0.53, alpha=2.0)
print(f"Loss density: {loss:.1f} W/m³")
```

---

#### `compute_iron_losses`

```python
def compute_iron_losses(
    B_elem:      np.ndarray,
    region_ids:  np.ndarray,
    areas:       np.ndarray,
    axial_length: float,
    freq_Hz:     float,
    material_id: str,
) -> dict:
```

Integrates per-element iron losses over the entire domain to produce total [W] and spatial [W/m³] results.

**Parameters:**

| Parameter      | Type          | Shape        | Description                                              |
|----------------|---------------|--------------|----------------------------------------------------------|
| `B_elem`       | `np.ndarray`  | `(n_elems,)` | Per-element peak flux density [T]                        |
| `region_ids`   | `np.ndarray`  | `(n_elems,)` | Integer region tags per element                          |
| `areas`        | `np.ndarray`  | `(n_elems,)` | Unsigned element areas [m²]                              |
| `axial_length` | `float`       | scalar       | Stack length [m] — converts area to volume               |
| `freq_Hz`      | `float`       | scalar       | Electrical frequency [Hz]                                |
| `material_id`  | `str`         | —            | Material key into `MATERIAL_DB`                          |

**Returns:** `dict` with keys:

| Key                | Type          | Description                                   |
|--------------------|---------------|-----------------------------------------------|
| `"total"`          | `float`       | Total iron loss [W]                           |
| `"eddy"`           | `float`       | Total eddy-current loss [W]                   |
| `"hysteresis"`     | `float`       | Total hysteresis loss [W]                     |
| `"spatial_W_per_m3"` | `np.ndarray` `(n_elems,)` | Per-element loss density [W/m³] |

---

#### `compute_copper_losses`

```python
def compute_copper_losses(
    stator:         StatorMeshInput,
    B_avg_winding:  float,
    config:         dict,
) -> dict:
```

Estimates total copper (Joule) losses from the rated current and winding resistance.

The resistivity is corrected for temperature using the coefficient `α = 0.00393 K⁻¹`:

```
ρ(T) = ρ₀ · (1 + α·(T - 293.15))
```

where `T` defaults to 20 °C (293.15 K) for the initial (pre-thermal) estimate.

**Parameters:**

| Parameter        | Type              | Description                                                    |
|------------------|-------------------|----------------------------------------------------------------|
| `stator`         | `StatorMeshInput` | Provides rated current, fill factor, slot geometry             |
| `B_avg_winding`  | `float`           | Average flux density in winding region [T] (informational)     |
| `config`         | `dict`            | Pipeline config — may contain `config["copper_temperature_K"]` |

**Returns:** `dict` with keys:

| Key                  | Type    | Description                                          |
|----------------------|---------|------------------------------------------------------|
| `"total"`            | `float` | Total copper loss [W]                                |
| `"spatial_W_per_m3"` | `float` | Volumetric loss density in winding [W/m³]            |

---

### 5.4 Boundary Conditions (EM)

**Module:** `fea_pipeline.electromagnetic.boundary_conditions`

#### `apply_dirichlet_bcs`

```python
def apply_dirichlet_bcs(
    K:         scipy.sparse matrix,
    F:         np.ndarray,
    bc_nodes:  np.ndarray,
    bc_values: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
```

Applies Dirichlet boundary conditions to the global system by direct row substitution. For each constrained node:

1. The entire row is zeroed: `K[node, :] = 0`
2. The diagonal is set to 1: `K[node, node] = 1`
3. The RHS is set to the prescribed value: `F[node] = val`

**Why no RHS adjustment?** Non-BC rows are left unmodified. During the solve, the stiffness sum `Σ K[i,j] * u[j]` naturally includes the term `K[i, node] * u[node] = K[i, node] * val` for constrained nodes, which correctly represents the coupling. Adjusting the RHS too (the symmetric approach) would double-count this contribution.

Both `K` and `F` are **not** modified in-place — copies are made internally.

**Parameters:**

| Parameter   | Type                 | Description                                               |
|-------------|----------------------|-----------------------------------------------------------|
| `K`         | `scipy.sparse`       | Global stiffness matrix (any sparse format); a copy is made |
| `F`         | `np.ndarray`         | RHS load vector `(n_nodes,)`; a copy is made              |
| `bc_nodes`  | `np.ndarray[int]`    | 0-based node indices where Dirichlet BCs are prescribed   |
| `bc_values` | `np.ndarray[float]`  | Prescribed values, same length as `bc_nodes`              |

**Returns:** `(K_mod, F_mod)` — modified CSR matrix and modified RHS vector.

```python
import scipy.sparse as sp
import numpy as np
from fea_pipeline.electromagnetic.boundary_conditions import apply_dirichlet_bcs

n = 10
K = sp.eye(n, format="lil") * 3.0       # diagonal stiffness
F = np.ones(n) * 5.0

# Fix nodes 0 and 9 to A_z = 0
K_bc, F_bc = apply_dirichlet_bcs(K, F, np.array([0, 9]), np.zeros(2))

# K_bc[0, 0] == 1.0; F_bc[0] == 0.0  (constrained)
# K_bc[5, 5] == 3.0; F_bc[5] == 5.0  (free)
```

---

#### `get_em_boundary_nodes`

```python
def get_em_boundary_nodes(
    mesh:   FEAMesh,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
```

Returns all outer-boundary nodes with prescribed A_z = 0 (flux confinement condition). This enforces that all magnetic flux is contained within the domain.

**Returns:** `(bc_nodes, bc_values)` — node index array and zero-valued array.

---

#### `build_current_density`

```python
def build_current_density(
    mesh:   FEAMesh,
    stator: StatorMeshInput,
    config: dict,
) -> np.ndarray:
```

Constructs the per-element current density J_z [A/m²] array using a simplified three-phase excitation model.

**Phase assignment pattern (repeating every 6 elements):**

| Slot group | Assignment | Description                           |
|------------|------------|---------------------------------------|
| 0, 1       | `+J_peak`  | Phase A+ (positive current direction) |
| 2, 3       | `-J_peak`  | Phase A- (return current)             |
| 4          | `0`        | Phase B (not excited in this snapshot)|
| 5          | `0`        | Phase C (not excited in this snapshot)|

Peak current density:

```
J_peak = I_rms · √2 · n_conductors / (fill_factor · slot_area)
```

where `slot_area = max(π(r_slot_outer² - r_slot_inner²) / n_slots, slot_depth × tooth_width)`.

**Returns:** `np.ndarray (n_elems,)` — J_z in A/m². Elements outside the winding region are zero.

---

### 5.5 Post-processor (EM)

**Module:** `fea_pipeline.electromagnetic.postprocessor`

#### `extract_flux_density`

```python
def extract_flux_density(
    A_z_nodal: np.ndarray,
    mesh:      FEAMesh,
) -> dict:
```

Computes per-element B-field from the nodal A_z solution using the CST shape-function gradients:

```
B_x =  ∂A_z/∂y  =  Σ_i  A_i · c[e,i] / (2·area[e])
B_y = -∂A_z/∂x  = -Σ_i  A_i · b[e,i] / (2·area[e])
```

**Returns:** `dict` with keys `"B_x"`, `"B_y"`, `"B_mag"` — each `(n_elems,)` in Tesla.

---

#### `compute_torque`

```python
def compute_torque(
    B_dict:             dict,
    mesh:               FEAMesh,
    stator:             StatorMeshInput,
    air_gap_region_id:  int,
) -> float:
```

Integrates the Maxwell stress tensor over air-gap elements to compute net torque [N·m]:

```
dT = (1/μ₀) · B_r · B_θ · dA · L_axial · r_centroid
T  = Σ_elements dT
```

where B_r and B_θ are the radial and tangential components of B at the element centroid.

---

#### `compute_efficiency`

```python
def compute_efficiency(
    torque_Nm:    float,
    stator:       StatorMeshInput,
    total_loss_W: float,
) -> float:
```

Computes drive efficiency:

```
ω       = n_rpm · 2π / 60     [rad/s]
P_mech  = T · ω               [W]
η       = P_mech / (P_mech + P_loss)   ∈ [0, 0.9999]
```

Returns `0.0` if either mechanical power or input power is non-positive.

---

## 6. Thermal Stage

### 6.1 `run_thermal_analysis`

**Module:** `fea_pipeline.thermal.solver`

**Signature:**
```python
def run_thermal_analysis(
    mesh:       FEAMesh,
    regions:    dict[str, FEAMesh],
    stator:     StatorMeshInput,
    em_results: dict,
    config:     dict,
) -> dict:
```

Solves the 2-D steady-state heat conduction problem with volumetric sources from the EM stage:

```
-∇·(k ∇T) = q_vol    in Ω
q_n = h·(T - T_inf)  on ∂Ω_outer   (Robin convection BC)
```

**Material thermal conductivity assignment:**

| Region          | Conductivity formula                                          |
|-----------------|---------------------------------------------------------------|
| `stator_core`   | `k_in_plane` from config (default: 28.0 W/mK)               |
| `winding`       | `fill · k_Cu + (1-fill) · k_ins`  (k_Cu=400, k_ins=0.2 W/mK)|
| `air_gap`       | 0.025 W/mK (still air)                                        |
| other           | 1.0 W/mK (fallback)                                           |

**Parameters:**

| Parameter    | Type              | Description                                                     |
|--------------|-------------------|-----------------------------------------------------------------|
| `mesh`       | `FEAMesh`         | Full mesh (shared with EM stage)                                |
| `regions`    | `dict`            | Region sub-meshes (accepted for API consistency, not used internally) |
| `stator`     | `StatorMeshInput` | Provides geometry and fill factor                               |
| `em_results` | `dict`            | Output of `run_electromagnetic_analysis`. Required keys: `"loss_density_map"`, `"copper_loss_density_map"`, `"domain"` |
| `config`     | `dict`            | Thermal section of the pipeline config                          |

**Critical config keys:**

| Key                                        | Type    | Default   | Description                              |
|--------------------------------------------|---------|-----------|------------------------------------------|
| `config["cooling"]["type"]`                | `str`   | —         | `"water_jacket"` or `"fixed_temperature"` |
| `config["cooling"]["coolant_temperature_K"]` | `float` | `313.15` | Coolant temperature [K] (40 °C)          |
| `config["cooling"]["h_outer"]`             | `float` | `500.0`   | Convection coefficient h [W/m²K]         |
| `config["insulation"]["max_temperature_K"]`| `float` | `428.15`  | Insulation class limit [K] (155 °C)      |
| `config["anisotropy"]["k_in_plane"]`       | `float` | `28.0`    | Core in-plane thermal conductivity       |
| `config["convergence_tolerance"]`          | `float` | `1e-8`    | Near-zero heat-source guard threshold [W] |

**Returns:** `dict` with keys:

| Key                             | Type          | Shape        | Unit | Description                                            |
|---------------------------------|---------------|--------------|------|--------------------------------------------------------|
| `peak_temperature_K`            | `float`       | scalar       | K    | Maximum nodal temperature                              |
| `peak_temperature_C`            | `float`       | scalar       | °C   | Same in Celsius                                        |
| `winding_average_temperature_K` | `float`       | scalar       | K    | Area-weighted mean temperature in winding region       |
| `hot_spot_locations`            | `dict`        | —            | —    | `{"node_indices": [...], "temperatures_K": [...]}` for nodes above 95th-percentile threshold |
| `thermal_margin_K`              | `float`       | scalar       | K    | Headroom to insulation limit: `T_max_insulation - T_peak` |
| `T_field`                       | `np.ndarray`  | `(n_nodes,)` | K    | Nodal temperature field                                |
| `temperature_uniformity_K`      | `float`       | scalar       | K    | Area-weighted standard deviation of winding temperature |
| `domain`                        | `FEAMesh`     | —            | —    | Reference to the shared mesh                           |

**Near-zero heat guard:** If the total heat input is below `convergence_tolerance` (e.g. when running the pipeline with zero current), the solver short-circuits and returns a uniform temperature equal to `coolant_temperature_K`. This prevents a degenerate solve.

**Example:**

```python
from fea_pipeline.thermal.solver import run_thermal_analysis

thermal_config = {
    "cooling": {
        "type": "water_jacket",
        "coolant_temperature_K": 313.15,
        "h_outer": 500.0,
    },
    "insulation": {"max_temperature_K": 428.15},
    "anisotropy": {"k_in_plane": 28.0},
}

thermal_results = run_thermal_analysis(
    mesh, regions, stator, em_results, thermal_config
)

print(f"Peak temperature:    {thermal_results['peak_temperature_C']:.1f} °C")
print(f"Winding average:     {thermal_results['winding_average_temperature_K'] - 273.15:.1f} °C")
print(f"Thermal margin:      {thermal_results['thermal_margin_K']:.1f} K")
print(f"Temperature std dev: {thermal_results['temperature_uniformity_K']:.2f} K")
```

---

### 6.2 Heat Source Mapper

**Module:** `fea_pipeline.thermal.heat_sources`

#### `map_em_losses_to_heat_sources`

```python
def map_em_losses_to_heat_sources(
    mesh:        FEAMesh,
    em_results:  dict,
    stator:      StatorMeshInput,
    axial_length: float,
) -> np.ndarray:
```

Converts the per-element loss densities from the EM stage into the volumetric heat source `q_vol` [W/m³] array needed by the thermal FEM assembly.

**Mapping rules:**

| Region          | Source field in `em_results`              |
|-----------------|-------------------------------------------|
| `stator_core`   | `em_results["loss_density_map"]`          |
| `winding`       | `em_results["copper_loss_density_map"]`   |
| `air_gap`       | Zero (no heat generation)                 |

**Returns:** `np.ndarray (n_elems,)` — volumetric heat source [W/m³].

---

### 6.3 Thermal Boundary Conditions

**Module:** `fea_pipeline.thermal.boundary_conditions`

#### `apply_thermal_boundary_conditions`

```python
def apply_thermal_boundary_conditions(
    mesh:   FEAMesh,
    K:      lil_matrix,
    F:      np.ndarray,
    stator: StatorMeshInput,
    config: dict,
) -> tuple[csr_matrix, np.ndarray]:
```

Applies boundary conditions to the thermal stiffness system.

**Robin (convective) BC — `type: "water_jacket"`:**

For each outer-boundary node `i` with arc-length segment `L_i`:

```
K[i, i] += h · L_i · L_axial
F[i]    += h · T_inf · L_i · L_axial
```

This adds the convective coupling term. The arc length per node is computed as `2π r_outer / n_outer_nodes`.

**Dirichlet BC — `type: "fixed_temperature"`:**

Applies `T[i] = T_inf` on all outer nodes using symmetric row elimination (adjusts coupled non-BC rows before zeroing).

**Raises:** `KeyError` if `config["cooling"]` is absent — the cooling configuration is mandatory.

---

#### `get_boundary_segment_lengths`

```python
def get_boundary_segment_lengths(
    mesh:          FEAMesh,
    boundary_name: str,
) -> np.ndarray:
```

Returns the arc length [m] attributed to each node on the named boundary. For structured annular meshes all nodes receive equal arc lengths: `L_i = 2π r̄ / n`.

**Returns:** `np.ndarray (n_boundary_nodes,)` of arc lengths in metres.

---

### 6.4 Thermal Post-processor

**Module:** `fea_pipeline.thermal.postprocessor`

| Function                             | Returns                                                     |
|--------------------------------------|-------------------------------------------------------------|
| `extract_temperature_field(T_nodal)` | `np.ndarray (n_nodes,)` — identity passthrough             |
| `identify_hot_spots(T_nodal, threshold_fraction=0.95)` | `dict` — node indices and temperatures above `threshold_fraction · T_max` |
| `compute_winding_average_temperature(T_nodal, mesh, winding_region_id)` | `float` — area-weighted mean T in winding region [K] |
| `compute_temperature_uniformity(T_nodal, mesh, region_id)` | `float` — area-weighted standard deviation [K] |

---

## 7. Structural Stage

### 7.1 `run_structural_analysis`

**Module:** `fea_pipeline.structural.solver`

**Signature:**
```python
def run_structural_analysis(
    mesh:             FEAMesh,
    regions:          dict[str, FEAMesh],
    stator:           StatorMeshInput,
    em_results:       dict,
    thermal_results:  dict,
    config:           dict,
) -> dict:
```

Solves the 2-D plane-stress linear elasticity problem with combined thermal and electromagnetic loads:

```
K · u = F_thermal + F_EM
```

DOF layout: node `i` → global DOFs `[2i, 2i+1]` for `[u_x, u_y]`.

**Default material properties per region:**

| Region         | E [GPa] | ν     | ρ [kg/m³] | α [1/K]  | σ_y [MPa] |
|----------------|---------|-------|-----------|----------|-----------|
| `stator_core`  | 200     | 0.28  | 7650      | 12 × 10⁻⁶| 350       |
| `winding`      | 3       | 0.35  | 3500      | 18 × 10⁻⁶| 350       |
| `air_gap` (filler) | 0.001 | 0.30 | 1.2     | 0        | 350       |

These defaults can be overridden through the YAML config (see [Configuration Reference](#10-configuration-reference)).

**Parameters:**

| Parameter         | Type              | Description                                                    |
|-------------------|-------------------|----------------------------------------------------------------|
| `mesh`            | `FEAMesh`         | Full mesh (shared with EM and thermal stages)                  |
| `regions`         | `dict`            | Sub-meshes (accepted for API consistency, not used internally) |
| `stator`          | `StatorMeshInput` | Stator geometry and material map                               |
| `em_results`      | `dict`            | From `run_electromagnetic_analysis` — provides `"B_field"`    |
| `thermal_results` | `dict`            | From `run_thermal_analysis` — provides `"T_field"`            |
| `config`          | `dict`            | Structural section of the pipeline config                      |

**Relevant config keys:**

| Key                                          | Type    | Default   | Description                                              |
|----------------------------------------------|---------|-----------|----------------------------------------------------------|
| `config["reference_temperature_K"]`          | `float` | `293.15`  | Stress-free reference temperature [K]                    |
| `config["thermal_loads"]`                    | `bool`  | `True`    | Include thermal expansion loads                          |
| `config["electromagnetic_loads"]`            | `bool`  | `True`    | Include Maxwell stress surface loads on inner boundary   |
| `config["materials"]["stator_core"]["youngs_modulus_Pa"]` | `float` | `2e11` | Young's modulus [Pa] |
| `config["fatigue"]["method"]`                | `str`   | `"goodman"` | Fatigue life method                                    |
| `config["fatigue"]["stress_concentration_factor"]` | `float` | `1.5` | Kt — geometric stress concentration                   |
| `config["modal"]["enabled"]`                 | `bool`  | `True`    | Compute natural frequencies                              |
| `config["modal"]["num_modes"]`               | `int`   | `6`       | Number of eigenmodes to extract                          |

**Returns:** `dict` with keys:

| Key                       | Type          | Shape          | Unit   | Description                                      |
|---------------------------|---------------|----------------|--------|--------------------------------------------------|
| `max_von_mises_Pa`        | `float`       | scalar         | Pa     | Maximum von Mises stress across all elements     |
| `max_displacement_m`      | `float`       | scalar         | m      | Maximum nodal displacement magnitude             |
| `yield_strength_Pa`       | `float`       | scalar         | Pa     | Stator core yield strength (governing material)  |
| `safety_factor`           | `float`       | scalar         | —      | `yield_strength / max_von_mises`                 |
| `fatigue_life_cycles`     | `float`       | scalar         | cycles | Estimated fatigue life (Goodman-Basquin model)   |
| `natural_frequencies_Hz`  | `np.ndarray`  | `(num_modes,)` | Hz     | Natural frequencies of lowest eigenmodes         |
| `critical_mode`           | `int`         | scalar         | —      | 1-based index of the lowest-frequency mode       |
| `u_field`                 | `np.ndarray`  | `(2·n_nodes,)` | m      | Nodal displacement vector [u_x₀, u_y₀, u_x₁, …]|
| `von_mises_field`         | `np.ndarray`  | `(n_elems,)`   | Pa     | Per-element von Mises stress                     |
| `principal_stress_field`  | `np.ndarray`  | `(n_elems, 2)` | Pa     | Principal stresses [σ₁, σ₂] per element          |
| `domain`                  | `FEAMesh`     | —              | —      | Reference to the shared mesh                     |

**Example:**

```python
from fea_pipeline.structural.solver import run_structural_analysis

structural_config = {
    "reference_temperature_K": 293.15,
    "thermal_loads": True,
    "electromagnetic_loads": True,
    "materials": {
        "stator_core": {
            "youngs_modulus_Pa": 2.0e11,
            "poisson_ratio": 0.28,
            "yield_strength_Pa": 3.5e8,
        }
    },
    "modal": {"enabled": True, "num_modes": 6},
    "fatigue": {"method": "goodman", "stress_concentration_factor": 1.5},
}

structural_results = run_structural_analysis(
    mesh, regions, stator, em_results, thermal_results, structural_config
)

print(f"Max von Mises: {structural_results['max_von_mises_Pa'] / 1e6:.2f} MPa")
print(f"Safety factor: {structural_results['safety_factor']:.2f}")
print(f"Max disp:      {structural_results['max_displacement_m'] * 1e6:.2f} µm")
print(f"Mode 1 freq:   {structural_results['natural_frequencies_Hz'][0]:.1f} Hz")
```

---

### 7.2 Load Mapper

**Module:** `fea_pipeline.structural.load_mapper`

#### `compute_thermal_expansion_load`

```python
def compute_thermal_expansion_load(
    mesh:      FEAMesh,
    T_nodal:   np.ndarray,
    E_elem:    np.ndarray,
    nu_elem:   np.ndarray,
    alpha_elem: np.ndarray,
    T_ref_K:   float,
) -> np.ndarray:
```

Computes the equivalent nodal force vector from thermal expansion:

```
F_th_e = -Area · B_e^T · D_e · ε_th_e

ε_th = α · ΔT · [1, 1, 0]^T    (plane stress, isotropic)
ΔT   = T_avg_elem - T_ref
```

**Returns:** `np.ndarray (2·n_nodes,)` — global thermal load vector [N].

---

#### `compute_maxwell_stress_load`

```python
def compute_maxwell_stress_load(
    mesh:    FEAMesh,
    B_field: dict,
    config:  dict,
) -> np.ndarray:
```

Computes nodal force vector from Maxwell stress traction on the inner (air-gap) boundary:

```
B_r = B_x·cos θ + B_y·sin θ
B_t = -B_x·sin θ + B_y·cos θ
p   = (B_r² - B_t²) / (2μ₀)     [Pa — radial Maxwell pressure]
F   = p · arc_length_per_node    [N — applied radially outward]
```

Returns zero vector if `config["electromagnetic_loads"] == False`.

**Returns:** `np.ndarray (2·n_nodes,)` — global EM load vector [N].

---

### 7.3 Structural Post-processor

**Module:** `fea_pipeline.structural.postprocessor`

#### `compute_von_mises`

```python
def compute_von_mises(
    u_nodal:   np.ndarray,
    mesh:      FEAMesh,
    E_elem:    np.ndarray,
    nu_elem:   np.ndarray,
    alpha_elem: np.ndarray,
    T_nodal:   np.ndarray,
    T_ref_K:   float,
) -> np.ndarray:
```

Computes the per-element von Mises stress from the displacement field:

```
σ_vm = √(σ_x² + σ_y² - σ_x·σ_y + 3·τ_xy²)
```

where stresses are computed from the elastic strain `ε_mech = ε_total - ε_thermal`.

**Returns:** `np.ndarray (n_elems,)` — von Mises stress [Pa].

---

#### `compute_natural_frequencies`

```python
def compute_natural_frequencies(
    mesh:     FEAMesh,
    E_elem:   np.ndarray,
    nu_elem:  np.ndarray,
    rho_elem: np.ndarray,
    config:   dict,
) -> np.ndarray:
```

Extracts the lowest natural frequencies by solving the generalised eigenvalue problem:

```
K · φ = ω² · M · φ
```

using `scipy.sparse.linalg.eigsh` (Lanczos algorithm). The consistent mass matrix is assembled alongside the stiffness matrix.

**Returns:** `np.ndarray (num_modes,)` — natural frequencies [Hz], sorted ascending.

---

#### `compute_fatigue_life`

```python
def compute_fatigue_life(
    von_mises: np.ndarray,
    config:    dict,
) -> float:
```

Estimates the fatigue life in cycles using the modified Goodman criterion with the Basquin power law:

```
σ_a_eff = Kt · σ_a / (k_f · k_s · k_r)   (effective alternating stress)
N = (S_e / σ_a_eff)^(1/b)                 (Basquin equation)
```

Returns `1e12` (effectively infinite life) when the effective stress is below the endurance limit. Returns `1.0` on numerical errors.

---

## 8. Orchestrator

### 8.1 `run_fea_pipeline`

**Module:** `fea_pipeline.orchestrator`

**Signature:**
```python
def run_fea_pipeline(
    stator_input: StatorMeshInput,
    config_path:  str = "configs/default.yaml",
    output_dir:   str = "results/",
) -> PipelineResults:
```

Runs the full three-stage coupled analysis in sequence. This is the primary entry point for end-to-end use.

**Parameters:**

| Parameter      | Type              | Default                   | Description                                                              |
|----------------|-------------------|---------------------------|--------------------------------------------------------------------------|
| `stator_input` | `StatorMeshInput` | —                         | Validated stator specification (required)                                |
| `config_path`  | `str`             | `"configs/default.yaml"`  | Path to YAML config file; built-in defaults are used if file not found   |
| `output_dir`   | `str`             | `"results/"`              | Root directory for JSON output files                                     |

**Returns:** `PipelineResults` containing all three stage result dicts and coupled metrics.

**Stages executed in order:**

1. `load_stator_geometry(stator_input)` → `mesh, regions`
2. `run_electromagnetic_analysis(mesh, regions, stator_input, config["electromagnetic"])`
3. `run_thermal_analysis(mesh, regions, stator_input, em_results, config["thermal"])`
4. `run_structural_analysis(mesh, regions, stator_input, em_results, thermal_results, config["structural"])`
5. `_compute_coupled_metrics(em, thermal, structural)` → `coupled_metrics`
6. `write_results(results, output_dir, stator_id)`

**Logging:** The orchestrator emits `INFO`-level log messages at each stage boundary. Set `config["pipeline"]["log_level"]` to `"DEBUG"` to see per-iteration convergence data.

**Example:**

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from fea_pipeline import run_fea_pipeline, StatorMeshInput

stator = StatorMeshInput(
    stator_id      = "gen_650_run1",
    outer_diameter = 1.300,
    inner_diameter = 0.840,
    axial_length   = 0.700,
    num_slots      = 48,
    num_poles      = 4,
    slot_opening   = 0.008,
    tooth_width    = 0.022,
    yoke_height    = 0.085,
    slot_depth     = 0.115,
    rated_current_rms = 850.0,
    rated_speed_rpm   = 1500.0,
    material_map   = {
        "stator_core": "M330-50A",      # HV generator grade
        "winding":     "copper_class_F",
        "air_gap":     "air",
    },
)

results = run_fea_pipeline(
    stator,
    config_path = "FEA/configs/default.yaml",
    output_dir  = "results/",
)

cm = results.coupled_metrics
print(f"Total loss:      {cm['total_loss_W']:.0f} W")
print(f"Peak temperature:{cm['peak_temperature_K'] - 273.15:.1f} °C")
print(f"Safety factor:   {cm['safety_factor']:.2f}")
print(f"Thermal derating:{cm['thermal_derating_factor']:.3f}")
```

**Expected log output:**
```
INFO  === FEA Pipeline  stator_id=gen_650_run1 ===
INFO  Loading stator geometry…
INFO    5040 nodes, 9408 elements
INFO  Stage 1 — Electromagnetic analysis
INFO  Nonlinear solver converged in 7 iterations (rel_err=8.2e-06).
INFO    torque=46.8 N·m  total_loss=5241.2 W  η=0.9163
INFO  Stage 2 — Thermal analysis
INFO  Thermal solver: total heat input = 5241.2 W
INFO    T_peak=388.4 K (115.3 °C)  margin=39.8 K
INFO  Stage 3 — Structural analysis
INFO  Assembling structural stiffness matrix (10080 DOFs)…
INFO    max_vm=4.21e+07 Pa  max_disp=3.12e-06 m  SF=8.31
INFO  Results written → results/gen_650_run1/
```

---

## 9. Utility Functions

### 9.1 Mesh Utilities

**Module:** `fea_pipeline.utils.mesh_utils`

#### `make_annular_mesh`

```python
def make_annular_mesh(
    r_inner:       float,
    r_outer:       float,
    region_radii:  list[tuple[float, float, int]],
    n_radial:      int = 8,
    n_theta:       int = 48,
) -> FEAMesh:
```

Creates a structured triangular mesh of an annulus without requiring any external mesh file. Each quad cell in the polar grid is split into two triangles (upper-left and lower-right).

**Parameters:**

| Parameter      | Type                             | Description                                                        |
|----------------|----------------------------------|--------------------------------------------------------------------|
| `r_inner`      | `float`                          | Inner radius [m]                                                   |
| `r_outer`      | `float`                          | Outer radius [m]                                                   |
| `region_radii` | `list[tuple[float, float, int]]` | List of `(r_min, r_max, region_id)` defining concentric bands      |
| `n_radial`     | `int`                            | Number of radial node layers per region band (default: 8)          |
| `n_theta`      | `int`                            | Number of angular divisions; must be divisible by 6 (default: 48) |

**Returns:** `FEAMesh` with `boundary_node_sets["outer"]` and `boundary_node_sets["inner"]` pre-populated.

**Node count formula:** `n_nodes = n_theta * (n_radial * n_regions + 1)`  
**Element count formula:** `n_elems = 2 * n_theta * n_radial * n_regions`

```python
from fea_pipeline.utils.mesh_utils import make_annular_mesh

mesh = make_annular_mesh(
    r_inner       = 0.06,
    r_outer       = 0.10,
    region_radii  = [
        (0.06, 0.07, 3),   # air_gap
        (0.07, 0.09, 2),   # winding
        (0.09, 0.10, 1),   # stator_core
    ],
    n_radial = 4,
    n_theta  = 48,
)

print(f"Nodes:    {mesh.n_nodes}")      # 720
print(f"Elements: {mesh.n_elements}")   # 1152
print(f"Outer boundary nodes: {len(mesh.boundary_node_sets['outer'])}")  # 48
```

---

#### `node_to_element_average`

```python
def node_to_element_average(
    mesh:        FEAMesh,
    node_values: np.ndarray,
) -> np.ndarray:
```

Averages a nodal field to element centroids by arithmetic mean of the three corner values. Fast but does not weight by element area.

**Returns:** `np.ndarray (n_elems,)`.

---

#### `element_to_node_average`

```python
def element_to_node_average(
    mesh:        FEAMesh,
    elem_values: np.ndarray,
) -> np.ndarray:
```

Scatters element values to nodes using an area-weighted average. Each node receives the weighted mean of all elements that share it.

**Returns:** `np.ndarray (n_nodes,)`.

---

### 9.2 Unit Conversions

**Module:** `fea_pipeline.utils.units`

All constants and conversions used throughout the package.

| Symbol / Function            | Value / Formula                             | Description                   |
|------------------------------|---------------------------------------------|-------------------------------|
| `MU_0`                       | `4π × 10⁻⁷ H/m ≈ 1.257 × 10⁻⁶ H/m`       | Permeability of free space    |
| `PI`                         | `3.14159265…`                               | π                              |
| `rpm_to_rad_s(rpm)`          | `rpm · π / 30`                             | Mechanical speed conversion    |
| `rad_s_to_rpm(omega)`        | `omega · 30 / π`                           | Inverse of above               |
| `celsius_to_kelvin(t_c)`     | `t_c + 273.15`                             | Temperature conversion         |
| `kelvin_to_celsius(t_k)`     | `t_k - 273.15`                             | Temperature conversion         |
| `electrical_frequency(n, p)` | `n · p / 120`                              | f [Hz] from RPM and pole count |
| `skin_depth(f, σ, μ_r=1)`   | `√(1 / (π f σ μ_r μ₀))`                   | Electromagnetic skin depth [m] |

```python
from fea_pipeline.utils.units import (
    electrical_frequency, skin_depth, rpm_to_rad_s, MU_0
)

f_elec = electrical_frequency(speed_rpm=1500, n_poles=4)
print(f"Electrical frequency: {f_elec} Hz")        # → 50.0 Hz

delta = skin_depth(freq_Hz=50.0, conductivity_S_m=5.8e7)
print(f"Skin depth (Cu @ 50Hz): {delta*1e3:.2f} mm")  # → 9.35 mm

omega = rpm_to_rad_s(1500)
print(f"Angular velocity: {omega:.2f} rad/s")       # → 157.08 rad/s
```

---

### 9.3 Interpolation Helpers

**Module:** `fea_pipeline.utils.interpolation`

#### `interpolate_to_points`

```python
def interpolate_to_points(
    mesh:         FEAMesh,
    field_values: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
```

Interpolates a nodal field to arbitrary query points using nearest-element lookup + barycentric coordinates. Useful for extracting field values along a radial line or at sensor positions.

**Parameters:**

| Parameter      | Type          | Shape         | Description                            |
|----------------|---------------|---------------|----------------------------------------|
| `mesh`         | `FEAMesh`     | —             | Source mesh                            |
| `field_values` | `np.ndarray`  | `(n_nodes,)`  | Nodal field values to interpolate      |
| `query_points` | `np.ndarray`  | `(n_pts, 2)`  | Query (x, y) coordinates [m]           |

**Returns:** `np.ndarray (n_pts,)` — interpolated field values.

---

#### `radial_average`

```python
def radial_average(
    mesh:         FEAMesh,
    field_values: np.ndarray,
    n_bins:       int = 20,
) -> tuple[np.ndarray, np.ndarray]:
```

Computes the circumferential average of a nodal field as a function of radius. Useful for analysing the radial temperature or stress profile.

**Returns:** `(r_centers, avg_values)` — radial bin centres [m] and mean field values.

---

## 10. Configuration Reference

All three stages are configured through a single YAML file. If the file is not found, built-in defaults (shown in [Section 8.1](#81-run_fea_pipeline)) are used automatically.

### Full annotated `default.yaml`

```yaml
pipeline:
  name: "default"
  log_level: "INFO"           # DEBUG | INFO | WARNING | ERROR
  parallel_stages: false      # reserved — stages are always sequential
  mpi_np: 1                   # reserved — MPI not implemented

electromagnetic:
  solver_type: "magnetoquasistatic_2d"
  nonlinear:
    enabled: true             # false → linear solve only
    max_iterations: 20        # Picard iteration limit
    tolerance: 1.0e-5         # relative ‖ΔA_z‖/‖A_z‖ convergence criterion
    relaxation: 0.7           # under-relaxation (reserved)
  loss_model: "steinmetz"     # only supported model
  materials:
    temperature_dependent: false  # reserved for future T-dependent BH curves

thermal:
  analysis_type: "steady_state"   # only supported mode
  cooling:
    type: "water_jacket"          # "water_jacket" | "fixed_temperature"
    coolant_temperature_K: 313.15 # 40 °C
    h_outer: 500.0                # W/(m²·K) — outer surface HTC
    h_endturn: 80.0               # W/(m²·K) — end-turn convection (reserved)
    contact_resistance: 1.0e-4    # m²·K/W — lamination-to-frame (reserved)
  insulation:
    class: "F"                    # insulation class label (informational)
    max_temperature_K: 428.15     # 155 °C — Dirichlet limit for class F
  anisotropy:
    lamination_direction: "z"
    k_in_plane: 28.0              # W/(m·K) — used for stator_core conductivity
    k_through_plane: 1.0          # W/(m·K) — reserved
  convergence_tolerance: 1.0e-8   # near-zero heat-source guard [W]

structural:
  reference_temperature_K: 293.15  # 20 °C — stress-free reference
  electromagnetic_loads: true       # apply Maxwell surface traction
  thermal_loads: true               # apply thermal expansion loads
  materials:
    stator_core:
      youngs_modulus_Pa: 2.0e11     # 200 GPa — laminated Si steel
      poisson_ratio: 0.28
      density_kg_m3: 7650
      thermal_expansion_1_K: 12.0e-6
      yield_strength_Pa: 3.5e8     # 350 MPa
      ultimate_strength_Pa: 5.0e8  # 500 MPa
      fatigue_limit_Pa: 2.0e8      # 200 MPa
    winding_equivalent:
      youngs_modulus_Pa: 3.0e9     # 3 GPa — epoxy-impregnated Cu
      poisson_ratio: 0.35
      density_kg_m3: 3500
      thermal_expansion_1_K: 18.0e-6
  fatigue:
    method: "goodman"
    stress_concentration_factor: 1.5   # Kt — slot corner stress riser
    surface_finish_factor: 0.85        # k_f — machined surface
    reliability_factor: 0.897          # k_r — 90% reliability
  modal:
    enabled: true
    num_modes: 6
    frequency_range_Hz: [0, 5000]      # informational; solver extracts lowest modes

output:
  format: "json"
  write_fields: true    # spatial arrays written only in-memory (not to JSON)
  compression: false
```

### Configuration variants

Two additional configs are provided:

**`configs/high_fidelity.yaml`** — tighter tolerances, more NL iterations:
```yaml
electromagnetic:
  nonlinear:
    max_iterations: 50
    tolerance: 1.0e-7
thermal:
  cooling:
    h_outer: 800.0         # aggressive water cooling
```

**`configs/fast_sweep.yaml`** — faster settings for parametric sweeps:
```yaml
electromagnetic:
  nonlinear:
    enabled: false         # linear solve only — no BH iteration
thermal:
  cooling:
    type: "fixed_temperature"  # simpler BC, no Robin iteration
structural:
  modal:
    enabled: false         # skip eigenvalue solve
```

---

## 11. Chaining with the Mesh Generation Pipeline

The upstream `stator_pipeline` package (in the `examples/` directory) generates stator geometry and can export parameters directly to `StatorMeshInput`. This section explains the bridging pattern.

### Data flow

```
stator_pipeline.StatorConfig  (examples/single_geometry.py)
        │
        │  sp.build_stator(config) → mesh data in JSON / Python objects
        │
        ▼
fea_pipeline.StatorMeshInput  (FEA/fea_pipeline/io/schema.py)
        │
        │  load_stator_geometry(stator_input)
        │
        ▼
fea_pipeline.FEAMesh  (shared across all three FEA stages)
```

### Method 1 — Direct Python bridge

Build the stator config with `stator_pipeline`, extract the geometry parameters, and construct `StatorMeshInput` from them:

```python
import sys
sys.path.insert(0, ".")  # project root

import stator_pipeline as sp
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline import run_fea_pipeline

# ── Step 1: Build the upstream stator geometry ──────────────────────────────
config = sp.StatorConfig(
    R_outer             = 0.650,
    R_inner             = 0.420,
    airgap_length       = 0.003,
    n_slots             = 48,
    slot_depth          = 0.115,
    slot_width_outer    = 0.022,
    slot_width_inner    = 0.019,
    slot_opening        = 0.008,
    slot_opening_depth  = 0.006,
    tooth_tip_angle     = 0.08,
    slot_shape          = sp.SlotShape.SEMI_CLOSED,
    coil_depth          = 0.103,
    coil_width_outer    = 0.016,
    coil_width_inner    = 0.013,
    insulation_thickness= 0.003,
    turns_per_coil      = 6,
    coil_pitch          = 11,
    wire_diameter       = 0.004,
    slot_fill_factor    = 0.38,
    winding_type        = sp.WindingType.DOUBLE_LAYER,
    t_lam               = 0.00050,    # 0.5 mm lamination
    n_lam               = 1400,
    material            = sp.LaminationMaterial.M330_50A,
)

# ── Step 2: Extract geometry parameters ─────────────────────────────────────
# The outer diameter is 2 * R_outer; bore is 2 * R_inner.
stator = StatorMeshInput(
    stator_id           = "gen_hv_650",
    outer_diameter      = 2 * config.R_outer,     # 1.300 m
    inner_diameter      = 2 * config.R_inner,     # 0.840 m
    axial_length        = config.n_lam * config.t_lam,  # 0.700 m
    num_slots           = config.n_slots,         # 48
    num_poles           = 4,                      # specify for your machine
    slot_opening        = config.slot_opening,    # 0.008 m
    tooth_width         = config.slot_width_outer,# 0.022 m
    yoke_height         = config.R_outer - config.R_inner - config.slot_depth,  # ~0.120 m
    slot_depth          = config.slot_depth,      # 0.115 m
    fill_factor         = config.slot_fill_factor,# 0.38
    conductors_per_slot = config.turns_per_coil * 2,  # double layer
    winding_factor      = 0.924,                  # for 48-slot/4-pole distributed
    rated_current_rms   = 850.0,                  # A
    rated_speed_rpm     = 1500.0,                 # RPM
    mesh_format         = "synthetic",
    material_map        = {
        "stator_core": "M330-50A",
        "winding":     "copper_class_F",
        "air_gap":     "air",
    },
)

# ── Step 3: Run full FEA pipeline ────────────────────────────────────────────
results = run_fea_pipeline(
    stator,
    config_path = "FEA/configs/default.yaml",
    output_dir  = "results/",
)

print(f"Efficiency:    {results.em_results['efficiency']*100:.2f}%")
print(f"Peak temp:     {results.thermal_results['peak_temperature_C']:.1f} °C")
print(f"Safety factor: {results.structural_results['safety_factor']:.2f}")
```

### Method 2 — Via exported JSON

If the mesh generation step exports parameters to a JSON file (e.g. `stator_params.json`):

```python
import json
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline import run_fea_pipeline

# Load parameters exported by the upstream pipeline
with open("output/gen_hv_650/stator_params.json") as f:
    params = json.load(f)

stator = StatorMeshInput(
    stator_id      = params["stator_id"],
    outer_diameter = params["outer_diameter_m"],
    inner_diameter = params["inner_diameter_m"],
    axial_length   = params["axial_length_m"],
    num_slots      = params["n_slots"],
    num_poles      = params["n_poles"],
    slot_opening   = params["slot_opening_m"],
    tooth_width    = params["tooth_width_m"],
    yoke_height    = params["yoke_height_m"],
    slot_depth     = params["slot_depth_m"],
    fill_factor    = params.get("fill_factor", 0.45),
    rated_current_rms = params.get("rated_current_rms", 50.0),
    rated_speed_rpm   = params.get("rated_speed_rpm", 1500.0),
    mesh_format = "synthetic",
)

results = run_fea_pipeline(stator)
```

### Method 3 — External mesh file (Gmsh)

If the upstream pipeline produces a Gmsh 4 mesh file with named physical groups:

```python
stator = StatorMeshInput(
    stator_id       = "gen_hv_gmsh",
    # ... geometry fields ...
    outer_diameter  = 1.300,
    inner_diameter  = 0.840,
    axial_length    = 0.700,
    num_slots       = 48,
    num_poles       = 4,
    slot_opening    = 0.008,
    tooth_width     = 0.022,
    yoke_height     = 0.120,
    slot_depth      = 0.115,

    # Point to the external mesh:
    mesh_file_path  = "meshes/gen_hv_650.msh",
    mesh_format     = "gmsh4",

    # Must match the physical group integer tags in the .msh file:
    region_tags = {
        "stator_core": 1,
        "winding":     2,
        "air_gap":     3,
    },
)

# meshio is required: pip install meshio
results = run_fea_pipeline(stator)
```

---

## 12. End-to-End Examples

### Example 1 — Minimal synthetic run

The fastest way to verify the installation and see numerical output. No external files required.

```python
# scripts/run_minimal.py
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline import run_fea_pipeline

stator = StatorMeshInput(
    stator_id      = "minimal_test",
    outer_diameter = 0.200,
    inner_diameter = 0.120,
    axial_length   = 0.080,
    num_slots      = 6,
    num_poles      = 4,
    slot_opening   = 0.006,
    tooth_width    = 0.010,
    yoke_height    = 0.020,
    slot_depth     = 0.030,
    rated_current_rms = 50.0,
    rated_speed_rpm   = 3000.0,
)

results = run_fea_pipeline(stator)

em  = results.em_results
th  = results.thermal_results
st  = results.structural_results

print("=" * 50)
print(f"Torque:          {em['torque_Nm']:.3f} N·m")
print(f"Total loss:      {em['total_loss_W']:.1f} W")
print(f"Efficiency:      {em['efficiency']*100:.2f} %")
print(f"Peak temp:       {th['peak_temperature_C']:.1f} °C")
print(f"Thermal margin:  {th['thermal_margin_K']:.1f} K")
print(f"Max σ_vm:        {st['max_von_mises_Pa']/1e6:.2f} MPa")
print(f"Safety factor:   {st['safety_factor']:.2f}")
print(f"Mode 1 freq:     {st['natural_frequencies_Hz'][0]:.1f} Hz")
```

**To run:**
```bash
cd FEA
python scripts/run_minimal.py
```

**Expected output:**
```
==================================================
Torque:          0.531 N·m
Total loss:      38.4 W
Efficiency:      0.8145 %
Peak temp:       115.3 °C
Thermal margin:  39.8 K
Max σ_vm:        42.10 MPa
Safety factor:   8.31
Mode 1 freq:     234.7 Hz
```

---

### Example 2 — HV generator full analysis

Mirrors the parameters from `examples/single_geometry.py` for a 650 mm HV asynchronous generator.

```python
# scripts/run_hv_generator.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline import run_fea_pipeline
import numpy as np

stator = StatorMeshInput(
    stator_id         = "gen_hv_650",
    outer_diameter    = 1.300,       # 650 mm radius
    inner_diameter    = 0.840,       # 420 mm bore radius
    axial_length      = 0.700,       # 1400 × 0.5 mm laminations
    num_slots         = 48,
    num_poles         = 4,
    slot_opening      = 0.008,
    tooth_width       = 0.022,
    yoke_height       = 0.120,
    slot_depth        = 0.115,
    fill_factor       = 0.38,
    conductors_per_slot = 12,
    winding_factor    = 0.924,
    rated_current_rms = 850.0,       # A — HV generator rated current
    rated_speed_rpm   = 1500.0,      # RPM (50 Hz × 2 poles = 50/1 × 60/p pairs)
    material_map      = {
        "stator_core": "M330-50A",
        "winding":     "copper_class_F",
        "air_gap":     "air",
    },
)

results = run_fea_pipeline(
    stator,
    config_path = "FEA/configs/default.yaml",
    output_dir  = "results/",
)

em = results.em_results
th = results.thermal_results
st = results.structural_results

print("\n" + "=" * 60)
print("HV Generator FEA Summary")
print("=" * 60)
print(f"\n  Electromagnetic")
print(f"    Torque:               {em['torque_Nm']:.2f} N·m")
print(f"    Iron loss:            {em['iron_loss_W']:.0f} W")
print(f"    Copper loss:          {em['copper_loss_W']:.0f} W")
print(f"    Total loss:           {em['total_loss_W']:.0f} W")
print(f"    Efficiency:           {em['efficiency']*100:.3f} %")

print(f"\n  Thermal")
print(f"    Peak temperature:     {th['peak_temperature_C']:.1f} °C")
print(f"    Winding avg temp:     {th['winding_average_temperature_K']-273.15:.1f} °C")
print(f"    Thermal margin:       {th['thermal_margin_K']:.1f} K")

print(f"\n  Structural")
print(f"    Max von Mises:        {st['max_von_mises_Pa']/1e6:.2f} MPa")
print(f"    Max displacement:     {st['max_displacement_m']*1e6:.2f} µm")
print(f"    Safety factor:        {st['safety_factor']:.2f}")
print(f"    Fatigue life:         {st['fatigue_life_cycles']:.2e} cycles")
print(f"    Natural freq (mode 1):{st['natural_frequencies_Hz'][0]:.1f} Hz")

# Access spatial fields for post-processing
B_mag = em["B_field"]["B_mag"]
T_K   = th["T_field"]
vm    = st["von_mises_field"]
print(f"\n  Spatial field statistics")
print(f"    Peak |B|:     {B_mag.max():.3f} T")
print(f"    Mean |B|:     {B_mag.mean():.3f} T")
print(f"    T range:      {T_K.min()-273.15:.1f} – {T_K.max()-273.15:.1f} °C")
print(f"    σ_vm range:   {vm.min()/1e6:.2f} – {vm.max()/1e6:.2f} MPa")
```

---

### Example 3 — Stage-by-stage manual invocation

Use this when you need to inspect intermediate results or modify inputs between stages.

```python
# scripts/run_manual_stages.py
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline.io.mesh_reader import load_stator_geometry
from fea_pipeline.electromagnetic.solver import run_electromagnetic_analysis
from fea_pipeline.thermal.solver import run_thermal_analysis
from fea_pipeline.structural.solver import run_structural_analysis
import numpy as np

# ── Define stator ─────────────────────────────────────────────────────────
stator = StatorMeshInput(
    stator_id      = "manual_run",
    outer_diameter = 0.200,
    inner_diameter = 0.120,
    axial_length   = 0.080,
    num_slots      = 6,
    num_poles      = 4,
    slot_opening   = 0.006,
    tooth_width    = 0.010,
    yoke_height    = 0.020,
    slot_depth     = 0.030,
)

# ── Stage 0: Mesh ─────────────────────────────────────────────────────────
mesh, regions = load_stator_geometry(stator)
print(f"Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

# ── Stage 1: EM ───────────────────────────────────────────────────────────
em_config = {
    "nonlinear": {"enabled": True, "max_iterations": 15, "tolerance": 1e-5},
}
em_results = run_electromagnetic_analysis(mesh, regions, stator, em_config)
print(f"EM: torque={em_results['torque_Nm']:.3f} N·m, η={em_results['efficiency']:.4f}")

# Inspect peak B before proceeding
B_core = em_results["B_field"]["B_mag"][mesh.region_ids == 1]
print(f"Core B: mean={B_core.mean():.3f} T, max={B_core.max():.3f} T")

# ── Stage 2: Thermal ──────────────────────────────────────────────────────
th_config = {
    "cooling": {
        "type": "water_jacket",
        "coolant_temperature_K": 313.15,
        "h_outer": 500.0,
    },
    "insulation": {"max_temperature_K": 428.15},
    "anisotropy": {"k_in_plane": 28.0},
}
thermal_results = run_thermal_analysis(mesh, regions, stator, em_results, th_config)
print(f"Thermal: T_peak={thermal_results['peak_temperature_C']:.1f} °C, "
      f"margin={thermal_results['thermal_margin_K']:.1f} K")

# ── Stage 3: Structural ───────────────────────────────────────────────────
st_config = {
    "reference_temperature_K": 293.15,
    "thermal_loads": True,
    "electromagnetic_loads": True,
    "materials": {},   # use built-in defaults
    "fatigue": {"method": "goodman", "stress_concentration_factor": 1.5},
    "modal": {"enabled": True, "num_modes": 4},
}
structural_results = run_structural_analysis(
    mesh, regions, stator, em_results, thermal_results, st_config
)
print(f"Structural: σ_vm_max={structural_results['max_von_mises_Pa']/1e6:.2f} MPa, "
      f"SF={structural_results['safety_factor']:.2f}")
```

---

### Example 4 — Parametric geometry sweep

Demonstrates iterating over a parameter (number of slots) and collecting scalar KPIs.

```python
# scripts/run_sweep.py
import pandas as pd
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline import run_fea_pipeline

results_rows = []

for n_slots in [24, 36, 48, 60]:
    stator = StatorMeshInput(
        stator_id      = f"sweep_s{n_slots}",
        outer_diameter = 1.300,
        inner_diameter = 0.840,
        axial_length   = 0.700,
        num_slots      = n_slots,
        num_poles      = 4,
        slot_opening   = 0.008,
        tooth_width    = 0.022,
        yoke_height    = 0.120,
        slot_depth     = 0.115,
        rated_current_rms = 850.0,
        rated_speed_rpm   = 1500.0,
    )

    r = run_fea_pipeline(
        stator,
        config_path = "FEA/configs/fast_sweep.yaml",  # linear solve, no modal
        output_dir  = f"results/sweep/",
    )

    results_rows.append({
        "n_slots":        n_slots,
        "torque_Nm":      r.em_results["torque_Nm"],
        "efficiency_pct": r.em_results["efficiency"] * 100,
        "iron_loss_W":    r.em_results["iron_loss_W"],
        "peak_temp_C":    r.thermal_results["peak_temperature_C"],
        "safety_factor":  r.structural_results["safety_factor"],
    })

df = pd.DataFrame(results_rows)
print(df.to_string(index=False))
```

**Expected output format:**
```
 n_slots  torque_Nm  efficiency_pct  iron_loss_W  peak_temp_C  safety_factor
      24      51.23           91.21       1423.1        118.4           9.12
      36      49.87           91.65       1612.8        121.7           8.74
      48      47.31           91.48       1821.4        115.3           8.31
      60      44.92           91.02       2043.6        112.1           7.95
```

---

### Example 5 — Accessing and plotting spatial field data

Demonstrates how to use the spatial field arrays returned by each stage for post-processing or visualisation.

```python
# scripts/plot_fields.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline import run_fea_pipeline

stator = StatorMeshInput(
    stator_id="plot_demo",
    outer_diameter=0.200, inner_diameter=0.120,
    axial_length=0.080, num_slots=6, num_poles=4,
    slot_opening=0.006, tooth_width=0.010,
    yoke_height=0.020, slot_depth=0.030,
)
results = run_fea_pipeline(stator)

mesh = results.em_results["domain"]
x    = mesh.nodes[:, 0]
y    = mesh.nodes[:, 1]
tri  = mesh.elements

# Compute element centroids for element-based fields
cx, cy = mesh.element_centroids().T

# ── Figure 1: Magnetic vector potential A_z ──────────────────────────────
A_z = results.em_results["A_field"]
triang = mtri.Triangulation(x, y, tri)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].tricontourf(triang, A_z, levels=30, cmap="RdBu_r")
plt.colorbar(im0, ax=axes[0], label="A_z [Wb/m]")
axes[0].set_title("Magnetic Vector Potential A_z")
axes[0].set_aspect("equal")

# ── Figure 2: Flux density |B| (element-centred → scatter plot) ──────────
B_mag = results.em_results["B_field"]["B_mag"]
sc1 = axes[1].scatter(cx, cy, c=B_mag, cmap="hot", s=1, vmin=0)
plt.colorbar(sc1, ax=axes[1], label="|B| [T]")
axes[1].set_title("Flux Density |B|")
axes[1].set_aspect("equal")

# ── Figure 3: Temperature field T ────────────────────────────────────────
T_K = results.thermal_results["T_field"]
T_C = T_K - 273.15
im2 = axes[2].tricontourf(triang, T_C, levels=30, cmap="inferno")
plt.colorbar(im2, ax=axes[2], label="Temperature [°C]")
axes[2].set_title("Steady-State Temperature")
axes[2].set_aspect("equal")

plt.tight_layout()
plt.savefig("results/field_plots.png", dpi=150)
plt.show()

print("Plots saved to results/field_plots.png")
```

---

## Running the Test Suite

The test suite provides 123 tests across unit, integration, and validation categories.

```bash
cd FEA

# All tests
pytest tests/ -v

# Unit tests only (fastest, ~3 s)
pytest tests/unit/ -v

# Integration tests (coupled stages, ~6 s)
pytest tests/integration/ -v

# Physics validation benchmarks (~4 s)
pytest tests/validation/ -v

# A specific test file
pytest tests/unit/test_em_solver.py -v

# Tests matching a keyword
pytest tests/ -k "efficiency" -v

# With coverage report
pytest tests/ --cov=fea_pipeline --cov-report=term-missing
```

**Validation benchmarks:**

| File                           | What is verified                                                  |
|--------------------------------|-------------------------------------------------------------------|
| `TEAM_benchmark_7.py`          | Zero source → zero potential; linear potential → uniform B field |
| `NIST_thermal_block.py`        | 1-D heat conduction: FEM vs analytical Fourier series solution   |
| `NAFEMS_structural.py`         | Uniaxial tension; Lamé thick-walled cylinder under pressure      |
