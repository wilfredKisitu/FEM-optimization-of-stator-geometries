# FEA Analysis Pipeline — Full Specification

## Overview

This document provides complete instructions for constructing, configuring, and testing a multi-physics Finite Element Analysis (FEA) pipeline that receives stator geometries from the upstream mesh construction module and executes three sequential but tightly coupled analyses:

1. **Electromagnetic Analysis** — magnetic flux, eddy current losses, torque ripple, iron losses
2. **Thermal Analysis** — steady-state and transient heat distribution driven by EM loss maps
3. **Structural Analysis** — stress, deformation, and fatigue driven by thermal and electromagnetic loads

The pipeline is designed to be modular: each analysis stage exposes a well-defined input/output contract so that individual solvers can be swapped, upgraded, or parallelised independently.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Dependencies and Environment Setup](#2-dependencies-and-environment-setup)
3. [Input Contract — Stator Geometry Interface](#3-input-contract--stator-geometry-interface)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Stage 1 — Electromagnetic Analysis](#5-stage-1--electromagnetic-analysis)
6. [Stage 2 — Thermal Analysis](#6-stage-2--thermal-analysis)
7. [Stage 3 — Structural Analysis](#7-stage-3--structural-analysis)
8. [Inter-Stage Data Coupling](#8-inter-stage-data-coupling)
9. [Output Schema](#9-output-schema)
10. [Testing Strategy](#10-testing-strategy)
11. [Validation Against Reference Cases](#11-validation-against-reference-cases)
12. [Configuration Reference](#12-configuration-reference)
13. [Error Handling and Logging](#13-error-handling-and-logging)
14. [Integration Checklist](#14-integration-checklist)

---

## 1. Repository Structure

```
fea_pipeline/
├── fea_pipeline/
│   ├── __init__.py
│   ├── orchestrator.py          # Top-level pipeline runner
│   ├── io/
│   │   ├── __init__.py
│   │   ├── mesh_reader.py       # Reads stator mesh from upstream module
│   │   ├── result_writer.py     # Writes HDF5 / VTK / JSON outputs
│   │   └── schema.py            # Pydantic models for all I/O contracts
│   ├── electromagnetic/
│   │   ├── __init__.py
│   │   ├── solver.py            # EM FEA solver wrapper
│   │   ├── boundary_conditions.py
│   │   ├── material_library.py
│   │   ├── loss_calculator.py   # Iron loss, eddy current loss
│   │   └── postprocessor.py     # Flux density maps, torque, force
│   ├── thermal/
│   │   ├── __init__.py
│   │   ├── solver.py            # Thermal FEA solver wrapper
│   │   ├── boundary_conditions.py
│   │   ├── heat_sources.py      # Maps EM losses → heat generation
│   │   └── postprocessor.py     # Temperature maps, hot spots
│   ├── structural/
│   │   ├── __init__.py
│   │   ├── solver.py            # Structural FEA solver wrapper
│   │   ├── boundary_conditions.py
│   │   ├── load_mapper.py       # Maps thermal + EM loads → mechanical
│   │   └── postprocessor.py     # Stress, strain, deformation, fatigue
│   └── utils/
│       ├── mesh_utils.py        # Mesh manipulation helpers
│       ├── interpolation.py     # Field interpolation between meshes
│       └── units.py             # Unit conversion registry
├── tests/
│   ├── unit/
│   │   ├── test_mesh_reader.py
│   │   ├── test_em_solver.py
│   │   ├── test_thermal_solver.py
│   │   └── test_structural_solver.py
│   ├── integration/
│   │   ├── test_em_to_thermal_coupling.py
│   │   ├── test_thermal_to_structural_coupling.py
│   │   └── test_full_pipeline.py
│   ├── validation/
│   │   ├── TEAM_benchmark_7.py  # Standard EM benchmark
│   │   ├── NIST_thermal_block.py
│   │   └── NAFEMS_structural.py
│   └── fixtures/
│       ├── stator_simple.h5      # Minimal stator: 6 slots, no winding
│       ├── stator_ipmsm_24s.h5   # 24-slot IPMSM stator
│       └── expected_outputs/
│           ├── em_reference.json
│           ├── thermal_reference.json
│           └── structural_reference.json
├── configs/
│   ├── default.yaml
│   ├── high_fidelity.yaml
│   └── fast_sweep.yaml
├── docs/
│   └── FEA.md                   # This file
├── pyproject.toml
└── README.md
```

---

## 2. Dependencies and Environment Setup

### 2.1 Python Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2.2 Core Python Dependencies (`pyproject.toml`)

```toml
[project]
name = "fea_pipeline"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Mesh and geometry
    "meshio>=5.3",           # Universal mesh I/O (reads .msh, .vtk, .h5, .xdmf)
    "gmsh>=4.12",            # Mesh generation and manipulation
    "numpy>=1.26",
    "scipy>=1.12",
    "h5py>=3.10",

    # FEA solver backends
    "fenics-dolfinx>=0.8",   # Primary FEM kernel (EM + thermal + structural)
    "petsc4py>=3.21",        # Linear algebra backend for dolfinx
    "slepc4py>=3.21",        # Eigenvalue problems (for EM modal analysis)

    # Electromagnetic specific
    "opencascade-pythonocc>=7.7",   # CAD kernel for geometry queries
    "magnetics-toolkit>=0.4",        # Material B-H curve fitting, iron loss models

    # Thermal specific
    "pyfluids>=2.6",         # Coolant property tables (water, oil, air)

    # Structural specific
    "pandas>=2.2",
    "scikit-learn>=1.4",     # For surrogate model post-processing

    # I/O and schema
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "vtk>=9.3",
    "rich>=13.7",            # CLI progress and logging

    # Visualization
    "matplotlib>=3.8",
    "pyvista>=0.43",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",    # Parallel test execution
    "mypy>=1.9",
    "ruff>=0.4",
]
```

### 2.3 External Solver Dependencies

Some solver backends require system-level installation:

| Solver            | Purpose                                 | Install Command                               |
| ----------------- | --------------------------------------- | --------------------------------------------- |
| FEniCSx (dolfinx) | Primary FEM kernel                      | `conda install -c conda-forge fenics-dolfinx` |
| Elmer FEM         | Optional: alternative structural solver | `apt install elmer` / brew                    |
| OpenFOAM          | Optional: CFD-coupled thermal           | See openfoam.com                              |
| GMSH              | Mesh generation and refinement          | bundled via `gmsh` pip package                |

> **Note:** FEniCSx requires MPI. Install OpenMPI before pip install:
> `apt install libopenmpi-dev` (Linux) or `brew install open-mpi` (macOS).

### 2.4 Hardware Requirements

| Analysis Type   | Minimum RAM | Recommended | Notes                                       |
| --------------- | ----------- | ----------- | ------------------------------------------- |
| Electromagnetic | 8 GB        | 32 GB       | Full B-H nonlinear: 3D mesh scales hard     |
| Thermal         | 4 GB        | 16 GB       | Steady-state is light; transient needs more |
| Structural      | 8 GB        | 32 GB       | Fine mesh near slot teeth critical          |
| Full pipeline   | 16 GB       | 64 GB       | All three coupled, parallel                 |

---

## 3. Input Contract — Stator Geometry Interface

This is the **critical integration boundary** between the upstream mesh construction module and this pipeline. Every field is mandatory unless marked optional.

### 3.1 Pydantic Schema

```python
# fea_pipeline/io/schema.py

from pydantic import BaseModel, Field, model_validator
from typing import Optional
import numpy as np

class StatorMeshInput(BaseModel):
    """
    Contract for stator geometry received from the mesh construction module.
    The upstream module must serialize to this schema before calling the pipeline.
    """

    # --- Identity ---
    stator_id: str                     # Unique identifier from upstream module
    geometry_source: str               # e.g. "parametric_v2", "imported_cad"

    # --- Mesh File References ---
    mesh_file_path: str                # Absolute path to .h5 or .msh file
    mesh_format: str                   # "gmsh4", "hdf5_xdmf", "vtk"

    # --- Physical Geometry (SI units throughout) ---
    outer_diameter: float              # [m]
    inner_diameter: float              # [m]
    axial_length: float                # [m]
    num_slots: int
    num_poles: int
    slot_opening: float                # [m]
    tooth_width: float                 # [m]
    yoke_height: float                 # [m]
    slot_depth: float                  # [m]

    # --- Winding Description ---
    winding_type: str                  # "distributed", "concentrated", "hairpin"
    num_layers: int                    # 1 or 2
    conductors_per_slot: int
    winding_factor: float              # Typically 0.85–0.96
    fill_factor: float                 # Copper fill factor, typically 0.40–0.65
    wire_diameter: Optional[float]     # [m] — None if litz wire or custom

    # --- Named Mesh Regions ---
    # These must match the physical group names tagged in the mesh file
    region_tags: dict[str, int]        # e.g. {"stator_core": 1, "winding": 2, "air_gap": 3}

    # --- Material Assignment ---
    material_map: dict[str, str]       # region_tag_name → material_id from library
    # Example: {"stator_core": "M250-35A", "winding": "copper_class_F", "air_gap": "air"}

    # --- Operating Point ---
    rated_current_rms: float           # [A]
    rated_speed_rpm: float             # [rpm]
    rated_torque: float                # [N·m]
    dc_bus_voltage: float              # [V]

    # --- Mesh Quality Metadata (from upstream module) ---
    min_element_quality: float         # Jacobian ratio; should be > 0.3
    max_element_size: float            # [m]
    num_elements: int
    num_nodes: int

    # --- Optional Symmetry ---
    symmetry_factor: Optional[int] = None   # e.g. 4 for quarter-symmetry model
    periodic_boundary_pairs: Optional[list[tuple[int, int]]] = None

    @model_validator(mode='after')
    def check_region_tags_cover_material_map(self):
        missing = set(self.material_map.keys()) - set(self.region_tags.keys())
        if missing:
            raise ValueError(f"material_map references unknown regions: {missing}")
        return self
```

### 3.2 Loading From Upstream Module

```python
# fea_pipeline/io/mesh_reader.py

import h5py
import meshio
from pathlib import Path
from .schema import StatorMeshInput

def load_stator_geometry(input: StatorMeshInput):
    """
    Reads the mesh file referenced in the StatorMeshInput and returns:
    - mesh: meshio.Mesh object with all physical groups populated
    - region_submeshes: dict[str, meshio.Mesh] keyed by region name
    """
    path = Path(input.mesh_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = meshio.read(str(path))

    # Validate that all declared region tags exist in the mesh file
    available_tags = {
        tag
        for cell_block in mesh.cells
        for tag in (mesh.cell_tags.get(cell_block.type, {}).values()
                    if hasattr(mesh, 'cell_tags') else [])
    }

    region_submeshes = {}
    for region_name, tag_id in input.region_tags.items():
        sub = _extract_region(mesh, tag_id)
        region_submeshes[region_name] = sub

    return mesh, region_submeshes


def _extract_region(mesh: meshio.Mesh, tag_id: int) -> meshio.Mesh:
    """Extracts cells belonging to a specific physical group tag."""
    cells = []
    cell_data = {}
    for cell_block in mesh.cells:
        mask = (mesh.cell_data.get("gmsh:physical", [None] * len(mesh.cells))[
                    mesh.cells.index(cell_block)] == tag_id)
        if mask.any():
            cells.append(meshio.CellBlock(cell_block.type, cell_block.data[mask]))
    return meshio.Mesh(points=mesh.points, cells=cells)
```

---

## 4. Pipeline Architecture

### 4.1 Orchestrator

```python
# fea_pipeline/orchestrator.py

from dataclasses import dataclass
from pathlib import Path
import yaml

from .io.schema import StatorMeshInput, PipelineConfig
from .io.mesh_reader import load_stator_geometry
from .electromagnetic.solver import run_electromagnetic_analysis
from .thermal.solver import run_thermal_analysis
from .structural.solver import run_structural_analysis
from .io.result_writer import write_results


@dataclass
class PipelineResults:
    em_results: dict
    thermal_results: dict
    structural_results: dict
    coupled_metrics: dict


def run_fea_pipeline(
    stator_input: StatorMeshInput,
    config_path: str = "configs/default.yaml",
    output_dir: str = "results/",
) -> PipelineResults:
    """
    Main entry point. Receives a StatorMeshInput and runs all three
    FEA stages in sequence, passing inter-stage data automatically.
    """
    config = _load_config(config_path)
    mesh, region_meshes = load_stator_geometry(stator_input)

    # --- Stage 1: Electromagnetic ---
    em_results = run_electromagnetic_analysis(
        mesh=mesh,
        regions=region_meshes,
        stator=stator_input,
        config=config["electromagnetic"],
    )

    # --- Stage 2: Thermal (driven by EM loss maps) ---
    thermal_results = run_thermal_analysis(
        mesh=mesh,
        regions=region_meshes,
        stator=stator_input,
        em_results=em_results,       # loss maps injected here
        config=config["thermal"],
    )

    # --- Stage 3: Structural (driven by thermal + EM loads) ---
    structural_results = run_structural_analysis(
        mesh=mesh,
        regions=region_meshes,
        stator=stator_input,
        em_results=em_results,
        thermal_results=thermal_results,
        config=config["structural"],
    )

    coupled_metrics = _compute_coupled_metrics(
        em_results, thermal_results, structural_results
    )

    results = PipelineResults(
        em_results=em_results,
        thermal_results=thermal_results,
        structural_results=structural_results,
        coupled_metrics=coupled_metrics,
    )

    write_results(results, output_dir, stator_input.stator_id)
    return results


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_coupled_metrics(em, thermal, structural) -> dict:
    return {
        "total_loss_W": em["total_loss_W"],
        "peak_temperature_K": thermal["peak_temperature_K"],
        "max_von_mises_Pa": structural["max_von_mises_Pa"],
        "thermal_derating_factor": _thermal_derating(thermal["peak_temperature_K"]),
        "safety_factor": structural["max_von_mises_Pa"] / structural["yield_strength_Pa"],
    }


def _thermal_derating(T_peak_K: float) -> float:
    """Simple linear derating above 120°C (393 K) for class F insulation."""
    T_max = 393.0  # 120°C in K
    if T_peak_K <= T_max:
        return 1.0
    return max(0.0, 1.0 - (T_peak_K - T_max) / 50.0)
```

---

## 5. Stage 1 — Electromagnetic Analysis

### 5.1 Physics Formulation

The electromagnetic problem is governed by the **magnetoquasistatic A-V formulation** (time-harmonic for steady-state, full transient for loss calculation):

```
∇ × (ν ∇ × A) + σ(∂A/∂t) = J_source
```

Where:

- `A` — magnetic vector potential [Wb/m]
- `ν` — magnetic reluctivity (inverse of permeability), nonlinear for iron regions
- `σ` — electrical conductivity [S/m]
- `J_source` — applied current density from winding excitation [A/m²]

### 5.2 Solver Implementation

```python
# fea_pipeline/electromagnetic/solver.py

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from .boundary_conditions import apply_em_boundary_conditions
from .material_library import get_material_properties
from .loss_calculator import compute_iron_losses, compute_copper_losses
from .postprocessor import (
    extract_flux_density,
    compute_torque,
    compute_cogging_torque,
)


def run_electromagnetic_analysis(mesh, regions, stator, config) -> dict:
    """
    Runs 2D magnetoquasistatic FEA on the stator cross-section.
    Returns field results and scalar metrics.
    """
    comm = MPI.COMM_WORLD

    # --- Build FEniCSx mesh from meshio object ---
    domain = _meshio_to_dolfinx(mesh, comm)
    V = fem.functionspace(domain, ("Nedelec 1st kind H(curl)", 1))

    # --- Material properties ---
    materials = {
        region: get_material_properties(
            material_id=stator.material_map[region],
            config=config["materials"]
        )
        for region in stator.region_tags
    }

    # --- Bilinear and linear forms ---
    A = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Reluctivity field (nonlinear for iron)
    nu = _build_reluctivity_field(domain, regions, materials)

    # Current density excitation
    J_src = _build_current_density(domain, stator, config)

    a = ufl.inner(nu * ufl.curl(A), ufl.curl(v)) * ufl.dx
    L = ufl.inner(J_src, v) * ufl.dx

    # --- Boundary conditions ---
    bcs = apply_em_boundary_conditions(domain, V, config)

    # --- Solve ---
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    A_sol = problem.solve()

    # --- Post-processing ---
    B_field = extract_flux_density(A_sol, domain)
    torque = compute_torque(B_field, stator, domain)
    cogging = compute_cogging_torque(domain, stator, config)
    iron_losses = compute_iron_losses(B_field, domain, materials, stator.rated_speed_rpm)
    copper_losses = compute_copper_losses(stator, config)

    return {
        # Scalar performance metrics
        "torque_Nm": float(torque),
        "cogging_torque_Nm": float(cogging),
        "iron_loss_W": float(iron_losses["total"]),
        "eddy_current_loss_W": float(iron_losses["eddy"]),
        "hysteresis_loss_W": float(iron_losses["hysteresis"]),
        "copper_loss_W": float(copper_losses),
        "total_loss_W": float(iron_losses["total"] + copper_losses),
        "efficiency": _compute_efficiency(torque, stator, iron_losses, copper_losses),

        # Spatial field data (passed to thermal stage)
        "A_field": A_sol,
        "B_field": B_field,
        "loss_density_map": iron_losses["spatial_W_per_m3"],  # fem.Function
        "copper_loss_density_map": copper_losses["spatial_W_per_m3"],

        # Mesh reference
        "domain": domain,
    }
```

### 5.3 Material Library

The material library must define the following for each iron grade:

```python
# fea_pipeline/electromagnetic/material_library.py

MATERIAL_DB = {
    "M250-35A": {
        "density_kg_m3": 7650,
        "electrical_conductivity_S_m": 1.96e6,
        "specific_heat_J_kgK": 460,
        "thermal_conductivity_W_mK": 28.0,
        # Steinmetz coefficients for iron loss: P = k_h * f * B^alpha + k_e * f^2 * B^2
        "steinmetz_kh": 143.0,
        "steinmetz_ke": 0.53,
        "steinmetz_alpha": 2.0,
        # B-H curve: list of (H [A/m], B [T]) pairs — nonlinear reluctivity
        "BH_curve": [
            (0, 0), (238.7, 0.2), (318.3, 0.4), (397.9, 0.6),
            (477.5, 0.8), (636.6, 1.0), (955.0, 1.2), (1591.5, 1.4),
            (3978.9, 1.6), (7957.7, 1.7), (15915.5, 1.8), (31831.0, 1.9),
        ],
    },
    "copper_class_F": {
        "electrical_conductivity_S_m": 5.8e7,
        "density_kg_m3": 8960,
        "specific_heat_J_kgK": 385,
        "thermal_conductivity_W_mK": 400.0,
        "resistivity_temperature_coefficient": 0.00393,   # per K
        "max_operating_temperature_K": 428.15,            # 155°C class F
    },
    "air": {
        "relative_permeability": 1.0,
        "electrical_conductivity_S_m": 0.0,
        "thermal_conductivity_W_mK": 0.025,
        "density_kg_m3": 1.204,
        "specific_heat_J_kgK": 1005,
    },
}
```

### 5.4 Boundary Conditions

| Boundary                     | Type                     | Value | Notes                         |
| ---------------------------- | ------------------------ | ----- | ----------------------------- |
| Outer stator boundary        | Dirichlet                | A = 0 | Flux confinement              |
| Symmetry plane (if periodic) | Periodic / Anti-periodic | ±A    | Depends on pole pair count    |
| Air gap inner boundary       | Natural (Neumann)        | —     | Continuity enforced naturally |

### 5.5 Key EM Configuration Parameters (`configs/default.yaml` section)

```yaml
electromagnetic:
  solver_type: "magnetoquasistatic_2d" # or "magnetostatic_3d" for 3D runs
  time_stepping:
    enabled: false # true for transient loss computation
    periods: 3
    steps_per_period: 72
  nonlinear:
    enabled: true # BH curve iteration
    max_iterations: 50
    tolerance: 1.0e-6
    relaxation: 0.7
  rotor_position_deg: 0.0 # Initial rotor position
  rotor_sweep:
    enabled: false
    start_deg: 0.0
    end_deg: 30.0 # One electrical period for 12-pole
    steps: 60
  materials:
    temperature_dependent: true # Update resistivity with thermal feedback
  mesh_refinement:
    air_gap_layers: 3 # Minimum layers across air gap
    tooth_tip_refinement: true
```

---

## 6. Stage 2 — Thermal Analysis

### 6.1 Physics Formulation

Steady-state heat conduction with volumetric heat sources from Stage 1:

```
-∇ · (k ∇T) = q_vol
```

Where:

- `T` — temperature field [K]
- `k` — thermal conductivity tensor [W/(m·K)] — anisotropic in laminated regions
- `q_vol` — volumetric heat generation [W/m³], sourced directly from EM loss maps

For transient analysis, the full parabolic problem:

```
ρ c_p ∂T/∂t - ∇ · (k ∇T) = q_vol(t)
```

### 6.2 Solver Implementation

```python
# fea_pipeline/thermal/solver.py

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
import numpy as np

from .boundary_conditions import apply_thermal_boundary_conditions
from .heat_sources import map_em_losses_to_heat_sources
from .postprocessor import (
    extract_temperature_field,
    identify_hot_spots,
    compute_winding_average_temperature,
)


def run_thermal_analysis(mesh, regions, stator, em_results, config) -> dict:
    """
    Runs steady-state (and optionally transient) thermal FEA.
    Heat sources are mapped directly from EM loss density fields.
    """
    comm = MPI.COMM_WORLD
    domain = em_results["domain"]   # Reuse same FEniCSx mesh

    V_T = fem.functionspace(domain, ("Lagrange", 1))

    # --- Map EM loss maps → volumetric heat sources ---
    q_iron = map_em_losses_to_heat_sources(
        em_results["loss_density_map"], domain, regions
    )
    q_copper = map_em_losses_to_heat_sources(
        em_results["copper_loss_density_map"], domain, regions
    )

    # --- Thermal conductivity (anisotropic for laminated core) ---
    k_tensor = _build_conductivity_tensor(domain, regions, stator, config)

    # --- Variational problem ---
    T = ufl.TrialFunction(V_T)
    v = ufl.TestFunction(V_T)

    a_th = ufl.inner(k_tensor * ufl.grad(T), ufl.grad(v)) * ufl.dx
    L_th = (q_iron + q_copper) * v * ufl.dx

    # --- Boundary conditions ---
    bcs_th = apply_thermal_boundary_conditions(domain, V_T, stator, config)

    # --- Solve ---
    problem = dolfinx.fem.petsc.LinearProblem(
        a_th, L_th, bcs=bcs_th,
        petsc_options={"ksp_type": "cg", "pc_type": "amg"},
    )
    T_sol = problem.solve()

    # --- Post-processing ---
    T_field = extract_temperature_field(T_sol)
    hot_spots = identify_hot_spots(T_field, threshold_fraction=0.95)
    T_winding_avg = compute_winding_average_temperature(T_field, regions)

    peak_T = float(T_field.x.array.max())

    return {
        "peak_temperature_K": peak_T,
        "peak_temperature_C": peak_T - 273.15,
        "winding_average_temperature_K": float(T_winding_avg),
        "hot_spot_locations": hot_spots,
        "thermal_margin_K": config["insulation"]["max_temperature_K"] - peak_T,

        # Spatial field (passed to structural stage)
        "T_field": T_sol,
        "domain": domain,
    }
```

### 6.3 Thermal Boundary Conditions

| Surface                 | Condition Type              | Default Value               | Config Key                      |
| ----------------------- | --------------------------- | --------------------------- | ------------------------------- |
| Outer stator OD         | Convection (Newton cooling) | h = 50 W/(m²·K), T_∞ = 40°C | `cooling.h_outer`               |
| Winding end-turn region | Convection                  | h = 25 W/(m²·K)             | `cooling.h_endturn`             |
| Frame interface         | Contact resistance          | R_c = 1.0e-4 m²·K/W         | `cooling.contact_resistance`    |
| Water jacket surface    | Fixed temperature           | T_coolant from config       | `cooling.coolant_temperature_K` |
| Symmetry planes         | Zero flux (Neumann)         | ∂T/∂n = 0                   | automatic                       |

### 6.4 Key Thermal Configuration

```yaml
thermal:
  analysis_type: "steady_state" # or "transient"
  transient:
    total_time_s: 300.0
    time_step_s: 1.0
    output_interval_s: 10.0
  cooling:
    type: "water_jacket" # or "air_forced", "air_natural", "oil_spray"
    coolant_temperature_K: 313.15 # 40°C
    h_outer: 500.0 # W/(m²·K) — water jacket
    h_endturn: 80.0
    contact_resistance: 1.0e-4 # m²·K/W, frame-stator interface
  insulation:
    class: "F" # B, F, H
    max_temperature_K: 428.15 # 155°C for class F
  anisotropy:
    lamination_direction: "z" # Stack axis
    k_in_plane: 28.0 # W/(m·K)
    k_through_plane: 1.0 # W/(m·K) — reduced due to lamination gaps
```

---

## 7. Stage 3 — Structural Analysis

### 7.1 Physics Formulation

Linear elasticity with thermal expansion and electromagnetic pressure loads:

```
∇ · σ + f_body = 0
σ = C : (ε - ε_thermal - ε_EM)
ε_thermal = α (T - T_ref) I
```

Where:

- `σ` — Cauchy stress tensor [Pa]
- `ε` — total strain tensor
- `C` — fourth-order elasticity tensor (Hooke)
- `α` — thermal expansion coefficient [1/K]
- `f_body` — body force from magnetic pressure (Maxwell stress tensor)

### 7.2 Solver Implementation

```python
# fea_pipeline/structural/solver.py

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI
import numpy as np

from .boundary_conditions import apply_structural_boundary_conditions
from .load_mapper import (
    compute_thermal_expansion_strain,
    compute_maxwell_stress_load,
)
from .postprocessor import (
    compute_von_mises,
    compute_principal_stresses,
    compute_fatigue_life,
    compute_natural_frequencies,
)


def run_structural_analysis(mesh, regions, stator, em_results, thermal_results, config) -> dict:
    """
    Runs linear elastostatic FEA with thermal and electromagnetic loads.
    """
    comm = MPI.COMM_WORLD
    domain = thermal_results["domain"]

    # Vector function space for displacement field
    V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    # --- Material properties ---
    E_mod = _build_youngs_modulus_field(domain, regions, stator, config)
    nu_mat = _build_poisson_ratio_field(domain, regions, stator, config)
    alpha_th = _build_thermal_expansion_field(domain, regions, stator, config)
    T_ref_K = config["reference_temperature_K"]

    # --- Constitutive relations ---
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u, T_field):
        eps = epsilon(u)
        eps_thermal = alpha_th * (T_field - T_ref_K) * ufl.Identity(len(u))
        lam = E_mod * nu_mat / ((1 + nu_mat) * (1 - 2 * nu_mat))
        mu = E_mod / (2 * (1 + nu_mat))
        return lam * ufl.tr(eps - eps_thermal) * ufl.Identity(len(u)) + \
               2 * mu * (eps - eps_thermal)

    # --- Maxwell stress body force from EM ---
    f_em = compute_maxwell_stress_load(em_results["B_field"], domain)

    a_s = ufl.inner(sigma(u, thermal_results["T_field"]), epsilon(v)) * ufl.dx
    L_s = ufl.inner(f_em, v) * ufl.dx

    bcs_s = apply_structural_boundary_conditions(domain, V_u, config)

    problem = dolfinx.fem.petsc.LinearProblem(
        a_s, L_s, bcs=bcs_s,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
    )
    u_sol = problem.solve()

    # --- Post-processing ---
    vm_stress = compute_von_mises(u_sol, domain, E_mod, nu_mat)
    principal = compute_principal_stresses(u_sol, domain, E_mod, nu_mat)
    max_vm = float(vm_stress.x.array.max())
    max_disp = float(np.max(np.linalg.norm(
        u_sol.x.array.reshape(-1, domain.geometry.dim), axis=1)))

    yield_strength = config["materials"]["stator_core"]["yield_strength_Pa"]
    fatigue = compute_fatigue_life(vm_stress, config)
    nat_freqs = compute_natural_frequencies(domain, V_u, E_mod, nu_mat, stator, config)

    return {
        "max_von_mises_Pa": max_vm,
        "max_displacement_m": max_disp,
        "yield_strength_Pa": yield_strength,
        "safety_factor": yield_strength / max_vm,
        "fatigue_life_cycles": fatigue,
        "natural_frequencies_Hz": nat_freqs,
        "critical_mode": int(np.argmin(nat_freqs)) + 1,

        # Spatial fields
        "u_field": u_sol,
        "von_mises_field": vm_stress,
        "principal_stress_field": principal,
        "domain": domain,
    }
```

### 7.3 Structural Material Properties

```yaml
structural:
  reference_temperature_K: 293.15 # 20°C stress-free reference
  materials:
    stator_core:
      material_id: "M250-35A_struct"
      youngs_modulus_Pa: 2.0e11 # 200 GPa for silicon steel
      poisson_ratio: 0.28
      density_kg_m3: 7650
      thermal_expansion_1_K: 12.0e-6
      yield_strength_Pa: 3.5e8 # 350 MPa
      ultimate_strength_Pa: 5.0e8
      fatigue_limit_Pa: 2.0e8 # Endurance limit
    winding_equivalent:
      # Homogenized winding + insulation composite
      youngs_modulus_Pa: 3.0e9
      poisson_ratio: 0.35
      density_kg_m3: 3500
      thermal_expansion_1_K: 18.0e-6
  boundary_conditions:
    fixed_surfaces: ["outer_frame_contact"]
    symmetry_planes: ["x_symmetry", "y_symmetry"]
  fatigue:
    method: "goodman" # or "gerber", "soderberg"
    stress_concentration_factor: 1.5
    surface_finish_factor: 0.85
    reliability_factor: 0.897 # 90% reliability
  modal:
    enabled: true
    num_modes: 10
    frequency_range_Hz: [0, 5000]
```

---

## 8. Inter-Stage Data Coupling

### 8.1 EM → Thermal Coupling

The electromagnetic loss maps are the primary thermal source. The coupling is one-way (no thermal feedback into EM for steady-state; optional two-way for temperature-dependent conductivity):

```
q_iron(x) = P_iron_density(x)   [W/m³]  from EM iron loss model
q_copper(x) = J²(x) / σ(T(x))  [W/m³]  current density from EM, conductivity temp-dependent
```

For temperature-dependent copper resistivity, enable `electromagnetic.materials.temperature_dependent: true`. This triggers a **staggered iteration**:

1. Run EM at T = T_ref
2. Run Thermal with EM losses
3. Update σ(T) in EM material model
4. Re-run EM
5. Repeat until `‖T_new - T_old‖ < tol_coupling`

### 8.2 Thermal → Structural Coupling

The thermal field is mapped as a prestrain load. The spatial temperature field `T(x)` from Stage 2 is directly fed into Stage 3 as a fem.Function; no interpolation is needed since both stages operate on the same FEniCSx mesh.

The thermal mismatch strain at each material interface (core–winding, core–frame) requires a **contact gap element** if debonding is of concern. Enable via `structural.contact.enabled: true`.

### 8.3 EM → Structural Coupling

Magnetic pressure (Maxwell stress) is computed from the flux density `B` field:

```python
# fea_pipeline/structural/load_mapper.py

import ufl

def compute_maxwell_stress_load(B_field, domain):
    """
    Computes Maxwell stress tensor divergence as a body force:
    f_i = (1/μ₀) * (B_j * ∂B_i/∂x_j - 0.5 * ∂B²/∂x_i)
    """
    mu_0 = 4 * 3.14159e-7  # H/m
    B = B_field
    T_maxwell = (1 / mu_0) * (
        ufl.outer(B, B) - 0.5 * ufl.inner(B, B) * ufl.Identity(domain.geometry.dim)
    )
    return ufl.div(T_maxwell)
```

---

## 9. Output Schema

All results are written to an HDF5 archive with a consistent structure:

```
results/{stator_id}/
├── metadata.json              # Input parameters, solver versions, run timestamp
├── electromagnetic/
│   ├── scalars.json           # torque, losses, efficiency
│   ├── A_field.h5             # Magnetic vector potential (nodal)
│   ├── B_field.h5             # Flux density (element-wise)
│   └── loss_density.h5        # Iron + copper loss density maps
├── thermal/
│   ├── scalars.json           # Peak temp, average temp, thermal margin
│   └── T_field.h5             # Temperature field (nodal)
├── structural/
│   ├── scalars.json           # von Mises, displacement, safety factor, fatigue
│   ├── u_field.h5             # Displacement field (nodal, vector)
│   ├── von_mises.h5           # von Mises stress (element-wise)
│   └── natural_frequencies.json
└── coupled_metrics.json       # Cross-physics summary metrics
```

The `scalars.json` files are all machine-readable and follow this structure:

```json
{
  "stage": "electromagnetic",
  "stator_id": "ipmsm_24s_run_001",
  "solver_version": "0.1.0",
  "timestamp_utc": "2025-04-01T14:32:00Z",
  "results": {
    "torque_Nm": 142.7,
    "cogging_torque_Nm": 1.23,
    "iron_loss_W": 312.4,
    "copper_loss_W": 894.1,
    "total_loss_W": 1206.5,
    "efficiency": 0.9234
  }
}
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

Run the unit test suite:

```bash
pytest tests/unit/ -v --cov=fea_pipeline --cov-report=term-missing
```

Each unit test isolates a single function using synthetic, analytically tractable inputs.

**Example: Iron loss unit test**

```python
# tests/unit/test_em_solver.py

import numpy as np
import pytest
from fea_pipeline.electromagnetic.loss_calculator import steinmetz_iron_loss

def test_steinmetz_loss_zero_frequency():
    """Iron loss must be zero at zero frequency."""
    loss = steinmetz_iron_loss(B_peak=1.5, freq_Hz=0.0, kh=143.0, ke=0.53, alpha=2.0)
    assert loss == pytest.approx(0.0, abs=1e-10)

def test_steinmetz_loss_known_value():
    """Cross-check against hand-calculated value for M250-35A at 50 Hz, 1T."""
    # P = 143 * 50 * 1.0^2 + 0.53 * 50^2 * 1.0^2 = 7150 + 1325 = 8475 W/m³
    loss = steinmetz_iron_loss(B_peak=1.0, freq_Hz=50.0, kh=143.0, ke=0.53, alpha=2.0)
    assert loss == pytest.approx(8475.0, rel=1e-4)

def test_steinmetz_loss_b_squared_dependence():
    """Doubling B should approximately quadruple loss (alpha=2)."""
    l1 = steinmetz_iron_loss(B_peak=0.5, freq_Hz=50.0, kh=143.0, ke=0.53, alpha=2.0)
    l2 = steinmetz_iron_loss(B_peak=1.0, freq_Hz=50.0, kh=143.0, ke=0.53, alpha=2.0)
    assert l2 / l1 == pytest.approx(4.0, rel=0.01)
```

**Example: Thermal boundary condition unit test**

```python
# tests/unit/test_thermal_solver.py

import pytest
from unittest.mock import MagicMock
from fea_pipeline.thermal.boundary_conditions import apply_thermal_boundary_conditions

def test_thermal_bc_raises_on_missing_cooling_config():
    """Missing cooling section in config must raise a KeyError."""
    domain = MagicMock()
    V_T = MagicMock()
    stator = MagicMock()
    with pytest.raises(KeyError, match="cooling"):
        apply_thermal_boundary_conditions(domain, V_T, stator, config={})

def test_thermal_bc_water_jacket_sets_fixed_temperature(monkeypatch):
    """Water jacket cooling must result in at least one Dirichlet BC."""
    ...
```

### 10.2 Integration Tests

```bash
pytest tests/integration/ -v
```

**EM → Thermal coupling test (abbreviated):**

```python
# tests/integration/test_em_to_thermal_coupling.py

import pytest
from pathlib import Path
from fea_pipeline.io.schema import StatorMeshInput
from fea_pipeline.io.mesh_reader import load_stator_geometry
from fea_pipeline.electromagnetic.solver import run_electromagnetic_analysis
from fea_pipeline.thermal.solver import run_thermal_analysis

FIXTURE_PATH = Path("tests/fixtures/stator_simple.h5")

@pytest.fixture
def simple_stator():
    return StatorMeshInput(
        stator_id="test_simple",
        geometry_source="test_fixture",
        mesh_file_path=str(FIXTURE_PATH),
        mesh_format="hdf5_xdmf",
        outer_diameter=0.200,
        inner_diameter=0.120,
        axial_length=0.080,
        num_slots=6,
        num_poles=4,
        slot_opening=0.005,
        tooth_width=0.012,
        yoke_height=0.015,
        slot_depth=0.025,
        winding_type="distributed",
        num_layers=2,
        conductors_per_slot=20,
        winding_factor=0.866,
        fill_factor=0.45,
        wire_diameter=0.001,
        region_tags={"stator_core": 1, "winding": 2, "air_gap": 3},
        material_map={"stator_core": "M250-35A", "winding": "copper_class_F", "air_gap": "air"},
        rated_current_rms=50.0,
        rated_speed_rpm=3000.0,
        rated_torque=50.0,
        dc_bus_voltage=400.0,
        min_element_quality=0.6,
        max_element_size=0.003,
        num_elements=8500,
        num_nodes=4320,
    )

def test_em_thermal_coupling_produces_positive_temperature(simple_stator):
    """Thermal stage output must be above ambient with any positive EM losses."""
    mesh, regions = load_stator_geometry(simple_stator)
    em = run_electromagnetic_analysis(mesh, regions, simple_stator, config=_default_em_config())
    assert em["total_loss_W"] > 0.0

    thermal = run_thermal_analysis(mesh, regions, simple_stator, em, config=_default_thermal_config())
    assert thermal["peak_temperature_K"] > 313.15   # Must exceed coolant temp

def test_em_thermal_loss_conservation(simple_stator):
    """Heat flowing out through boundaries must match total loss input within 2%."""
    ...
```

### 10.3 Full Pipeline Test

```bash
pytest tests/integration/test_full_pipeline.py -v -s
```

```python
def test_full_pipeline_runs_without_error(simple_stator, tmp_path):
    from fea_pipeline.orchestrator import run_fea_pipeline
    results = run_fea_pipeline(
        stator_input=simple_stator,
        config_path="configs/default.yaml",
        output_dir=str(tmp_path),
    )
    assert results.em_results["torque_Nm"] > 0
    assert results.thermal_results["peak_temperature_K"] > 293.0
    assert results.structural_results["safety_factor"] > 0.0
    assert (tmp_path / simple_stator.stator_id / "coupled_metrics.json").exists()

def test_pipeline_metrics_physically_reasonable(simple_stator, tmp_path):
    from fea_pipeline.orchestrator import run_fea_pipeline
    results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
    m = results.coupled_metrics
    # Efficiency must be between 70% and 99.9%
    assert 0.70 < results.em_results["efficiency"] < 0.999
    # Safety factor must be positive
    assert m["safety_factor"] > 0
    # Temperature derating factor between 0 and 1
    assert 0 <= m["thermal_derating_factor"] <= 1.0
```

---

## 11. Validation Against Reference Cases

### 11.1 TEAM Benchmark Problem 7 (Electromagnetic)

The TEAM (Testing Electromagnetic Analysis Methods) Problem 7 is a standard validation for 2D magnetostatic solvers.

```bash
python tests/validation/TEAM_benchmark_7.py
```

Expected output:

- `B_y` at measurement point P1 = 0.951 T ± 1%
- `B_x` at measurement point P2 = 0.423 T ± 1%

### 11.2 Thermal Validation — Isothermal Slab

```python
# tests/validation/NIST_thermal_block.py
"""
Validates thermal solver against analytical solution for a 1D slab:
  -k d²T/dx² = q
  T(0) = T_L, T(L) = T_R
  Analytical: T(x) = T_L + (T_R - T_L)*x/L + q/(2k) * x*(L-x)
"""
```

Expected: max error < 0.1% of analytical solution across all nodes.

### 11.3 Structural Validation — NAFEMS Benchmark

The NAFEMS LE1 benchmark (linear elastic plane stress) validates stress computation:

```bash
python tests/validation/NAFEMS_structural.py
```

Expected: σ_y at point D = 92.7 MPa ± 2%.

---

## 12. Configuration Reference

### `configs/default.yaml` — Annotated

```yaml
# Pipeline-wide settings
pipeline:
  name: "default"
  log_level: "INFO" # DEBUG, INFO, WARNING, ERROR
  parallel_stages: false # true requires MPI across stages
  mpi_np: 4 # MPI process count per stage

electromagnetic:
  solver_type: "magnetoquasistatic_2d"
  nonlinear:
    enabled: true
    max_iterations: 50
    tolerance: 1.0e-6
    relaxation: 0.7
  time_stepping:
    enabled: false
    periods: 3
    steps_per_period: 72
  rotor_position_deg: 0.0
  rotor_sweep:
    enabled: false
    start_deg: 0.0
    end_deg: 30.0
    steps: 60
  loss_model: "steinmetz" # or "bertotti" for frequency-dependent
  materials:
    temperature_dependent: false

thermal:
  analysis_type: "steady_state"
  cooling:
    type: "water_jacket"
    coolant_temperature_K: 313.15
    h_outer: 500.0
    h_endturn: 80.0
    contact_resistance: 1.0e-4
  insulation:
    class: "F"
    max_temperature_K: 428.15
  anisotropy:
    k_in_plane: 28.0
    k_through_plane: 1.0
  convergence_tolerance: 1.0e-8

structural:
  reference_temperature_K: 293.15
  electromagnetic_loads: true
  thermal_loads: true
  materials:
    stator_core:
      youngs_modulus_Pa: 2.0e11
      poisson_ratio: 0.28
      yield_strength_Pa: 3.5e8
      thermal_expansion_1_K: 12.0e-6
    winding_equivalent:
      youngs_modulus_Pa: 3.0e9
      poisson_ratio: 0.35
      thermal_expansion_1_K: 18.0e-6
  fatigue:
    method: "goodman"
    stress_concentration_factor: 1.5
  modal:
    enabled: true
    num_modes: 10

output:
  format: "hdf5" # or "vtk", "both"
  write_fields: true # Set false for scalar-only sweeps
  compression: true
```

---

## 13. Error Handling and Logging

```python
# fea_pipeline/utils/logging.py

import logging
from rich.logging import RichHandler

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(name)
```

### Common Errors and Resolutions

| Error                               | Likely Cause                                           | Resolution                                                    |
| ----------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------- |
| `MeshRegionNotFoundError`           | Region tag mismatch between input schema and mesh file | Verify `region_tags` dict matches physical group IDs in GMSH  |
| `NonlinearConvergenceError`         | BH curve is outside extrapolation range                | Check B field saturation; reduce relaxation factor            |
| `NegativeJacobianError`             | Poor mesh quality from upstream module                 | Set `min_element_quality > 0.3` in StatorMeshInput            |
| `ThermalDivergenceError`            | Heat source exceeds boundary capacity                  | Check h values; reduce loss inputs                            |
| `SingularMatrixError` in structural | Underconstrained model                                 | Add at least one Dirichlet BC per degree of freedom direction |
| `MemoryError`                       | 3D mesh too large for available RAM                    | Enable symmetry reduction or use distributed MPI solver       |

---

## 14. Integration Checklist

Before connecting the upstream mesh construction module to this pipeline, verify the following:

### From Upstream Module (mesh construction side)

- [ ] Mesh file is written in one of: `.msh` (GMSH 4.x), `.h5` + `.xdmf` pair, or `.vtk`
- [ ] All physical groups declared in `region_tags` are present in the mesh file
- [ ] Physical group integer IDs are stable across mesh regenerations
- [ ] Air gap region is included as a separate physical group
- [ ] Mesh includes at least 3 element layers across the minimum feature (tooth tip or air gap)
- [ ] `min_element_quality` has been computed and is > 0.3
- [ ] All geometric parameters (OD, ID, slot_depth, etc.) match the actual mesh dimensions

### From This Pipeline (FEA side)

- [ ] All materials referenced in `material_map` exist in `MATERIAL_DB`
- [ ] BH curves cover the expected B range (check at 1.8 T minimum)
- [ ] Boundary condition surfaces are tagged in the mesh
- [ ] Config YAML loads without validation errors: `python -c "from fea_pipeline.orchestrator import _load_config; _load_config('configs/default.yaml')"`
- [ ] Unit tests pass: `pytest tests/unit/ -v`
- [ ] Integration tests pass on the simple fixture: `pytest tests/integration/ -v`
- [ ] Validation benchmarks within tolerance: `python tests/validation/TEAM_benchmark_7.py`

### Runtime Verification

After the first real stator geometry is fed in:

- [ ] `em_results["total_loss_W"]` is in the range 0.1–10% of rated power
- [ ] `thermal_results["peak_temperature_K"]` is above coolant temperature
- [ ] `structural_results["safety_factor"]` is > 1.0 (otherwise redesign is needed)
- [ ] All output files are written to `results/{stator_id}/`
- [ ] `coupled_metrics.json` contains all five keys

---

_End of FEA Pipeline Specification. Version 0.1.0 — April 2026._
