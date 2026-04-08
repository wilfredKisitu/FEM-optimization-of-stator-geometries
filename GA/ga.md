# GA_PIPELINE.md — Agent Build & Test Instructions

## Genetic Algorithm Optimization Pipeline for Stator Generator Design

---

## Purpose of This Document

This file is a complete, step-by-step instruction set for an **autonomous agent** to build, wire, and test a multi-objective genetic algorithm (GA) optimization pipeline that sits as a top-level orchestration layer above two existing modules:

- **Stator generation pipeline** — takes geometric parameters and produces a meshed stator geometry (`StatorMeshInput`)
- **FEA analysis pipeline** — takes a `StatorMeshInput` and returns electromagnetic, thermal, and structural analysis results

The agent must not modify either of those modules. The GA pipeline calls into them through their existing public interfaces only.

---

## Table of Contents

1. [Pre-Build Verification — Existing Module Interfaces](#1-pre-build-verification--existing-module-interfaces)
2. [Repository Extension — New Directory Structure](#2-repository-extension--new-directory-structure)
3. [Dependencies](#3-dependencies)
4. [Design Space — Chromosome Encoding](#4-design-space--chromosome-encoding)
5. [Objective Functions](#5-objective-functions)
6. [Constraint Handling](#6-constraint-handling)
7. [Core GA Components](#7-core-ga-components)
   - 7.1 Population initialisation
   - 7.2 Fitness evaluation (FEA call)
   - 7.3 Non-dominated sorting (NSGA-II)
   - 7.4 Crowding distance
   - 7.5 Tournament selection
   - 7.6 Simulated binary crossover (SBX)
   - 7.7 Polynomial mutation
   - 7.8 Pareto archive
8. [Termination Conditions](#8-termination-conditions)
9. [GA Orchestrator](#9-ga-orchestrator)
10. [Parallelisation Strategy](#10-parallelisation-strategy)
11. [Persistence and Checkpointing](#11-persistence-and-checkpointing)
12. [Output Schema](#12-output-schema)
13. [Configuration Reference](#13-configuration-reference)
14. [Build Order — Step-by-Step Agent Instructions](#14-build-order--step-by-step-agent-instructions)
15. [Testing Strategy](#15-testing-strategy)
16. [Validation Benchmarks](#16-validation-benchmarks)
17. [Integration Checklist](#17-integration-checklist)
18. [Error Handling Reference](#18-error-handling-reference)

---

## 1. Pre-Build Verification — Existing Module Interfaces

**Before writing a single line of GA code**, the agent must verify that the two downstream modules expose the expected interfaces. Run the following checks and abort with a descriptive error if any fail.

### 1.1 Stator generation module check

```python
# scripts/verify_stator_interface.py
"""
Run this first. If it raises, the stator generation module is not ready
to receive GA-generated parameters. Fix the interface mismatch before
building any GA code.
"""
from stator_generation.api import generate_stator   # adjust import path
from stator_generation.schema import StatorParams    # pydantic model

# Probe with a known-good minimal parameter set
probe = StatorParams(
    outer_diameter=0.200,
    inner_diameter=0.120,
    axial_length=0.080,
    num_slots=24,
    num_poles=8,
    tooth_width_fraction=0.45,
    yoke_height_fraction=0.35,
    slot_depth=0.030,
    slot_opening=0.004,
    winding_type="distributed",
    num_layers=2,
    conductors_per_slot=20,
    fill_factor=0.45,
)
result = generate_stator(probe)
assert hasattr(result, 'mesh_file_path'), "generate_stator must return object with mesh_file_path"
assert hasattr(result, 'region_tags'),   "generate_stator must return object with region_tags"
print("Stator generation interface OK")
```

### 1.2 FEA pipeline check

```python
# scripts/verify_fea_interface.py
from fea_pipeline.orchestrator import run_fea_pipeline
from fea_pipeline.io.schema import StatorMeshInput

# Use the simple fixture bundled with the FEA pipeline
stator = StatorMeshInput(
    stator_id="ga_probe_001",
    mesh_file_path="tests/fixtures/stator_simple.h5",
    # ... fill all required fields from FEA.md section 3.1
)
results = run_fea_pipeline(stator, config_path="configs/default.yaml", output_dir="/tmp/ga_probe")
assert results.em_results["torque_Nm"] > 0
assert results.thermal_results["peak_temperature_K"] > 293.0
assert results.structural_results["safety_factor"] > 0.0
print("FEA pipeline interface OK")
```

**The agent must not proceed past this section until both scripts print "OK".**

---

## 2. Repository Extension — New Directory Structure

The GA pipeline lives entirely inside a new top-level `ga_optimizer/` package. Do **not** modify anything outside it.

```
ga_optimizer/
├── __init__.py
├── orchestrator.py              # Top-level GA loop
├── chromosome.py                # Encoding / decoding / bounds
├── population.py                # Population dataclass + initialisation
├── objectives.py                # Objective extraction from FEA results
├── constraints.py               # Feasibility checks before FEA is called
├── operators/
│   ├── __init__.py
│   ├── selection.py             # Binary tournament
│   ├── crossover.py             # SBX (real-valued)
│   ├── mutation.py              # Polynomial mutation
│   └── repair.py                # Clamp out-of-bounds genes after mutation
├── pareto/
│   ├── __init__.py
│   ├── nsga2.py                 # Non-dominated sorting + crowding distance
│   └── archive.py               # Persistent Pareto archive across generations
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py             # Calls stator_generation + FEA; returns ObjectiveVector
│   └── cache.py                 # Hash-keyed result cache (skip duplicate individuals)
├── io/
│   ├── __init__.py
│   ├── checkpoint.py            # Save / restore full GA state to HDF5
│   ├── result_writer.py         # Final Pareto front export
│   └── schema.py                # Pydantic models for GA I/O
├── utils/
│   ├── logger.py
│   └── metrics.py               # Hypervolume, generational distance, spread
├── configs/
│   ├── default_ga.yaml
│   └── fast_ga.yaml             # Reduced pop/gen for CI testing
└── tests/
    ├── unit/
    │   ├── test_chromosome.py
    │   ├── test_nsga2.py
    │   ├── test_operators.py
    │   └── test_cache.py
    ├── integration/
    │   ├── test_evaluator.py
    │   ├── test_full_ga_fast.py  # 3 generations, pop=10, uses fast_ga.yaml
    │   └── test_pareto_archive.py
    └── fixtures/
        ├── mock_fea_results.json
        └── reference_pareto_front.json
```

---

## 3. Dependencies

Add these to `pyproject.toml` under `[project.dependencies]`. Do not remove any existing dependency.

```toml
# GA optimizer additions
"deap>=1.4",            # Optional: used only for reference comparison in tests
"pymoo>=0.6",           # Reference NSGA-II for validation (test use only)
"joblib>=1.4",          # Parallel population evaluation
"optuna>=3.6",          # Optional: surrogate-assisted acceleration (phase 2)
"tqdm>=4.66",           # Progress bars
"plotly>=5.20",         # Pareto front visualisation
```

All existing dependencies from `fea_pipeline` and `stator_generation` are inherited. The agent must confirm there are no version conflicts after adding these.

---

## 4. Design Space — Chromosome Encoding

### 4.1 Gene vector definition

Each individual is a real-valued vector of 12 genes. The agent must implement this exactly — the bounds and types here are the authoritative specification.

```python
# ga_optimizer/chromosome.py

from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np

@dataclass
class GeneDefinition:
    name: str
    lower: float
    upper: float
    dtype: str          # "float" or "int"
    unit: str
    description: str

GENE_DEFINITIONS: list[GeneDefinition] = [
    # Index 0
    GeneDefinition("outer_diameter",        0.150, 0.400, "float", "m",
                   "Stator outer diameter"),
    # Index 1
    GeneDefinition("bore_ratio",            0.50,  0.72,  "float", "—",
                   "inner_diameter / outer_diameter; enforces ID<OD"),
    # Index 2
    GeneDefinition("axial_length",          0.050, 0.200, "float", "m",
                   "Active axial stack length"),
    # Index 3
    GeneDefinition("num_slots",             12,    72,    "int",   "—",
                   "Number of stator slots; must be divisible by num_poles/2*3"),
    # Index 4
    GeneDefinition("num_poles",             4,     20,    "int",   "—",
                   "Number of rotor poles; must be even"),
    # Index 5
    GeneDefinition("tooth_width_fraction",  0.35,  0.65,  "float", "—",
                   "Tooth width as fraction of slot pitch"),
    # Index 6
    GeneDefinition("yoke_height_fraction",  0.20,  0.55,  "float", "—",
                   "Yoke height as fraction of (OD-ID)/2"),
    # Index 7
    GeneDefinition("slot_depth_fraction",   0.30,  0.65,  "float", "—",
                   "Slot depth as fraction of (OD-ID)/2"),
    # Index 8
    GeneDefinition("conductors_per_slot",   8,     64,    "int",   "—",
                   "Number of conductors per slot"),
    # Index 9
    GeneDefinition("fill_factor",           0.35,  0.65,  "float", "—",
                   "Copper fill factor"),
    # Index 10
    GeneDefinition("slot_opening_fraction", 0.10,  0.40,  "float", "—",
                   "Slot opening as fraction of tooth width"),
    # Index 11
    GeneDefinition("axial_length_ratio",    0.50,  3.00,  "float", "—",
                   "axial_length / outer_diameter; aspect ratio constraint"),
]

N_GENES = len(GENE_DEFINITIONS)
LOWER_BOUNDS = np.array([g.lower for g in GENE_DEFINITIONS])
UPPER_BOUNDS = np.array([g.upper for g in GENE_DEFINITIONS])


def decode_chromosome(genes: np.ndarray) -> dict:
    """
    Converts a raw gene vector to a StatorParams-compatible dict.
    Handles int rounding, unit derivations, and consistency enforcement.
    Raises ValueError if any derived quantity is physically impossible.
    """
    g = genes.copy()

    OD = float(np.clip(g[0], LOWER_BOUNDS[0], UPPER_BOUNDS[0]))
    bore_ratio = float(np.clip(g[1], LOWER_BOUNDS[1], UPPER_BOUNDS[1]))
    ID = OD * bore_ratio

    axial = float(np.clip(g[2], LOWER_BOUNDS[2], UPPER_BOUNDS[2]))

    # Integer genes: round to nearest valid int
    num_poles = int(round(g[4]))
    num_poles = max(4, num_poles - (num_poles % 2))   # enforce even
    # slots must be multiple of 3 * (num_poles/2) for standard 3-phase winding
    q_slots_min = 3 * (num_poles // 2)
    num_slots_raw = int(round(g[3]))
    num_slots = max(q_slots_min, round(num_slots_raw / q_slots_min) * q_slots_min)

    air_gap_radial = (OD - ID) / 2.0
    tooth_w = float(g[5]) * (np.pi * ID / num_slots)   # absolute tooth width [m]
    yoke_h  = float(g[6]) * air_gap_radial
    slot_d  = float(g[7]) * air_gap_radial
    slot_op = float(g[10]) * tooth_w

    conductors = int(round(g[8]))
    conductors = max(2, conductors - (conductors % 2))  # enforce even
    fill_factor = float(np.clip(g[9], LOWER_BOUNDS[9], UPPER_BOUNDS[9]))

    # Consistency check: slot_depth + yoke_height must fit inside the radial build
    if slot_d + yoke_h > air_gap_radial * 0.95:
        raise ValueError(
            f"slot_depth ({slot_d:.4f}) + yoke_height ({yoke_h:.4f}) "
            f"exceeds radial build ({air_gap_radial:.4f})"
        )

    return {
        "outer_diameter":        OD,
        "inner_diameter":        ID,
        "axial_length":          axial,
        "num_slots":             num_slots,
        "num_poles":             num_poles,
        "tooth_width":           tooth_w,
        "yoke_height":           yoke_h,
        "slot_depth":            slot_d,
        "slot_opening":          slot_op,
        "conductors_per_slot":   conductors,
        "fill_factor":           fill_factor,
    }


def random_individual(rng: np.random.Generator) -> np.ndarray:
    """Samples a random gene vector uniformly within bounds."""
    return rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
```

### 4.2 Repair after crossover / mutation

```python
# ga_optimizer/operators/repair.py

import numpy as np
from ..chromosome import LOWER_BOUNDS, UPPER_BOUNDS

def clamp(genes: np.ndarray) -> np.ndarray:
    """Hard clamp all genes to [lower, upper]. Always call after crossover and mutation."""
    return np.clip(genes, LOWER_BOUNDS, UPPER_BOUNDS)
```

---

## 5. Objective Functions

The GA optimises three objectives simultaneously. All three are to be **minimised** (negate where necessary).

```python
# ga_optimizer/objectives.py

from dataclasses import dataclass
import numpy as np

@dataclass
class ObjectiveVector:
    """
    Three-objective minimisation problem.
    All values must be finite floats; any NaN or Inf marks the individual as infeasible.
    """
    neg_efficiency: float        # −η  (minimise → maximise efficiency)
    total_loss_W: float          # Total EM + thermal losses [W]
    neg_power_density: float     # −(torque × speed / volume)  [W/m³ negated]

    # Constraint violations (≤0 means feasible)
    temperature_violation_K: float    # max(0, T_peak − T_limit)
    safety_factor_violation: float    # max(0, SF_min − SF_actual)

    @property
    def objective_array(self) -> np.ndarray:
        return np.array([
            self.neg_efficiency,
            self.total_loss_W,
            self.neg_power_density,
        ])

    @property
    def is_feasible(self) -> bool:
        return (
            self.temperature_violation_K <= 0.0 and
            self.safety_factor_violation  <= 0.0 and
            np.all(np.isfinite(self.objective_array))
        )


def extract_objectives(fea_results, stator_params: dict, config: dict) -> ObjectiveVector:
    """
    Maps FEA pipeline results → ObjectiveVector.
    fea_results is the PipelineResults object returned by run_fea_pipeline().
    """
    em  = fea_results.em_results
    th  = fea_results.thermal_results
    st  = fea_results.structural_results

    torque  = em["torque_Nm"]
    speed   = stator_params.get("rated_speed_rpm", 3000.0) * (2 * np.pi / 60)
    OD      = stator_params["outer_diameter"]
    axial   = stator_params["axial_length"]
    volume  = np.pi * (OD / 2) ** 2 * axial   # Bounding cylinder volume [m³]

    power_out = torque * speed
    power_in  = power_out + em["total_loss_W"]
    efficiency = power_out / max(power_in, 1e-9)
    power_density = power_out / max(volume, 1e-9)

    T_limit = config["constraints"]["max_winding_temperature_K"]
    SF_min  = config["constraints"]["min_safety_factor"]

    return ObjectiveVector(
        neg_efficiency        = -efficiency,
        total_loss_W          = float(em["total_loss_W"]),
        neg_power_density     = -power_density,
        temperature_violation_K = max(0.0, th["peak_temperature_K"] - T_limit),
        safety_factor_violation = max(0.0, SF_min - st["safety_factor"]),
    )
```

---

## 6. Constraint Handling

Two types of constraints are applied. **Geometric constraints** are checked before FEA is called (cheap). **Physical constraints** are checked after FEA (expensive, only for feasible geometries).

```python
# ga_optimizer/constraints.py

import numpy as np
from .chromosome import decode_chromosome

class GeometricConstraintViolation(Exception):
    pass

def check_geometric_constraints(genes: np.ndarray, config: dict) -> None:
    """
    Raises GeometricConstraintViolation if genes produce an unphysical stator.
    Called before invoking the stator generation module.
    All checks are O(1) — no mesh generation required.
    """
    params = decode_chromosome(genes)
    OD = params["outer_diameter"]
    ID = params["inner_diameter"]
    air_gap = (OD - ID) / 2.0

    # 1. Air gap must be at least min_air_gap_m
    min_ag = config["constraints"]["min_air_gap_m"]
    if air_gap < min_ag:
        raise GeometricConstraintViolation(
            f"Air gap {air_gap*1e3:.2f} mm < min {min_ag*1e3:.2f} mm"
        )

    # 2. Tooth width must leave a minimum slot width
    slot_pitch = np.pi * ID / params["num_slots"]
    min_slot_w = config["constraints"]["min_slot_width_m"]
    slot_w = slot_pitch - params["tooth_width"]
    if slot_w < min_slot_w:
        raise GeometricConstraintViolation(
            f"Slot width {slot_w*1e3:.2f} mm < min {min_slot_w*1e3:.2f} mm"
        )

    # 3. Yoke must be thick enough for flux containment
    min_yoke = config["constraints"]["min_yoke_height_m"]
    if params["yoke_height"] < min_yoke:
        raise GeometricConstraintViolation(
            f"Yoke height {params['yoke_height']*1e3:.2f} mm < min {min_yoke*1e3:.2f} mm"
        )

    # 4. Conductors per slot must be even (for two-layer winding)
    if params["conductors_per_slot"] % 2 != 0:
        raise GeometricConstraintViolation("conductors_per_slot must be even")
```

Individuals that raise `GeometricConstraintViolation` are assigned a **penalty objective vector** (all objectives set to `+∞`) and are **not** evaluated by FEA. They survive in the population subject to the standard selection pressure — they will be quickly dominated and crowded out but are not discarded, preserving genetic diversity.

```python
INFEASIBLE_OBJECTIVES = ObjectiveVector(
    neg_efficiency=1e9,
    total_loss_W=1e9,
    neg_power_density=1e9,
    temperature_violation_K=1e9,
    safety_factor_violation=1e9,
)
```

---

## 7. Core GA Components

### 7.1 Population Initialisation

```python
# ga_optimizer/population.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .chromosome import random_individual, N_GENES
from .objectives import ObjectiveVector

@dataclass
class Individual:
    genes: np.ndarray                        # shape (N_GENES,)
    objectives: Optional[ObjectiveVector] = None
    rank: int = -1                           # NSGA-II rank (0 = Pareto front)
    crowding_distance: float = 0.0
    stator_id: Optional[str] = None          # set after evaluation
    evaluated: bool = False

Population = list[Individual]


def initialise_population(pop_size: int, rng: np.random.Generator,
                           seed_designs: list[np.ndarray] | None = None) -> Population:
    """
    Creates the initial population.
    If seed_designs are provided (e.g. hand-crafted baseline designs), they
    are inserted first and the remainder is filled with random individuals.
    """
    pop: Population = []

    if seed_designs:
        for genes in seed_designs[:pop_size]:
            pop.append(Individual(genes=genes.copy()))

    rng_count = pop_size - len(pop)
    for _ in range(rng_count):
        pop.append(Individual(genes=random_individual(rng)))

    return pop
```

### 7.2 Fitness Evaluation

```python
# ga_optimizer/evaluation/evaluator.py

import hashlib
import numpy as np
import logging

from stator_generation.api import generate_stator
from stator_generation.schema import StatorParams
from fea_pipeline.orchestrator import run_fea_pipeline
from fea_pipeline.io.schema import StatorMeshInput

from ..chromosome import decode_chromosome
from ..constraints import check_geometric_constraints, GeometricConstraintViolation
from ..objectives import extract_objectives, INFEASIBLE_OBJECTIVES, ObjectiveVector
from .cache import EvaluationCache

log = logging.getLogger(__name__)


def evaluate_individual(
    genes: np.ndarray,
    generation: int,
    individual_index: int,
    config: dict,
    cache: EvaluationCache,
    fea_config_path: str,
    fea_output_dir: str,
) -> ObjectiveVector:
    """
    Full evaluation pipeline for one individual:
      1. Geometric constraint check (fast)
      2. Cache lookup (skip if already evaluated)
      3. Stator generation
      4. FEA analysis
      5. Objective extraction
    """
    # Step 1: geometric feasibility
    try:
        check_geometric_constraints(genes, config)
    except GeometricConstraintViolation as e:
        log.debug(f"Individual {individual_index} infeasible (geometric): {e}")
        return INFEASIBLE_OBJECTIVES

    # Step 2: cache lookup by gene hash
    gene_key = hashlib.sha256(genes.tobytes()).hexdigest()
    cached = cache.get(gene_key)
    if cached is not None:
        log.debug(f"Individual {individual_index} cache hit")
        return cached

    # Step 3: decode → stator params
    try:
        params = decode_chromosome(genes)
    except ValueError as e:
        log.warning(f"Individual {individual_index} decode failed: {e}")
        return INFEASIBLE_OBJECTIVES

    stator_id = f"ga_gen{generation:04d}_ind{individual_index:04d}"

    try:
        # Step 3a: generate stator geometry
        stator_params_obj = StatorParams(**params,
            winding_type="distributed",
            num_layers=2,
            rated_current_rms=config["operating_point"]["current_A"],
            rated_speed_rpm=config["operating_point"]["speed_rpm"],
            rated_torque=config["operating_point"]["torque_Nm"],
            dc_bus_voltage=config["operating_point"]["voltage_V"],
        )
        mesh_result = generate_stator(stator_params_obj)

        # Step 3b: wrap as StatorMeshInput for FEA
        stator_input = StatorMeshInput(
            stator_id=stator_id,
            geometry_source="ga_optimizer",
            mesh_file_path=mesh_result.mesh_file_path,
            mesh_format=mesh_result.mesh_format,
            region_tags=mesh_result.region_tags,
            material_map=config["materials"],
            min_element_quality=mesh_result.min_element_quality,
            max_element_size=mesh_result.max_element_size,
            num_elements=mesh_result.num_elements,
            num_nodes=mesh_result.num_nodes,
            **params,
            winding_type="distributed",
            num_layers=2,
            winding_factor=mesh_result.winding_factor,
            rated_current_rms=config["operating_point"]["current_A"],
            rated_speed_rpm=config["operating_point"]["speed_rpm"],
            rated_torque=config["operating_point"]["torque_Nm"],
            dc_bus_voltage=config["operating_point"]["voltage_V"],
        )

        # Step 4: run FEA
        fea_results = run_fea_pipeline(
            stator_input=stator_input,
            config_path=fea_config_path,
            output_dir=f"{fea_output_dir}/{stator_id}",
        )

        # Step 5: extract objectives
        obj = extract_objectives(fea_results, params, config)

    except Exception as e:
        log.error(f"Individual {individual_index} evaluation failed: {e}", exc_info=True)
        return INFEASIBLE_OBJECTIVES

    # Store in cache
    cache.put(gene_key, obj)
    return obj
```

### 7.3 Non-Dominated Sorting (NSGA-II)

```python
# ga_optimizer/pareto/nsga2.py

import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..population import Individual, Population


def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """
    Returns True iff obj_a Pareto-dominates obj_b.
    All objectives are minimisation. Infeasible individuals (1e9 values) are
    never dominated-over by feasible ones automatically due to value magnitude.
    """
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def fast_non_dominated_sort(population: "Population") -> list[list[int]]:
    """
    NSGA-II fast non-dominated sort.
    Returns fronts: list of lists of individual indices.
    Front 0 is the current Pareto front (best).
    """
    n = len(population)
    domination_count = [0] * n        # how many individuals dominate i
    dominated_set    = [[] for _ in range(n)]   # individuals that i dominates
    fronts           = [[]]

    objs = [
        ind.objectives.objective_array
        for ind in population
    ]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objs[i], objs[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif dominates(objs[j], objs[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)
            population[i].rank = 0

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 1
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance_assignment(population: "Population", front: list[int]) -> None:
    """
    Assigns crowding distance to all individuals in a front.
    Boundary individuals receive distance = +inf.
    """
    n_obj = len(population[0].objectives.objective_array)
    n     = len(front)
    if n == 0:
        return

    for idx in front:
        population[idx].crowding_distance = 0.0

    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: population[i].objectives.objective_array[m])
        # Boundary individuals
        population[sorted_front[0]].crowding_distance  = float("inf")
        population[sorted_front[-1]].crowding_distance = float("inf")

        obj_min = population[sorted_front[0]].objectives.objective_array[m]
        obj_max = population[sorted_front[-1]].objectives.objective_array[m]
        obj_range = obj_max - obj_min if obj_max > obj_min else 1e-10

        for k in range(1, n - 1):
            left  = population[sorted_front[k - 1]].objectives.objective_array[m]
            right = population[sorted_front[k + 1]].objectives.objective_array[m]
            population[sorted_front[k]].crowding_distance += (right - left) / obj_range
```

### 7.4 Tournament Selection

```python
# ga_optimizer/operators/selection.py

import random
from ..population import Individual, Population


def crowded_tournament(population: Population, rng) -> Individual:
    """
    Binary tournament selection using NSGA-II crowded comparison operator.
    Individual a is preferred over b if:
      - a has a lower rank, OR
      - they share the same rank and a has a greater crowding distance.
    """
    a, b = rng.choice(len(population), size=2, replace=False)
    ind_a, ind_b = population[a], population[b]

    if ind_a.rank < ind_b.rank:
        return ind_a
    if ind_b.rank < ind_a.rank:
        return ind_b
    if ind_a.crowding_distance > ind_b.crowding_distance:
        return ind_a
    return ind_b
```

### 7.5 Simulated Binary Crossover (SBX)

```python
# ga_optimizer/operators/crossover.py

import numpy as np
from ..chromosome import LOWER_BOUNDS, UPPER_BOUNDS, N_GENES
from .repair import clamp


def sbx_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    eta_c: float,
    rng: np.random.Generator,
    p_crossover: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) for real-valued chromosomes.
    eta_c: distribution index (typically 2–20; higher = offspring closer to parents).
    Returns two offspring after clamping to bounds.
    """
    if rng.random() > p_crossover:
        return parent_a.copy(), parent_b.copy()

    child_a = parent_a.copy()
    child_b = parent_b.copy()

    for i in range(N_GENES):
        if rng.random() <= 0.5:
            continue   # skip this gene with 50% probability


        x1, x2 = min(parent_a[i], parent_b[i]), max(parent_a[i], parent_b[i])
        if abs(x2 - x1) < 1e-14:
            continue

        lb, ub = LOWER_BOUNDS[i], UPPER_BOUNDS[i]
        u = rng.random()
        beta_q = _beta_q(u, x1, x2, lb, ub, eta_c)

        child_a[i] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))
        child_b[i] = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

    return clamp(child_a), clamp(child_b)


def _beta_q(u, x1, x2, lb, ub, eta_c):
    beta = 1.0 + (2.0 * (x1 - lb) / (x2 - x1))
    alpha = 2.0 - beta ** (-(eta_c + 1.0))
    if u <= 1.0 / alpha:
        return (u * alpha) ** (1.0 / (eta_c + 1.0))
    else:
        return (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
```

### 7.6 Polynomial Mutation

```python
# ga_optimizer/operators/mutation.py

import numpy as np
from ..chromosome import LOWER_BOUNDS, UPPER_BOUNDS, N_GENES
from .repair import clamp


def polynomial_mutation(
    genes: np.ndarray,
    eta_m: float,
    rng: np.random.Generator,
    p_mutation: float | None = None,
) -> np.ndarray:
    """
    Polynomial mutation. Default p_mutation = 1/N_GENES (standard NSGA-II).
    eta_m: distribution index (typically 20; higher = smaller perturbations).
    """
    if p_mutation is None:
        p_mutation = 1.0 / N_GENES

    mutant = genes.copy()
    for i in range(N_GENES):
        if rng.random() > p_mutation:
            continue

        lb, ub = LOWER_BOUNDS[i], UPPER_BOUNDS[i]
        x = genes[i]
        delta1 = (x - lb) / (ub - lb)
        delta2 = (ub - x) / (ub - lb)
        u = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)

        if u < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
            delta_q = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
            delta_q = 1.0 - val ** mut_pow

        mutant[i] = x + delta_q * (ub - lb)

    return clamp(mutant)
```

### 7.7 Pareto Archive

The Pareto archive stores every non-dominated individual **ever evaluated**, not just the current generation's front. It is updated every generation and is the source of the final output.

```python
# ga_optimizer/pareto/archive.py

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from ..population import Individual
from .nsga2 import dominates


class ParetoArchive:
    """
    Maintains a running archive of all non-dominated individuals seen so far.
    Individuals are added if they are non-dominated with respect to all
    current archive members. Dominated archive members are pruned on each update.
    """

    def __init__(self):
        self._members: list[Individual] = []

    def update(self, candidates: list[Individual]) -> int:
        """
        Adds any non-dominated candidates to the archive, pruning newly dominated
        archive members. Returns count of new entries added.
        """
        added = 0
        for candidate in candidates:
            if candidate.objectives is None or not candidate.objectives.is_feasible:
                continue
            if self._is_non_dominated(candidate):
                self._members = [
                    m for m in self._members
                    if not dominates(
                        candidate.objectives.objective_array,
                        m.objectives.objective_array
                    )
                ]
                self._members.append(candidate)
                added += 1
        return added

    def _is_non_dominated(self, candidate: Individual) -> bool:
        c_obj = candidate.objectives.objective_array
        return not any(
            dominates(m.objectives.objective_array, c_obj)
            for m in self._members
        )

    @property
    def members(self) -> list[Individual]:
        return list(self._members)

    @property
    def size(self) -> int:
        return len(self._members)
```

---

## 8. Termination Conditions

The GA terminates when **any one** of the following is true. All conditions are evaluated at the end of each generation.

```python
# ga_optimizer/orchestrator.py  (termination section)

def check_termination(state: "GAState", config: dict) -> tuple[bool, str]:
    """
    Returns (should_terminate, reason_string).
    Evaluated once per generation after archive update.
    """
    tc = config["termination"]

    # Condition 1: Maximum generations reached
    if state.generation >= tc["max_generations"]:
        return True, f"max_generations ({tc['max_generations']}) reached"

    # Condition 2: Maximum FEA evaluations reached
    if state.total_evaluations >= tc["max_evaluations"]:
        return True, f"max_evaluations ({tc['max_evaluations']}) reached"

    # Condition 3: Pareto front has not changed for N generations
    if state.generation >= tc["stagnation_window"]:
        recent_sizes = state.archive_size_history[-tc["stagnation_window"]:]
        if len(set(recent_sizes)) == 1:
            # Also check hypervolume stagnation
            recent_hv = state.hypervolume_history[-tc["stagnation_window"]:]
            hv_change = abs(recent_hv[-1] - recent_hv[0]) / max(abs(recent_hv[0]), 1e-12)
            if hv_change < tc["hypervolume_stagnation_tolerance"]:
                return True, (
                    f"Hypervolume stagnated for {tc['stagnation_window']} generations "
                    f"(change={hv_change:.2e} < tol={tc['hypervolume_stagnation_tolerance']})"
                )

    # Condition 4: Target hypervolume indicator reached
    if tc.get("target_hypervolume") and state.hypervolume_history:
        if state.hypervolume_history[-1] >= tc["target_hypervolume"]:
            return True, f"Target hypervolume {tc['target_hypervolume']:.4f} reached"

    return False, ""
```

---

## 9. GA Orchestrator

```python
# ga_optimizer/orchestrator.py

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import joblib

from .population import initialise_population, Individual, Population
from .evaluation.evaluator import evaluate_individual
from .evaluation.cache import EvaluationCache
from .pareto.nsga2 import fast_non_dominated_sort, crowding_distance_assignment
from .pareto.archive import ParetoArchive
from .operators.selection import crowded_tournament
from .operators.crossover import sbx_crossover
from .operators.mutation import polynomial_mutation
from .io.checkpoint import save_checkpoint, load_checkpoint
from .io.result_writer import write_pareto_results
from .utils.metrics import compute_hypervolume
from .utils.logger import setup_logger

log = logging.getLogger(__name__)


@dataclass
class GAState:
    generation: int = 0
    total_evaluations: int = 0
    archive_size_history: list[int] = field(default_factory=list)
    hypervolume_history:  list[float] = field(default_factory=list)
    population: Population = field(default_factory=list)
    archive: ParetoArchive = field(default_factory=ParetoArchive)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )


def run_ga(
    config_path: str = "ga_optimizer/configs/default_ga.yaml",
    fea_config_path: str = "configs/default.yaml",
    output_dir: str = "ga_results/",
    resume_from_checkpoint: str | None = None,
) -> ParetoArchive:
    """
    Main GA entry point. Returns the final Pareto archive.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    setup_logger(config.get("log_level", "INFO"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fea_output_dir = str(Path(output_dir) / "fea_evaluations")

    # Resume or initialise state
    if resume_from_checkpoint:
        state = load_checkpoint(resume_from_checkpoint)
        log.info(f"Resumed from checkpoint at generation {state.generation}")
    else:
        state = GAState(rng=np.random.default_rng(config["random_seed"]))
        state.population = initialise_population(
            pop_size=config["population_size"],
            rng=state.rng,
            seed_designs=config.get("seed_designs"),
        )

    cache = EvaluationCache()
    pop_size = config["population_size"]

    while True:
        t_gen_start = time.perf_counter()
        log.info(f"=== Generation {state.generation} ===")

        # ── Evaluate population ──────────────────────────────────────────────
        unevaluated = [ind for ind in state.population if not ind.evaluated]
        if unevaluated:
            results = joblib.Parallel(n_jobs=config["parallel_workers"])(
                joblib.delayed(evaluate_individual)(
                    genes=ind.genes,
                    generation=state.generation,
                    individual_index=i,
                    config=config,
                    cache=cache,
                    fea_config_path=fea_config_path,
                    fea_output_dir=fea_output_dir,
                )
                for i, ind in enumerate(unevaluated)
            )
            for ind, obj in zip(unevaluated, results):
                ind.objectives = obj
                ind.evaluated = True
                if obj.is_feasible:
                    state.total_evaluations += 1

        # ── NSGA-II sort ─────────────────────────────────────────────────────
        fronts = fast_non_dominated_sort(state.population)
        for front in fronts:
            crowding_distance_assignment(state.population, front)

        # ── Archive update ────────────────────────────────────────────────────
        front_0_individuals = [state.population[i] for i in fronts[0]]
        new_entries = state.archive.update(front_0_individuals)
        log.info(f"Archive size: {state.archive.size} (+{new_entries} new)")

        # ── Metrics ───────────────────────────────────────────────────────────
        if state.archive.size > 0:
            hv = compute_hypervolume(state.archive.members, config["hypervolume_reference_point"])
        else:
            hv = 0.0
        state.hypervolume_history.append(hv)
        state.archive_size_history.append(state.archive.size)

        log.info(f"Hypervolume: {hv:.6f}  |  Time: {time.perf_counter()-t_gen_start:.1f}s")

        # ── Termination check ─────────────────────────────────────────────────
        terminate, reason = check_termination(state, config)
        if terminate:
            log.info(f"Termination condition met: {reason}")
            break

        # ── Generate offspring ────────────────────────────────────────────────
        offspring: Population = []
        while len(offspring) < pop_size:
            p1 = crowded_tournament(state.population, state.rng)
            p2 = crowded_tournament(state.population, state.rng)
            c1_genes, c2_genes = sbx_crossover(
                p1.genes, p2.genes,
                eta_c=config["operators"]["eta_c"],
                rng=state.rng,
            )
            c1_genes = polynomial_mutation(c1_genes, config["operators"]["eta_m"], state.rng)
            c2_genes = polynomial_mutation(c2_genes, config["operators"]["eta_m"], state.rng)
            offspring.append(Individual(genes=c1_genes))
            offspring.append(Individual(genes=c2_genes))

        # ── Survivor selection (elitist) ──────────────────────────────────────
        combined = state.population + offspring[:pop_size]
        # Evaluate offspring (un-evaluated)
        unevaluated_offspring = [ind for ind in offspring[:pop_size] if not ind.evaluated]
        off_results = joblib.Parallel(n_jobs=config["parallel_workers"])(
            joblib.delayed(evaluate_individual)(
                genes=ind.genes,
                generation=state.generation,
                individual_index=i,
                config=config,
                cache=cache,
                fea_config_path=fea_config_path,
                fea_output_dir=fea_output_dir,
            )
            for i, ind in enumerate(unevaluated_offspring)
        )
        for ind, obj in zip(unevaluated_offspring, off_results):
            ind.objectives = obj
            ind.evaluated = True
            if obj.is_feasible:
                state.total_evaluations += 1

        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            crowding_distance_assignment(combined, front)

        # Fill next generation front-by-front
        next_pop: Population = []
        for front in combined_fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend([combined[i] for i in front])
            else:
                # Fill remainder sorted by crowding distance (descending)
                remaining = sorted(
                    front,
                    key=lambda i: combined[i].crowding_distance,
                    reverse=True
                )
                slots = pop_size - len(next_pop)
                next_pop.extend([combined[i] for i in remaining[:slots]])
                break

        state.population = next_pop
        state.generation += 1

        # ── Checkpoint ────────────────────────────────────────────────────────
        if state.generation % config["checkpoint_every_n_generations"] == 0:
            ckpt_path = str(Path(output_dir) / f"checkpoint_gen{state.generation:04d}.h5")
            save_checkpoint(state, ckpt_path)
            log.info(f"Checkpoint saved: {ckpt_path}")

    # ── Final output ──────────────────────────────────────────────────────────
    write_pareto_results(state.archive, output_dir, config)
    log.info(f"Optimization complete. Final Pareto front: {state.archive.size} solutions.")
    return state.archive
```

---

## 10. Parallelisation Strategy

The GA uses `joblib.Parallel` for population evaluation. Each FEA call is fully independent — no shared state between evaluations.

| Workers | Strategy                 | Notes                                                                 |
| ------- | ------------------------ | --------------------------------------------------------------------- |
| 1       | Serial                   | For debugging and CI                                                  |
| 2–4     | `joblib` multiprocessing | Each worker loads FEA modules independently                           |
| 4–16    | MPI via `mpi4py`         | Recommended for HPC; each MPI rank handles a subset of the population |
| 16+     | Ray cluster              | Advanced: set `parallel_backend: "ray"` in config                     |

For MPI-based execution, the agent must implement `ga_optimizer/parallel/mpi_evaluator.py` using a manager/worker pattern: rank 0 runs the GA loop and dispatches genes; ranks 1..N evaluate FEA and return `ObjectiveVector` results.

The default `joblib` backend is sufficient for single-node execution. The `parallel_workers` config key controls the number of processes.

---

## 11. Persistence and Checkpointing

```python
# ga_optimizer/io/checkpoint.py

import h5py
import pickle
import numpy as np
from pathlib import Path
from ..orchestrator import GAState


def save_checkpoint(state: GAState, path: str) -> None:
    """
    Saves full GA state to HDF5. Includes:
    - All gene vectors and objective arrays
    - NSGA-II rank and crowding distance
    - Archive members
    - Hypervolume and archive size history
    - RNG state (for reproducibility of resumed runs)
    """
    with h5py.File(path, "w") as f:
        f.attrs["generation"]        = state.generation
        f.attrs["total_evaluations"] = state.total_evaluations

        # Population
        pop_grp = f.create_group("population")
        genes_arr = np.array([ind.genes for ind in state.population])
        pop_grp.create_dataset("genes", data=genes_arr)
        # Objectives
        obj_arr = np.array([
            ind.objectives.objective_array if ind.evaluated else np.full(3, np.nan)
            for ind in state.population
        ])
        pop_grp.create_dataset("objectives", data=obj_arr)
        pop_grp.create_dataset("ranks",
            data=np.array([ind.rank for ind in state.population]))
        pop_grp.create_dataset("crowding",
            data=np.array([ind.crowding_distance for ind in state.population]))
        pop_grp.create_dataset("evaluated",
            data=np.array([ind.evaluated for ind in state.population], dtype=bool))

        # Archive
        if state.archive.size > 0:
            arc_grp = f.create_group("archive")
            arc_genes = np.array([m.genes for m in state.archive.members])
            arc_objs  = np.array([m.objectives.objective_array for m in state.archive.members])
            arc_grp.create_dataset("genes",      data=arc_genes)
            arc_grp.create_dataset("objectives", data=arc_objs)

        # Histories
        f.create_dataset("hypervolume_history",
            data=np.array(state.hypervolume_history))
        f.create_dataset("archive_size_history",
            data=np.array(state.archive_size_history))

        # RNG state (pickle serialised into bytes dataset)
        rng_bytes = pickle.dumps(state.rng)
        f.create_dataset("rng_state",
            data=np.frombuffer(rng_bytes, dtype=np.uint8))
```

---

## 12. Output Schema

### 12.1 File structure

```
ga_results/
├── pareto_front.json              # All Pareto-optimal designs, objectives, genes
├── pareto_front.csv               # Same, tabular
├── hypervolume_history.json       # Per-generation hypervolume indicator
├── run_metadata.json              # Config snapshot, git hash, timestamps
├── checkpoint_gen0010.h5          # Periodic checkpoint (resumable)
├── checkpoint_gen0020.h5
│   ...
├── fea_evaluations/
│   ├── ga_gen0000_ind0000/        # One directory per evaluated individual
│   │   ├── electromagnetic/
│   │   ├── thermal/
│   │   └── structural/
│   └── ...
└── plots/
    ├── pareto_front_2d_eff_loss.html
    ├── pareto_front_2d_eff_pd.html
    └── pareto_front_3d.html
```

### 12.2 `pareto_front.json` format

```json
{
  "run_id": "ga_run_2025_04_01_143200",
  "final_generation": 42,
  "total_fea_evaluations": 3840,
  "solutions": [
    {
      "rank": 0,
      "stator_id": "ga_gen0038_ind0014",
      "genes": [0.24, 0.618, 0.11, 36, 8, 0.48, 0.32, 0.51, 20, 0.5, 0.22, 1.1],
      "decoded_params": {
        "outer_diameter": 0.24,
        "inner_diameter": 0.148,
        "num_slots": 36,
        "num_poles": 8
      },
      "objectives": {
        "efficiency": 0.9412,
        "total_loss_W": 894.2,
        "power_density_W_m3": 1.84e6
      },
      "constraints": {
        "peak_temperature_K": 392.1,
        "safety_factor": 2.34,
        "feasible": true
      },
      "fea_dir": "fea_evaluations/ga_gen0038_ind0014"
    }
  ]
}
```

---

## 13. Configuration Reference

### `ga_optimizer/configs/default_ga.yaml`

```yaml
random_seed: 42
log_level: "INFO"

# Population and generation budget
population_size: 100
parallel_workers: 8 # joblib workers for parallel FEA evaluation

# NSGA-II operator parameters
operators:
  eta_c: 15.0 # SBX distribution index (higher = tighter offspring)
  eta_m: 20.0 # Polynomial mutation distribution index
  p_crossover: 1.0 # Probability of applying SBX
  # p_mutation defaults to 1/N_GENES per gene

# Termination
termination:
  max_generations: 100
  max_evaluations: 8000
  stagnation_window: 15 # Generations without HV improvement → terminate
  hypervolume_stagnation_tolerance: 1.0e-4
  target_hypervolume: null # Set to a float to terminate on HV target

# Hypervolume reference point (must dominate all feasible solutions)
# Set each to the worst acceptable objective value
hypervolume_reference_point: [0.0, 5000.0, 0.0]
# [neg_efficiency=0 means 0% eff, total_loss=5000W, neg_power_density=0]

# Constraint bounds
constraints:
  min_air_gap_m: 0.0008 # 0.8 mm
  min_slot_width_m: 0.003 # 3 mm
  min_yoke_height_m: 0.008 # 8 mm
  max_winding_temperature_K: 428.15 # 155°C class F
  min_safety_factor: 1.5

# Operating point (same for all individuals in this run)
operating_point:
  current_A: 80.0
  speed_rpm: 3000.0
  torque_Nm: 120.0
  voltage_V: 400.0

# Material assignments (passed to FEA StatorMeshInput)
materials:
  stator_core: "M250-35A"
  winding: "copper_class_F"
  air_gap: "air"

# Checkpoint
checkpoint_every_n_generations: 10

# Optional: seed designs (list of gene vectors)
seed_designs: null
```

### `ga_optimizer/configs/fast_ga.yaml` (CI / quick test)

```yaml
random_seed: 0
population_size: 10
parallel_workers: 2
operators:
  eta_c: 15.0
  eta_m: 20.0
  p_crossover: 1.0
termination:
  max_generations: 3
  max_evaluations: 40
  stagnation_window: 3
  hypervolume_stagnation_tolerance: 1.0e-4
hypervolume_reference_point: [0.0, 5000.0, 0.0]
constraints:
  min_air_gap_m: 0.0008
  min_slot_width_m: 0.003
  min_yoke_height_m: 0.008
  max_winding_temperature_K: 428.15
  min_safety_factor: 1.5
operating_point:
  current_A: 50.0
  speed_rpm: 3000.0
  torque_Nm: 50.0
  voltage_V: 400.0
materials:
  stator_core: "M250-35A"
  winding: "copper_class_F"
  air_gap: "air"
checkpoint_every_n_generations: 2
seed_designs: null
```

---

## 14. Build Order — Step-by-Step Agent Instructions

The agent must follow these steps in sequence. Each step has a verification gate. **Do not advance to the next step if the gate fails.**

```
STEP 1  — Verify existing interfaces
  Action:  Run scripts/verify_stator_interface.py and scripts/verify_fea_interface.py
  Gate:    Both print "OK" without exceptions

STEP 2  — Scaffold directory structure
  Action:  Create all directories and __init__.py files listed in Section 2
  Gate:    `find ga_optimizer -name "*.py" | wc -l` returns ≥ 20

STEP 3  — Implement chromosome.py
  Action:  Write GENE_DEFINITIONS, LOWER_BOUNDS, UPPER_BOUNDS, decode_chromosome(), random_individual()
  Gate:    pytest tests/unit/test_chromosome.py — all pass

STEP 4  — Implement constraints.py
  Action:  Write check_geometric_constraints()
  Gate:    pytest tests/unit/test_chromosome.py::test_geometric_constraints — all pass

STEP 5  — Implement objectives.py
  Action:  Write ObjectiveVector, extract_objectives(), INFEASIBLE_OBJECTIVES
  Gate:    pytest tests/unit/test_objectives.py — all pass

STEP 6  — Implement operators/ (all four files)
  Action:  Write repair.py, selection.py, crossover.py, mutation.py
  Gate:    pytest tests/unit/test_operators.py — all pass

STEP 7  — Implement pareto/nsga2.py
  Action:  Write dominates(), fast_non_dominated_sort(), crowding_distance_assignment()
  Gate:    pytest tests/unit/test_nsga2.py — all pass
           Specifically: test_nsga2_known_2d_front must verify ranks match pymoo reference

STEP 8  — Implement pareto/archive.py
  Action:  Write ParetoArchive class
  Gate:    pytest tests/unit/test_pareto_archive.py — all pass

STEP 9  — Implement evaluation/cache.py
  Action:  Write EvaluationCache (dict-backed, thread-safe for joblib)
  Gate:    pytest tests/unit/test_cache.py — all pass

STEP 10 — Implement evaluation/evaluator.py
  Action:  Write evaluate_individual() — full stator_generation + FEA call chain
  Gate:    pytest tests/integration/test_evaluator.py::test_evaluator_with_mock_fea — passes
           (Uses a monkeypatched FEA that returns mock_fea_results.json — no real FEA needed)

STEP 11 — Implement io/checkpoint.py and io/result_writer.py
  Action:  Write save_checkpoint(), load_checkpoint(), write_pareto_results()
  Gate:    pytest tests/unit/test_checkpoint.py — all pass

STEP 12 — Implement utils/metrics.py
  Action:  Write compute_hypervolume() using pygmo or manual WFG algorithm
  Gate:    pytest tests/unit/test_metrics.py::test_hypervolume_known_value — passes

STEP 13 — Implement GA orchestrator (orchestrator.py)
  Action:  Write GAState, run_ga(), check_termination()
  Gate:    pytest tests/integration/test_full_ga_fast.py — completes 3 generations,
           produces pareto_front.json, no exceptions raised

STEP 14 — End-to-end integration test with real FEA (if compute allows)
  Action:  Run run_ga(config_path="ga_optimizer/configs/fast_ga.yaml") on the test fixture stator
  Gate:    Final Pareto archive has ≥ 1 feasible solution
           pareto_front.json is written and validates against schema
           No NaN or Inf in any objective value

STEP 15 — Validation against pymoo NSGA-II reference (Section 16)
  Action:  Run tests/validation/test_nsga2_vs_pymoo.py on ZDT1 benchmark
  Gate:    IGD (Inverted Generational Distance) < 0.01 after 50 generations
```

---

## 15. Testing Strategy

### 15.1 Unit tests

```bash
pytest ga_optimizer/tests/unit/ -v --cov=ga_optimizer --cov-report=term-missing
```

#### Key unit test: NSGA-II rank correctness

```python
# ga_optimizer/tests/unit/test_nsga2.py

import numpy as np
import pytest
from ga_optimizer.pareto.nsga2 import dominates, fast_non_dominated_sort, crowding_distance_assignment
from ga_optimizer.population import Individual
from ga_optimizer.objectives import ObjectiveVector


def _make_ind(obj: list[float]) -> Individual:
    ov = ObjectiveVector(
        neg_efficiency=obj[0],
        total_loss_W=obj[1],
        neg_power_density=obj[2],
        temperature_violation_K=0.0,
        safety_factor_violation=0.0,
    )
    return Individual(genes=np.zeros(12), objectives=ov, evaluated=True)


def test_dominates_basic():
    assert dominates(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
    assert not dominates(np.array([1.0, 2.0, 1.0]), np.array([2.0, 1.0, 2.0]))
    assert not dominates(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))

def test_non_dominated_sort_two_fronts():
    """
    Three individuals: A dominates C; B dominates C; A and B are non-dominated.
    Expected: front 0 = [A, B], front 1 = [C]
    """
    A = _make_ind([1.0, 2.0, 1.0])
    B = _make_ind([2.0, 1.0, 1.0])
    C = _make_ind([3.0, 3.0, 3.0])
    pop = [A, B, C]
    fronts = fast_non_dominated_sort(pop)
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]

def test_crowding_distance_boundary_is_inf():
    """Boundary members of a front must receive infinite crowding distance."""
    pop = [_make_ind([float(i), float(10 - i), 0.0]) for i in range(5)]
    front = list(range(5))
    for ind in pop:
        ind.rank = 0
    fast_non_dominated_sort(pop)
    crowding_distance_assignment(pop, front)
    assert pop[0].crowding_distance == float("inf") or pop[4].crowding_distance == float("inf")
```

#### Key unit test: SBX offspring bounds

```python
# ga_optimizer/tests/unit/test_operators.py

import numpy as np
import pytest
from ga_optimizer.chromosome import LOWER_BOUNDS, UPPER_BOUNDS, random_individual
from ga_optimizer.operators.crossover import sbx_crossover
from ga_optimizer.operators.mutation import polynomial_mutation

def test_sbx_offspring_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        p1 = random_individual(rng)
        p2 = random_individual(rng)
        c1, c2 = sbx_crossover(p1, p2, eta_c=15.0, rng=rng)
        assert np.all(c1 >= LOWER_BOUNDS) and np.all(c1 <= UPPER_BOUNDS)
        assert np.all(c2 >= LOWER_BOUNDS) and np.all(c2 <= UPPER_BOUNDS)

def test_polynomial_mutation_within_bounds():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        original = random_individual(rng)
        mutant = polynomial_mutation(original, eta_m=20.0, rng=rng)
        assert np.all(mutant >= LOWER_BOUNDS) and np.all(mutant <= UPPER_BOUNDS)

def test_mutation_changes_genes():
    """Setting p_mutation=1.0 must alter at least one gene."""
    rng = np.random.default_rng(42)
    genes = random_individual(rng)
    mutant = polynomial_mutation(genes, eta_m=20.0, rng=rng, p_mutation=1.0)
    assert not np.allclose(genes, mutant)
```

### 15.2 Integration tests

```bash
pytest ga_optimizer/tests/integration/ -v
```

#### Full pipeline test with mocked FEA

```python
# ga_optimizer/tests/integration/test_full_ga_fast.py

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ga_optimizer.orchestrator import run_ga
from ga_optimizer.objectives import ObjectiveVector


def mock_evaluate_individual(genes, generation, individual_index, config, cache,
                              fea_config_path, fea_output_dir):
    """
    Returns a synthetic ObjectiveVector based on gene values.
    Simulates a real Pareto trade-off: higher efficiency = higher losses.
    """
    import numpy as np
    from ga_optimizer.chromosome import decode_chromosome, LOWER_BOUNDS, UPPER_BOUNDS

    norm = (genes - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS + 1e-9)
    eff = 0.85 + 0.10 * norm[0]
    loss = 2000.0 - 1000.0 * norm[1]
    pd = 1.0e6 + 5.0e5 * norm[2]
    return ObjectiveVector(
        neg_efficiency=-eff,
        total_loss_W=loss,
        neg_power_density=-pd,
        temperature_violation_K=0.0,
        safety_factor_violation=0.0,
    )


@patch("ga_optimizer.evaluation.evaluator.evaluate_individual", side_effect=mock_evaluate_individual)
def test_fast_ga_runs_3_generations(mock_eval, tmp_path):
    archive = run_ga(
        config_path="ga_optimizer/configs/fast_ga.yaml",
        fea_config_path="configs/default.yaml",
        output_dir=str(tmp_path),
    )
    assert archive.size >= 1, "Archive must have at least one solution"
    pf_path = tmp_path / "pareto_front.json"
    assert pf_path.exists(), "pareto_front.json must be written"
    with open(pf_path) as f:
        pf = json.load(f)
    assert len(pf["solutions"]) >= 1

@patch("ga_optimizer.evaluation.evaluator.evaluate_individual", side_effect=mock_evaluate_individual)
def test_checkpoint_resume(mock_eval, tmp_path):
    """Run 2 generations, checkpoint, resume for 1 more, verify final state."""
    import yaml
    cfg = yaml.safe_load(open("ga_optimizer/configs/fast_ga.yaml"))
    cfg["termination"]["max_generations"] = 2
    cfg_path = tmp_path / "cfg_2gen.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    archive_phase1 = run_ga(
        config_path=str(cfg_path),
        output_dir=str(tmp_path / "phase1"),
    )
    ckpt = list((tmp_path / "phase1").glob("checkpoint_gen*.h5"))
    assert len(ckpt) >= 1

    cfg["termination"]["max_generations"] = 3
    cfg_path_resume = tmp_path / "cfg_3gen.yaml"
    with open(cfg_path_resume, "w") as f:
        yaml.dump(cfg, f)
    archive_phase2 = run_ga(
        config_path=str(cfg_path_resume),
        output_dir=str(tmp_path / "phase2"),
        resume_from_checkpoint=str(sorted(ckpt)[-1]),
    )
    assert archive_phase2.size >= archive_phase1.size
```

---

## 16. Validation Benchmarks

### 16.1 NSGA-II on ZDT1 (standard benchmark)

This validates the GA loop independently of the FEA pipeline.

```python
# ga_optimizer/tests/validation/test_nsga2_vs_pymoo.py
"""
Runs the GA operator stack (SBX + polynomial mutation + NSGA-II sort) on the
ZDT1 benchmark (2 objectives, 30 variables) and compares the final Pareto front
to the pymoo reference using the IGD metric.
Expected: IGD < 0.01 after 50 generations with population_size=100.
"""
import numpy as np
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD

def test_ga_operators_on_zdt1():
    # ... implementation uses GA components directly on ZDT1 without FEA
    # IGD computed against pymoo's analytically sampled reference Pareto front
    assert igd_value < 0.01, f"IGD {igd_value:.4f} ≥ 0.01; GA operators may be incorrect"
```

### 16.2 Hypervolume monotonicity test

```python
def test_hypervolume_non_decreasing():
    """
    Hypervolume must be non-decreasing across generations (due to elitist archive).
    A single decrease indicates a bug in archive update or sorting.
    """
    hv_hist = state.hypervolume_history
    for i in range(1, len(hv_hist)):
        assert hv_hist[i] >= hv_hist[i-1] - 1e-10, \
            f"Hypervolume decreased at generation {i}: {hv_hist[i-1]:.6f} → {hv_hist[i]:.6f}"
```

---

## 17. Integration Checklist

### Upstream readiness (stator generation module)

- [ ] `generate_stator(StatorParams)` callable and returns object with `mesh_file_path`, `region_tags`, `winding_factor`, `num_elements`, `num_nodes`, `min_element_quality`, `mesh_format`
- [ ] `StatorParams` accepts all 11 geometric fields listed in Section 4.1 decoded output
- [ ] Mesh generation is deterministic for the same `StatorParams`
- [ ] Mesh generation fails fast (raises within 5 s) for geometrically invalid parameters
- [ ] verify_stator_interface.py prints "OK"

### Downstream readiness (FEA pipeline)

- [ ] `run_fea_pipeline(StatorMeshInput)` callable and returns `PipelineResults`
- [ ] `PipelineResults.em_results["torque_Nm"]` is a positive float
- [ ] `PipelineResults.thermal_results["peak_temperature_K"]` > ambient
- [ ] `PipelineResults.structural_results["safety_factor"]` > 0
- [ ] verify_fea_interface.py prints "OK"

### GA pipeline readiness

- [ ] All unit tests pass: `pytest ga_optimizer/tests/unit/ -v`
- [ ] Integration test with mocked FEA passes: `pytest ga_optimizer/tests/integration/test_full_ga_fast.py`
- [ ] ZDT1 validation IGD < 0.01: `pytest ga_optimizer/tests/validation/test_nsga2_vs_pymoo.py`
- [ ] Hypervolume monotonicity test passes
- [ ] `pareto_front.json` is written and conforms to schema in Section 12.2
- [ ] Checkpoint file is readable and `run_ga(..., resume_from_checkpoint=...)` completes

---

## 18. Error Handling Reference

| Error                                    | Source               | Recovery                                                                          |
| ---------------------------------------- | -------------------- | --------------------------------------------------------------------------------- |
| `GeometricConstraintViolation`           | `constraints.py`     | Assign `INFEASIBLE_OBJECTIVES`; continue                                          |
| `ValueError` from `decode_chromosome`    | `chromosome.py`      | Assign `INFEASIBLE_OBJECTIVES`; log warning                                       |
| `MeshGenerationError` from stator module | `generate_stator()`  | Assign `INFEASIBLE_OBJECTIVES`; log error                                         |
| `FEAConvergenceError` from FEA pipeline  | `run_fea_pipeline()` | Assign `INFEASIBLE_OBJECTIVES`; log error; increment `failed_evaluations` counter |
| `MemoryError` during parallel eval       | joblib worker        | Reduce `parallel_workers`; restart from last checkpoint                           |
| `KeyboardInterrupt`                      | User                 | Save emergency checkpoint to `output_dir/emergency_checkpoint.h5` before exit     |
| Pareto archive empty after N generations | Archive logic        | Log warning; inspect constraint bounds — may be too tight                         |
| Hypervolume = 0 for all generations      | Metrics              | Check `hypervolume_reference_point` — must dominate all feasible objectives       |

### Handling `KeyboardInterrupt` gracefully

```python
# In orchestrator.py main loop
try:
    while True:
        # ... GA loop body ...
except KeyboardInterrupt:
    emergency_path = str(Path(output_dir) / "emergency_checkpoint.h5")
    save_checkpoint(state, emergency_path)
    log.warning(f"Interrupted at generation {state.generation}. "
                f"Emergency checkpoint saved to {emergency_path}")
    write_pareto_results(state.archive, output_dir, config)
    raise
```

---

_End of GA_PIPELINE.md — Agent Build and Test Instructions. Version 0.1.0 — April 2026._
