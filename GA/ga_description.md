# GA Optimizer — API Reference

Complete API documentation for every module, class, and function in the
`ga_optimizer` package, with the corresponding test cases for each.

---

## Table of Contents

1. [Package Entry Point](#1-package-entry-point)
2. [Orchestrator](#2-orchestrator)
3. [Chromosome](#3-chromosome)
4. [Population](#4-population)
5. [Objectives](#5-objectives)
6. [Constraints](#6-constraints)
7. [Operators](#7-operators)
   - [Repair](#71-repair)
   - [Crossover](#72-crossover)
   - [Mutation](#73-mutation)
   - [Selection](#74-selection)
8. [Pareto](#8-pareto)
   - [NSGA-II Core](#81-nsga-ii-core)
   - [Archive](#82-pareto-archive)
9. [Evaluation](#9-evaluation)
   - [Cache](#91-evaluation-cache)
   - [Evaluator](#92-evaluator)
10. [I/O](#10-io)
    - [Checkpointing](#101-checkpointing)
    - [Result Writer](#102-result-writer)
11. [Utilities](#11-utilities)
    - [Metrics](#111-metrics)
    - [Logger](#112-logger)
12. [Test Suite Reference](#12-test-suite-reference)

---

## 1. Package Entry Point

**File:** `ga_optimizer/__init__.py`

Exposes the full public API so that consumers only need to import from `ga_optimizer`.

```python
from ga_optimizer import (
    run_ga, GAState, check_termination,
    ParetoArchive,
    decode_chromosome, random_individual, GENE_DEFINITIONS, N_GENES,
    ObjectiveVector, extract_objectives, INFEASIBLE_OBJECTIVES,
    Individual, Population, initialise_population,
)
```

All symbols are documented in the sections below where they are defined.

---

## 2. Orchestrator

**File:** `ga_optimizer/orchestrator.py`

The top-level entry point for running the full NSGA-II loop. Coordinates
population initialisation, parallel evaluation, NSGA-II ranking, elitist
archive updates, checkpointing, and result writing.

---

### `GAState`

```python
@dataclass
class GAState:
    generation:            int
    total_evaluations:     int
    archive_size_history:  list[int]
    hypervolume_history:   list[float]
    population:            Population
    archive:               ParetoArchive
    rng:                   np.random.Generator
```

Mutable container for all state that must survive a checkpoint/resume cycle.
Every field is written to disk by `save_checkpoint` and reconstructed by
`load_checkpoint`.

| Field | Type | Description |
|---|---|---|
| `generation` | `int` | Current generation index (0-based) |
| `total_evaluations` | `int` | Cumulative FEA calls across all generations |
| `archive_size_history` | `list[int]` | Archive size recorded after each generation |
| `hypervolume_history` | `list[float]` | Hypervolume indicator recorded after each generation |
| `population` | `Population` | Current live population (list of `Individual`) |
| `archive` | `ParetoArchive` | All-time elitist non-dominated archive |
| `rng` | `np.random.Generator` | Seeded random generator — serialised to preserve reproducibility |

---

### `check_termination`

```python
def check_termination(state: GAState, config: dict) -> tuple[bool, str]
```

Evaluates all stopping criteria in order and returns as soon as one is met.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `state` | `GAState` | Current GA state |
| `config` | `dict` | Full GA YAML config dict |

**Returns** `(should_stop: bool, reason: str)`

**Termination conditions checked (in order)**

| Condition | Config key | Description |
|---|---|---|
| Generation limit | `termination.max_generations` | Stop when `state.generation >= max_generations` |
| Evaluation limit | `termination.max_evaluations` | Stop when `state.total_evaluations >= max_evaluations` |
| HV stagnation | `termination.stagnation_generations` | Stop when HV has not improved in the last N generations |
| Target HV | `termination.target_hypervolume` | Stop when HV exceeds the target threshold |

**Example**

```python
done, reason = check_termination(state, config)
if done:
    print(f"Stopping: {reason}")
```

---

### `run_ga`

```python
def run_ga(
    config_path:              str = "ga_optimizer/configs/default_ga.yaml",
    fea_config_path:          str = "FEA/configs/default.yaml",
    output_dir:               str = "ga_results/",
    resume_from_checkpoint:   str | None = None,
) -> ParetoArchive
```

Runs the full NSGA-II optimisation loop and returns the final elitist archive.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `config_path` | `str` | Path to the GA YAML configuration file |
| `fea_config_path` | `str` | Path to the FEA YAML configuration file (forwarded to `evaluate_individual`) |
| `output_dir` | `str` | Root directory for checkpoints, per-individual FEA outputs, and final results |
| `resume_from_checkpoint` | `str \| None` | Path to a `.h5` or `.pkl` checkpoint file; when provided the loop resumes from the saved generation |

**Returns** `ParetoArchive` — the final all-time non-dominated archive.

**Algorithm outline**

```
load config → initialise population (or restore from checkpoint)
loop until check_termination():
    evaluate unevaluated individuals (parallel with joblib)
    fast_non_dominated_sort → crowding_distance_assignment
    update elitist archive
    save checkpoint (every N generations)
    produce next generation via tournament selection + SBX + polynomial mutation
write_pareto_results()
return archive
```

**Example**

```python
from ga_optimizer import run_ga

archive = run_ga(
    config_path="GA/ga_optimizer/configs/default_ga.yaml",
    fea_config_path="FEA/configs/default.yaml",
    output_dir="results/run_01",
)
print(f"Pareto front size: {archive.size}")
```

**Resuming from a checkpoint**

```python
archive = run_ga(
    config_path="GA/ga_optimizer/configs/default_ga.yaml",
    fea_config_path="FEA/configs/default.yaml",
    output_dir="results/run_01_resumed",
    resume_from_checkpoint="results/run_01/checkpoint_gen010.h5",
)
```

---

## 3. Chromosome

**File:** `ga_optimizer/chromosome.py`

Defines the gene encoding for stator geometries and provides decode/sample
utilities. Each of the 12 genes encodes one design parameter either as an
absolute physical dimension or as a fraction of a derived dimension.

---

### `GeneDefinition`

```python
@dataclass
class GeneDefinition:
    name:        str
    lower:       float
    upper:       float
    dtype:       str      # "float" or "int"
    unit:        str
    description: str
```

Metadata for a single gene. Not normally instantiated by user code; access
via `GENE_DEFINITIONS`.

---

### `GENE_DEFINITIONS`

```python
GENE_DEFINITIONS: list[GeneDefinition]  # length 12
```

Authoritative definition of the 12-gene chromosome. The index order is fixed
and must not be changed without updating `LOWER_BOUNDS`, `UPPER_BOUNDS`, and
all decode logic.

| Index | Name | Range | Unit | Notes |
|---|---|---|---|---|
| 0 | `outer_diameter` | [0.150, 0.400] | m | Stator OD |
| 1 | `bore_ratio` | [0.50, 0.72] | — | ID / OD |
| 2 | `axial_length` | [0.050, 0.200] | m | Stack length |
| 3 | `num_slots` | [12, 72] | — | Integer |
| 4 | `num_poles` | [4, 20] | — | Integer, forced even |
| 5 | `tooth_width_fraction` | [0.35, 0.65] | — | Of slot pitch |
| 6 | `yoke_height_fraction` | [0.20, 0.55] | — | Of radial build |
| 7 | `slot_depth_fraction` | [0.30, 0.65] | — | Of radial build |
| 8 | `conductors_per_slot` | [8, 64] | — | Integer, forced even |
| 9 | `fill_factor` | [0.35, 0.65] | — | Copper fill ratio |
| 10 | `slot_opening_fraction` | [0.10, 0.40] | — | Of slot pitch |
| 11 | `axial_length_ratio` | [0.50, 3.00] | — | axial / pole_pitch |

---

### `LOWER_BOUNDS` / `UPPER_BOUNDS`

```python
LOWER_BOUNDS: np.ndarray  # shape (12,)
UPPER_BOUNDS: np.ndarray  # shape (12,)
```

Derived from `GENE_DEFINITIONS`. Used by operators and the constraint checker
to enforce gene bounds.

---

### `N_GENES`

```python
N_GENES: int = 12
```

The number of genes. Used throughout the package wherever an array of shape
`(N_GENES,)` is expected.

---

### `decode_chromosome`

```python
def decode_chromosome(genes: np.ndarray) -> dict
```

Converts a raw gene vector into a dict of physical SI-unit parameters ready
for geometric constraint checking and `StatorMeshInput` construction.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `genes` | `np.ndarray` | Shape `(12,)` gene vector |

**Returns** `dict` with keys:

| Key | Type | Unit | Derivation |
|---|---|---|---|
| `outer_diameter` | `float` | m | `genes[0]` (clipped to bounds) |
| `inner_diameter` | `float` | m | `outer_diameter * bore_ratio` |
| `axial_length` | `float` | m | `genes[2]` |
| `num_slots` | `int` | — | Rounded; multiple of `3 * (num_poles // 2)` |
| `num_poles` | `int` | — | Rounded to nearest even integer |
| `tooth_width` | `float` | m | `tooth_width_fraction * slot_pitch_outer` |
| `yoke_height` | `float` | m | `yoke_height_fraction * radial_build` |
| `slot_depth` | `float` | m | `slot_depth_fraction * radial_build` |
| `slot_opening` | `float` | m | `slot_opening_fraction * slot_pitch_inner` |
| `conductors_per_slot` | `int` | — | Rounded to nearest even integer |
| `fill_factor` | `float` | — | `genes[9]` |

**Integer constraints enforced**
- `num_poles` is forced to the nearest even integer (≥ 4).
- `num_slots` is forced to the nearest multiple of `3 * (num_poles // 2)` (standard three-phase winding requirement).
- `conductors_per_slot` is forced to the nearest even integer (two-layer winding).

**Example**

```python
import numpy as np
from ga_optimizer.chromosome import decode_chromosome, LOWER_BOUNDS, UPPER_BOUNDS

genes = (LOWER_BOUNDS + UPPER_BOUNDS) / 2   # midpoint of all bounds
params = decode_chromosome(genes)
print(params["outer_diameter"])   # 0.275 m
print(params["num_poles"])        # 12 (even)
```

---

### `random_individual`

```python
def random_individual(rng: np.random.Generator) -> np.ndarray
```

Samples a uniformly random gene vector within `[LOWER_BOUNDS, UPPER_BOUNDS]`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `rng` | `np.random.Generator` | Seeded NumPy random generator |

**Returns** `np.ndarray` of shape `(12,)`.

**Example**

```python
rng = np.random.default_rng(seed=42)
genes = random_individual(rng)
assert genes.shape == (12,)
assert np.all(genes >= LOWER_BOUNDS)
assert np.all(genes <= UPPER_BOUNDS)
```

---

## 4. Population

**File:** `ga_optimizer/population.py`

---

### `Individual`

```python
@dataclass
class Individual:
    genes:             np.ndarray              # shape (N_GENES,)
    objectives:        Optional[ObjectiveVector] = None
    rank:              int                     = -1
    crowding_distance: float                   = 0.0
    stator_id:         Optional[str]           = None
    evaluated:         bool                    = False
```

One candidate stator design. The `genes` array is the only mandatory field at
construction time; all other fields are populated during the GA loop.

**Equality and hashing** are identity-based (`self is other`), allowing safe use of `in` and `set()` on populations that contain numpy arrays.

| Field | Set by | Description |
|---|---|---|
| `genes` | `initialise_population` | Raw gene vector |
| `objectives` | `evaluate_individual` | Multi-objective result after FEA |
| `rank` | `fast_non_dominated_sort` | Non-domination rank (0 = Pareto front) |
| `crowding_distance` | `crowding_distance_assignment` | NSGA-II diversity estimate |
| `stator_id` | `evaluate_individual` | UUID string linking to FEA output directory |
| `evaluated` | `evaluate_individual` | Flag preventing redundant FEA calls |

#### `Individual.copy`

```python
def copy(self) -> Individual
```

Returns a new `Individual` with a copied `genes` array and shared references
to all other fields (shallow copy).

---

### `Population`

```python
Population = list[Individual]
```

Type alias used throughout the package. No behaviour beyond a plain list.

---

### `initialise_population`

```python
def initialise_population(
    pop_size:     int,
    rng:          np.random.Generator,
    seed_designs: Optional[list[np.ndarray]] = None,
) -> Population
```

Creates the initial population, optionally inserting hand-crafted designs first.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `pop_size` | `int` | Target population size |
| `rng` | `np.random.Generator` | Random generator (shared with the rest of the GA loop) |
| `seed_designs` | `list[np.ndarray] \| None` | Up to `pop_size` gene vectors to insert before random fill |

**Returns** `Population` of `pop_size` un-evaluated `Individual` objects.

**Example**

```python
rng = np.random.default_rng(0)
pop = initialise_population(pop_size=100, rng=rng)
assert len(pop) == 100
assert not pop[0].evaluated
```

---

## 5. Objectives

**File:** `ga_optimizer/objectives.py`

---

### `ObjectiveVector`

```python
@dataclass
class ObjectiveVector:
    neg_efficiency:           float   # -η          (minimise → maximise η)
    total_loss_W:             float   # W           (minimise)
    neg_power_density:        float   # -(P/V)      (minimise → maximise P/V)
    temperature_violation_K:  float   # K above limit (0 if feasible)
    safety_factor_violation:  float   # structural margin shortfall (0 if feasible)
```

Stores the result of evaluating one stator design. The three primary
objectives are encoded so that minimisation always improves the design.
The two violation fields are constraint penalties; an individual is feasible
only when both equal zero and all primary objectives are finite.

#### `ObjectiveVector.objective_array`

```python
@property
def objective_array(self) -> np.ndarray  # shape (3,)
```

Returns `[neg_efficiency, total_loss_W, neg_power_density]` as a contiguous
array for use in dominance checks and hypervolume calculations.

#### `ObjectiveVector.is_feasible`

```python
@property
def is_feasible(self) -> bool
```

`True` iff `temperature_violation_K == 0`, `safety_factor_violation == 0`,
and all three primary objectives are finite (not NaN or ±inf).

#### `ObjectiveVector.to_dict`

```python
def to_dict(self) -> dict
```

Serialises all fields to a plain Python dict suitable for JSON export. Derived
human-readable values (e.g., `efficiency_pct`, `power_density_kW_m3`) are
included alongside the raw fields.

---

### `INFEASIBLE_OBJECTIVES`

```python
INFEASIBLE_OBJECTIVES: ObjectiveVector
```

Sentinel returned by `evaluate_individual` whenever geometry checking, mesh
generation, or FEA fails. All primary objectives are set to `1e9` and both
violation fields are `1e9`, so the individual is always dominated and always
infeasible.

---

### `extract_objectives`

```python
def extract_objectives(
    fea_results,
    stator_params: dict,
    config:        dict,
) -> ObjectiveVector
```

Maps the raw output of `run_fea_pipeline` and the decoded chromosome
parameters into an `ObjectiveVector`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `fea_results` | `PipelineResults` | Return value of `run_fea_pipeline` |
| `stator_params` | `dict` | Decoded chromosome dict from `decode_chromosome` |
| `config` | `dict` | Full GA config (reads `objectives` and `constraints` sub-dicts) |

**Returns** `ObjectiveVector` with all five fields populated.

---

## 6. Constraints

**File:** `ga_optimizer/constraints.py`

---

### `GeometricConstraintViolation`

```python
class GeometricConstraintViolation(Exception)
```

Raised by `check_geometric_constraints` on the first violated geometric rule.
Callers catch this and return `INFEASIBLE_OBJECTIVES` without invoking FEA.

---

### `check_geometric_constraints`

```python
def check_geometric_constraints(genes: np.ndarray, config: dict) -> None
```

Performs seven O(1) geometric feasibility checks before invoking mesh
generation. Raises `GeometricConstraintViolation` on the first failure.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `genes` | `np.ndarray` | Shape `(12,)` gene vector |
| `config` | `dict` | Full GA config; reads `config["constraints"]` |

**Constraints checked (in order)**

| # | Check | Config key | Default |
|---|---|---|---|
| 1 | Stator wall `(OD - ID)/2 >= min_air_gap_m` | `min_air_gap_m` | 0.8 mm |
| 2 | Slot width `>= min_slot_width_m` | `min_slot_width_m` | 3 mm |
| 3 | Yoke height `>= min_yoke_height_m` | `min_yoke_height_m` | 8 mm |
| 4 | `conductors_per_slot` must be even | — | — |
| 5 | `slot_opening < slot_width` | — | — |
| 6 | `slot_depth + yoke_height <= radial_build` | — | — |
| 7 | `slot_depth > slot_opening` | — | — |

**Example**

```python
from ga_optimizer.constraints import check_geometric_constraints, GeometricConstraintViolation

try:
    check_geometric_constraints(genes, config)
except GeometricConstraintViolation as exc:
    print(f"Infeasible: {exc}")
    return INFEASIBLE_OBJECTIVES
```

---

## 7. Operators

### 7.1 Repair

**File:** `ga_optimizer/operators/repair.py`

---

#### `clamp`

```python
def clamp(genes: np.ndarray) -> np.ndarray
```

Clips every gene to `[LOWER_BOUNDS[i], UPPER_BOUNDS[i]]`. Called by SBX and
polynomial mutation after generating offspring to guarantee feasibility of the
gene vector.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `genes` | `np.ndarray` | Shape `(N_GENES,)` gene vector (not modified in-place) |

**Returns** A new `np.ndarray` of shape `(N_GENES,)` with all genes within bounds.

**Example**

```python
from ga_optimizer.operators.repair import clamp
from ga_optimizer.chromosome import LOWER_BOUNDS, UPPER_BOUNDS

out_of_range = UPPER_BOUNDS + 5.0
clamped = clamp(out_of_range)
assert np.all(clamped == UPPER_BOUNDS)
```

---

### 7.2 Crossover

**File:** `ga_optimizer/operators/crossover.py`

---

#### `sbx_crossover`

```python
def sbx_crossover(
    parent_a:    np.ndarray,
    parent_b:    np.ndarray,
    eta_c:       float,
    rng:         np.random.Generator,
    p_crossover: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]
```

Applies Simulated Binary Crossover (SBX) to two parent gene vectors and
returns two offspring. SBX mimics single-point binary crossover on real-valued
vectors; the distribution index `eta_c` controls the spread of offspring
relative to parents.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `parent_a` | `np.ndarray` | — | Shape `(N_GENES,)` first parent |
| `parent_b` | `np.ndarray` | — | Shape `(N_GENES,)` second parent |
| `eta_c` | `float` | — | Distribution index; higher = offspring closer to parents |
| `rng` | `np.random.Generator` | — | Random generator |
| `p_crossover` | `float` | `1.0` | Probability that crossover is applied at all |

**Returns** `tuple[np.ndarray, np.ndarray]` — two offspring, each clamped to
`[LOWER_BOUNDS, UPPER_BOUNDS]`.

**Properties**
- Each gene is crossed independently with probability 0.5.
- When `|parent_a[i] - parent_b[i]| < 1e-14`, gene `i` is copied unchanged.
- When `p_crossover < rng.random()`, unmodified copies of both parents are returned.

**Example**

```python
rng = np.random.default_rng(0)
child_a, child_b = sbx_crossover(parent_a, parent_b, eta_c=15.0, rng=rng)
```

---

#### `_beta_q` (internal)

```python
def _beta_q(u: float, x1: float, x2: float, lb: float, ub: float, eta_c: float) -> float
```

Computes the boundary-aware SBX spread factor β_q for a single gene. The
calculation adjusts for proximity to the lower bound so that offspring cannot
violate bounds before clamping. Not intended for direct use.

---

### 7.3 Mutation

**File:** `ga_optimizer/operators/mutation.py`

---

#### `polynomial_mutation`

```python
def polynomial_mutation(
    genes:      np.ndarray,
    eta_m:      float,
    rng:        np.random.Generator,
    p_mutation: float | None = None,
) -> np.ndarray
```

Applies polynomial mutation to a gene vector. Each gene is mutated
independently with probability `p_mutation`. The perturbation is bounded so
that mutated values remain within `[LOWER_BOUNDS[i], UPPER_BOUNDS[i]]`.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `genes` | `np.ndarray` | — | Shape `(N_GENES,)` input gene vector (not modified) |
| `eta_m` | `float` | — | Distribution index; higher = smaller perturbations |
| `rng` | `np.random.Generator` | — | Random generator |
| `p_mutation` | `float \| None` | `1/N_GENES` | Per-gene mutation probability |

**Returns** A new `np.ndarray` of shape `(N_GENES,)`.

**Example**

```python
rng = np.random.default_rng(1)
mutant = polynomial_mutation(genes, eta_m=20.0, rng=rng)
assert np.all(mutant >= LOWER_BOUNDS)
assert np.all(mutant <= UPPER_BOUNDS)
```

---

### 7.4 Selection

**File:** `ga_optimizer/operators/selection.py`

---

#### `crowded_tournament`

```python
def crowded_tournament(
    population: Population,
    rng:        np.random.Generator,
) -> Individual
```

Selects one individual from `population` using the NSGA-II crowded-comparison
operator in a binary tournament. Two individuals are drawn at random; the
better one by the crowded-comparison operator is returned.

**Crowded-comparison rule**

1. Prefer the individual with the lower `rank` (closer to the Pareto front).
2. If ranks are equal, prefer the individual with the higher `crowding_distance`
   (more isolated in objective space, promoting diversity).

**Parameters**

| Name | Type | Description |
|---|---|---|
| `population` | `Population` | Current population with `rank` and `crowding_distance` set |
| `rng` | `np.random.Generator` | Random generator |

**Returns** One `Individual` object from `population` (not a copy).

**Example**

```python
parent = crowded_tournament(population, rng)
```

---

## 8. Pareto

### 8.1 NSGA-II Core

**File:** `ga_optimizer/pareto/nsga2.py`

---

#### `dominates`

```python
def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool
```

Returns `True` iff `obj_a` Pareto-dominates `obj_b` under minimisation:
every component of `obj_a` is ≤ the corresponding component of `obj_b`, and
at least one component is strictly less.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `obj_a` | `np.ndarray` | Objective vector of candidate A |
| `obj_b` | `np.ndarray` | Objective vector of candidate B |

**Returns** `bool`

**Example**

```python
from ga_optimizer.pareto.nsga2 import dominates
import numpy as np

assert dominates(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
assert not dominates(np.array([1.0, 2.0, 1.0]), np.array([2.0, 1.0, 2.0]))
```

---

#### `fast_non_dominated_sort`

```python
def fast_non_dominated_sort(population: Population) -> list[list[int]]
```

Partitions the population into non-domination fronts using the classic O(M N²)
algorithm from Deb et al. (2002). Sets `individual.rank` in-place on every
member.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `population` | `Population` | List of evaluated `Individual` objects |

**Returns** `list[list[int]]` — list of fronts, each front being a list of
population indices. `fronts[0]` contains the non-dominated (rank-0) members.

**Side effects** Sets `individual.rank` (int) on every member of `population`.

**Example**

```python
fronts = fast_non_dominated_sort(population)
pareto_members = [population[i] for i in fronts[0]]
```

---

#### `crowding_distance_assignment`

```python
def crowding_distance_assignment(
    population: Population,
    front:      list[int],
) -> None
```

Computes and assigns crowding distance (a diversity estimate) to each member
of one non-domination front. Boundary individuals (with the smallest or
largest value on any objective) receive `+inf`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `population` | `Population` | Full population list |
| `front` | `list[int]` | Indices of members belonging to one front |

**Side effects** Sets `individual.crowding_distance` (float) on every member
in `front`.

**Example**

```python
for front in fronts:
    crowding_distance_assignment(population, front)
```

---

### 8.2 Pareto Archive

**File:** `ga_optimizer/pareto/archive.py`

---

### `ParetoArchive`

```python
class ParetoArchive()
```

Maintains the running set of all-time non-dominated feasible solutions across
all generations. On each update, newly dominated existing members are removed
and new non-dominated feasible candidates are inserted.

---

#### `ParetoArchive.update`

```python
def update(self, candidates: list[Individual]) -> int
```

Attempts to add each candidate to the archive.

**Algorithm per candidate**
1. Skip if `candidate.objectives` is `None` or `not is_feasible`.
2. Check if the candidate is dominated by any existing archive member → skip.
3. Remove all existing archive members dominated by the candidate.
4. Insert the candidate.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `candidates` | `list[Individual]` | New individuals to consider for the archive |

**Returns** `int` — the number of candidates actually added.

---

#### `ParetoArchive.members`

```python
@property
def members(self) -> list[Individual]
```

Read-only view of the current archive members. Do not mutate the returned list.

---

#### `ParetoArchive.size`

```python
@property
def size(self) -> int
```

Number of solutions in the archive.

---

#### `ParetoArchive.objective_matrix`

```python
def objective_matrix(self) -> np.ndarray
```

**Returns** `np.ndarray` of shape `(archive.size, 3)` — the stacked
`objective_array` of every member. Used by metric computation and plotting.

**Example**

```python
archive = ParetoArchive()
archive.update(population)
mat = archive.objective_matrix()    # (n, 3)
print(f"Best efficiency: {-mat[:, 0].min():.3f}")
```

---

## 9. Evaluation

### 9.1 Evaluation Cache

**File:** `ga_optimizer/evaluation/cache.py`

---

### `EvaluationCache`

```python
class EvaluationCache(max_size: Optional[int] = None)
```

Thread-safe in-memory cache that maps a SHA-256 hash of gene bytes to an
`ObjectiveVector`. Prevents redundant FEA calls for duplicate individuals
produced by selection pressure or crossover. Evicts entries in FIFO order
when `max_size` is reached.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `max_size` | `int \| None` | `None` | Maximum number of cached entries; `None` = unlimited |

---

#### `EvaluationCache.get`

```python
def get(self, genes: np.ndarray) -> Optional[ObjectiveVector]
```

Look up a cached result. Returns `None` on a cache miss. Increments internal
hit/miss counters.

**Gene hashing** The gene vector is converted to `float64` before computing
its SHA-256 digest, so `float32` and `float64` arrays with identical values
return the same result.

---

#### `EvaluationCache.put`

```python
def put(self, genes: np.ndarray, obj: ObjectiveVector) -> None
```

Store an evaluation result. If the cache is at `max_size`, the oldest entry
is evicted before insertion (FIFO). If the key already exists, the value is
overwritten.

---

#### `EvaluationCache.size`

```python
@property
def size(self) -> int
```

Number of entries currently stored.

---

#### `EvaluationCache.stats`

```python
def stats(self) -> dict
```

Returns a dict with keys `size`, `hits`, `misses`, `hit_rate`.

**Example**

```python
cache = EvaluationCache(max_size=500)
result = cache.get(genes)         # miss
if result is None:
    result = run_fea(...)
    cache.put(genes, result)
print(cache.stats())              # {'size': 1, 'hits': 0, 'misses': 1, 'hit_rate': 0.0}
```

---

### 9.2 Evaluator

**File:** `ga_optimizer/evaluation/evaluator.py`

---

#### `evaluate_individual`

```python
def evaluate_individual(
    genes:           np.ndarray,
    generation:      int,
    individual_index: int,
    config:          dict,
    cache:           EvaluationCache,
    fea_config_path: str,
    fea_output_dir:  str,
) -> ObjectiveVector
```

The single-individual evaluation pipeline. This is the function patched by
integration tests and called in parallel by the orchestrator via `joblib`.

**Execution sequence**

```
1. check_geometric_constraints(genes, config)
       → on failure: return INFEASIBLE_OBJECTIVES

2. cache.get(genes)
       → on hit: return cached ObjectiveVector

3. decode_chromosome(genes) → params

4. Build StatorParams from params
   validate_and_derive(StatorParams) → stator_params
       → on failure: return INFEASIBLE_OBJECTIVES

5. Construct StatorMeshInput(mesh_format="synthetic", ...)

6. run_fea_pipeline(mesh_input, config_path=fea_config_path, ...)
       → on failure: return INFEASIBLE_OBJECTIVES

7. extract_objectives(fea_results, stator_params, config) → ObjectiveVector

8. cache.put(genes, ObjectiveVector)

9. return ObjectiveVector
```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `genes` | `np.ndarray` | Shape `(12,)` gene vector |
| `generation` | `int` | Current generation number (for directory naming) |
| `individual_index` | `int` | Position in population (for directory naming) |
| `config` | `dict` | Full GA config dict |
| `cache` | `EvaluationCache` | Shared cache across all workers |
| `fea_config_path` | `str` | Path to the FEA YAML config |
| `fea_output_dir` | `str` | Root directory for per-individual FEA outputs |

**Returns** `ObjectiveVector` (feasible or `INFEASIBLE_OBJECTIVES`).

**Thread safety** All cache operations are protected by a `threading.Lock`.
`joblib` with `prefer="threads"` is safe.

---

## 10. I/O

### 10.1 Checkpointing

**File:** `ga_optimizer/io/checkpoint.py`

---

#### `save_checkpoint`

```python
def save_checkpoint(state: GAState, path: str) -> None
```

Serialises the full `GAState` to disk. Tries HDF5 (via `h5py`) first;
falls back to pickle if `h5py` is not installed or the write fails.

**HDF5 structure**

```
/generation             scalar int
/total_evaluations      scalar int
/archive_size_history   1-D int array
/hypervolume_history    1-D float array
/population/
    genes               (pop_size, 12) float64
    objectives          (pop_size, 5) float64   [neg_eff, loss, neg_pd, temp_v, sf_v]
    ranks               (pop_size,) int
    crowding_distances  (pop_size,) float64
/archive/
    genes               (archive_size, 12) float64
    objectives          (archive_size, 5) float64
/rng_state              bytes dataset (pickled RNG state)
```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `state` | `GAState` | Full GA state to serialise |
| `path` | `str` | Output file path; `.h5` or `.pkl` extension |

---

#### `load_checkpoint`

```python
def load_checkpoint(path: str) -> GAState
```

Restores a `GAState` from a checkpoint file. Auto-detects format from the
file extension (`.h5` → HDF5, anything else → pickle).

**Parameters**

| Name | Type | Description |
|---|---|---|
| `path` | `str` | Path to checkpoint file written by `save_checkpoint` |

**Returns** `GAState` with all fields reconstructed, including the RNG state
(so subsequent random draws are identical to a non-interrupted run).

**Example**

```python
state = load_checkpoint("results/run_01/checkpoint_gen010.h5")
print(f"Resuming from generation {state.generation}")
```

---

### 10.2 Result Writer

**File:** `ga_optimizer/io/result_writer.py`

---

#### `write_pareto_results`

```python
def write_pareto_results(
    archive:    ParetoArchive,
    output_dir: str,
    config:     dict,
    state:      GAState | None = None,
    run_id:     str | None = None,
) -> dict[str, str]
```

Writes all result artefacts for one completed GA run.

**Output files**

| File | Format | Description |
|---|---|---|
| `pareto_front.json` | JSON | All archive members with full objective dicts and gene vectors; includes `pareto_front_size` and `solutions` keys |
| `pareto_front.csv` | CSV | Tabular view of the same data, one row per solution |
| `hypervolume_history.json` | JSON | Per-generation HV and archive size history (from `state`) |
| `run_metadata.json` | JSON | Config snapshot, run ID, wall-clock time, git hash if available |
| `plots/pareto_front.html` | HTML | Interactive 3-D Pareto scatter (requires `plotly`; skipped silently if absent) |

**Parameters**

| Name | Type | Description |
|---|---|---|
| `archive` | `ParetoArchive` | Final Pareto archive |
| `output_dir` | `str` | Directory to write into (created if absent) |
| `config` | `dict` | Full GA config dict (embedded in `run_metadata.json`) |
| `state` | `GAState \| None` | If provided, HV/archive-size history is written; optional |
| `run_id` | `str \| None` | Human-readable run identifier; auto-generated UUID if `None` |

**Returns** `dict[str, str]` mapping result type to absolute file path.

**Example**

```python
paths = write_pareto_results(archive, output_dir="results/run_01", config=cfg, state=state)
print(paths["pareto_front_json"])
```

---

## 11. Utilities

### 11.1 Metrics

**File:** `ga_optimizer/utils/metrics.py`

---

#### `compute_hypervolume`

```python
def compute_hypervolume(
    members:         list[Individual],
    reference_point: list[float],
) -> float
```

Computes the hypervolume indicator of an archive using the WFG (Walking Fish
Group) recursive algorithm, which is exact for any number of objectives.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `members` | `list[Individual]` | Archive members; infeasible individuals are automatically excluded |
| `reference_point` | `list[float]` | Point that is dominated by every Pareto-optimal solution; length must equal the number of objectives (3) |

**Returns** `float` — hypervolume value; `0.0` if archive is empty or all members
are infeasible.

**Internal helpers (not part of the public API)**

| Helper | Description |
|---|---|
| `_wfg_hypervolume(pts, ref)` | Entry point for the recursive WFG computation |
| `_hv_2d(pts, ref)` | Exact 2-D hypervolume via left-to-right sweep with running y-minimum |
| `_hv_nd(pts, ref)` | N-D WFG recursion — sweeps last-objective slices from worst to best |
| `_limit(pts, limit)` | Component-wise maximum (element-level worst case) used during WFG recursion |

**Example**

```python
hv = compute_hypervolume(archive.members, reference_point=[0.0, 5000.0, 0.0])
```

---

#### `compute_igd`

```python
def compute_igd(
    approx_front:     np.ndarray,
    reference_front:  np.ndarray,
) -> float
```

Inverted Generational Distance — average minimum distance from each reference
point to the nearest point in the approximation front. Lower is better;
`IGD = 0` means the approximation is a superset of the reference front.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `approx_front` | `np.ndarray` | `(n, m)` approximated Pareto front |
| `reference_front` | `np.ndarray` | `(r, m)` true or reference Pareto front |

**Returns** `float`; returns `inf` if either input is empty.

---

#### `compute_gd`

```python
def compute_gd(
    approx_front:     np.ndarray,
    reference_front:  np.ndarray,
) -> float
```

Generational Distance — average minimum distance from each approximation
point to the nearest reference point. Measures how far the computed front
is from the true front (but not whether the reference is covered).

**Returns** `float`; returns `inf` if either input is empty.

---

#### `compute_spread`

```python
def compute_spread(
    approx_front:     np.ndarray,
    reference_front:  np.ndarray,
) -> float
```

The Δ (spread) metric — measures diversity of the approximated Pareto front.
A value of 0 means perfectly uniform spacing; higher values indicate
clustering or missing regions.

**Returns** `float` in [0, ∞). Returns `inf` if `approx_front` has fewer than
two points.

---

### 11.2 Logger

**File:** `ga_optimizer/utils/logger.py`

---

#### `setup_logger`

```python
def setup_logger(level: str = "INFO") -> logging.Logger
```

Configures the root `ga_optimizer` logger with a human-readable
`%(asctime)s  %(levelname)-8s  %(name)s — %(message)s` format and returns
the package-level logger.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `level` | `str` | Logging level string: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

**Returns** `logging.Logger` for the `ga_optimizer` namespace.

---

## 12. Test Suite Reference

All tests live under `GA/ga_optimizer/tests/`. Run from the **project root**:

```bash
python -m pytest GA/ga_optimizer/tests/ -v
```

---

### `tests/unit/test_chromosome.py`

Tests for `chromosome.py` — gene definitions, bounds, decode logic, and
geometric constraint checking.

| Test | Function under test | What it verifies |
|---|---|---|
| `test_gene_count` | `N_GENES`, `GENE_DEFINITIONS` | `len(GENE_DEFINITIONS) == 12` |
| `test_bounds_are_valid` | `LOWER_BOUNDS`, `UPPER_BOUNDS` | `LOWER_BOUNDS < UPPER_BOUNDS` for all genes |
| `test_gene_names_unique` | `GENE_DEFINITIONS` | All gene names are unique strings |
| `test_random_individual_shape` | `random_individual` | Output shape is `(12,)` |
| `test_random_individual_within_bounds` | `random_individual` | All values in `[LOWER_BOUNDS, UPPER_BOUNDS]` over 1000 samples |
| `test_decode_returns_required_keys` | `decode_chromosome` | All required keys present in output dict |
| `test_decode_inner_lt_outer` | `decode_chromosome` | `inner_diameter < outer_diameter` |
| `test_decode_num_poles_is_even` | `decode_chromosome` | `num_poles % 2 == 0` |
| `test_decode_conductors_is_even` | `decode_chromosome` | `conductors_per_slot % 2 == 0` |
| `test_decode_num_slots_multiple_of_pole_group` | `decode_chromosome` | `num_slots % (3 * num_poles // 2) == 0` |
| `test_decode_fill_factor_in_range` | `decode_chromosome` | `0 < fill_factor < 1` |
| `test_decode_known_gene` | `decode_chromosome` | Specific gene vector → specific expected output values |
| `test_feasible_gene_passes_constraints` | `check_geometric_constraints` | A known-good gene does not raise |
| `test_small_yoke_height_fails` | `check_geometric_constraints` | High bore_ratio on small OD triggers `"Yoke height"` violation |
| `test_large_tooth_fraction_fails` | `check_geometric_constraints` | Oversize tooth fraction triggers `"Slot width"` violation |
| `test_zero_yoke_height_fraction_fails` | `check_geometric_constraints` | Zero yoke fraction triggers `"Yoke height"` violation |

---

### `tests/unit/test_nsga2.py`

Tests for `pareto/nsga2.py` and `pareto/archive.py` — dominance logic,
non-dominated sorting, crowding distance, and archive update behaviour.

**`TestDominates`**

| Test | What it verifies |
|---|---|
| `test_clear_dominance` | `[1,1,1]` dominates `[2,2,2]` |
| `test_non_dominated_equal` | Equal vectors do not dominate each other |
| `test_non_dominated_tradeoff` | Trade-off vectors are mutually non-dominated |
| `test_one_better_rest_equal` | One strictly better component is sufficient for dominance |
| `test_one_worse_rest_better` | One worse component prevents dominance |
| `test_two_fronts` | Three-point example: two rank-0 members, one rank-1 |

**`TestFastNonDominatedSort`**

| Test | What it verifies |
|---|---|
| `test_single_individual` | Single member always goes to `fronts[0]` with `rank=0` |
| `test_two_fronts_3obj` | A, B non-dominated; C dominated by both → 2 fronts |
| `test_three_fronts` | Chain A dominates B dominates C → 3 fronts |
| `test_all_non_dominated` | All members on Pareto curve → single front |
| `test_ranks_set_in_place` | `individual.rank` is mutated correctly |
| `test_large_population` | 50-point Pareto curve → all rank 0 |

**`TestCrowdingDistance`**

| Test | What it verifies |
|---|---|
| `test_single_individual_gets_inf` | Single member gets `+inf` |
| `test_two_individuals_get_inf` | Two-member front → both `+inf` |
| `test_boundary_individuals_get_inf` | 5-member front → ≥ 2 boundary `+inf` members |
| `test_interior_individuals_have_positive_distance` | Interior members have finite non-negative distance |
| `test_distances_are_non_negative` | All crowding distances ≥ 0 |

**`TestParetoArchive`**

| Test | What it verifies |
|---|---|
| `test_empty_archive` | `size == 0`, `members == []` on construction |
| `test_add_feasible_individual` | First feasible member is added; `update` returns `1` |
| `test_dominated_individual_not_added` | A dominated candidate is rejected; archive stays at size 1 |
| `test_new_dominator_prunes_archive` | A dominating candidate replaces the dominated member |
| `test_infeasible_individual_not_added` | Infeasible candidate (violation > 0) is rejected |
| `test_objective_matrix_shape` | `objective_matrix()` returns `(n, 3)` array |

---

### `tests/unit/test_operators.py`

Tests for `operators/repair.py`, `operators/crossover.py`,
`operators/mutation.py`, and `operators/selection.py`.

**`TestClamp`**

| Test | What it verifies |
|---|---|
| `test_within_bounds_unchanged` | Genes already in bounds are not changed |
| `test_below_lower_clamped` | Values below `LOWER_BOUNDS` are clipped to `LOWER_BOUNDS` |
| `test_above_upper_clamped` | Values above `UPPER_BOUNDS` are clipped to `UPPER_BOUNDS` |
| `test_does_not_modify_input` | Input array is not mutated (returns new array) |

**`TestSBXCrossover`**

| Test | What it verifies |
|---|---|
| `test_offspring_shape` | Output arrays have shape `(N_GENES,)` |
| `test_offspring_within_bounds` | Over 500 pairs, no offspring gene violates bounds |
| `test_p_crossover_zero_returns_copies` | `p_crossover=0.0` → offspring equal parents |
| `test_identical_parents_return_copies` | Identical parents produce identical offspring |
| `test_different_eta_affects_spread` | `eta_c=2` produces larger out-of-interval excursions than `eta_c=20` |

**`TestPolynomialMutation`**

| Test | What it verifies |
|---|---|
| `test_output_shape` | Output shape is `(N_GENES,)` |
| `test_within_bounds_1000_trials` | 1000 mutations never produce out-of-bounds genes |
| `test_p_mutation_1_always_changes` | `p_mutation=1.0` → at least one gene changes |
| `test_does_not_modify_input` | Input gene array is not mutated |
| `test_default_mutation_rate` | Default rate `1/N_GENES` → most genes unchanged per call |

**`TestTournamentSelection`**

| Test | What it verifies |
|---|---|
| `test_lower_rank_wins` | Rank-0 individual is selected when competing with rank-1 |
| `test_same_rank_higher_crowding_wins` | Higher crowding distance wins majority of tournaments |
| `test_returns_individual_from_population` | Selected individual is always a reference from the input list |

---

### `tests/unit/test_cache.py`

Tests for `evaluation/cache.py` — thread-safe gene-result caching.

| Test | What it verifies |
|---|---|
| `test_initial_size_zero` | Fresh cache has `size == 0` |
| `test_miss_returns_none` | `get` on empty cache returns `None` |
| `test_put_and_get` | Stored result can be retrieved with same gene vector |
| `test_different_genes_dont_collide` | Two different gene vectors stored independently |
| `test_duplicate_put_overwrites` | Second `put` for same genes replaces the first |
| `test_size_after_puts` | `size` tracks the number of unique entries |
| `test_hit_rate_tracking` | `stats()` correctly reports `hits`, `misses`, `hit_rate` |
| `test_max_size_eviction` | `size <= max_size` after inserting more entries than `max_size` |
| `test_infeasible_objectives_cached` | `INFEASIBLE_OBJECTIVES` sentinel can be stored and retrieved |
| `test_numpy_dtype_independence` | `float32` and `float64` arrays with same values share one cache entry |

---

### `tests/unit/test_metrics.py`

Tests for `utils/metrics.py` — hypervolume, IGD, and GD computation.

**`TestHV2D`** — exact values for the internal 2-D sweep

| Test | What it verifies |
|---|---|
| `test_single_point` | HV of `[0,0]` with ref `[2,2]` = 4.0 |
| `test_two_points_no_overlap` | HV of `{[0,1],[1,0]}` with ref `[2,2]` = 3.0 |
| `test_dominated_point_not_counted` | Adding a dominated point does not increase HV |

**`TestComputeHypervolume`** — Individual-based interface

| Test | What it verifies |
|---|---|
| `test_empty_archive_returns_zero` | `compute_hypervolume([])` returns `0.0` |
| `test_single_feasible_individual` | Feasible individual with objectives below ref gives HV > 0 |
| `test_infeasible_individual_not_counted` | Infeasible individual (temp violation > 0) contributes HV = 0 |
| `test_hypervolume_increases_with_better_solutions` | Adding a Pareto-better solution increases HV |
| `test_two_non_dominated_solutions` | Two non-dominated solutions give larger HV than either alone |

**`TestIGD`**

| Test | What it verifies |
|---|---|
| `test_perfect_approximation_gives_zero` | When approximation equals reference, IGD = 0 |
| `test_shifted_approximation` | A shifted approximation gives IGD > 0 |
| `test_empty_inputs_give_inf` | Either empty input → `inf` |
| `test_gd_and_igd_order` | GD measures approx→ref, IGD measures ref→approx (both > 0 for a single interior point) |

---

### `tests/integration/test_full_ga_fast.py`

End-to-end integration tests using a **mocked evaluator** that replaces
`ga_optimizer.orchestrator.evaluate_individual` via `unittest.mock.patch`.
All tests use `fast_ga.yaml` (10 individuals, 3 generations, 1 worker) and
`tmp_path` for output isolation.

**Mock evaluator** returns a synthetic `ObjectiveVector` with a smooth
Pareto trade-off determined by gene values:

```python
def _mock_evaluate_individual(genes, ...):
    norm = (genes - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS + 1e-9)
    eff  = 0.80 + 0.15 * norm[0]          # higher OD → higher efficiency
    loss = 2000.0 - 1000.0 * norm[1]      # higher bore_ratio → lower loss
    pd   = 0.5e6 + 1.5e6 * norm[2]        # longer stator → higher power density
    return ObjectiveVector(neg_efficiency=-eff, total_loss_W=loss,
                           neg_power_density=-pd,
                           temperature_violation_K=0.0, safety_factor_violation=0.0)
```

| Test | What it verifies |
|---|---|
| `test_fast_ga_runs_3_generations` | `run_ga` completes without error; `archive.size >= 1`; `pareto_front.json` is written with correct structure |
| `test_all_archive_members_feasible` | Every member of the final archive has `objectives.is_feasible == True` |
| `test_pareto_front_non_dominated` | No archive member dominates another (elitist archive is clean) |
| `test_hypervolume_history_non_decreasing` | HV recorded in `hypervolume_history.json` never decreases across generations |
| `test_checkpoint_and_resume` | Phase-1 run (2 gen) writes at least one checkpoint; Phase-2 resumes from it and archive does not shrink |
| `test_result_files_written` | `pareto_front.json`, `pareto_front.csv`, and `run_metadata.json` all exist after `run_ga` |
| `test_no_nan_in_objectives` | All archive members have fully finite `objective_array` values |
