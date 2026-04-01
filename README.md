# FEM Optimization of Stator Geometries

A pure-Python pipeline for parametric finite-element mesh generation of electric
motor and generator stator geometries.  The full chain — dimensioned parameters,
geometry construction, physical-group assignment, adaptive mesh sizing, and
multi-format export — is exposed through a single importable package with zero
runtime dependencies.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Package Reference](#package-reference)
   - [Enumerations](#enumerations)
   - [StatorParams / StatorConfig](#statorparams--statorconfig)
   - [params — validation and factories](#params--validation-and-factories)
   - [geometry_builder](#geometry_builder)
   - [topology_registry](#topology_registry)
   - [gmsh_backend](#gmsh_backend)
   - [mesh_generator](#mesh_generator)
   - [export_engine](#export_engine)
   - [pipeline — high-level API](#pipeline--high-level-api)
   - [batch_scheduler](#batch_scheduler)
   - [visualiser](#visualiser)
5. [Examples](#examples)
   - [single_geometry.py](#single_geometrypy)
   - [batch_scheduler.py](#batch_schedulerpy)
6. [Tests](#tests)

---

## Architecture

```
Python caller
     │
     ▼
 pipeline.py          ← validate_config / generate_single / generate_batch
     │
     ├─ params.py             validate_and_derive (15 constraint rules)
     ├─ geometry_builder.py   GeometryBuilder.build()
     ├─ topology_registry.py  TopologyRegistry
     ├─ mesh_generator.py     MeshGenerator.generate()
     └─ export_engine.py      ExportEngine.write_all()
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                     ▼
            .msh               _meta.json              .vtk / .h5
```

`batch_scheduler.py` wraps the pipeline in a `ProcessPoolExecutor` for
parameter-sweep workloads.

---

## Repository Layout

```
FEM-optimization-of-stator-geometries/
├── stator_pipeline/
│   ├── __init__.py            Public API exports
│   ├── params.py              StatorParams dataclass + validation
│   ├── geometry_builder.py    2-D geometry construction (GMSH OCC)
│   ├── topology_registry.py   Thread-safe tag → region mapping
│   ├── gmsh_backend.py        Abstract backend + in-memory stub
│   ├── mesh_generator.py      Physical groups + mesh sizing + generation
│   ├── export_engine.py       MSH / VTK / HDF5 / JSON writers
│   ├── pipeline.py            High-level generate_single / generate_batch
│   ├── batch_scheduler.py     Parallel batch execution
│   └── visualiser.py          Cross-section and mesh plotting
├── examples/
│   ├── single_geometry.py     Single stator — HV generator example
│   ├── single_geometry.md     Design notes for the example above
│   └── batch_scheduler.py     15-job parameter sweep
├── tests/
│   └── test_stator.py         ~140 pytest test cases
└── pyproject.toml
```

---

## Installation

```bash
# Editable install (zero runtime dependencies)
pip install -e .

# With visualization support
pip install -e ".[vis]"          # matplotlib + numpy
pip install -e ".[vis,vtk]"      # + VTK library for mesh loading
pip install -e ".[vis,pyvista]"  # + PyVista for 3-D rendering

# Development (adds pytest)
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.

---

## Package Reference

All public symbols are importable directly from `stator_pipeline`:

```python
import stator_pipeline as sp
```

---

### Enumerations

#### `SlotShape`

```
class SlotShape(IntEnum)
```

Slot cross-section profile.

| Member | Value | Description |
|---|---|---|
| `RECTANGULAR` | 0 | Uniform width from bore to yoke |
| `TRAPEZOIDAL` | 1 | Linearly tapered — inner narrower than outer |
| `ROUND_BOTTOM` | 2 | Straight sides, circular arc at bottom |
| `SEMI_CLOSED` | 3 | Narrow mouth opening + full-width body |

---

#### `WindingType`

```
class WindingType(IntEnum)
```

Coil layout strategy.

| Member | Value | Description |
|---|---|---|
| `SINGLE_LAYER` | 0 | One coil group per slot |
| `DOUBLE_LAYER` | 1 | Upper and lower coil groups per slot |
| `CONCENTRATED` | 2 | Coils concentrated in adjacent slot pairs |
| `DISTRIBUTED` | 3 | Coils spread uniformly around stator |

---

#### `LaminationMaterial`

```
class LaminationMaterial(IntEnum)
```

Silicon-steel grade.

| Member | Value | Grade | Typical use |
|---|---|---|---|
| `M270_35A` | 0 | 0.35 mm, low loss | Small/medium motors |
| `M330_50A` | 1 | 0.50 mm, general | Large industrial machines |
| `M400_50A` | 2 | 0.50 mm, higher flux | Cost-sensitive large frames |
| `NO20` | 3 | Specialty amorphous | High-efficiency applications |
| `CUSTOM` | 4 | User-defined B-H curve | Research / non-standard |

---

#### `ExportFormat`

```
class ExportFormat(IntFlag)
```

Bitmask controlling which output files are written.  Combine with `|`.

| Member | Bit | Extension | Content |
|---|---|---|---|
| `NONE` | 0 | — | No files |
| `MSH` | 1 | `.msh` | GMSH native mesh |
| `VTK` | 2 | `.vtk` | VTK legacy ASCII |
| `HDF5` | 4 | `.h5` | HDF5 mesh + scalars |
| `JSON` | 8 | `_meta.json` | Parameter + mesh metadata |
| `ALL` | 15 | all above | All four formats |

```python
# Examples
sp.ExportFormat.JSON
sp.ExportFormat.MSH | sp.ExportFormat.JSON
sp.ExportFormat.ALL
```

---

#### `RegionType`

```
class RegionType(IntEnum)
```

Physical region identifiers stored in the topology registry.

| Member | Value | Description |
|---|---|---|
| `UNKNOWN` | 0 | Unassigned |
| `YOKE` | 1 | Back-iron annulus |
| `TOOTH` | 2 | Tooth body |
| `SLOT_AIR` | 3 | Slot air cavity |
| `SLOT_INS` | 4 | Slot insulation layer |
| `COIL_A_POS` | 5 | Phase A, current into page |
| `COIL_A_NEG` | 6 | Phase A, current out of page |
| `COIL_B_POS` | 7 | Phase B, positive |
| `COIL_B_NEG` | 8 | Phase B, negative |
| `COIL_C_POS` | 9 | Phase C, positive |
| `COIL_C_NEG` | 10 | Phase C, negative |
| `BORE_AIR` | 11 | Air in bore |
| `BOUNDARY_BORE` | 12 | Bore circle boundary curve |
| `BOUNDARY_OUTER` | 13 | Outer circle boundary curve |

---

### StatorParams / StatorConfig

```
class StatorParams
```

Central parameter dataclass.  `StatorConfig` is an alias for `StatorParams`.

All dimensional fields are in **SI units (metres)** unless noted otherwise.

#### Radial geometry

| Field | Type | Default | Description |
|---|---|---|---|
| `R_outer` | `float` | `0.25` | Outer stator radius (m) |
| `R_inner` | `float` | `0.15` | Inner bore radius (m) |
| `airgap_length` | `float` | `0.001` | Radial air-gap (m) |

#### Slot geometry

| Field | Type | Default | Description |
|---|---|---|---|
| `n_slots` | `int` | `36` | Number of stator slots; must be even and ≥ 6 |
| `slot_depth` | `float` | `0.06` | Radial depth of slot (m) |
| `slot_width_outer` | `float` | `0.012` | Slot width at yoke edge (m) |
| `slot_width_inner` | `float` | `0.010` | Slot width at bore edge (m) |
| `slot_opening` | `float` | `0.004` | Mouth opening width — `SEMI_CLOSED` only (m) |
| `slot_opening_depth` | `float` | `0.003` | Depth of mouth region — `SEMI_CLOSED` only (m) |
| `tooth_tip_angle` | `float` | `0.1` | Chamfer angle of tooth tips (radians); must be in `[0, π/4)` |
| `slot_shape` | `SlotShape` | `SEMI_CLOSED` | Cross-section profile |

#### Coil and winding

| Field | Type | Default | Description |
|---|---|---|---|
| `coil_depth` | `float` | `0.05` | Radial coil depth (m); must satisfy constraint below |
| `coil_width_outer` | `float` | `0.008` | Coil width at outer edge (m) |
| `coil_width_inner` | `float` | `0.007` | Coil width at inner edge (m) |
| `insulation_thickness` | `float` | `0.001` | Groundwall insulation thickness (m) |
| `turns_per_coil` | `int` | `10` | Number of conductor turns per coil |
| `coil_pitch` | `int` | `5` | Coil pitch in slot numbers |
| `wire_diameter` | `float` | `0.001` | Round conductor diameter (m) |
| `slot_fill_factor` | `float` | `0.45` | Target copper fill fraction `(0, 1)` |
| `winding_type` | `WindingType` | `DOUBLE_LAYER` | Coil layout |

**Coil depth constraint:**

```
coil_depth ≤ slot_depth − slot_opening_depth − 2 × insulation_thickness
```

#### Lamination stack

| Field | Type | Default | Description |
|---|---|---|---|
| `t_lam` | `float` | `0.00035` | Individual lamination thickness (m) |
| `n_lam` | `int` | `200` | Number of laminations; must be > 0 |
| `z_spacing` | `float` | `0.0` | Inter-lamination gap (m); must be ≥ 0 |
| `insulation_coating_thickness` | `float` | `0.00005` | Conductor coating thickness (m) |
| `material` | `LaminationMaterial` | `M270_35A` | Electrical steel grade |
| `material_file` | `str` | `""` | Path to B-H curve file; required when `material=CUSTOM` |

#### Mesh sizing

| Field | Type | Default | Description |
|---|---|---|---|
| `mesh_yoke` | `float` | `0.006` | Target element size in yoke (m) |
| `mesh_slot` | `float` | `0.003` | Target element size in slot (m) |
| `mesh_coil` | `float` | `0.0015` | Target element size in coil (m) |
| `mesh_ins` | `float` | `0.0007` | Target element size in insulation (m) |
| `mesh_boundary_layers` | `int` | `3` | Number of boundary layers at bore |
| `mesh_curvature` | `float` | `0.3` | Curvature-based refinement factor `(0, 1)` |
| `mesh_transition_layers` | `int` | `2` | Transition layer count between regions |

Mesh sizes must satisfy: `mesh_ins ≤ mesh_coil ≤ mesh_slot ≤ mesh_yoke`.

#### Derived fields (read-only after validation)

| Field | Formula | Description |
|---|---|---|
| `yoke_height` | `R_outer − R_inner − slot_depth` | Radial back-iron height (m) |
| `tooth_width` | `R_inner × (2π/n_slots) − slot_width_inner` | Tooth body width (m) |
| `slot_pitch` | `2π / n_slots` | Angular slot pitch (radians) |
| `stack_length` | `n_lam × t_lam + (n_lam−1) × z_spacing` | Total axial stack (m) |
| `fill_factor` | `coil_area / slot_area` | Actual copper fill fraction |

---

### params — validation and factories

#### `validate_and_derive`

```python
stator_pipeline.validate_and_derive(p: StatorParams) -> StatorParams
```

Validates all geometric constraints and computes derived fields.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `p` | `StatorParams` | Parameter set to validate |

**Returns** `StatorParams` — new instance with all five derived fields populated.

**Raises** `ValueError` with a descriptive message if any of the 15 constraint
rules is violated.

**Validation rules**

| # | Rule |
|---|---|
| 1 | All dimensional fields > 0 |
| 2 | `R_inner < R_outer` |
| 3 | `slot_depth < (R_outer − R_inner)` |
| 4 | `n_slots ≥ 6` and even |
| 5 | `slot_width_inner < R_inner × 2π/n_slots` |
| 6 | `SEMI_CLOSED`: `slot_opening < slot_width_inner` and `slot_opening_depth < slot_depth` |
| 7 | `coil_depth ≤ slot_depth − slot_opening_depth − 2×insulation_thickness` |
| 8 | `coil_width_inner ≤ slot_width_inner − 2×insulation_thickness` |
| 9 | `n_lam > 0` |
| 10 | `z_spacing ≥ 0` |
| 11 | `insulation_coating_thickness ≥ 0` |
| 12 | `material == CUSTOM` requires non-empty `material_file` |
| 13 | `mesh_ins ≤ mesh_coil ≤ mesh_slot ≤ mesh_yoke` |
| 14 | `tooth_tip_angle ∈ [0, π/4)` |
| 15 | Computed `fill_factor ∈ (0, 1)` |

---

#### `make_reference_params`

```python
stator_pipeline.make_reference_params() -> StatorParams
```

Factory returning a validated 36-slot reference configuration (SEMI_CLOSED,
DOUBLE_LAYER, M270_35A, 200 laminations × 0.35 mm).

**Returns** `StatorParams` — validated, derived fields populated.

---

#### `make_minimal_params`

```python
stator_pipeline.make_minimal_params() -> StatorParams
```

Factory returning a validated 12-slot minimal configuration (RECTANGULAR,
SINGLE_LAYER, smaller radii) suitable for unit-test workloads.

**Returns** `StatorParams` — validated, derived fields populated.

---

### geometry_builder

#### `SlotProfile`

```
class SlotProfile(dataclass)
```

Geometry tags produced for one slot during `GeometryBuilder.build()`.

| Field | Type | Description |
|---|---|---|
| `slot_idx` | `int` | Slot index (0-based) |
| `angle` | `float` | Rotation angle in the cross-section (radians) |
| `slot_surface` | `int` | GMSH surface tag of the slot air cavity |
| `coil_upper_sf` | `int` | Upper coil surface tag |
| `coil_lower_sf` | `int` | Lower coil tag; `-1` for `SINGLE_LAYER` |
| `ins_upper_sf` | `int` | Upper insulation surface tag |
| `ins_lower_sf` | `int` | Lower insulation surface tag |
| `mouth_curve_bot` | `int` | Bore-facing slot mouth edge tag |
| `mouth_curve_top` | `int` | Inner edge of slot body (SEMI_CLOSED only) |

---

#### `GeometryBuildResult`

```
class GeometryBuildResult(dataclass)
```

Return value of `GeometryBuilder.build()`.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | `True` if geometry was built without error |
| `yoke_surface` | `int` | GMSH surface tag of the yoke annulus |
| `bore_curve` | `int` | GMSH curve tag of the bore circle |
| `outer_curve` | `int` | GMSH curve tag of the outer circle |
| `n_slots` | `int` | Number of slots built |
| `slots` | `list[SlotProfile]` | One `SlotProfile` per slot |
| `error_message` | `str` | Non-empty only when `success=False` |

---

#### `GeometryBuilder`

```
class GeometryBuilder
```

Constructs the 2-D stator cross-section inside a GMSH OCC session.

```python
GeometryBuilder(backend: GmshBackend) -> GeometryBuilder
```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `backend` | `GmshBackend` | Active GMSH backend (real or stub) |

---

```python
GeometryBuilder.build(p: StatorParams) -> GeometryBuildResult
```

Builds the full cross-section for the given parameter set.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `p` | `StatorParams` | Validated stator parameters |

**Returns** `GeometryBuildResult`

**Build sequence**

1. Add outer and inner circles.
2. Boolean-cut to produce the yoke annulus.
3. For each slot, call the shape-specific builder (`_build_rectangular`,
   `_build_trapezoidal`, `_build_round_bottom`, or `_build_semi_closed`).
4. Boolean-cut all slot cavities from the yoke.
5. Build coil surfaces (`_build_coil`) and insulation surfaces
   (`_build_insulation`) inside each slot.
6. Call `backend.synchronize()`.

---

### topology_registry

#### `SlotWindingAssignment`

```
class SlotWindingAssignment(dataclass)
```

Winding phase assignment for one slot, produced by
`TopologyRegistry.assign_winding_layout()`.

| Field | Type | Description |
|---|---|---|
| `slot_idx` | `int` | Slot index (0-based) |
| `upper_tag` | `int` | GMSH surface tag of upper coil |
| `lower_tag` | `int` | GMSH surface tag of lower coil; `-1` for `SINGLE_LAYER` |
| `upper_phase` | `RegionType` | Phase assignment for upper coil |
| `lower_phase` | `RegionType` | Phase assignment for lower coil; `UNKNOWN` for `SINGLE_LAYER` |

---

#### `TopologyRegistry`

```
class TopologyRegistry
```

Thread-safe registry mapping GMSH entity tags to stator regions.  Protected
internally by a reentrant lock (`threading.RLock`).

```python
TopologyRegistry(n_slots: int) -> TopologyRegistry
```

**Parameters**

| Name | Type | Description |
|---|---|---|
| `n_slots` | `int` | Number of stator slots; must be > 0 |

**Raises** `ValueError` if `n_slots ≤ 0`.

---

```python
TopologyRegistry.register_surface(
    region_type: RegionType,
    gmsh_tag: int,
    slot_idx: int = -1
) -> None
```

Register a GMSH 2-D surface tag.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `region_type` | `RegionType` | — | Region the surface belongs to |
| `gmsh_tag` | `int` | — | GMSH surface tag |
| `slot_idx` | `int` | `-1` | Slot index for slot-specific regions |

---

```python
TopologyRegistry.register_slot_coil(
    slot_idx: int,
    upper_tag: int,
    lower_tag: int
) -> None
```

Register the coil surface tags for one slot.

| Parameter | Type | Description |
|---|---|---|
| `slot_idx` | `int` | Slot index |
| `upper_tag` | `int` | Upper coil GMSH surface tag |
| `lower_tag` | `int` | Lower coil GMSH surface tag (`-1` = none) |

**Raises** `IndexError` if `slot_idx` is out of range.

---

```python
TopologyRegistry.register_boundary_curve(
    region_type: RegionType,
    gmsh_curve: int
) -> None
```

Register a boundary curve tag.

| Parameter | Type | Description |
|---|---|---|
| `region_type` | `RegionType` | Must be `BOUNDARY_BORE` or `BOUNDARY_OUTER` |
| `gmsh_curve` | `int` | GMSH curve tag |

**Raises** `ValueError` if `region_type` is not a boundary type.

---

```python
TopologyRegistry.assign_winding_layout(winding_type: WindingType) -> None
```

Compute and store the 3-phase winding assignment for all registered slots.

| Parameter | Type | Description |
|---|---|---|
| `winding_type` | `WindingType` | Winding configuration |

**Phase patterns (slot index mod 6):**

| Index mod 6 | DISTRIBUTED / DOUBLE_LAYER | CONCENTRATED | SINGLE_LAYER |
|---|---|---|---|
| 0 | A+ | A+ | A+ (lower = UNKNOWN) |
| 1 | B− | A− | B+ |
| 2 | C+ | B+ | C+ |
| 3 | A− | B− | A+ |
| 4 | B+ | C+ | B+ |
| 5 | C− | C− | C+ |

**Raises** `RuntimeError` if no coil surfaces have been registered.

---

```python
TopologyRegistry.get_surfaces(region_type: RegionType) -> list[int]
```

**Returns** list of GMSH surface tags registered under `region_type`.

---

```python
TopologyRegistry.get_boundary_curves(region_type: RegionType) -> list[int]
```

**Returns** list of boundary curve tags for `region_type`.

---

```python
TopologyRegistry.get_slot_assignment(slot_idx: int) -> SlotWindingAssignment
```

**Returns** winding assignment for one slot.

**Raises** `RuntimeError` if `assign_winding_layout` has not been called.
**Raises** `IndexError` if `slot_idx` is out of range.

**Properties**

| Name | Type | Description |
|---|---|---|
| `total_surfaces` | `int` | Total count of registered surfaces |
| `winding_assigned` | `bool` | Whether winding layout has been assigned |
| `winding_assignments` | `list[Optional[SlotWindingAssignment]]` | All slot assignments |

---

### gmsh_backend

#### `GmshBackend` (abstract)

```
class GmshBackend(ABC)
```

Abstract interface for all geometry and mesh backends.  Concrete
implementations are the real GMSH backend (when `gmsh` is installed) and the
`StubGmshBackend` used for testing.

**Session lifecycle**

```python
GmshBackend.initialize(model_name: str) -> None
GmshBackend.synchronize() -> None
GmshBackend.finalize() -> None
GmshBackend.set_option(name: str, value: float) -> None
```

**Geometry primitives** — each returns a new integer entity tag.

```python
GmshBackend.add_point(x: float, y: float, z: float, mesh_size: float) -> int
GmshBackend.add_line(start: int, end: int) -> int
GmshBackend.add_circle(cx: float, cy: float, cz: float, r: float) -> int
GmshBackend.add_arc(start: int, centre: int, end: int) -> int
GmshBackend.add_curve_loop(tags: list[int]) -> int
GmshBackend.add_plane_surface(loop_tags: list[int]) -> int
```

**Boolean operations** — return surviving `(dim, tag)` pairs.

```python
GmshBackend.boolean_cut(
    objects: list[tuple[int, int]],
    tools: list[tuple[int, int]],
    remove_tool: bool = False
) -> list[tuple[int, int]]

GmshBackend.boolean_fragment(
    objects: list[tuple[int, int]],
    tools: list[tuple[int, int]]
) -> list[tuple[int, int]]
```

**Physical groups and mesh fields**

```python
GmshBackend.add_physical_group(dim: int, tags: list[int], name: str, tag: int = -1) -> int
GmshBackend.add_math_eval_field(expr: str) -> int
GmshBackend.add_constant_field(value: float, surfaces: list[int]) -> int
GmshBackend.set_background_field(field_tag: int) -> None
```

**Mesh I/O**

```python
GmshBackend.generate_mesh(dim: int) -> None
GmshBackend.write_mesh(filename: str) -> None
GmshBackend.get_entities_2d() -> list[tuple[int, int]]
```

---

#### `StubGmshBackend`

```
class StubGmshBackend(GmshBackend)
```

In-memory stub — no GMSH installation required.  Used by the test suite and
as the default when `gmsh` is not importable.

- `add_point`, `add_line`, `add_circle`, `add_arc` return monotonically
  increasing integer tags.
- `boolean_cut` returns the `objects` list unchanged.
- `boolean_fragment` returns the union of `objects` and `tools`.
- All other methods are no-ops or record state for inspection.

---

#### `make_default_backend`

```python
stator_pipeline.make_default_backend() -> GmshBackend
```

Returns a `StubGmshBackend`.  When a real GMSH backend is registered it will
be returned instead.

---

### mesh_generator

#### `MeshConfig`

```
class MeshConfig(dataclass)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `algorithm_2d` | `int` | `6` | GMSH 2-D meshing algorithm (6 = Frontal-Delaunay) |
| `smoothing_passes` | `int` | `5` | Laplacian smoothing iterations after generation |

---

#### `MeshResult`

```
class MeshResult(dataclass)
```

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | `True` if meshing completed without error |
| `n_nodes` | `int` | Total node count |
| `n_elements_2d` | `int` | Number of 2-D triangles / quads |
| `n_elements_3d` | `int` | Number of 3-D elements (`0` for 2-D-only meshes) |
| `min_quality` | `float` | Minimum element quality metric `[0, 1]` |
| `avg_quality` | `float` | Average element quality metric `[0, 1]` |
| `n_phys_groups` | `int` | Number of physical groups assigned |
| `error_message` | `str` | Non-empty only when `success=False` |

---

#### `MeshGenerator`

```
class MeshGenerator
```

```python
MeshGenerator(
    backend: GmshBackend,
    config: MeshConfig | None = None
) -> MeshGenerator
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `GmshBackend` | — | Active GMSH backend |
| `config` | `MeshConfig \| None` | `None` | Uses `MeshConfig()` if `None` |

---

```python
MeshGenerator.assign_physical_groups(
    p: StatorParams,
    geo: GeometryBuildResult,
    registry: TopologyRegistry
) -> None
```

Registers all GMSH physical groups from a completed geometry build.

| Parameter | Type | Description |
|---|---|---|
| `p` | `StatorParams` | Stator parameters |
| `geo` | `GeometryBuildResult` | Geometry build result |
| `registry` | `TopologyRegistry` | Registry to populate |

**Actions**
1. Register yoke surface as `YOKE`.
2. Register bore and outer boundary curves.
3. Register slot air and insulation surfaces.
4. Aggregate coil surfaces by phase and register them.
5. Add each region as a named GMSH physical group.

---

```python
MeshGenerator.generate(
    p: StatorParams,
    geo: GeometryBuildResult,
    registry: TopologyRegistry
) -> MeshResult
```

Full mesh generation pipeline.

| Parameter | Type | Description |
|---|---|---|
| `p` | `StatorParams` | Stator parameters |
| `geo` | `GeometryBuildResult` | Completed geometry |
| `registry` | `TopologyRegistry` | Topology registry (populated in place) |

**Returns** `MeshResult`

**Generation sequence**

1. Check `geo.success`; return failure result immediately if false.
2. Call `assign_physical_groups`.
3. Apply Layer A — constant element-size fields per region.
4. Apply Layer B — mouth transition threshold field.
5. Apply Layer C — bore boundary-layer field.
6. Combine fields with Min operation and set as background field.
7. Set algorithm and smoothing options.
8. Generate 2-D mesh (`dim=2`).
9. If `n_lam > 1`, extrude and generate 3-D mesh.

---

### export_engine

#### `ExportConfig`

```
class ExportConfig(dataclass)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | `"/tmp/stator_out"` | Output directory (created if absent) |
| `formats` | `ExportFormat` | `ExportFormat.JSON` | Format bitmask |

---

#### `ExportResult`

```
class ExportResult(dataclass)
```

Result for a single written file.

| Field | Type | Description |
|---|---|---|
| `format` | `ExportFormat` | Which format this result covers |
| `path` | `str` | Absolute path of the written file |
| `success` | `bool` | Write succeeded |
| `write_time_ms` | `float` | Wall-clock time for the write (ms) |
| `error_message` | `str` | Non-empty only when `success=False` |

---

#### `ExportEngine`

```
class ExportEngine
```

```python
ExportEngine(backend: GmshBackend) -> ExportEngine
```

---

```python
ExportEngine.write_all(
    p: StatorParams,
    mesh: MeshResult,
    cfg: ExportConfig
) -> list[ExportResult]
```

Write all requested output formats.

| Parameter | Type | Description |
|---|---|---|
| `p` | `StatorParams` | Stator parameters (encoded in file names and JSON) |
| `mesh` | `MeshResult` | Mesh statistics to embed in JSON |
| `cfg` | `ExportConfig` | Output directory and format selection |

**Returns** `list[ExportResult]` — one entry per format bit set in `cfg.formats`.

**Output file naming** — files share a deterministic 8-character stem derived
from the SHA-256 hash of the serialised parameters:

```
stator_<hash8>.msh
stator_<hash8>.vtk
stator_<hash8>.h5
stator_<hash8>_meta.json
```

**JSON metadata structure**

```json
{
  "stem": "stator_a1b2c3d4",
  "params": { "R_outer": 0.65, "n_slots": 48, ... },
  "derived": {
    "yoke_height": 0.11,
    "tooth_width": 0.023,
    "slot_pitch": 0.1309,
    "stack_length": 0.70,
    "fill_factor": 0.38
  },
  "mesh": {
    "n_nodes": 0,
    "n_elements_2d": 0,
    "min_quality": 0.0,
    "avg_quality": 0.0
  }
}
```

---

#### `sha256`

```python
stator_pipeline.sha256(data: str) -> str
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `str` | UTF-8 string to hash |

**Returns** 64-character lowercase hexadecimal SHA-256 digest.

---

#### `compute_stem`

```python
stator_pipeline.compute_stem(p: StatorParams) -> str
```

| Parameter | Type | Description |
|---|---|---|
| `p` | `StatorParams` | Stator parameters |

**Returns** `"stator_" + sha256(json_dump(p))[:8]` — deterministic 15-character
stem used in output file names.

---

#### `outputs_exist`

```python
stator_pipeline.outputs_exist(p: StatorParams, cfg: ExportConfig) -> bool
```

Check whether all requested output files for `p` already exist in `cfg.output_dir`.

**Returns** `True` only if every file corresponding to every set bit in
`cfg.formats` is present on disk.

---

### pipeline — high-level API

The pipeline module provides the recommended entry point for most use cases.

#### `validate_config`

```python
stator_pipeline.validate_config(cfg: StatorParams) -> dict[str, Any]
```

Validate a parameter set and return a result dictionary.

| Parameter | Type | Description |
|---|---|---|
| `cfg` | `StatorParams` | Parameters to validate |

**Returns** `dict` with the following keys:

| Key | Type | Present when | Description |
|---|---|---|---|
| `success` | `bool` | always | `True` if all 15 rules passed |
| `error` | `str` | `success=False` | Human-readable error message |
| `yoke_height` | `float` | `success=True` | Derived yoke height (m) |
| `tooth_width` | `float` | `success=True` | Derived tooth width (m) |
| `slot_pitch` | `float` | `success=True` | Derived slot pitch (radians) |
| `stack_length` | `float` | `success=True` | Derived stack length (m) |
| `fill_factor` | `float` | `success=True` | Derived fill factor |

---

#### `generate_single`

```python
stator_pipeline.generate_single(
    config: StatorParams,
    output_dir: str,
    formats: str | int = "JSON"
) -> dict[str, Any]
```

Run the full pipeline for a single stator configuration.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `StatorParams` | — | Stator parameters |
| `output_dir` | `str` | — | Output directory (created if absent) |
| `formats` | `str \| int` | `"JSON"` | Format string, pipe-separated (`"MSH\|VTK"`) or integer bitmask |

**Recognized format strings** (case-insensitive, pipe-separated):
`"MSH"`, `"VTK"`, `"HDF5"`, `"JSON"`, `"ALL"`

**Returns** `dict` with the following keys:

| Key | Type | Present when | Description |
|---|---|---|---|
| `success` | `bool` | always | |
| `error` | `str` | `success=False` | |
| `yoke_height` | `float` | `success=True` | |
| `tooth_width` | `float` | `success=True` | |
| `slot_pitch` | `float` | `success=True` | |
| `stack_length` | `float` | `success=True` | |
| `fill_factor` | `float` | `success=True` | |
| `output_dir` | `str` | `success=True` | |
| `formats` | `int` | `success=True` | Resolved bitmask |
| `stem` | `str` | `success=True` | File name stem |
| `json_path` | `str` | JSON in formats | Absolute path to `_meta.json` |

**Pipeline steps**

1. `validate_config` — fail fast on invalid parameters.
2. Create `output_dir`.
3. Compute deterministic stem via `compute_stem`.
4. Write JSON metadata.
5. If formats include MSH / VTK / HDF5: run geometry → mesh → export.
6. Finalize the GMSH backend.

---

#### `generate_batch`

```python
stator_pipeline.generate_batch(
    configs: list[StatorParams],
    output_dir: str,
    max_parallel: int = 0,
    formats: str | int = "MSH|VTK|HDF5|JSON",
    progress_callback: Callable[[int, int, str], None] | None = None,
    skip_existing: bool = True,
    job_timeout_sec: int = 300
) -> list[dict[str, Any]]
```

Run the pipeline for a list of configurations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `configs` | `list[StatorParams]` | — | Parameter sets |
| `output_dir` | `str` | — | Shared output directory |
| `max_parallel` | `int` | `0` | Reserved for future parallel execution |
| `formats` | `str \| int` | `"MSH\|VTK\|HDF5\|JSON"` | Format specification |
| `progress_callback` | `Callable \| None` | `None` | Called after each job: `callback(done, total, job_id)` |
| `skip_existing` | `bool` | `True` | Skip if output files already exist |
| `job_timeout_sec` | `int` | `300` | Reserved; not currently enforced |

**Returns** `list[dict]` — one result dict per input config, in the same order.
Each dict has all keys from `generate_single` plus `job_id: str`
(`"batch_0"`, `"batch_1"`, …).

---

### batch_scheduler

The `BatchScheduler` provides parallel execution via `ProcessPoolExecutor`.

#### `BatchJob`

```
class BatchJob(dataclass)
```

| Field | Type | Description |
|---|---|---|
| `job_id` | `str` | User-assigned identifier |
| `params` | `StatorParams` | Stator parameters |
| `export_config` | `ExportConfig` | Output directory and format selection |

---

#### `BatchConfig`

```
class BatchConfig(dataclass)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_parallel` | `int` | `0` | Worker count; `0` = CPU count |
| `skip_existing` | `bool` | `True` | Skip jobs whose output files exist |
| `job_timeout_sec` | `int` | `300` | Per-job timeout (reserved) |
| `write_summary` | `bool` | `True` | Write `batch_summary.json` on completion |

---

#### `BatchResult`

```
class BatchResult(dataclass)
```

| Field | Type | Description |
|---|---|---|
| `job_id` | `str` | Job identifier |
| `success` | `bool` | Job completed without error |
| `error` | `str` | Error message when `success=False` |
| `msh_path` | `str` | Path to `.msh` output (empty if not requested) |
| `vtk_path` | `str` | Path to `.vtk` output |
| `hdf5_path` | `str` | Path to `.h5` output |
| `json_path` | `str` | Path to `_meta.json` output |

---

#### `BatchScheduler`

```
class BatchScheduler
```

```python
BatchScheduler() -> BatchScheduler
```

---

```python
BatchScheduler.cancel() -> None
```

Signal all pending jobs to abort.  Jobs already running in subprocesses are
not interrupted; only jobs not yet submitted are skipped.

---

```python
BatchScheduler.run(
    jobs: list[BatchJob],
    config: BatchConfig,
    progress_callback: Callable[[int, int, bool, str], None] | None = None
) -> list[BatchResult]
```

Execute jobs in parallel.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `jobs` | `list[BatchJob]` | — | Jobs to run |
| `config` | `BatchConfig` | — | Scheduling configuration |
| `progress_callback` | `Callable \| None` | `None` | Called after each job: `callback(done, total, success, job_id)` |

**Returns** `list[BatchResult]` in the same order as `jobs`.

**Execution sequence**

1. Filter `skip_existing` jobs — mark as success immediately, don't submit.
2. Submit remaining jobs to `ProcessPoolExecutor`.
3. Collect `Future` results as they complete.
4. Optionally write `batch_summary.json`.
5. Return all results in input order.

---

### visualiser

#### `StatorVisualiser`

```
class StatorVisualiser
```

```python
StatorVisualiser() -> StatorVisualiser
```

Raises `ImportError` if `matplotlib` is not installed.

---

```python
StatorVisualiser.plot_cross_section(
    vtk_path: str,
    output_png: str | None = None
) -> None
```

Render the 2-D cross-section coloured by physical region.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vtk_path` | `str` | — | Path to a `.vtk` file produced by `ExportEngine` |
| `output_png` | `str \| None` | `None` | Save to this path; if `None`, call `plt.show()` |

**Region colour scheme**

| Region | Colour |
|---|---|
| Yoke | `#4a90d9` (blue) |
| Slot air | `#b0d4f1` (light blue) |
| Insulation | `#f5e642` (yellow) |
| Coil A+ | `#e63946` (red) |
| Coil A− | `#ff8fa3` (light red) |
| Coil B+ | `#2a9d8f` (teal) |
| Coil B− | `#80cdc1` (light teal) |
| Coil C+ | `#f4a261` (orange) |
| Coil C− | `#ffd6a5` (light orange) |

---

```python
StatorVisualiser.plot_mesh(
    vtk_path: str,
    show_quality: bool = True
) -> None
```

Render each mesh element coloured by quality.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vtk_path` | `str` | — | Path to `.vtk` file |
| `show_quality` | `bool` | `True` | Colour by quality; `False` colours by region scalar |

Requires `matplotlib` and `numpy`.

---

## Examples

### single_geometry.py

Generates and visualises one stator cross-section sized for a large high-voltage
asynchronous generator (4-pole, 6–11 kV, power-station duty).

See [examples/single_geometry.md](examples/single_geometry.md) for full design
rationale.

**Usage**

```bash
# Default run — 48 slots, 1400 laminations
python examples/single_geometry.py

# Custom output directory, skip visualisation
python examples/single_geometry.py --output /tmp/my_stator --no-plot

# Custom slot and lamination count
python examples/single_geometry.py --slots 72 --lam 1600
```

**CLI options**

| Flag | Default | Description |
|---|---|---|
| `--output PATH` | `/tmp/stator_single` | Directory for outputs |
| `--no-plot` | off | Skip gmsh GUI and PNG export |
| `--slots N` | `48` | Number of stator slots |
| `--lam N` | `1400` | Number of laminations |

**Default geometry (no flags)**

| Parameter | Value |
|---|---|
| Outer radius | 650 mm |
| Inner radius | 420 mm |
| Air-gap | 3 mm |
| Slots | 48 |
| Slot depth | 115 mm |
| Lamination thickness | 0.50 mm (M330-50A) |
| Stack length | 700 mm (1400 × 0.5 mm) |
| Insulation | 3 mm class-F groundwall |

**Expected console output**

```
Validating parameters …
  OK

==============================================================
  STATOR GEOMETRY — VALIDATION REPORT
==============================================================
  Outer radius                      650.0 mm
  Inner radius                      420.0 mm
  Air-gap                             3.00 mm
  Slots                                 48
  Slot shape                    SEMI_CLOSED
  Winding type                 DOUBLE_LAYER
  Laminations                1400 × 0.500 mm
  Material                        M330-50A
--------------------------------------------------------------
  Yoke height                       115.00 mm
  Tooth width                        23.XXX mm
  Slot pitch                           7.50 °  (0.1309 rad)
  Stack length                        700.0 mm
  Fill factor                         0.XXX  (XX.X %)
--------------------------------------------------------------
  Mesh — yoke                         20.00 mm
  Mesh — slot                         10.00 mm
  Mesh — coil                          6.00 mm
  Mesh — insulation                    3.00 mm
==============================================================

Running pipeline …
  Stem      : stator_XXXXXXXX
  Metadata  : /tmp/stator_single/stator_XXXXXXXX_meta.json
  SHA-256   : XXXXXXXXXXXXXXXXXXXX…

Rendering 2-D cross-section …
  Cross-section saved → /tmp/stator_single/stator_cross_section.png

Rendering 3-D lamination stack …
  Showing 8/1400 laminations  (t_lam=0.500 mm, vis stack ≈ 4.00 mm)
  3-D stack saved → /tmp/stator_single/stator_3d_stack.png

Done.  Output → /tmp/stator_single
```

**Output files**

| File | Description |
|---|---|
| `stator_XXXXXXXX_meta.json` | Full parameter set + derived geometry + mesh statistics |
| `stator_cross_section.png` | 2-D cross-section coloured by region (yoke, phases A/B/C, insulation) |
| `stator_3d_stack.png` | Isometric 3-D view of up to 8 laminations with semi-transparent yoke |

---

### batch_scheduler.py

Runs a 5 × 3 parameter sweep over slot count and lamination count (15 jobs
total) with a live progress bar and optional matplotlib summary chart.

**Usage**

```bash
# Full run — generate all 15 geometries
python examples/batch_scheduler.py

# Validate only, skip generation
python examples/batch_scheduler.py --dry-run

# Custom output, skip chart
python examples/batch_scheduler.py --output /tmp/sweep --no-plot
```

**CLI options**

| Flag | Default | Description |
|---|---|---|
| `--output PATH` | `/tmp/stator_batch` | Directory for all job outputs |
| `--no-plot` | off | Skip matplotlib sweep chart |
| `--dry-run` | off | Validate all configs only; skip generation |

**Parameter sweep**

| Axis | Values |
|---|---|
| Slot counts | 24, 36, 48, 60, 72 |
| Lamination counts | 100, 200, 400 |
| Total jobs | 15 |

Winding type and slot shape are assigned per slot count:

| Slots | Winding | Shape |
|---|---|---|
| 24 | `CONCENTRATED` | `RECTANGULAR` |
| 36 | `DOUBLE_LAYER` | `SEMI_CLOSED` |
| 48 | `DOUBLE_LAYER` | `SEMI_CLOSED` |
| 60 | `DISTRIBUTED` | `TRAPEZOIDAL` |
| 72 | `DISTRIBUTED` | `TRAPEZOIDAL` |

**Expected console output**

```
Building parameter sweep …  15 jobs

Validating all configurations …
  s24_lam100  OK
  s24_lam200  OK
  ...
  s72_lam400  OK

Running batch …
[████████████████████████████] 100.0%  15/15  eta 0s  (s72_lam400)

JOB_ID        SLOTS  LAM  FILL%  STACK mm  YOKE mm  OK
s24_lam100       24  100   XX.X      35.0     XX.X   ✓
s24_lam200       24  200   XX.X      70.0     XX.X   ✓
s24_lam400       24  400   XX.X     140.0     XX.X   ✓
s36_lam100       36  100   XX.X      35.0     XX.X   ✓
...
s72_lam400       72  400   XX.X     140.0     XX.X   ✓

Batch summary → /tmp/stator_batch/batch_summary.json
Sweep chart   → /tmp/stator_batch/batch_sweep_chart.png
```

**Output files**

| File | Description |
|---|---|
| `batch_summary.json` | Array of all 15 job results (success, stem, paths, derived fields) |
| `batch_sweep_chart.png` | Two-panel chart: fill factor vs slot count; yoke height vs slot count |
| `stator_XXXXXXXX_meta.json` × 15 | Per-job metadata |

**Sweep chart panels**

- **Panel 1** — Fill factor (%) vs slot count.  One line per lamination count.
  Shows how slot geometry changes with slot count for a fixed radial envelope.
- **Panel 2** — Yoke height (mm) vs slot count.  Scatter plot coloured by
  lamination count.  Illustrates trade-off between slot depth and back-iron.

---

## Tests

```bash
# Install dev dependencies then run the full suite
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite covers ~140 cases across all modules:

| Test class | Scope |
|---|---|
| `TestParams` | Default construction, field values, enum members |
| `TestValidation` | All 15 geometric constraint rules (pass and fail) |
| `TestDerivedFields` | Derived field formulas |
| `TestSlotShapes` | All four slot shapes produce valid params |
| `TestWindingTypes` | All four winding types validate |
| `TestMaterials` | All material grades including CUSTOM with/without file |
| `TestSHA256` | Known-answer hash tests, determinism, output format |
| `TestGmshBackend` | StubGmshBackend tag counters, booleans, physical groups |
| `TestGeometryBuilder` | All shapes, single/double-layer, slot count, tag assignment |
| `TestTopologyRegistry` | Registration, winding assignment, thread safety (6 threads × 6 slots) |
| `TestMeshGenerator` | Successful generation, graceful failure on bad geometry |
| `TestExportEngine` | Stem determinism, JSON structure |
| `TestPipeline` | validate_config, generate_single, generate_batch, progress callback |
| `TestPublicAPI` | All expected names present in `__all__` |

**Expected output (abridged)**

```
========================= test session starts ==========================
platform linux -- Python 3.12.x, pytest-8.x.x
collected 140 items

tests/test_stator.py::TestParams::test_default_construction    PASSED
tests/test_stator.py::TestParams::test_all_enums               PASSED
tests/test_stator.py::TestValidation::test_valid_default        PASSED
tests/test_stator.py::TestValidation::test_rule7_coil_depth    PASSED
...
tests/test_stator.py::TestPipeline::test_generate_batch_progress_callback PASSED

========================= 140 passed in X.XXs ==========================
```
