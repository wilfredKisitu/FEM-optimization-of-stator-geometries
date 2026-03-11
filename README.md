# Stator Mesh Construction Pipeline

A high-performance C++20 library for parametric finite-element mesh generation of electric motor stator geometries. The pipeline covers the full chain from dimensioned parameters through geometry construction, physical-group assignment, mesh sizing, and multi-format file export. A pybind11 Python layer exposes the complete API for integration with optimisation loops (genetic algorithms, Bayesian optimisation, etc.).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Layout](#repository-layout)
3. [Build Instructions](#build-instructions)
4. [C++ Components](#c-components)
   - [StatorParams](#statorparams)
   - [IGmshBackend / StubGmshBackend](#igmshbackend--stubgmshbackend)
   - [GeometryBuilder](#geometrybuilder)
   - [TopologyRegistry](#topologyregistry)
   - [MeshGenerator](#meshgenerator)
   - [ExportEngine](#exportengine)
   - [BatchScheduler](#batchscheduler)
5. [Python API](#python-api)
   - [StatorConfig](#statorconfig)
   - [generate_single](#generate_single)
   - [generate_batch](#generate_batch)
   - [StatorVisualiser](#statorvisualiser)
   - [Low-level _stator_core bindings](#low-level-_stator_core-bindings)
6. [Enumerations Reference](#enumerations-reference)
7. [Output File Formats](#output-file-formats)
8. [Testing](#testing)
9. [Examples](#examples)

---

## Architecture Overview

```
Python caller (optimiser / script)
        │
        │  pybind11 (GIL released during C++ execution)
        ▼
BatchScheduler  ──fork()──►  worker process 1
                ──fork()──►  worker process 2  ...
                             │
                             ▼
                   [validate params]
                         │
                         ▼
                   GeometryBuilder   (GMSH OCC kernel)
                         │
                         ▼
                   TopologyRegistry  (thread-safe, shared_mutex)
                         │
                         ▼
                   MeshGenerator     (3-layer size-field strategy)
                         │
                         ▼
                   ExportEngine      (async: .msh / .vtk / .h5 / .json)
```

The core library always compiles. When `STATOR_WITH_GMSH=OFF` (the default for CI/tests) it substitutes `StubGmshBackend`, which records all API calls and allows the full pipeline to be unit-tested without a GMSH installation.

---

## Repository Layout

```
FEM/
├── CMakeLists.txt
├── include/
│   └── stator/
│       ├── params.hpp            # StatorParams struct + enums
│       ├── gmsh_backend.hpp      # IGmshBackend interface + StubGmshBackend
│       ├── geometry_builder.hpp  # GeometryBuilder + SlotProfile
│       ├── topology_registry.hpp # TopologyRegistry + RegionType
│       ├── mesh_generator.hpp    # MeshGenerator + MeshConfig + MeshResult
│       ├── export_engine.hpp     # ExportEngine + ExportFormat + sha256
│       └── batch_scheduler.hpp   # BatchScheduler + BatchJob + BatchResult
├── src/
│   ├── params.cpp
│   ├── gmsh_backend.cpp
│   ├── geometry_builder.cpp
│   ├── topology_registry.cpp
│   ├── mesh_generator.cpp
│   ├── export_engine.cpp
│   └── batch_scheduler.cpp
├── bindings/
│   └── python_bindings.cpp       # pybind11 module _stator_core
├── python/
│   └── stator_pipeline/
│       ├── __init__.py
│       ├── pipeline.py           # generate_single, generate_batch, StatorConfig
│       ├── params.py             # (reserved)
│       └── visualiser.py         # StatorVisualiser
├── tests/
│   └── test_stator.cpp           # 143 hand-rolled unit/integration tests
└── examples/
    ├── single_geometry.py
    └── batch_ga_integration.py
```

---

## Build Instructions

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.22 |
| GCC or Clang | GCC 13 / Clang 16 (`-std=c++20`) |
| GMSH *(optional)* | 4.11 |
| Intel TBB *(optional)* | 2021 |
| HDF5 + HighFive *(optional)* | HDF5 1.12 |
| pybind11 *(optional)* | 2.11 |

### Stub build (no GMSH required — default)

```bash
cmake -B build
cmake --build build --parallel
```

This produces `build/libstator_core.so` and (if `STATOR_BUILD_TESTS=ON`) `build/test_stator`.

### Full build with all optional components

```bash
cmake -B build \
  -DSTATOR_WITH_GMSH=ON \
  -DSTATOR_WITH_TBB=ON \
  -DSTATOR_WITH_HDF5=ON \
  -DSTATOR_WITH_PYTHON=ON
cmake --build build --parallel
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `STATOR_WITH_GMSH` | `OFF` | Link against the real GMSH C++ API |
| `STATOR_WITH_TBB` | `OFF` | Enable Intel TBB for parallel topology work |
| `STATOR_WITH_HDF5` | `OFF` | Enable HDF5 mesh export via HighFive |
| `STATOR_WITH_PYTHON` | `OFF` | Build the `_stator_core` pybind11 module |
| `STATOR_BUILD_TESTS` | `ON` | Build `test_stator` executable |

### Installing the Python package

```bash
# After building with -DSTATOR_WITH_PYTHON=ON:
pip install -e python/
# or
python python/setup.py install
```

---

## C++ Components

All symbols live in the `stator` namespace. All units are SI (metres, radians) unless noted.

---

### StatorParams

**Header:** `include/stator/params.hpp`

Plain-old-data struct representing the complete parametric description of a stator cross-section. Set fields then call `validate_and_derive()` before passing the struct to any other component.

#### Fields

**Section 1 — Core radii & air gap**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `R_outer` | `double` | `0.25` | Outer radius of the stator lamination (m) |
| `R_inner` | `double` | `0.15` | Inner (bore-facing) radius (m) |
| `airgap_length` | `double` | `0.001` | Radial air-gap between rotor and stator bore (m) |

**Section 2 — Slot geometry**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_slots` | `int` | `36` | Number of stator slots |
| `slot_depth` | `double` | `0.06` | Radial depth of each slot (m) |
| `slot_width_outer` | `double` | `0.012` | Slot width at the outer (yoke) edge (m) |
| `slot_width_inner` | `double` | `0.010` | Slot width at the bore edge (m) |
| `slot_opening` | `double` | `0.004` | Mouth opening width (SEMI_CLOSED only) (m) |
| `slot_opening_depth` | `double` | `0.003` | Depth of the slot mouth region (m) |
| `tooth_tip_angle` | `double` | `0.1` | Tooth-tip chamfer angle (radians) |
| `slot_shape` | `SlotShape` | `SEMI_CLOSED` | Cross-sectional slot profile |

**Section 3 — Coil / winding**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `coil_depth` | `double` | `0.05` | Radial depth of the coil conductor region (m) |
| `coil_width_outer` | `double` | `0.008` | Coil width at outer edge (m) |
| `coil_width_inner` | `double` | `0.007` | Coil width at inner edge (m) |
| `insulation_thickness` | `double` | `0.001` | Thickness of slot insulation wrap (m) |
| `turns_per_coil` | `int` | `10` | Number of conductor turns per coil |
| `coil_pitch` | `int` | `5` | Coil pitch in slot numbers |
| `wire_diameter` | `double` | `0.001` | Individual wire diameter (m) |
| `slot_fill_factor` | `double` | `0.45` | Target copper fill ratio (0–1); used for validation cross-check |
| `winding_type` | `WindingType` | `DOUBLE_LAYER` | Winding arrangement |

**Section 4 — Lamination stack**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `t_lam` | `double` | `0.00035` | Single lamination sheet thickness (m) |
| `n_lam` | `int` | `200` | Number of lamination sheets |
| `z_spacing` | `double` | `0.0` | Axial gap between sheets (m) |
| `insulation_coating_thickness` | `double` | `0.00005` | Inter-lamination coating thickness (m) |
| `material` | `LaminationMaterial` | `M270_35A` | Electrical steel grade |
| `material_file` | `std::string` | `""` | Path to custom B-H curve file (used when `material=CUSTOM`) |

**Section 5 — Mesh sizing**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mesh_yoke` | `double` | `0.006` | Target element size in the yoke region (m) |
| `mesh_slot` | `double` | `0.003` | Target element size in slot air regions (m) |
| `mesh_coil` | `double` | `0.0015` | Target element size in coil conductor regions (m) |
| `mesh_ins` | `double` | `0.0007` | Target element size in insulation regions (m) |
| `mesh_boundary_layers` | `int` | `3` | Number of boundary-layer inflation rows at the bore |
| `mesh_curvature` | `double` | `0.3` | Curvature-based refinement factor (0–1) |
| `mesh_transition_layers` | `int` | `2` | Number of transition element layers between regions |

**Section 6 — Derived fields (read-only after `validate_and_derive()`)**

| Field | Formula | Description |
|-------|---------|-------------|
| `yoke_height` | `R_outer - R_inner - slot_depth` | Radial height of the back-iron yoke (m) |
| `tooth_width` | `R_inner * slot_pitch - slot_width_inner` | Tooth width at the bore (m) |
| `slot_pitch` | `2π / n_slots` | Angular pitch between slot centres (radians) |
| `stack_length` | `n_lam * t_lam + (n_lam-1) * z_spacing` | Total axial stack length (m) |
| `fill_factor` | `coil_area / slot_area` | Computed copper fill fraction |

#### Methods

```cpp
void validate_and_derive();
```
Validates all 16 geometric constraint rules and computes the 5 derived fields. Throws `std::invalid_argument` or `std::logic_error` with a descriptive message on failure. Must be called before passing a `StatorParams` to any other component.

**Validation rules enforced:**
- `R_outer > R_inner > 0`
- `airgap_length > 0` and `airgap_length < R_inner`
- `n_slots >= 3`
- `slot_depth > 0` and `slot_depth < (R_outer - R_inner)`
- `slot_width_inner > 0`, `slot_width_outer > 0`
- `slot_opening > 0` and `slot_opening < slot_width_inner`
- `coil_depth > 0` and `coil_depth < slot_depth`
- `insulation_thickness > 0`
- `turns_per_coil >= 1`
- `wire_diameter > 0`
- `slot_fill_factor` in `(0, 1)`
- `t_lam > 0`, `n_lam >= 1`
- `yoke_height > 0` (derived check)
- `tooth_width > 0` (derived check)
- `fill_factor` in `(0, 1)` (derived check)

```cpp
std::string to_json() const;
```
Returns a single-line JSON string with all user-settable fields plus a `"_derived"` sub-object. Used as input to SHA-256 for deterministic output file naming.

#### Factory functions

```cpp
StatorParams make_reference_params();
```
Returns a validated 36-slot SEMI_CLOSED DOUBLE_LAYER reference design (the default field values). Used as a test baseline.

```cpp
StatorParams make_minimal_params();
```
Returns a validated 12-slot RECTANGULAR SINGLE_LAYER minimal design suitable for quick integration tests.

---

### IGmshBackend / StubGmshBackend

**Header:** `include/stator/gmsh_backend.hpp`

Abstract interface decoupling all geometry code from the GMSH C++ library. The same `GeometryBuilder` / `MeshGenerator` code operates against either backend.

#### IGmshBackend — pure virtual interface

```cpp
// Session lifecycle
void initialize(const std::string& model_name);
void synchronize();
void finalize();
void set_option(const std::string& name, double value);

// OCC geometry primitives — return new entity tag
int add_point(double x, double y, double z, double mesh_size = 0.0);
int add_line(int start, int end);
int add_circle(double cx, double cy, double cz, double radius);
int add_arc(int start, int centre, int end);
int add_curve_loop(const std::vector<int>& tags);
int add_plane_surface(const std::vector<int>& loop_tags);

// Boolean operations — return surviving (dim, tag) pairs
std::vector<std::pair<int,int>> boolean_cut(objects, tools, remove_tool=true);
std::vector<std::pair<int,int>> boolean_fragment(objects, tools);

// Physical groups
int add_physical_group(int dim, const std::vector<int>& tags,
                        const std::string& name, int tag = -1);

// Mesh size fields — return new field tag
int  add_math_eval_field(const std::string& expr);
int  add_constant_field(double value, const std::vector<int>& surfaces);
void set_background_field(int field_tag);

// Mesh generation
void generate_mesh(int dim);
void write_mesh(const std::string& filename);
std::vector<std::pair<int,int>> get_entities_2d();
```

#### StubGmshBackend

Fully functional in-process stub. All primitive calls auto-increment counters and return monotonically increasing tags. `boolean_cut` returns objects unchanged; `boolean_fragment` returns the union. No files are written by `write_mesh`.

**Inspection methods (available in tests):**

```cpp
int  point_count()          const;  // total add_point calls
int  line_count()           const;  // total add_line + add_arc calls
int  surface_count()        const;  // total add_plane_surface calls
int  field_count()          const;  // total add_*_field calls
int  physical_group_count() const;
bool was_initialized()      const;
bool was_synchronized()     const;
bool was_finalized()        const;
int  sync_count()           const;
bool mesh_generated()       const;
int  background_field()     const;
const std::string& last_write_path() const;
const std::vector<PhysGroupRecord>& physical_groups() const;

void reset();  // clear all state; reuse between test cases
```

#### Factory

```cpp
std::unique_ptr<IGmshBackend> make_default_backend();
```
Returns `RealGmshBackend` when compiled with `-DSTATOR_WITH_GMSH=ON`, otherwise returns `StubGmshBackend` (with a stderr warning).

---

### GeometryBuilder

**Header:** `include/stator/geometry_builder.hpp`

Constructs the complete 2-D stator cross-section in the GMSH OCC kernel.

#### Construction

```cpp
explicit GeometryBuilder(IGmshBackend* backend);
```
Throws `std::invalid_argument` if `backend` is null.

#### Primary method

```cpp
GeometryBuildResult build(const StatorParams& p);
```

Executes the full geometry build sequence:

1. Add outer circle (`R_outer`) and inner (bore) circle (`R_inner`)
2. Create the annular yoke surface via `add_plane_surface` with both curve loops
3. For each of the `n_slots` slots, call the shape-specific builder (see below) to produce a `SlotProfile`
4. Boolean-cut all slot cavities out of the yoke surface
5. Call `build_coil_inside_slot()` and `build_insulation()` for each slot
6. `synchronize()` the backend

Returns a `GeometryBuildResult`. On failure, `result.success == false` and `result.error_message` describes the problem.

**GeometryBuildResult fields:**

| Field | Description |
|-------|-------------|
| `success` | `true` if geometry was built without errors |
| `error_message` | Non-empty string on failure |
| `yoke_surface` | GMSH surface tag of the trimmed back-iron yoke |
| `bore_curve` | GMSH tag of the inner bore circle |
| `outer_curve` | GMSH tag of the outer circle |
| `slots` | `std::vector<SlotProfile>` — one entry per slot |

**SlotProfile fields:**

| Field | Description |
|-------|-------------|
| `slot_idx` | Zero-based slot index |
| `slot_surface` | GMSH tag of the main slot air-cavity surface |
| `coil_upper_sf` | GMSH tag of the upper (or only) coil conductor surface |
| `coil_lower_sf` | GMSH tag of the lower coil surface; `-1` for SINGLE_LAYER |
| `ins_upper_sf` | GMSH tag of the insulation surface around the upper coil |
| `ins_lower_sf` | GMSH tag of the insulation surface around the lower coil |
| `mouth_curve_bot` | GMSH tag of the bore-facing slot-mouth edge (boundary-layer seed) |
| `mouth_curve_top` | GMSH tag of the inner edge of the slot body (SEMI_CLOSED only) |
| `angle` | Rotation angle of this slot's centre (radians) |

#### Slot shape builders

All slot builders operate in a local frame (slot axis on +x) then rotate by `slot_angle(k, n_slots) = 2π*k/n_slots` before adding to GMSH.

| Shape | Points | Notes |
|-------|--------|-------|
| `RECTANGULAR` | 4 corners | Uniform width = `slot_width_outer` |
| `TRAPEZOIDAL` | 4 corners | Inner width `slot_width_inner`, outer `slot_width_outer` |
| `ROUND_BOTTOM` | 4 points + 2 arc centres | Bottom edge replaced by circular arc |
| `SEMI_CLOSED` | 6 points | Adds a narrow mouth region of width `slot_opening` and depth `slot_opening_depth` |

#### Static helpers

```cpp
static std::pair<double,double> rotate(double x, double y, double theta) noexcept;
```
Rotates a 2-D point by `theta` radians. Returns `(x cos θ − y sin θ, x sin θ + y cos θ)`.

---

### TopologyRegistry

**Header:** `include/stator/topology_registry.hpp`

Thread-safe registry that maps GMSH entity tags to named physical regions. Uses `std::shared_mutex` — multiple concurrent readers, exclusive writers.

#### Construction

```cpp
explicit TopologyRegistry(int n_slots);
```

#### Registration (write-locked)

```cpp
void register_surface(RegionType type, int gmsh_tag, int slot_idx = -1);
```
Associates a GMSH surface tag with a `RegionType`. `slot_idx = -1` for non-slot regions (e.g. `YOKE`).

```cpp
void register_slot_coil(int slot_idx, int upper_tag, int lower_tag = -1);
```
Records coil surface tags for a slot. `lower_tag = -1` for SINGLE_LAYER.

```cpp
void register_boundary_curve(RegionType type, int gmsh_curve);
```
Records a 1-D boundary curve (e.g. `BOUNDARY_BORE`, `BOUNDARY_OUTER`).

#### Winding assignment

```cpp
void assign_winding_layout(WindingType wt);
```
Computes the 3-phase winding assignment for all registered slots and stores one `SlotWindingAssignment` per slot. Throws `std::logic_error` if coils have not been registered first.

**Phase sequences:**

| WindingType | Slot % 6 pattern (upper phase) |
|-------------|-------------------------------|
| `DISTRIBUTED` | A+, B−, C+, A−, B+, C− |
| `CONCENTRATED` | A+, A−, B+, B−, C+, C− |
| `SINGLE_LAYER` | A+, B+, C+, A+, B+, C+ (repeating, lower = UNKNOWN) |
| `DOUBLE_LAYER` | Same as DISTRIBUTED with both upper and lower assigned |

#### Queries (read-locked)

```cpp
std::vector<int> get_surfaces(RegionType type) const;
```
Returns all GMSH surface tags registered under `type`.

```cpp
std::vector<int> get_boundary_curves(RegionType type) const;
```
Returns boundary curve tags for the given region type.

```cpp
const SlotWindingAssignment& get_slot_assignment(int slot_idx) const;
const std::vector<SlotWindingAssignment>& get_winding_assignments() const;
```
Access the computed winding assignments (available after `assign_winding_layout()`).

```cpp
int  total_registered_surfaces() const;
bool winding_assigned() const noexcept;
void dump(std::ostream& os) const;  // diagnostic print
```

#### SlotWindingAssignment fields

| Field | Description |
|-------|-------------|
| `slot_idx` | Zero-based slot index |
| `upper_phase` | `RegionType` for the upper coil conductor |
| `lower_phase` | `RegionType` for the lower coil; `UNKNOWN` for SINGLE_LAYER |
| `upper_tag` | GMSH surface tag of the upper coil |
| `lower_tag` | GMSH surface tag of the lower coil; `-1` for SINGLE_LAYER |

---

### MeshGenerator

**Header:** `include/stator/mesh_generator.hpp`

Assigns GMSH physical groups and generates the 2-D (and optionally 3-D extruded) mesh using a three-layer size-field strategy.

#### MeshConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `algorithm_2d` | `int` | `5` | GMSH 2-D algorithm (5=Delaunay, 6=Frontal-Delaunay, 8=Delaunay-quads) |
| `algorithm_3d` | `int` | `10` | GMSH 3-D algorithm (10=HXT, 1=Delaunay, 4=Frontal) |
| `smoothing_passes` | `int` | `3` | Laplacian smoothing iterations after meshing |
| `optimiser` | `std::string` | `"Netgen"` | Post-meshing optimiser name |
| `min_quality_threshold` | `double` | `0.3` | Minimum acceptable element quality (0–1); elements below this trigger warnings |
| `periodic` | `bool` | `false` | Enable periodic meshing (sector periodicity) |
| `layers_per_lam` | `int` | `2` | Number of element layers per lamination in 3-D extrusion |

#### Construction

```cpp
MeshGenerator(IGmshBackend* backend, const MeshConfig& config = {});
```

#### Primary method

```cpp
MeshResult generate(const StatorParams& p,
                    const GeometryBuildResult& geo,
                    TopologyRegistry& registry);
```

Full meshing pipeline:
1. Calls `assign_physical_groups()` — registers YOKE, BOUNDARY_BORE, BOUNDARY_OUTER, SLOT_AIR, SLOT_INS, COIL_A/B/C ±
2. Applies **Layer A** constant size fields per region type
3. Applies **Layer B** mouth-transition Distance+Threshold field
4. Applies **Layer C** bore boundary-layer field
5. Combines all fields into a Min field and sets it as the background
6. Sets `Mesh.Algorithm` and `Mesh.Smoothing`
7. Calls `generate_mesh(2)` for 2-D triangulation
8. If `n_lam > 1`, calls `generate_mesh(3)` for 3-D extrusion

Returns a `MeshResult`.

**MeshResult fields:**

| Field | Description |
|-------|-------------|
| `success` | `true` if meshing completed without errors |
| `error_message` | Non-empty on failure |
| `n_nodes` | Total node count |
| `n_elements_2d` | 2-D triangle / quad element count |
| `n_elements_3d` | 3-D element count (0 for 2-D-only meshes) |
| `min_quality` | Minimum element quality (scaled Jacobian, 0–1) |
| `avg_quality` | Mean element quality |
| `n_phys_groups` | Number of assigned physical groups |

#### Size-field layers

| Layer | Method | Effect |
|-------|--------|--------|
| A | `add_constant_fields()` | Per-surface constant size: YOKE→`mesh_yoke`, SLOT_AIR→`mesh_slot`, coils→`mesh_coil`, insulation→`mesh_ins` |
| B | `add_mouth_transition_fields()` | Distance+Threshold field grading element size from `mesh_slot` to `mesh_yoke` across the slot-mouth transition zone |
| C | `add_bore_boundary_layer()` | Boundary-layer inflation field seeded from the bore curve with `mesh_boundary_layers` inflation rows and ratio 1.2 |

Physical group canonical integer tags (matching `RegionType` values):

| Region | Tag |
|--------|-----|
| YOKE | 100 |
| TOOTH | 101 |
| SLOT_AIR | 200 |
| SLOT_INS | 201 |
| COIL_A_POS | 301 |
| COIL_A_NEG | 302 |
| COIL_B_POS | 303 |
| COIL_B_NEG | 304 |
| COIL_C_POS | 305 |
| COIL_C_NEG | 306 |
| BORE_AIR | 400 |
| BOUNDARY_BORE | 500 |
| BOUNDARY_OUTER | 501 |

---

### ExportEngine

**Header:** `include/stator/export_engine.hpp`

Writes mesh output in up to four formats, each in a separate `std::async` task.

#### Construction

```cpp
explicit ExportEngine(IGmshBackend* backend);
```
Throws `std::invalid_argument` if `backend` is null.

#### ExportFormat bitmask

```cpp
enum class ExportFormat : uint32_t {
    NONE = 0,
    MSH  = 1 << 0,   // GMSH native mesh
    VTK  = 1 << 1,   // VTK legacy ASCII
    HDF5 = 1 << 2,   // HDF5 / HighFive (placeholder when STATOR_WITH_HDF5=OFF)
    JSON = 1 << 3,   // metadata JSON
    ALL  = MSH | VTK | HDF5 | JSON
};
```
Combine formats with `|`: e.g. `ExportFormat::MSH | ExportFormat::JSON`.

#### ExportConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `formats` | `ExportFormat` | `ALL` | Bitmask of formats to write |
| `output_dir` | `std::string` | `"."` | Directory for output files |
| `msh_version` | `int` | `4` | GMSH mesh format version (2 or 4) |

#### Static methods

```cpp
static std::string compute_stem(const StatorParams& p);
```
Computes `"stator_" + sha256(p.to_json()).substr(0, 8)`. All output files for a given `StatorParams` share this stem, making outputs deterministically named and idempotent.

```cpp
static bool outputs_exist(const StatorParams& p, const ExportConfig& cfg);
```
Returns `true` if every file implied by `cfg.formats` already exists on disk. Used by `BatchScheduler` to skip jobs when `skip_existing=true`.

#### Write methods

```cpp
std::vector<ExportResult> write_all_sync(const StatorParams& p,
                                          const MeshResult& mesh,
                                          const ExportConfig& cfg);
```
Launches all format writers asynchronously then blocks until all complete. Returns one `ExportResult` per requested format.

```cpp
std::vector<std::future<ExportResult>> write_all_async(const StatorParams& p,
                                                        const MeshResult& mesh,
                                                        const ExportConfig& cfg);
```
Launches writers and returns futures immediately. Caller must join all futures before calling `backend->finalize()`.

**ExportResult fields:**

| Field | Description |
|-------|-------------|
| `success` | `true` if the file was written without error |
| `format` | Which format this result describes |
| `path` | Absolute path to the written file |
| `error_message` | Non-empty on failure |
| `write_time_ms` | Wall-clock write time in milliseconds |

#### Output filenames

| Format | Filename pattern |
|--------|-----------------|
| MSH | `<output_dir>/stator_<hash8>.msh` |
| VTK | `<output_dir>/stator_<hash8>.vtk` |
| HDF5 | `<output_dir>/stator_<hash8>.h5` |
| JSON | `<output_dir>/stator_<hash8>_meta.json` |

#### SHA-256 utility

```cpp
std::string sha256(const std::string& data);
```
Self-contained FIPS 180-4 SHA-256 implementation. Returns a 64-character lowercase hex string. Verifiable: `sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"`.

---

### BatchScheduler

**Header:** `include/stator/batch_scheduler.hpp`

Fork-based parallel job runner. Spawns one child process per job up to `max_parallel`, polls with `WNOHANG`, applies `SIGKILL` on timeout, and writes a `batch_summary.json` on completion.

#### BatchJob fields

| Field | Type | Description |
|-------|------|-------------|
| `params` | `StatorParams` | Must have `validate_and_derive()` called before passing |
| `export_config` | `ExportConfig` | Output directory and format mask |
| `mesh_config` | `MeshConfig` | Mesh algorithm settings |
| `job_id` | `std::string` | Caller-assigned identifier; echoed in status JSON and `BatchResult` |

#### BatchSchedulerConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_parallel` | `int` | `0` | Max concurrent child processes; `0` = `std::thread::hardware_concurrency()` |
| `skip_existing` | `bool` | `true` | Skip jobs where all output files already exist |
| `job_timeout_sec` | `int` | `300` | Send `SIGKILL` to child after this many seconds; `0` = no timeout |
| `write_summary` | `bool` | `true` | Write `batch_summary.json` to `jobs[0].export_config.output_dir` |

#### Key methods

```cpp
void set_progress_callback(ProgressCallback cb);
```
Register a callback invoked after each job completes:
```cpp
using ProgressCallback = std::function<void(
    int jobs_done,       // number of jobs completed so far
    int jobs_total,      // total number of jobs
    bool success,        // whether the just-completed job succeeded
    const std::string& job_id  // job identifier
)>;
```

```cpp
std::vector<BatchResult> run(const std::vector<BatchJob>& jobs,
                              const BatchSchedulerConfig& config = {});
```
Blocks until all jobs complete (or are cancelled). Returns one `BatchResult` per job in input order. Releases the Python GIL when called from Python.

```cpp
void cancel();
```
Sets the cancellation flag. Running children receive `SIGTERM`; pending jobs are marked with `error = "cancelled"`.

```cpp
bool is_running() const noexcept;
```
Returns `true` while `run()` is executing.

#### BatchResult fields

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `std::string` | Echoed from `BatchJob::job_id` |
| `success` | `bool` | `true` if the full pipeline completed without error |
| `error` | `std::string` | Error message; empty on success |
| `msh_path` | `std::string` | Path to `.msh` output; empty if not requested or failed |
| `vtk_path` | `std::string` | Path to `.vtk` output |
| `hdf5_path` | `std::string` | Path to `.h5` output |
| `json_path` | `std::string` | Path to `_meta.json` output |

#### batch_summary.json format

Written to `<output_dir>/batch_summary.json`:

```json
[
  {
    "job_id": "run_001",
    "success": true,
    "error": "",
    "msh_path": "/out/stator_a1b2c3d4.msh"
  },
  ...
]
```

---

## Python API

Install the package (requires `_stator_core` built with `-DSTATOR_WITH_PYTHON=ON`):

```bash
pip install -e python/
```

Import:

```python
from stator_pipeline import StatorConfig, generate_single, generate_batch
from stator_pipeline.visualiser import StatorVisualiser
```

---

### StatorConfig

```python
from stator_pipeline import StatorConfig
```

Python `dataclass` mirroring `StatorParams`. All fields are keyword arguments with the same defaults as the C++ struct. All units are SI (metres, radians).

```python
@dataclass
class StatorConfig:
    # Section 1 — Core radii & air gap
    R_outer: float = 0.25
    R_inner: float = 0.15
    airgap_length: float = 0.001

    # Section 2 — Slot geometry
    n_slots: int = 36
    slot_depth: float = 0.06
    slot_width_outer: float = 0.012
    slot_width_inner: float = 0.010
    slot_opening: float = 0.004
    slot_opening_depth: float = 0.003
    tooth_tip_angle: float = 0.1
    slot_shape: str = "SEMI_CLOSED"       # "RECTANGULAR" | "TRAPEZOIDAL" | "ROUND_BOTTOM" | "SEMI_CLOSED"

    # Section 3 — Coil / winding
    coil_depth: float = 0.05
    coil_width_outer: float = 0.008
    coil_width_inner: float = 0.007
    insulation_thickness: float = 0.001
    turns_per_coil: int = 10
    coil_pitch: int = 5
    wire_diameter: float = 0.001
    slot_fill_factor: float = 0.45
    winding_type: str = "DOUBLE_LAYER"    # "SINGLE_LAYER" | "DOUBLE_LAYER" | "CONCENTRATED" | "DISTRIBUTED"

    # Section 4 — Lamination stack
    t_lam: float = 0.00035
    n_lam: int = 200
    z_spacing: float = 0.0
    insulation_coating_thickness: float = 0.00005
    material: str = "M270_35A"            # "M270_35A" | "M330_50A" | "M400_50A" | "NO20" | "CUSTOM"
    material_file: str = ""

    # Section 5 — Mesh sizing
    mesh_yoke: float = 0.006
    mesh_slot: float = 0.003
    mesh_coil: float = 0.0015
    mesh_ins: float = 0.0007
    mesh_boundary_layers: int = 3
    mesh_curvature: float = 0.3
    mesh_transition_layers: int = 2
```

**Enum fields** (`slot_shape`, `winding_type`, `material`) accept case-sensitive string names matching the C++ enum value names listed above.

---

### generate_single

```python
def generate_single(
    config: StatorConfig,
    output_dir: str,
    formats: str = "JSON|HDF5",
) -> dict
```

Generate a mesh for a single stator configuration.

**Arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `StatorConfig` | required | Stator geometry and mesh settings |
| `output_dir` | `str` | required | Directory where output files are written (created if absent) |
| `formats` | `str` | `"JSON|HDF5"` | `\|`-separated list of export formats: any combination of `MSH`, `VTK`, `HDF5`, `JSON`, or `ALL` |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `success` | `bool` | `True` if the full pipeline completed |
| `output_dir` | `str` | Echo of the requested output directory |
| `job_id` | `str` | `"single"` |
| `error` | `str` | Error message; empty on success |
| `msh_path` | `str` | Path to `.msh` file (empty if not requested) |
| `vtk_path` | `str` | Path to `.vtk` file |
| `hdf5_path` | `str` | Path to `.h5` file |
| `json_path` | `str` | Path to `_meta.json` file |

**Raises:** `ImportError` if `_stator_core` is not available (library not built).

**Example:**

```python
from stator_pipeline import StatorConfig, generate_single

cfg = StatorConfig(n_slots=24, R_outer=0.20, R_inner=0.12, slot_shape="TRAPEZOIDAL")
result = generate_single(cfg, output_dir="/tmp/stator_out", formats="MSH|VTK|JSON")

if result["success"]:
    print("Mesh written to:", result["msh_path"])
else:
    print("Failed:", result["error"])
```

---

### generate_batch

```python
def generate_batch(
    configs: list[StatorConfig],
    output_dir: str,
    max_parallel: int = 0,
    formats: str = "MSH|VTK|HDF5|JSON",
    progress_callback: callable | None = None,
    skip_existing: bool = True,
    job_timeout_sec: int = 300,
) -> list[dict]
```

Generate meshes for a list of stator configurations in parallel using fork-based workers.

**Arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `configs` | `list[StatorConfig]` | required | One `StatorConfig` per job |
| `output_dir` | `str` | required | Common output directory for all jobs |
| `max_parallel` | `int` | `0` | Maximum worker processes; `0` = CPU count |
| `formats` | `str` | `"MSH\|VTK\|HDF5\|JSON"` | Export formats (see `generate_single`) |
| `progress_callback` | `callable` or `None` | `None` | Called after each job: `fn(jobs_done, jobs_total, success, job_id)` |
| `skip_existing` | `bool` | `True` | Skip jobs where output files already exist (idempotent re-runs) |
| `job_timeout_sec` | `int` | `300` | Kill worker after this many seconds; `0` = no timeout |

**Returns:** `list[dict]`, one entry per input config, in order. Each dict has the same keys as `generate_single`'s return value, plus `job_id` set to `"batch_<index>"`.

**Note:** The Python GIL is released for the duration of the C++ batch execution. The `progress_callback` is called from the C++ layer — ensure it is thread-safe if it mutates shared state.

**Example:**

```python
from stator_pipeline import StatorConfig, generate_batch

configs = [
    StatorConfig(n_slots=24, slot_depth=0.05 + i * 0.005)
    for i in range(8)
]

def on_progress(done, total, ok, job_id):
    print(f"[{done}/{total}] {job_id} {'OK' if ok else 'FAIL'}")

results = generate_batch(
    configs,
    output_dir="/tmp/batch_out",
    max_parallel=4,
    formats="MSH|JSON",
    progress_callback=on_progress,
)

passed = sum(1 for r in results if r["success"])
print(f"{passed}/{len(results)} jobs succeeded")
```

---

### StatorVisualiser

```python
from stator_pipeline.visualiser import StatorVisualiser
```

Renders stator cross-section and mesh quality plots from VTK output files. Requires `matplotlib`. Optionally uses the `vtk` library for parsing; falls back to a built-in VTK legacy ASCII parser.

#### Construction

```python
vis = StatorVisualiser()
```

No arguments. Availability of optional libraries (`matplotlib`, `vtk`) is checked at construction time.

#### Methods

```python
def plot_cross_section(vtk_path: str, output_png: str | None = None) -> None
```
Renders the 2-D stator cross-section with cells coloured by region type.

| Parameter | Description |
|-----------|-------------|
| `vtk_path` | Path to the `.vtk` file written by `ExportEngine` |
| `output_png` | If given, save the figure to this PNG path instead of displaying interactively |

Raises `ImportError` if `matplotlib` is not installed.

```python
def plot_mesh(vtk_path: str, show_quality: bool = True) -> None
```
Renders mesh elements coloured by element quality scalar (green = good, red = poor).

| Parameter | Description |
|-----------|-------------|
| `vtk_path` | Path to the `.vtk` file |
| `show_quality` | If `True`, colour by element quality; if `False`, colour by region |

**Region colour map:**

| Region | Hex colour |
|--------|-----------|
| YOKE | `#4a90d9` (steel blue) |
| TOOTH | `#2c5f8a` (dark blue) |
| SLOT_AIR | `#b0d4f1` (light blue) |
| SLOT_INS | `#f5e642` (yellow) |
| COIL_A_POS | `#e63946` (red) |
| COIL_A_NEG | `#ff8fa3` (pink) |
| COIL_B_POS | `#2a9d8f` (teal) |
| COIL_B_NEG | `#80cdc1` (light teal) |
| COIL_C_POS | `#f4a261` (orange) |
| COIL_C_NEG | `#ffd6a5` (pale orange) |
| BORE_AIR | `#e8f4f8` (near-white) |
| BOUNDARY_BORE | `#264653` (dark teal) |
| BOUNDARY_OUTER | `#1a1a2e` (near-black) |

---

### Low-level _stator_core bindings

The `_stator_core` extension module exposes the C++ types directly for advanced usage.

#### Enumerations

```python
import _stator_core as core

core.SlotShape.RECTANGULAR
core.SlotShape.TRAPEZOIDAL
core.SlotShape.ROUND_BOTTOM
core.SlotShape.SEMI_CLOSED

core.WindingType.SINGLE_LAYER
core.WindingType.DOUBLE_LAYER
core.WindingType.CONCENTRATED
core.WindingType.DISTRIBUTED

core.LaminationMaterial.M270_35A
core.LaminationMaterial.M330_50A
core.LaminationMaterial.M400_50A
core.LaminationMaterial.NO20
core.LaminationMaterial.CUSTOM

core.ExportFormat.NONE
core.ExportFormat.MSH
core.ExportFormat.VTK
core.ExportFormat.HDF5
core.ExportFormat.JSON
core.ExportFormat.ALL
```

#### StatorParams

```python
p = core.StatorParams()
p.R_outer = 0.20
p.n_slots  = 24
p.slot_shape = core.SlotShape.TRAPEZOIDAL
p.validate_and_derive()
json_str = p.to_json()
repr(p)  # "<StatorParams n_slots=24>"

# Read-only derived fields (after validate_and_derive):
p.yoke_height
p.tooth_width
p.slot_pitch
p.stack_length
p.fill_factor
```

#### ExportConfig

```python
cfg = core.ExportConfig()
cfg.formats    = core.ExportFormat.MSH | core.ExportFormat.JSON
cfg.output_dir = "/tmp/out"
cfg.msh_version = 4
```

#### MeshConfig

```python
mc = core.MeshConfig()
mc.algorithm_2d = 5
mc.smoothing_passes = 3
mc.periodic = False
```

#### BatchJob

```python
job = core.BatchJob()
job.params        = p          # core.StatorParams
job.export_config = cfg        # core.ExportConfig
job.mesh_config   = mc         # core.MeshConfig
job.job_id        = "my_job"
```

#### BatchScheduler

```python
sched = core.BatchScheduler()

# Optional progress callback
def cb(done, total, ok, job_id):
    print(done, total, ok, job_id)
sched.set_progress_callback(cb)

# Run — GIL is released during execution
results = sched.run(jobs, sched_cfg)   # list[BatchResult]

sched.cancel()
sched.is_running()  # bool
```

#### BatchSchedulerConfig

```python
sched_cfg = core.BatchSchedulerConfig()
sched_cfg.max_parallel    = 4
sched_cfg.skip_existing   = True
sched_cfg.job_timeout_sec = 120
sched_cfg.write_summary   = True
```

#### BatchResult (read-only)

```python
r = results[0]
r.job_id     # str
r.success    # bool
r.error      # str
r.msh_path   # str
r.vtk_path   # str
r.hdf5_path  # str
r.json_path  # str
```

#### Free functions

```python
p = core.make_reference_params()  # validated 36-slot reference design
p = core.make_minimal_params()    # validated 12-slot minimal design
h = core.sha256("hello")          # 64-char hex SHA-256 digest
```

---

## Enumerations Reference

### SlotShape

| Value | C++ | Python string | Description |
|-------|-----|---------------|-------------|
| `RECTANGULAR` | `SlotShape::RECTANGULAR` | `"RECTANGULAR"` | All four corners at 90°; uniform width `slot_width_outer` |
| `TRAPEZOIDAL` | `SlotShape::TRAPEZOIDAL` | `"TRAPEZOIDAL"` | Narrower at bore (`slot_width_inner`), wider at yoke |
| `ROUND_BOTTOM` | `SlotShape::ROUND_BOTTOM` | `"ROUND_BOTTOM"` | Trapezoidal with circular arc at the bottom |
| `SEMI_CLOSED` | `SlotShape::SEMI_CLOSED` | `"SEMI_CLOSED"` | Narrow mouth opening (`slot_opening`) for reduced cogging torque |

### WindingType

| Value | Description |
|-------|-------------|
| `SINGLE_LAYER` | One coil conductor per slot; lower coil tag is `-1` |
| `DOUBLE_LAYER` | Two coil conductors per slot (upper/lower); uses DISTRIBUTED phase sequence |
| `CONCENTRATED` | Short-pitch concentrated winding; phase sequence A+A−B+B−C+C− |
| `DISTRIBUTED` | Distributed winding; phase sequence A+B−C+A−B+C− |

### LaminationMaterial

| Value | Grade | Typical loss at 1T/50Hz |
|-------|-------|------------------------|
| `M270_35A` | 0.35 mm high-grade | 2.70 W/kg |
| `M330_50A` | 0.50 mm standard | 3.30 W/kg |
| `M400_50A` | 0.50 mm economy | 4.00 W/kg |
| `NO20` | 0.20 mm non-oriented | ~2.0 W/kg |
| `CUSTOM` | User-supplied B-H curve | Loaded from `material_file` |

### ExportFormat (bitmask)

| Value | Bit | File extension | Notes |
|-------|-----|---------------|-------|
| `NONE` | 0 | — | No export |
| `MSH` | 0 | `.msh` | GMSH native; version controlled by `msh_version` |
| `VTK` | 1 | `.vtk` | Legacy ASCII VTK UNSTRUCTURED_GRID |
| `HDF5` | 2 | `.h5` | HighFive HDF5; placeholder text file when `STATOR_WITH_HDF5=OFF` |
| `JSON` | 3 | `_meta.json` | Mesh statistics + parameters + output file paths |
| `ALL` | — | all four | Shorthand for `MSH\|VTK\|HDF5\|JSON` |

---

## Output File Formats

### _meta.json

```json
{
  "params": { ... all StatorParams fields ... },
  "mesh_stats": {
    "n_nodes": 12345,
    "n_elements_2d": 23456,
    "n_elements_3d": 0,
    "min_quality": 0.52,
    "avg_quality": 0.81
  },
  "output_files": {
    "msh":  "/out/stator_a1b2c3d4.msh",
    "vtk":  "/out/stator_a1b2c3d4.vtk",
    "hdf5": "/out/stator_a1b2c3d4.h5",
    "json": "/out/stator_a1b2c3d4_meta.json"
  }
}
```

### batch_summary.json

```json
[
  { "job_id": "batch_0", "success": true,  "error": "", "msh_path": "/out/stator_a1b2c3d4.msh" },
  { "job_id": "batch_1", "success": false, "error": "Geometry build failed: ...", "msh_path": "" }
]
```

---

## Testing

The test suite uses a hand-rolled framework (no external test library required).

```bash
# Build (stub backend, no GMSH required)
cmake -B build && cmake --build build --parallel

# Run all tests
./build/test_stator

# Via CTest
ctest --test-dir build --output-on-failure
```

**Expected output:**
```
=== Results: PASS=143 FAIL=0 ===
```

Test groups and counts:

| Group | Tests | What is verified |
|-------|-------|-----------------|
| `[PARAMS]` | 31 | All 16 validation rules, derived field computation, JSON serialisation, factory functions |
| `[TOPOLOGY]` | 22 | Registration, winding phase sequences for all 4 types, thread-safety, boundary curves |
| `[GEOMETRY]` | 32 | All 4 slot shapes, coil placement (single/double), insulation, rotation, backend call counts |
| `[MESH]` | 16 | Physical group assignment, size-field layers, algorithm settings, 3-D extrusion path |
| `[EXPORT]` | 22 | SHA-256 correctness, stem derivation, all 4 format writers, async/sync consistency, `outputs_exist` |
| `[BATCH]` | 10 | Fork dispatch, cancel flag, `skip_existing`, timeout path, summary JSON |
| `[INTEGRATION]` | 10 | Full pipeline from `make_reference_params()` through export |

The stub-backend warnings (`[stator] Warning: STATOR_WITH_GMSH not defined`) are expected and do not indicate test failures.

---

## Examples

### Single geometry (Python)

```python
from stator_pipeline import StatorConfig, generate_single

# 24-slot TRAPEZOIDAL single-layer design
cfg = StatorConfig(
    n_slots=24,
    R_outer=0.18,
    R_inner=0.11,
    slot_depth=0.048,
    slot_shape="TRAPEZOIDAL",
    winding_type="SINGLE_LAYER",
    material="M330_50A",
    mesh_yoke=0.005,
    mesh_slot=0.002,
)

result = generate_single(cfg, output_dir="output/single", formats="MSH|VTK|JSON")
print(result)
```

### Batch sweep for a genetic algorithm

```python
from stator_pipeline import StatorConfig, generate_batch
import json

# Parameter sweep: vary slot_depth and slot_width_outer
population = [
    StatorConfig(slot_depth=0.04 + i*0.004, slot_width_outer=0.010 + j*0.001)
    for i in range(4) for j in range(4)
]

results = generate_batch(
    population,
    output_dir="output/sweep",
    max_parallel=8,
    formats="JSON",                        # metadata only — fast
    skip_existing=True,                    # idempotent across GA generations
    progress_callback=lambda d,t,ok,jid: print(f"{d}/{t} {jid}"),
)

# Extract fitness proxy: avg_quality from the metadata JSON
fitnesses = []
for r in results:
    if r["success"] and r["json_path"]:
        with open(r["json_path"]) as f:
            meta = json.load(f)
        fitnesses.append(meta["mesh_stats"]["avg_quality"])
    else:
        fitnesses.append(0.0)

print("Best quality:", max(fitnesses))
```

### Direct C++ usage

```cpp
#include "stator/params.hpp"
#include "stator/gmsh_backend.hpp"
#include "stator/geometry_builder.hpp"
#include "stator/topology_registry.hpp"
#include "stator/mesh_generator.hpp"
#include "stator/export_engine.hpp"

using namespace stator;

int main() {
    // 1. Parameters
    auto p = make_reference_params();   // already validated

    // 2. Backend (real GMSH or stub)
    auto backend = make_default_backend();

    // 3. Geometry
    backend->initialize("my_stator");
    GeometryBuilder builder(backend.get());
    auto geo = builder.build(p);
    if (!geo.success) { /* handle error */ }

    // 4. Topology
    TopologyRegistry registry(p.n_slots);

    // 5. Mesh
    MeshGenerator mesher(backend.get());
    auto mesh = mesher.generate(p, geo, registry);

    // 6. Export
    ExportConfig cfg;
    cfg.output_dir = "/tmp/stator_out";
    cfg.formats    = ExportFormat::MSH | ExportFormat::JSON;
    ExportEngine exporter(backend.get());
    auto results = exporter.write_all_sync(p, mesh, cfg);

    backend->finalize();
    return mesh.success ? 0 : 1;
}
```
