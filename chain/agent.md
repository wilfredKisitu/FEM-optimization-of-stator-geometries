# Stator Mesh Construction Pipeline — Implementation Specification

## Purpose of This Document

This README is a **complete implementation brief** intended for Claude Opus 4.6. It describes a C++20 mesh construction pipeline for parametric high-voltage asynchronous generator stator geometries. Every module, header, source file, test case, and build rule must be implemented exactly as specified. Do not summarise, skip, or defer any section — implement all of it in full.

---

## Project Overview

The pipeline accepts a set of geometric and meshing parameters describing one stator geometry instance, constructs its 2-D cross-section as a boundary-representation (B-rep) model inside a GMSH OCC session, generates a finite-element mesh with per-region size control, and exports the result in multiple formats (`.msh`, `.vtk`, `.h5`, `.json`). A batch scheduler uses `fork()`-based multi-processing to generate many geometries in parallel for use by a genetic algorithm (GA) optimiser. A Python wrapper built with pybind11 makes the entire pipeline usable from Python.

### Key design constraints

- **Language:** C++20 (gcc ≥ 13, `-std=c++20`)
- **Geometry/Mesh:** GMSH C++ API (`gmsh.h`) via OpenCASCADE kernel
- **Intra-process concurrency:** Intel oneTBB (`tbb::parallel_for`, `tbb::task_group`)
- **Inter-process parallelism:** POSIX `fork()` — one GMSH session per child process (GMSH is not thread-safe across processes)
- **Python binding:** pybind11 with `py::gil_scoped_release` on all blocking batch calls
- **Serialisation:** nlohmann/json for params/metadata; HDF5 via HighFive for mesh arrays
- **Testing:** self-contained test binary, no external testing framework dependency
- **All SI units** (metres, radians) throughout

---

## Directory Structure to Implement

```
stator_pipeline/
├── CMakeLists.txt                         # Full build system
├── README.md                              # This file
├── include/
│   └── stator/
│       ├── params.hpp                     # Parameter struct + validation
│       ├── topology_registry.hpp          # Physical region tag registry
│       ├── gmsh_backend.hpp               # GMSH abstraction interface + stub
│       ├── geometry_builder.hpp           # 2-D B-rep construction
│       ├── mesh_generator.hpp             # Mesh sizing + generation
│       ├── export_engine.hpp              # Multi-format async export
│       └── batch_scheduler.hpp            # Fork-based batch parallelism
├── src/
│   ├── params.cpp
│   ├── topology_registry.cpp
│   ├── gmsh_backend.cpp
│   ├── geometry_builder.cpp
│   ├── mesh_generator.cpp
│   ├── export_engine.cpp
│   └── batch_scheduler.cpp
├── bindings/
│   └── python_bindings.cpp               # pybind11 module
├── python/
│   └── stator_pipeline/
│       ├── __init__.py
│       ├── pipeline.py                   # Python-facing API
│       ├── params.py                     # Python dataclass mirroring StatorParams
│       └── visualiser.py                 # VTK/REST visualisation helpers
├── tests/
│   └── test_stator.cpp                   # All unit + integration tests
└── examples/
    ├── single_geometry.py
    └── batch_ga_integration.py
```

---

## Parameter Specification (Complete)

All parameters in this section must be represented as fields in `StatorParams`. They are grouped by physical domain. SI units throughout.

### Section 1 — Core Radii & Air Gap

| Field | Type | Unit | Description |
|---|---|---|---|
| `R_outer` | `double` | m | Outer radius of stator core (back of yoke) |
| `R_inner` | `double` | m | Inner (bore) radius of stator core, air-gap surface |
| `airgap_length` | `double` | m | Radial clearance between stator bore and rotor OD |

### Section 2 — Slot Geometry

| Field | Type | Unit | Description |
|---|---|---|---|
| `n_slots` | `int` | — | Number of slots; must be even and ≥ 6 |
| `slot_depth` | `double` | m | Radial depth of main slot body (excluding mouth region) |
| `slot_width_outer` | `double` | m | Width at yoke side (bottom) of the slot |
| `slot_width_inner` | `double` | m | Width at bore side (top) of the slot |
| `slot_opening` | `double` | m | Mouth opening width (required when `slot_shape == SEMI_CLOSED`) |
| `slot_opening_depth` | `double` | m | Radial depth of the mouth/wedge region |
| `tooth_tip_angle` | `double` | rad | Chamfer angle at tooth tip corners (0 = no chamfer). Affects cogging torque and iron saturation at the tip |
| `slot_shape` | `SlotShape` | — | Enum: `RECTANGULAR`, `TRAPEZOIDAL`, `ROUND_BOTTOM`, `SEMI_CLOSED` |

### Section 3 — Coil / Winding

| Field | Type | Unit | Description |
|---|---|---|---|
| `coil_depth` | `double` | m | Radial depth of the coil conductor bundle |
| `coil_width_outer` | `double` | m | Bundle width at yoke side |
| `coil_width_inner` | `double` | m | Bundle width at bore side |
| `insulation_thickness` | `double` | m | Slot-liner clearance on each side of coil |
| `turns_per_coil` | `int` | — | Turns per coil side; cross-checked against fill factor |
| `coil_pitch` | `int` | slots | Coil span in slot pitches |
| `wire_diameter` | `double` | m | Conductor wire diameter (used for fill factor cross-check and homogenisation decision) |
| `slot_fill_factor` | `double` | — | Target fill factor in (0, 1); validated against computed coil geometry |
| `winding_type` | `WindingType` | — | Enum: `SINGLE_LAYER`, `DOUBLE_LAYER`, `CONCENTRATED`, `DISTRIBUTED` |

### Section 4 — Lamination Stack

| Field | Type | Unit | Description |
|---|---|---|---|
| `t_lam` | `double` | m | Thickness of one lamination sheet |
| `n_lam` | `int` | — | Number of laminations in the axial stack |
| `z_spacing` | `double` | m | Axial gap between adjacent laminations (≥ 0) |
| `insulation_coating_thickness` | `double` | m | Inter-lamination insulation coating thickness per sheet |
| `material` | `LaminationMaterial` | — | Enum: `M270_35A`, `M330_50A`, `M400_50A`, `NO20`, `CUSTOM` |
| `material_file` | `std::string` | — | Path to custom BH-curve CSV (required when `material == CUSTOM`) |

### Section 5 — Mesh Sizing

| Field | Type | Unit | Description |
|---|---|---|---|
| `mesh_yoke` | `double` | m | Target edge length in yoke (back-iron) region |
| `mesh_slot` | `double` | m | Target edge length in slot air region |
| `mesh_coil` | `double` | m | Target edge length inside coil conductor region |
| `mesh_ins` | `double` | m | Target edge length in slot insulation region (finest) |
| `mesh_boundary_layers` | `int` | — | Number of structured boundary-layer elements at bore and slot mouth |
| `mesh_curvature` | `double` | — | Curvature refinement factor on arcs (0 = disabled; typical: 0.3) |
| `mesh_transition_layers` | `int` | — | Number of transition layers between different-sized mesh regions |

### Section 6 — Derived Quantities (computed by `validate_and_derive()`, read-only)

| Field | Formula |
|---|---|
| `yoke_height` | `R_outer - R_inner - slot_depth` |
| `tooth_width` | `R_inner * slot_pitch - effective_bore_slot_width` |
| `slot_pitch` | `2π / n_slots` (radians) |
| `stack_length` | `n_lam * t_lam + (n_lam - 1) * z_spacing` |
| `fill_factor` | `coil_area / slot_area` (cross-check against `slot_fill_factor`) |

---

## Module Specifications

### 1. `params.hpp` / `params.cpp`

**Purpose:** Single source of truth for all geometry parameters. Plain-old-data struct, trivially copyable, no virtual members.

**Enumerations to define:**

```cpp
namespace stator {
  enum class SlotShape       : uint8_t { RECTANGULAR=0, TRAPEZOIDAL=1, ROUND_BOTTOM=2, SEMI_CLOSED=3 };
  enum class WindingType     : uint8_t { SINGLE_LAYER=0, DOUBLE_LAYER=1, CONCENTRATED=2, DISTRIBUTED=3 };
  enum class LaminationMaterial : uint8_t { M270_35A=0, M330_50A=1, M400_50A=2, NO20=3, CUSTOM=4 };
}
```

**`StatorParams::validate_and_derive()` must check:**

1. All `double` dimensions > 0
2. `R_inner < R_outer`
3. `slot_depth < (R_outer - R_inner)` — slot must fit radially inside the yoke
4. `slot_width_inner < R_inner * 2π / n_slots` — slot must not exceed the tooth pitch arc
5. For `SEMI_CLOSED`: `slot_opening < slot_width_inner` and `slot_opening_depth < slot_depth`
6. `coil_depth ≤ slot_depth - slot_opening_depth - 2 * insulation_thickness`
7. `coil_width_inner ≤ slot_width_inner - 2 * insulation_thickness`
8. `fill_factor` computed and in (0, 1)
9. `slot_fill_factor` matches computed fill_factor within 5% tolerance (warning, not error)
10. `t_lam > 0`, `n_lam > 0`, `z_spacing ≥ 0`
11. `material == CUSTOM` requires non-empty `material_file`
12. Mesh sizes: `mesh_ins ≤ mesh_coil ≤ mesh_slot ≤ mesh_yoke` (strict ordering, error on violation)
13. `n_slots ≥ 6` and `n_slots % 2 == 0`
14. `tooth_tip_angle ≥ 0` and `< π/4` (chamfer must be physically sensible)
15. `wire_diameter > 0`
16. `insulation_coating_thickness ≥ 0`

**Additional free functions to implement:**

```cpp
// Convert each enum to descriptive string (used in JSON + logging)
const char* to_string(SlotShape);
const char* to_string(WindingType);
const char* to_string(LaminationMaterial);

// Build a validated 36-slot reference design for tests
StatorParams make_reference_params();

// Build a minimal 12-slot design (another test baseline)
StatorParams make_minimal_params();
```

**`to_json()`** must serialise all user-settable fields plus a `"_derived"` sub-object containing computed fields. Output must be a single-line valid JSON string.

**`operator<<`** must print a human-readable multi-line summary including derived fields.

---

### 2. `topology_registry.hpp` / `topology_registry.cpp`

**Purpose:** Thread-safe registry mapping GMSH entity tags to named physical regions.

**`RegionType` enum (integer values are the canonical GMSH physical-group tags):**

```
YOKE=100, TOOTH=101, SLOT_AIR=200, SLOT_INS=201,
COIL_A_POS=301, COIL_A_NEG=302,
COIL_B_POS=303, COIL_B_NEG=304,
COIL_C_POS=305, COIL_C_NEG=306,
BORE_AIR=400, BOUNDARY_BORE=500, BOUNDARY_OUTER=501,
UNKNOWN=-1
```

**Thread-safety model:**
- All writes use `std::unique_lock<std::shared_mutex>`
- All reads use `std::shared_lock<std::shared_mutex>`
- The mutex is `mutable` so const read methods can acquire it

**Methods to implement (full signatures and semantics):**

```cpp
explicit TopologyRegistry(int n_slots);   // throws if n_slots <= 0

void register_surface(RegionType, int gmsh_tag, int slot_idx = -1);
void register_slot_coil(int slot_idx, int upper_tag, int lower_tag = -1);
void register_boundary_curve(RegionType, int gmsh_curve);  // only BORE/OUTER accepted

void assign_winding_layout(WindingType);   // throws if coils not registered

std::vector<int> get_surfaces(RegionType) const;
std::vector<int> get_boundary_curves(RegionType) const;
const SlotWindingAssignment& get_slot_assignment(int slot_idx) const;
const std::vector<SlotWindingAssignment>& get_winding_assignments() const;
int  total_registered_surfaces() const;
bool winding_assigned() const noexcept;
void dump(std::ostream&) const;
```

**Winding phase assignment logic:**

DISTRIBUTED (repeat every 6 slots):
```
slot % 6:  0→A+  1→B−  2→C+  3→A−  4→B+  5→C−
```

CONCENTRATED (repeat every 6 slots):
```
slot % 6:  0→A+  1→A−  2→B+  3→B−  4→C+  5→C−
```

SINGLE_LAYER and DOUBLE_LAYER: use DISTRIBUTED sequence. For DOUBLE_LAYER, both coil halves in the same slot receive the same phase assignment (simplified model; full short-pitch modelling is a future extension noted explicitly in the code with a `// TODO(v2):` comment).

---

### 3. `gmsh_backend.hpp` / `gmsh_backend.cpp`

**Purpose:** Decouple geometry code from the GMSH C library so that unit tests can run without GMSH installed.

**`IGmshBackend` interface** — implement ALL of these pure virtual methods:

```cpp
// Session lifecycle
virtual void initialize(const std::string& model_name) = 0;
virtual void synchronize() = 0;
virtual void finalize() = 0;
virtual void set_option(const std::string& name, double value) = 0;

// OCC geometry primitives
virtual int add_point(double x, double y, double z, double mesh_size = 0.0) = 0;
virtual int add_line(int start, int end) = 0;
virtual int add_circle(double cx, double cy, double cz, double radius) = 0;
virtual int add_arc(int start, int centre, int end) = 0;
virtual int add_curve_loop(const std::vector<int>& tags) = 0;
virtual int add_plane_surface(const std::vector<int>& loop_tags) = 0;

// Boolean operations
virtual std::vector<std::pair<int,int>> boolean_cut(
    const std::vector<std::pair<int,int>>& objects,
    const std::vector<std::pair<int,int>>& tools,
    bool remove_tool = true) = 0;

virtual std::vector<std::pair<int,int>> boolean_fragment(
    const std::vector<std::pair<int,int>>& objects,
    const std::vector<std::pair<int,int>>& tools) = 0;

// Physical groups
virtual int add_physical_group(int dim, const std::vector<int>& tags,
                               const std::string& name, int tag = -1) = 0;

// Mesh fields
virtual int  add_math_eval_field(const std::string& expr) = 0;
virtual int  add_constant_field(double value, const std::vector<int>& surfaces) = 0;
virtual void set_background_field(int field_tag) = 0;

// Mesh generation and I/O
virtual void generate_mesh(int dim) = 0;
virtual void write_mesh(const std::string& filename) = 0;
virtual std::vector<std::pair<int,int>> get_entities_2d() = 0;
```

**`StubGmshBackend`** — implement all overrides. Must:
- Return auto-incrementing tags from 1 for each entity type (points, curves, surfaces share separate counters)
- Record every `add_physical_group` call in a `std::vector<PhysGroupRecord>`
- Track: `initialized_`, `sync_count_`, `finalized_`, `mesh_generated_`, `last_write_path_`
- `boolean_cut` returns object list unchanged (no actual subtraction in stub)
- `boolean_fragment` returns union of both input lists
- Expose inspection methods: `point_count()`, `line_count()`, `surface_count()`, `physical_group_count()`, `was_initialized()`, `was_synchronized()`, `was_finalized()`, `sync_count()`, `mesh_generated()`, `last_write_path()`, `physical_groups()`
- `reset()` clears all state to allow reuse between test cases

**`RealGmshBackend`** (compiled only when `-DSTATOR_WITH_GMSH=ON`):
- Must be in a separate file `src/gmsh_real_backend.cpp` so the main library links without GMSH
- Delegates every method to the corresponding `gmsh::model::occ::*` or `gmsh::model::mesh::field::*` call
- Session lifecycle: `initialize()` calls `gmsh::initialize()` then `gmsh::model::add(name)`; `finalize()` calls `gmsh::finalize()`

**`make_default_backend()` factory:**
- Returns `RealGmshBackend` when `STATOR_WITH_GMSH` defined, else `StubGmshBackend` with a `std::cerr` warning

---

### 4. `geometry_builder.hpp` / `geometry_builder.cpp`

**Purpose:** Constructs the complete 2-D stator cross-section geometry in a GMSH OCC session.

**Coordinate convention:**
- Local frame: slot centred on +x axis. `x` = radial outward, `y` = tangential (CCW positive)
- Global frame: rotate by `θ_k = 2π·k / n_slots` via `rotate(x, y, θ)`
- All 2-D work at `z = 0`

**`rotate(x, y, θ)` helper:**
```cpp
static std::pair<double,double> rotate(double x, double y, double theta) noexcept {
    return { x*cos(θ) - y*sin(θ),  x*sin(θ) + y*cos(θ) };
}
```

**`build()` sequence (must follow exactly):**
1. Add outer circle (tag `c_outer`) and inner circle (tag `c_inner`)
2. Create `cl_outer = add_curve_loop({c_outer})` and `cl_inner = add_curve_loop({-c_inner})` (reversed = hole)
3. `yoke_sf = add_plane_surface({cl_outer, cl_inner})`
4. For `k` in `[0, n_slots)`: call `build_single_slot(p, k)` — stores in `result.slots[k]`
5. Boolean cut: cut all slot surfaces from yoke → trimmed yoke surface
6. `backend_->synchronize()`
7. Populate `GeometryBuildResult` and return

**`build_single_slot(p, k)` sequence:**
1. Compute `θ = slot_angle(k, p.n_slots)`
2. Dispatch to shape builder based on `p.slot_shape`
3. Call `build_coil_inside_slot(p, profile, θ)`
4. Call `build_insulation(p, profile, θ)`
5. Return populated `SlotProfile`

**Four slot shape builders — full geometry detail:**

**`RECTANGULAR`:**
```
Points (local frame, CCW):
  p1: (R_inner,        +hw_inner)    ← top-left
  p2: (R_inner,        −hw_inner)    ← top-right
  p3: (R_inner+depth,  −hw_inner)    ← bottom-right   (hw_inner == hw_outer for rect)
  p4: (R_inner+depth,  +hw_inner)    ← bottom-left
Lines: l1=p1→p2 (top), l2=p2→p3, l3=p3→p4 (bottom), l4=p4→p1
mouth_curve_bot = l1
```

**`TRAPEZOIDAL`:**
```
Points (local frame):
  p1: (R_inner,              +hw_inner)
  p2: (R_inner,              −hw_inner)
  p3: (R_inner+slot_depth,   −hw_outer)
  p4: (R_inner+slot_depth,   +hw_outer)
Lines: l1=p1→p2 (top/bore), l2=p2→p3, l3=p3→p4 (bottom), l4=p4→p1
mouth_curve_bot = l1
```

**`ROUND_BOTTOM`:**
```
Straight walls down to r_straight = R_inner + slot_depth − hw_outer.
Arc centre at (r_straight, 0), radius = hw_outer.
Use TWO arcs (GMSH arcs must be < 180°):
  arc1: p_right_join → p_bottom_mid  (via arc_centre)
  arc2: p_bottom_mid → p_left_join   (via arc_centre)
mouth_curve_bot = top line
```

**`SEMI_CLOSED`:**
```
Radial levels:
  r0 = R_inner                              ← bore
  r1 = R_inner + slot_opening_depth         ← shoulder
  r2 = R_inner + slot_opening_depth         ← bottom of slot
        + slot_depth

Points:
  p1: (r0, +hw_mouth)    p2: (r0, −hw_mouth)     ← mouth
  p3: (r1, +hw_shoulder) p4: (r1, −hw_shoulder)  ← shoulder
  p5: (r2, +hw_bottom)   p6: (r2, −hw_bottom)    ← bottom

Lines (CCW contour):
  l_mouth_top = p1→p2   (bore-facing, used for BL seeding)
  l_mouth_rhs = p2→p4
  l_wall_r    = p4→p6
  l_bottom    = p6→p5
  l_wall_l    = p5→p3
  l_mouth_lhs = p3→p1

mouth_curve_bot = l_mouth_top
mouth_curve_top = l_bottom
```

**`tooth_tip_angle` handling:**
When `tooth_tip_angle > 0`, add chamfer points at the tooth-tip corners (where slot wall meets the bore arc). This produces two extra lines per slot side, replacing the sharp corner with a short angled edge. Required for SEMI_CLOSED and TRAPEZOIDAL shapes.

**`build_coil_inside_slot(p, profile, θ)`:**

DOUBLE_LAYER:
```
half_depth = (coil_depth − 2*insulation_thickness) / 2
upper coil: r_u0 = R_inner+slot_opening_depth+ins, r_u1 = r_u0+half_depth
lower coil: r_l0 = r_u1 + 2*ins,                  r_l1 = r_l0+half_depth
Width: coil_width_inner at bore side, coil_width_outer at yoke side (trapezoidal bundle)
```

SINGLE_LAYER:
```
r_c0 = R_inner+slot_opening_depth+ins
r_c1 = r_c0 + coil_depth
coil_lower_sf = -1 (not used)
```

**`build_insulation(p, profile, θ)`:**
Create a slightly-expanded version of each coil outline (expanded by `insulation_thickness` on all sides) as a separate surface. These surfaces will be resolved into the insulation shell by the boolean fragment step.

---

### 5. `mesh_generator.hpp` / `mesh_generator.cpp`

**Purpose:** Configure GMSH mesh size fields per region and drive mesh generation.

**Three-layer size field strategy (all three are combined via a Min field):**

**Layer A — per-surface Constant fields:**
```
YOKE + TOOTH  → mesh_yoke
SLOT_AIR      → mesh_slot
COIL_*        → mesh_coil
SLOT_INS      → mesh_ins
```

**Layer B — mouth transition (Distance + Threshold):**
```
Seed Distance field on all mouth_curve_bot tags.
Threshold: size = mesh_slot at distance 0,
           size = mesh_yoke at distance = slot_depth/4.
One threshold field per slot (or combined into one Distance field with all curves).
```

**Layer C — bore boundary layer:**
```
BoundaryLayer field on BOUNDARY_BORE curve.
Size    = mesh_ins
Ratio   = 1.2 (growth ratio between layers)
NbLayers = mesh_boundary_layers
```

**Combine:** `Min{all Layer A fields, all Layer B fields, Layer C field}` → set as background field.

**`assign_physical_groups(registry)`** must:
- Register every non-empty surface region (YOKE, TOOTH, SLOT_AIR, SLOT_INS, all 6 COIL_* types, BORE_AIR) as a 2-D physical group
- Register BOUNDARY_BORE and BOUNDARY_OUTER as 1-D physical groups
- Use `canonical_tag(RegionType)` as the physical group integer tag
- Use `to_string(RegionType)` as the physical group name

**`MeshConfig` struct fields:**
```cpp
int         algorithm_2d         = 5;      // 5=Delaunay, 6=Frontal-Delaunay, 8=Delaunay-quads
int         algorithm_3d         = 10;     // 10=HXT (parallel), 1=Delaunay, 4=Frontal
int         smoothing_passes     = 3;
std::string optimiser            = "Netgen";
double      min_quality_threshold = 0.3;
bool        periodic             = false;
int         layers_per_lam       = 2;
```

**3-D extrusion** (when `n_lam > 1`):
Extrude the 2-D cross-section axially. Each lamination = one extrusion of height `t_lam` with `layers_per_lam` structured layers. If `z_spacing > 0`, insert an air-gap extrusion of height `z_spacing` between each pair of laminations.

---

### 6. `export_engine.hpp` / `export_engine.cpp`

**Purpose:** Write mesh data in multiple formats in parallel using `std::async`.

**`ExportFormat` bitmask enum:**
```cpp
enum class ExportFormat : uint32_t {
    NONE=0, MSH=1<<0, VTK=1<<1, HDF5=1<<2, JSON=1<<3, ALL=MSH|VTK|HDF5|JSON
};
```

**Format details:**

| Format | Content |
|---|---|
| `.msh` | GMSH MSH v4.1 (or v2.2 if `msh_version=2`). Includes physical groups. |
| `.vtk` | VTK Legacy unstructured grid. Includes element quality scalar field. |
| `.h5` | HDF5 via HighFive: `/mesh/nodes` (N×2 float64), `/mesh/elements` (M×3 int32), `/mesh/region_tags` (M int32), `/params` attributes, `/quality/per_element` (M float32). When HighFive unavailable, write a placeholder text file. |
| `_meta.json` | All StatorParams fields + mesh statistics (n_nodes, n_elements_2d, n_elements_3d, min/avg quality) |

**File naming:**
```
stem = "stator_" + sha256(params.to_json()).substr(0, 8)
files: {stem}.msh, {stem}.vtk, {stem}.h5, {stem}_meta.json
```

**SHA-256:** implement a self-contained version (FIPS 180-4) with no external cryptographic dependency. This keeps the library portable. Place implementation in an anonymous namespace inside `export_engine.cpp`.

**`write_all_sync()` / `write_all_async()` behaviour:**
- `write_all_sync` submits all format tasks via `std::async(std::launch::async, ...)` then joins all futures before returning
- `write_all_async` returns futures immediately (caller must join before `backend->finalize()`)
- Each writer captures params and config by value (safe for async lifetime)
- If any writer throws, capture in `ExportResult::error_message`; do not propagate across thread boundary

**`outputs_exist(params, config)`:** returns true iff all files for selected formats exist on disk for the given params hash.

---

### 7. `batch_scheduler.hpp` / `batch_scheduler.cpp`

**Purpose:** Fork-based parallel execution of geometry build batches.

**`BatchJob` struct:**
```cpp
struct BatchJob {
    StatorParams  params;
    ExportConfig  export_config;
    MeshConfig    mesh_config;
    std::string   job_id;        // caller-assigned identifier, echoed in status JSON
};
```

**`BatchSchedulerConfig` struct:**
```cpp
struct BatchSchedulerConfig {
    int  max_parallel      = 0;    // 0 = auto-detect via hardware_concurrency()
    bool skip_existing     = true; // idempotent re-runs
    int  job_timeout_sec   = 300;  // SIGKILL child after this; 0 = no timeout
    bool write_summary     = true; // batch_summary.json in output_dir
};
```

**`run()` algorithm:**
```
resolve max_parallel (hardware_concurrency if 0)
for each job:
  if skip_existing && outputs_exist → mark success, skip
  fork()
  child: call execute_job(job, status_path); _exit(rc)
  parent: record (pid, job_index, start_time) in active list
  if active.size() == max_parallel: poll with waitpid(WNOHANG) until slot free
while any children still running:
  sleep 100ms
  for each active child:
    check timeout → SIGKILL if exceeded
    waitpid(WNOHANG) → if done: read status JSON, invoke progress_cb, release slot
after all done:
  if write_summary: write batch_summary.json
return results vector
```

**`execute_job(job, status_path)` pipeline** (runs inside child process):
```
1.  params.validate_and_derive()           // throws on invalid
2.  backend = make_default_backend()
3.  backend->initialize("stator_" + stem)
4.  GeometryBuilder builder(backend)
5.  geo = builder.build(params)            // throws or returns error result
6.  TopologyRegistry registry(params.n_slots)
7.  Register yoke, boundary curves, per-slot surfaces from geo
8.  registry.assign_winding_layout(params.winding_type)
9.  MeshGenerator mesher(backend, job.mesh_config)
10. mesh_result = mesher.generate(params, geo, registry)
11. ExportEngine exporter(backend)
12. export_results = exporter.write_all_sync(params, mesh_result, job.export_config)
13. write status JSON with paths and success flag
14. backend->finalize()
15. return 0 on success, non-zero on any error
```

**Status JSON format** (written by each child):
```json
{
  "job_id": "...",
  "success": true,
  "error": "",
  "msh_path": "/path/to/stator_abcd1234.msh",
  "vtk_path": "/path/to/stator_abcd1234.vtk",
  "hdf5_path": "/path/to/stator_abcd1234.h5"
}
```

**Progress callback signature:**
```cpp
using ProgressCallback = std::function<void(int jobs_done, int jobs_total,
                                             bool success, const std::string& job_id)>;
```

**`cancel()`:** sets `cancel_flag_`, sends `SIGTERM` to all active children, waits for them. Unstarted jobs marked as cancelled.

---

### 8. `bindings/python_bindings.cpp`

Expose the following to Python via pybind11:

**Enums** (as `py::enum_`):
- `stator.SlotShape` with values `RECTANGULAR`, `TRAPEZOIDAL`, `ROUND_BOTTOM`, `SEMI_CLOSED`
- `stator.WindingType` with values `SINGLE_LAYER`, `DOUBLE_LAYER`, `CONCENTRATED`, `DISTRIBUTED`
- `stator.LaminationMaterial` with values `M270_35A`, `M330_50A`, `M400_50A`, `NO20`, `CUSTOM`
- `stator.ExportFormat` with values `NONE`, `MSH`, `VTK`, `HDF5`, `JSON`, `ALL`

**`StatorParams`** as a pybind11 value type with all fields exposed as `def_readwrite`. Expose `validate_and_derive()` and `to_json()` as methods.

**`BatchJob`** as a pybind11 value type.

**`BatchScheduler`** with:
```python
scheduler.set_progress_callback(callback)   # callable(done, total, success, job_id)
results = scheduler.run(jobs, config)        # releases GIL via gil_scoped_release
scheduler.cancel()
scheduler.is_running()
```

**`BatchSchedulerConfig`** with all fields exposed.

**`make_reference_params()`** as a free function.

**Module name:** `_stator_core`

---

### 9. `python/stator_pipeline/pipeline.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import _stator_core as _core

@dataclass
class StatorConfig:
    # All StatorParams fields exposed as Python dataclass fields
    # with Python-idiomatic names (snake_case matches C++ exactly)
    R_outer: float
    R_inner: float
    airgap_length: float
    n_slots: int
    slot_depth: float
    slot_width_outer: float
    slot_width_inner: float
    slot_opening: float = 0.0
    slot_opening_depth: float = 0.0
    tooth_tip_angle: float = 0.0
    slot_shape: str = "SEMI_CLOSED"
    coil_depth: float = 0.0
    coil_width_outer: float = 0.0
    coil_width_inner: float = 0.0
    insulation_thickness: float = 0.001
    turns_per_coil: int = 10
    coil_pitch: int = 5
    wire_diameter: float = 0.001
    slot_fill_factor: float = 0.45
    winding_type: str = "DOUBLE_LAYER"
    t_lam: float = 0.00035
    n_lam: int = 200
    z_spacing: float = 0.0
    insulation_coating_thickness: float = 0.00005
    material: str = "M270_35A"
    material_file: str = ""
    mesh_yoke: float = 0.006
    mesh_slot: float = 0.003
    mesh_coil: float = 0.0015
    mesh_ins: float = 0.0007
    mesh_boundary_layers: int = 3
    mesh_curvature: float = 0.3
    mesh_transition_layers: int = 2

def generate_single(
    config: StatorConfig,
    output_dir: str,
    formats: str = "JSON|HDF5"
) -> dict:
    """Generate mesh for one stator config. Returns dict with output paths."""

def generate_batch(
    configs: List[StatorConfig],
    output_dir: str,
    max_parallel: int = 0,
    formats: str = "MSH|VTK|HDF5|JSON",
    progress_callback: Optional[Callable] = None,
    skip_existing: bool = True,
    job_timeout_sec: int = 300
) -> List[dict]:
    """Generate meshes for a batch. Returns list of result dicts."""
```

---

### 10. `python/stator_pipeline/visualiser.py`

Implement a `StatorVisualiser` class that:
- Loads a `.vtk` file using `pyvtk` or `vtk` Python package (if available) or falls back to parsing the VTK legacy ASCII format manually
- Renders the 2-D cross-section using `matplotlib` with coloured regions (one colour per `RegionType`)
- Exposes `plot_cross_section(vtk_path, output_png=None)` — shows or saves the plot
- Exposes `plot_mesh(vtk_path, show_quality=True)` — colours elements by quality scalar

---

## CMakeLists.txt Specification

```cmake
cmake_minimum_required(VERSION 3.22)
project(StatorMeshPipeline VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(STATOR_WITH_GMSH    "Build with real GMSH backend"     OFF)
option(STATOR_WITH_TBB     "Enable Intel TBB parallelism"     OFF)
option(STATOR_WITH_HDF5    "Enable HDF5 mesh export"          OFF)
option(STATOR_WITH_PYTHON  "Build pybind11 Python bindings"   OFF)
option(STATOR_BUILD_TESTS  "Build test executable"            ON)

# Core library (always builds — uses stub backend when GMSH off)
add_library(stator_core SHARED
    src/params.cpp
    src/topology_registry.cpp
    src/gmsh_backend.cpp
    src/geometry_builder.cpp
    src/mesh_generator.cpp
    src/export_engine.cpp
    src/batch_scheduler.cpp)

target_include_directories(stator_core PUBLIC include)
target_compile_options(stator_core PRIVATE -Wall -Wextra -Wpedantic)

# Optional GMSH
if(STATOR_WITH_GMSH)
    find_package(GMSH REQUIRED)
    target_link_libraries(stator_core PUBLIC gmsh)
    target_compile_definitions(stator_core PUBLIC STATOR_WITH_GMSH)
    target_sources(stator_core PRIVATE src/gmsh_real_backend.cpp)
endif()

# Optional TBB
if(STATOR_WITH_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(stator_core PUBLIC TBB::tbb)
    target_compile_definitions(stator_core PUBLIC STATOR_WITH_TBB)
endif()

# Optional HDF5
if(STATOR_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    target_link_libraries(stator_core PUBLIC HDF5::HDF5)
    target_compile_definitions(stator_core PUBLIC STATOR_WITH_HDF5)
endif()

# pybind11 Python module
if(STATOR_WITH_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_stator_core bindings/python_bindings.cpp)
    target_link_libraries(_stator_core PRIVATE stator_core)
endif()

# Tests
if(STATOR_BUILD_TESTS)
    add_executable(test_stator tests/test_stator.cpp)
    target_link_libraries(test_stator PRIVATE stator_core)
    target_compile_options(test_stator PRIVATE -Wall -Wextra)
    enable_testing()
    add_test(NAME StatorPipelineTests COMMAND test_stator)
endif()
```

---

## Test Specification

The test binary is `tests/test_stator.cpp`. It must use a **hand-rolled test framework** (no Catch2, Google Test, or other external dependency) consisting of `TEST(name)`, `RUN(name)`, `EXPECT(cond)`, `EXPECT_THROWS(expr)`, `EXPECT_NEAR(a, b, tol)` macros. The `main()` function prints pass/fail per test and exits with the number of failures as the exit code.

### Test Groups and Required Test Cases

#### [PARAMS] — 25 test cases minimum

1. `params_reference_validates` — `make_reference_params()` does not throw; derived fields all > 0
2. `params_minimal_validates` — `make_minimal_params()` does not throw
3. `params_derived_yoke_height` — checks formula `R_outer - R_inner - slot_depth`
4. `params_derived_tooth_width` — checks formula `R_inner * slot_pitch - slot_width_inner`
5. `params_derived_slot_pitch` — checks `2π / n_slots`
6. `params_derived_stack_length` — checks `n_lam * t_lam + (n_lam-1) * z_spacing`
7. `params_derived_fill_factor_in_range` — (0, 1)
8. `params_rejects_zero_R_outer`
9. `params_rejects_negative_R_outer`
10. `params_rejects_R_inner_ge_R_outer`
11. `params_rejects_equal_radii`
12. `params_rejects_too_few_slots` — n_slots < 6
13. `params_rejects_odd_slot_count` — n_slots = 35
14. `params_rejects_slot_depth_exceeds_annulus` — slot_depth ≥ R_outer - R_inner
15. `params_rejects_slot_too_wide_for_pitch` — slot_width_inner ≥ R_inner * 2π/n_slots
16. `params_rejects_coil_depth_exceeds_slot`
17. `params_rejects_coil_too_wide` — coil_width_inner > slot_width_inner - 2*ins
18. `params_rejects_mesh_ins_coarser_than_mesh_coil`
19. `params_rejects_mesh_coil_coarser_than_mesh_slot`
20. `params_rejects_mesh_slot_coarser_than_mesh_yoke`
21. `params_rejects_custom_material_no_file`
22. `params_rejects_negative_airgap`
23. `params_rejects_zero_t_lam`
24. `params_to_json_contains_all_sections` — checks for "R_outer", "n_slots", "_derived", "fill_factor"
25. `params_stream_operator_contains_derived`
26. `params_to_string_all_slot_shapes` — all 4 values return non-empty, non-"UNKNOWN" strings
27. `params_to_string_all_winding_types` — all 4 values
28. `params_to_string_all_materials` — all 5 values
29. `params_fill_factor_consistent_double_layer` — double-layer fills approx 2× single
30. `params_rejects_negative_tooth_tip_angle`
31. `params_rejects_excessive_tooth_tip_angle` — angle ≥ π/4

#### [TOPOLOGY] — 20 test cases minimum

1. `topology_construction_succeeds`
2. `topology_rejects_zero_n_slots`
3. `topology_rejects_negative_n_slots`
4. `topology_register_and_query_yoke`
5. `topology_register_multiple_surfaces_same_region`
6. `topology_empty_query_unregistered_region`
7. `topology_register_boundary_bore_curve`
8. `topology_register_boundary_outer_curve`
9. `topology_register_boundary_rejects_non_boundary_type`
10. `topology_get_slot_assignment_before_winding_throws`
11. `topology_get_winding_before_assign_throws`
12. `topology_assign_winding_before_coil_registration_throws`
13. `topology_distributed_phase_sequence_6_slots`
14. `topology_distributed_phase_sequence_36_slots` — verify repeating pattern across all 36
15. `topology_concentrated_phase_sequence_6_slots`
16. `topology_register_slot_coil_out_of_range_throws`
17. `topology_total_registered_surfaces_count`
18. `topology_thread_safe_concurrent_surface_registration` — 8 threads, 4 registrations each
19. `topology_thread_safe_concurrent_read_during_write` — writers and readers simultaneously
20. `topology_dump_output_nonempty`
21. `topology_canonical_tag_values` — verify YOKE=100, SLOT_AIR=200, COIL_A_POS=301, etc.
22. `topology_all_coil_regions_registered_after_winding_assign`

#### [GEOMETRY] — 30 test cases minimum

1. `geometry_null_backend_throws`
2. `geometry_build_single_slot_rectangular` — slot_surface ≥ 0
3. `geometry_build_single_slot_trapezoidal`
4. `geometry_build_single_slot_round_bottom`
5. `geometry_build_single_slot_semi_closed` — also checks mouth_curve_bot ≥ 0
6. `geometry_slot_0_angle_is_zero` — first slot on +x axis
7. `geometry_slot_angles_all_distinct` — n_slots distinct angles
8. `geometry_slot_angles_span_full_circle` — last slot < 2π
9. `geometry_coil_surfaces_double_layer_both_populated`
10. `geometry_coil_surfaces_single_layer_lower_is_minus_one`
11. `geometry_insulation_surfaces_double_layer`
12. `geometry_insulation_surfaces_single_layer_lower_is_minus_one`
13. `geometry_semi_closed_has_mouth_curves`
14. `geometry_rectangular_mouth_curve_bot_set`
15. `geometry_trapezoidal_mouth_curve_bot_set`
16. `geometry_build_full_36_slot_success`
17. `geometry_build_full_36_slot_slot_count_correct`
18. `geometry_build_full_calls_synchronize`
19. `geometry_build_creates_yoke_surface`
20. `geometry_build_creates_bore_curve`
21. `geometry_build_creates_outer_curve`
22. `geometry_build_12_slot` — smaller machine
23. `geometry_build_48_slot` — larger machine
24. `geometry_build_all_four_shapes_succeed` — parametrised loop
25. `geometry_point_count_scales_with_slot_count` — more slots → more points
26. `geometry_surface_count_minimum` — at least n_slots coil surfaces + yoke
27. `geometry_stub_records_boolean_cut_call`
28. `geometry_single_slot_does_not_call_synchronize` — build_single_slot skips global sync
29. `geometry_rotate_helper_zero_angle` — rotate(x,y,0) = (x,y)
30. `geometry_rotate_helper_90_degrees` — rotate(1,0,π/2) ≈ (0,1)
31. `geometry_rotate_helper_180_degrees` — rotate(1,0,π) ≈ (-1,0)
32. `geometry_tooth_tip_chamfer_adds_extra_points` — when tooth_tip_angle > 0

#### [MESH] — 15 test cases minimum

1. `mesh_null_backend_throws`
2. `mesh_generate_success`
3. `mesh_generate_triggers_gmsh_mesh_generate`
4. `mesh_failed_geometry_propagates_error`
5. `mesh_physical_groups_assigned_after_generate`
6. `mesh_physical_group_count_minimum` — at least 6 groups (YOKE, 3 coil phases, BORE, OUTER)
7. `mesh_physical_group_names_nonempty`
8. `mesh_yoke_group_has_canonical_tag_100`
9. `mesh_slot_air_group_has_canonical_tag_200`
10. `mesh_coil_a_pos_group_has_canonical_tag_301`
11. `mesh_boundary_bore_is_1d_group`
12. `mesh_region_size_fields_created` — at least 4 constant fields
13. `mesh_mouth_transition_fields_created_for_semi_closed`
14. `mesh_background_field_set`
15. `mesh_3d_extrusion_called_when_n_lam_gt_1`
16. `mesh_quality_struct_populated`

#### [EXPORT] — 20 test cases minimum

1. `export_null_backend_throws`
2. `export_stem_deterministic` — same params → same stem
3. `export_stem_different_params_differ`
4. `export_stem_has_stator_prefix`
5. `export_stem_length_correct` — "stator_" + 8 hex = 15 chars
6. `export_write_json_creates_file`
7. `export_write_hdf5_creates_file`
8. `export_write_json_contains_R_outer`
9. `export_write_json_contains_n_slots`
10. `export_write_json_contains_mesh_stats`
11. `export_write_json_contains_output_file_paths`
12. `export_outputs_exist_false_before_write`
13. `export_outputs_exist_true_after_write`
14. `export_async_returns_correct_future_count_msh_only`
15. `export_async_returns_correct_future_count_all`
16. `export_all_formats_succeed_sync`
17. `export_write_time_ms_positive`
18. `export_result_format_field_correct`
19. `export_sha256_consistent` — sha256("") is known constant
20. `export_sha256_different_inputs_differ`
21. `export_msh_path_has_correct_extension`
22. `export_hdf5_path_has_h5_extension`

#### [BATCH] — 10 test cases minimum

1. `batch_execute_job_success_returns_zero`
2. `batch_execute_job_invalid_params_returns_nonzero`
3. `batch_execute_job_writes_status_json`
4. `batch_execute_job_status_json_success_field_true`
5. `batch_execute_job_status_json_failure_has_error`
6. `batch_execute_job_output_files_exist_on_success`
7. `batch_read_status_json_populates_result`
8. `batch_progress_callback_not_invoked_in_execute_job` — execute_job doesn't invoke cb itself
9. `batch_cancel_flag_prevents_new_forks` — set cancel before run, verify no children spawned
10. `batch_empty_job_list_returns_empty_result`

#### [INTEGRATION] — 10 test cases minimum

1. `integration_full_pipeline_semi_closed_double_layer` — all 6 pipeline steps, stub backend
2. `integration_full_pipeline_rectangular_single_layer`
3. `integration_full_pipeline_trapezoidal_distributed`
4. `integration_full_pipeline_round_bottom_concentrated`
5. `integration_all_slot_shapes_build_and_mesh` — parametrised over all 4 shapes
6. `integration_all_winding_types_assign_correctly` — parametrised over all 4 types
7. `integration_stub_reset_between_builds` — verify tag counters reset
8. `integration_12_slot_machine_full_pipeline`
9. `integration_48_slot_machine_full_pipeline`
10. `integration_to_string_no_region_type_returns_unknown` — verify all named RegionType values

---

## Critical Implementation Notes for Claude Opus 4.6

### Do not leave stubs or TODOs in required code

Every method body listed in this specification must be fully implemented. The only permitted `// TODO` comments are:

1. `// TODO(v2): full short-pitch coil modelling` in `assign_winding_layout()` where the simplified same-phase-for-both-halves assignment is used
2. `// TODO(v2): HighFive HDF5 real write` in the `write_hdf5()` placeholder section
3. `// TODO(v2): RealGmshBackend per-field GMSH API calls` in `add_mouth_transition_fields` and `add_bore_boundary_layer` when building the expression string

### SHA-256 implementation

Must be self-contained in `export_engine.cpp` inside an anonymous namespace. The constant `K[64]` table must be the standard FIPS 180-4 values. Test case `export_sha256_consistent` verifies:
```cpp
sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
```

### Thread-safety guarantee

`TopologyRegistry` must use `std::shared_mutex` (C++17). The `mutable std::shared_mutex mutex_` member enables const methods to acquire shared locks. All test cases in `[TOPOLOGY]` that exercise thread-safety must use `std::vector<std::thread>` and join all threads before asserting.

### Error propagation

No function may silently swallow exceptions except the top-level `execute_job()` which catches `std::exception` and writes the error to the status JSON file. All other error conditions must either:
- Return a result struct with `success = false` and populated `error_message` field (for value-returning functions like `build()` and `generate()`), or
- Throw `std::invalid_argument`, `std::out_of_range`, or `std::logic_error` with a descriptive message (for constructors and registration methods)

### Coordinate system consistency

All slot profiles must be built in the canonical local frame (slot centred on +x axis) and rotated to global coordinates via `rotate(x, y, θ)` before calling `add_point`. Never compute rotated coordinates inline — always go through the `rotate()` helper or `add_rotated_point()` wrapper. This must be enforced throughout all four slot shape builders.

### GMSH-free compilation

The project must compile and all tests must pass with `-DSTATOR_WITH_GMSH=OFF`. In this mode the `StubGmshBackend` is used everywhere. The `RealGmshBackend` class must only exist when `-DSTATOR_WITH_GMSH=ON` and must be in `src/gmsh_real_backend.cpp` which is only added to the build in that case.

### Test binary exit code

`main()` in `test_stator.cpp` must return `g_fail` (the count of failed tests). This allows `ctest` to detect failures. Print a final summary line:
```
=== Results: PASS=N FAIL=M ===
```

---

## Reference Parameter Values

These values define `make_reference_params()` — a typical 200 kW, 6-pole, 36-slot squirrel-cage induction generator:

```
R_outer                  = 0.250 m
R_inner                  = 0.165 m
airgap_length            = 0.001 m
n_slots                  = 36
slot_depth               = 0.050 m
slot_width_outer         = 0.014 m
slot_width_inner         = 0.010 m
slot_opening             = 0.004 m
slot_opening_depth       = 0.003 m
tooth_tip_angle          = 0.0 rad   (no chamfer in reference)
slot_shape               = SEMI_CLOSED
coil_depth               = 0.040 m
coil_width_outer         = 0.010 m
coil_width_inner         = 0.007 m
insulation_thickness     = 0.001 m
turns_per_coil           = 10
coil_pitch               = 5
wire_diameter            = 0.00133 m  (AWG 16 approx)
slot_fill_factor         = 0.45
winding_type             = DOUBLE_LAYER
t_lam                    = 0.00035 m  (0.35 mm)
n_lam                    = 200
z_spacing                = 0.0 m
insulation_coating_thickness = 0.00005 m (50 µm)
material                 = M270_35A
mesh_yoke                = 0.006 m
mesh_slot                = 0.003 m
mesh_coil                = 0.0015 m
mesh_ins                 = 0.0007 m
mesh_boundary_layers     = 3
mesh_curvature           = 0.3
mesh_transition_layers   = 2
```

These values define `make_minimal_params()` — a minimal 12-slot test machine:

```
R_outer                  = 0.120 m
R_inner                  = 0.075 m
airgap_length            = 0.0008 m
n_slots                  = 12
slot_depth               = 0.025 m
slot_width_outer         = 0.012 m
slot_width_inner         = 0.009 m
slot_opening             = 0.003 m
slot_opening_depth       = 0.002 m
tooth_tip_angle          = 0.0 rad
slot_shape               = SEMI_CLOSED
coil_depth               = 0.018 m
coil_width_outer         = 0.008 m
coil_width_inner         = 0.006 m
insulation_thickness     = 0.0008 m
turns_per_coil           = 6
coil_pitch               = 4
wire_diameter            = 0.0009 m
slot_fill_factor         = 0.40
winding_type             = SINGLE_LAYER
t_lam                    = 0.00050 m
n_lam                    = 80
z_spacing                = 0.0 m
insulation_coating_thickness = 0.00005 m
material                 = M330_50A
mesh_yoke                = 0.004 m
mesh_slot                = 0.002 m
mesh_coil                = 0.001 m
mesh_ins                 = 0.0005 m
mesh_boundary_layers     = 2
mesh_curvature           = 0.3
mesh_transition_layers   = 2
```

---

## Implementation Checklist

Before considering the implementation complete, verify:

- [ ] All 7 header files in `include/stator/` fully implemented with Doxygen comments
- [ ] All 7 source files in `src/` (plus `gmsh_real_backend.cpp`) fully implemented
- [ ] `bindings/python_bindings.cpp` implemented
- [ ] `python/stator_pipeline/pipeline.py` and `visualiser.py` implemented
- [ ] `tests/test_stator.cpp` contains ≥ 130 test cases across all groups
- [ ] Test binary compiles with `g++ -std=c++20` and zero warnings under `-Wall -Wextra -Wpedantic`
- [ ] `make_reference_params()` and `make_minimal_params()` both validate without throwing
- [ ] SHA-256 of empty string returns the known constant
- [ ] All thread-safety tests use `std::thread` and join before asserting
- [ ] `export_outputs_exist_true_after_write` verifies file exists on disk
- [ ] `CMakeLists.txt` builds the test binary with `ctest`-compatible exit codes
- [ ] No `#include "gmsh.h"` appears outside `gmsh_real_backend.cpp` and the conditional block in `gmsh_backend.hpp`
