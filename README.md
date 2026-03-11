GA Process (Python)
    │
    │  pybind11 / subprocess call
    ▼
StatorPipeline.generate_batch(params_list)   ← Python entry point
    │
    │  releases GIL, enters C++
    ▼
BatchScheduler (C++)
    │
    ├── multiprocessing: fork N worker processes (one per CPU core)
    │     Each worker owns:
    │       - its own GMSH context (not thread-safe across processes)
    │       - its own OCCT session
    │
    │  Within each worker process:
    ▼
GeometryWorker::run(StatorParams p)
    │
    ├── Thread 1: build_2d_cross_section()     ← GMSH occ kernel
    ├── Thread 2: validate_params() + compute_derived()
    │             (runs concurrently with geometry build)
    │
    ├── [barrier: geometry ready]
    │
    ├── Thread pool (TBB):
    │     ├── tag_physical_groups()            ← yoke, teeth, slots, coils
    │     ├── set_mesh_size_fields()           ← per-region MeshField
    │     └── compute_winding_layout()         ← slot→coil assignment
    │
    ├── [barrier: mesh spec ready]
    │
    ├── gmsh::model::mesh::generate(2)         ← single-threaded per GMSH instance
    │
    └── AsyncExporter:
          ├── write .msh  (thread A)
          ├── write .vtk  (thread B)
          └── write metadata.json (thread C)
