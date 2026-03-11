cmake_minimum_required(VERSION 3.22)
project(StatorMeshPipeline VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────────
option(STATOR_WITH_GMSH    "Build with real GMSH backend"     OFF)
option(STATOR_WITH_TBB     "Enable Intel TBB parallelism"     OFF)
option(STATOR_WITH_HDF5    "Enable HDF5 mesh export"          OFF)
option(STATOR_WITH_PYTHON  "Build pybind11 Python bindings"   OFF)
option(STATOR_BUILD_TESTS  "Build test executable"            ON)

# ─── Core library ─────────────────────────────────────────────────────────────
# Always builds — uses StubGmshBackend when STATOR_WITH_GMSH is OFF.
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

# ─── Optional: Real GMSH backend ─────────────────────────────────────────────
if(STATOR_WITH_GMSH)
    find_package(GMSH REQUIRED)
    target_link_libraries(stator_core PUBLIC gmsh)
    target_compile_definitions(stator_core PUBLIC STATOR_WITH_GMSH)
    target_sources(stator_core PRIVATE src/gmsh_real_backend.cpp)
endif()

# ─── Optional: Intel TBB ──────────────────────────────────────────────────────
if(STATOR_WITH_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(stator_core PUBLIC TBB::tbb)
    target_compile_definitions(stator_core PUBLIC STATOR_WITH_TBB)
endif()

# ─── Optional: HDF5 ───────────────────────────────────────────────────────────
if(STATOR_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    target_link_libraries(stator_core PUBLIC HDF5::HDF5)
    target_compile_definitions(stator_core PUBLIC STATOR_WITH_HDF5)
endif()

# ─── Optional: pybind11 Python module ─────────────────────────────────────────
if(STATOR_WITH_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_stator_core bindings/python_bindings.cpp)
    target_link_libraries(_stator_core PRIVATE stator_core)
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(STATOR_BUILD_TESTS)
    add_executable(test_stator tests/test_stator.cpp)
    target_link_libraries(test_stator PRIVATE stator_core)
    target_compile_options(test_stator PRIVATE -Wall -Wextra)
    enable_testing()
    add_test(NAME StatorPipelineTests COMMAND test_stator)
endif()
