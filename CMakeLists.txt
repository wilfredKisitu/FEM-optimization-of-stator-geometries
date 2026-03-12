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

# ─── 3-D Integration Test (requires GMSH with FLTK GUI support) ───────────────
# Build with:  cmake -DSTATOR_BUILD_INTEGRATION=ON ..
#              make integration_test
#              ./integration_test
option(STATOR_BUILD_INTEGRATION "Build 3-D GMSH integration test with live visualisation" OFF)

if(STATOR_BUILD_INTEGRATION)
    find_path(GMSH_INCLUDE_DIR gmsh.h
              PATHS /usr/include /usr/local/include
              DOC "GMSH C++ API header directory")

    find_library(GMSH_LIB gmsh
                 PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
                 DOC "GMSH shared library")

    if(NOT GMSH_INCLUDE_DIR)
        message(FATAL_ERROR
            "GMSH header (gmsh.h) not found. Install libgmsh-dev:\n"
            "  sudo apt install libgmsh-dev")
    endif()
    if(NOT GMSH_LIB)
        message(FATAL_ERROR
            "GMSH library (libgmsh.so) not found. Install libgmsh-dev:\n"
            "  sudo apt install libgmsh-dev")
    endif()

    message(STATUS "Integration test: GMSH include=${GMSH_INCLUDE_DIR}")
    message(STATUS "Integration test: GMSH library=${GMSH_LIB}")

    add_executable(integration_test tests/integration_test.cpp)
    target_include_directories(integration_test PRIVATE
        ${GMSH_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/include)
    # Link stator_core (params/validation) and gmsh (geometry + GUI)
    target_link_libraries(integration_test PRIVATE stator_core ${GMSH_LIB})
    target_compile_definitions(integration_test PRIVATE STATOR_WITH_GMSH)
    target_compile_options(integration_test PRIVATE -Wall -Wextra)
    target_compile_features(integration_test PRIVATE cxx_std_20)
endif()
