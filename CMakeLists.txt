cmake_minimum_required(VERSION 3.22)
project(StatorMeshPipeline VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────────
option(STATOR_WITH_GMSH   "Build with real GMSH C backend"  OFF)
option(STATOR_WITH_HDF5   "Enable HDF5 mesh export"         OFF)
option(STATOR_BUILD_TESTS "Build test executable"           ON)

# ─── C core library ───────────────────────────────────────────────────────────
add_library(stator_c_core SHARED
    csrc/params.c
    csrc/gmsh_backend.c
    csrc/geometry_builder.c
    csrc/topology_registry.c
    csrc/mesh_generator.c
    csrc/export_engine.c
    csrc/batch_scheduler.c)

target_include_directories(stator_c_core PUBLIC include)
target_compile_options(stator_c_core PRIVATE -Wall -Wextra)
target_link_libraries(stator_c_core PRIVATE m pthread)

if(STATOR_WITH_GMSH)
    target_compile_definitions(stator_c_core PUBLIC STATOR_WITH_GMSH)
endif()

if(STATOR_WITH_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C)
    target_link_libraries(stator_c_core PUBLIC HDF5::HDF5)
    target_compile_definitions(stator_c_core PUBLIC STATOR_WITH_HDF5)
endif()

# ─── Tests ────────────────────────────────────────────────────────────────────
if(STATOR_BUILD_TESTS)
    enable_testing()
    add_executable(test_stator_c tests/test_stator.c)
    target_link_libraries(test_stator_c PRIVATE stator_c_core m pthread)
    target_compile_options(test_stator_c PRIVATE -Wall -Wextra)
    add_test(NAME StatorCPipelineTests COMMAND test_stator_c)
endif()

# ─── C Examples ───────────────────────────────────────────────────────────────
option(STATOR_BUILD_EXAMPLES "Build C example executables" OFF)

if(STATOR_BUILD_EXAMPLES)
    # Helper macro: add an example executable linked to stator_c_core
    macro(stator_example name src)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE stator_c_core m pthread)
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endmacro()

    stator_example(single_geometry_c  examples/single_geometry_c.c)
    stator_example(batch_scheduler_c  examples/batch_scheduler_c.c)

    message(STATUS "C examples enabled: single_geometry_c  batch_scheduler_c")
endif()

# ─── Integration test (requires GMSH) ─────────────────────────────────────────
option(STATOR_BUILD_INTEGRATION "Build 3-D GMSH integration test" OFF)

if(STATOR_BUILD_INTEGRATION)
    find_library(GMSH_LIB gmsh HINTS /usr/lib /usr/lib/x86_64-linux-gnu
                                     /usr/local/lib REQUIRED)
    find_path(GMSH_INCLUDE_DIR gmshc.h HINTS /usr/include /usr/local/include REQUIRED)

    add_executable(integration_test tests/integration_test.c)
    target_include_directories(integration_test PRIVATE ${GMSH_INCLUDE_DIR})
    target_link_libraries(integration_test PRIVATE ${GMSH_LIB} m)
    target_compile_options(integration_test PRIVATE -Wall -Wextra)
    message(STATUS "GMSH integration test enabled (${GMSH_LIB})")
endif()
