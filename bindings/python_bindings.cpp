// python_bindings.cpp — pybind11 module: _stator_core
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "stator/params.hpp"
#include "stator/topology_registry.hpp"
#include "stator/gmsh_backend.hpp"
#include "stator/mesh_generator.hpp"
#include "stator/export_engine.hpp"
#include "stator/batch_scheduler.hpp"

namespace py = pybind11;
using namespace stator;

PYBIND11_MODULE(_stator_core, m) {
    m.doc() = "Stator mesh construction pipeline — C++ core";

    // ── Enums ─────────────────────────────────────────────────────────────────
    py::enum_<SlotShape>(m, "SlotShape")
        .value("RECTANGULAR",  SlotShape::RECTANGULAR)
        .value("TRAPEZOIDAL",  SlotShape::TRAPEZOIDAL)
        .value("ROUND_BOTTOM", SlotShape::ROUND_BOTTOM)
        .value("SEMI_CLOSED",  SlotShape::SEMI_CLOSED)
        .export_values();

    py::enum_<WindingType>(m, "WindingType")
        .value("SINGLE_LAYER", WindingType::SINGLE_LAYER)
        .value("DOUBLE_LAYER", WindingType::DOUBLE_LAYER)
        .value("CONCENTRATED", WindingType::CONCENTRATED)
        .value("DISTRIBUTED",  WindingType::DISTRIBUTED)
        .export_values();

    py::enum_<LaminationMaterial>(m, "LaminationMaterial")
        .value("M270_35A", LaminationMaterial::M270_35A)
        .value("M330_50A", LaminationMaterial::M330_50A)
        .value("M400_50A", LaminationMaterial::M400_50A)
        .value("NO20",     LaminationMaterial::NO20)
        .value("CUSTOM",   LaminationMaterial::CUSTOM)
        .export_values();

    py::enum_<ExportFormat>(m, "ExportFormat")
        .value("NONE", ExportFormat::NONE)
        .value("MSH",  ExportFormat::MSH)
        .value("VTK",  ExportFormat::VTK)
        .value("HDF5", ExportFormat::HDF5)
        .value("JSON", ExportFormat::JSON)
        .value("ALL",  ExportFormat::ALL)
        .export_values();

    // ── StatorParams ──────────────────────────────────────────────────────────
    py::class_<StatorParams>(m, "StatorParams")
        // Section 1
        .def_readwrite("R_outer",       &StatorParams::R_outer)
        .def_readwrite("R_inner",       &StatorParams::R_inner)
        .def_readwrite("airgap_length", &StatorParams::airgap_length)
        // Section 2
        .def_readwrite("n_slots",             &StatorParams::n_slots)
        .def_readwrite("slot_depth",          &StatorParams::slot_depth)
        .def_readwrite("slot_width_outer",    &StatorParams::slot_width_outer)
        .def_readwrite("slot_width_inner",    &StatorParams::slot_width_inner)
        .def_readwrite("slot_opening",        &StatorParams::slot_opening)
        .def_readwrite("slot_opening_depth",  &StatorParams::slot_opening_depth)
        .def_readwrite("tooth_tip_angle",     &StatorParams::tooth_tip_angle)
        .def_readwrite("slot_shape",          &StatorParams::slot_shape)
        // Section 3
        .def_readwrite("coil_depth",                 &StatorParams::coil_depth)
        .def_readwrite("coil_width_outer",           &StatorParams::coil_width_outer)
        .def_readwrite("coil_width_inner",           &StatorParams::coil_width_inner)
        .def_readwrite("insulation_thickness",       &StatorParams::insulation_thickness)
        .def_readwrite("turns_per_coil",             &StatorParams::turns_per_coil)
        .def_readwrite("coil_pitch",                 &StatorParams::coil_pitch)
        .def_readwrite("wire_diameter",              &StatorParams::wire_diameter)
        .def_readwrite("slot_fill_factor",           &StatorParams::slot_fill_factor)
        .def_readwrite("winding_type",               &StatorParams::winding_type)
        // Section 4
        .def_readwrite("t_lam",                        &StatorParams::t_lam)
        .def_readwrite("n_lam",                        &StatorParams::n_lam)
        .def_readwrite("z_spacing",                    &StatorParams::z_spacing)
        .def_readwrite("insulation_coating_thickness", &StatorParams::insulation_coating_thickness)
        .def_readwrite("material",                     &StatorParams::material)
        .def_readwrite("material_file",                &StatorParams::material_file)
        // Section 5
        .def_readwrite("mesh_yoke",              &StatorParams::mesh_yoke)
        .def_readwrite("mesh_slot",              &StatorParams::mesh_slot)
        .def_readwrite("mesh_coil",              &StatorParams::mesh_coil)
        .def_readwrite("mesh_ins",               &StatorParams::mesh_ins)
        .def_readwrite("mesh_boundary_layers",   &StatorParams::mesh_boundary_layers)
        .def_readwrite("mesh_curvature",         &StatorParams::mesh_curvature)
        .def_readwrite("mesh_transition_layers", &StatorParams::mesh_transition_layers)
        // Section 6 — derived (read-only from Python perspective)
        .def_readonly("yoke_height",  &StatorParams::yoke_height)
        .def_readonly("tooth_width",  &StatorParams::tooth_width)
        .def_readonly("slot_pitch",   &StatorParams::slot_pitch)
        .def_readonly("stack_length", &StatorParams::stack_length)
        .def_readonly("fill_factor",  &StatorParams::fill_factor)
        // Methods
        .def("validate_and_derive", &StatorParams::validate_and_derive)
        .def("to_json",             &StatorParams::to_json)
        .def("__repr__", [](const StatorParams& p) {
            return "<StatorParams n_slots=" + std::to_string(p.n_slots) + ">";
        });

    // ── ExportConfig ──────────────────────────────────────────────────────────
    py::class_<ExportConfig>(m, "ExportConfig")
        .def(py::init<>())
        .def_readwrite("formats",      &ExportConfig::formats)
        .def_readwrite("output_dir",   &ExportConfig::output_dir)
        .def_readwrite("msh_version",  &ExportConfig::msh_version);

    // ── MeshConfig ────────────────────────────────────────────────────────────
    py::class_<MeshConfig>(m, "MeshConfig")
        .def(py::init<>())
        .def_readwrite("algorithm_2d",          &MeshConfig::algorithm_2d)
        .def_readwrite("algorithm_3d",          &MeshConfig::algorithm_3d)
        .def_readwrite("smoothing_passes",      &MeshConfig::smoothing_passes)
        .def_readwrite("optimiser",             &MeshConfig::optimiser)
        .def_readwrite("min_quality_threshold", &MeshConfig::min_quality_threshold)
        .def_readwrite("periodic",              &MeshConfig::periodic)
        .def_readwrite("layers_per_lam",        &MeshConfig::layers_per_lam);

    // ── BatchJob ──────────────────────────────────────────────────────────────
    py::class_<BatchJob>(m, "BatchJob")
        .def(py::init<>())
        .def_readwrite("params",        &BatchJob::params)
        .def_readwrite("export_config", &BatchJob::export_config)
        .def_readwrite("mesh_config",   &BatchJob::mesh_config)
        .def_readwrite("job_id",        &BatchJob::job_id);

    // ── BatchSchedulerConfig ──────────────────────────────────────────────────
    py::class_<BatchSchedulerConfig>(m, "BatchSchedulerConfig")
        .def(py::init<>())
        .def_readwrite("max_parallel",    &BatchSchedulerConfig::max_parallel)
        .def_readwrite("skip_existing",   &BatchSchedulerConfig::skip_existing)
        .def_readwrite("job_timeout_sec", &BatchSchedulerConfig::job_timeout_sec)
        .def_readwrite("write_summary",   &BatchSchedulerConfig::write_summary);

    // ── BatchResult ───────────────────────────────────────────────────────────
    py::class_<BatchResult>(m, "BatchResult")
        .def_readonly("job_id",    &BatchResult::job_id)
        .def_readonly("success",   &BatchResult::success)
        .def_readonly("error",     &BatchResult::error)
        .def_readonly("msh_path",  &BatchResult::msh_path)
        .def_readonly("vtk_path",  &BatchResult::vtk_path)
        .def_readonly("hdf5_path", &BatchResult::hdf5_path)
        .def_readonly("json_path", &BatchResult::json_path);

    // ── BatchScheduler ────────────────────────────────────────────────────────
    py::class_<BatchScheduler>(m, "BatchScheduler")
        .def(py::init<>())
        .def("set_progress_callback", &BatchScheduler::set_progress_callback)
        .def("run", [](BatchScheduler& self,
                       const std::vector<BatchJob>& jobs,
                       const BatchSchedulerConfig& config) {
            py::gil_scoped_release release;
            return self.run(jobs, config);
        }, py::arg("jobs"), py::arg("config") = BatchSchedulerConfig{})
        .def("cancel",     &BatchScheduler::cancel)
        .def("is_running", &BatchScheduler::is_running);

    // ── Free functions ────────────────────────────────────────────────────────
    m.def("make_reference_params", &make_reference_params,
          "Return a validated 36-slot reference design.");
    m.def("make_minimal_params",   &make_minimal_params,
          "Return a validated 12-slot minimal design.");
    m.def("sha256", &sha256, py::arg("data"),
          "Compute SHA-256 hex digest of a string.");
}
