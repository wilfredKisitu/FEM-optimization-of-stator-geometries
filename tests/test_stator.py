"""test_stator.py — Comprehensive pytest suite for the pure-Python stator pipeline."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from stator_pipeline.params import (
    StatorParams, SlotShape, WindingType, LaminationMaterial,
    validate_and_derive, make_reference_params, make_minimal_params,
)
from stator_pipeline.gmsh_backend import StubGmshBackend, make_default_backend
from stator_pipeline.geometry_builder import GeometryBuilder
from stator_pipeline.topology_registry import TopologyRegistry, RegionType
from stator_pipeline.mesh_generator import MeshGenerator, MeshResult
from stator_pipeline.export_engine import (
    ExportFormat, ExportConfig, ExportEngine, sha256, compute_stem,
)
from stator_pipeline.pipeline import (
    validate_config, generate_single, generate_batch,
)
import stator_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# TestParams
# ─────────────────────────────────────────────────────────────────────────────

class TestParams:
    def test_default_construction(self):
        p = StatorParams()
        assert p.R_outer == 0.25
        assert p.R_inner == 0.15
        assert p.n_slots == 36
        assert p.slot_shape == SlotShape.SEMI_CLOSED
        assert p.winding_type == WindingType.DOUBLE_LAYER
        assert p.material == LaminationMaterial.M270_35A

    def test_field_values(self):
        p = StatorParams()
        assert p.airgap_length == 0.001
        assert p.slot_depth == 0.06
        assert p.slot_width_outer == 0.012
        assert p.slot_width_inner == 0.010
        assert p.slot_opening == 0.004
        assert p.slot_opening_depth == 0.003
        assert p.tooth_tip_angle == 0.1
        assert p.coil_depth == 0.05
        assert p.coil_width_outer == 0.008
        assert p.coil_width_inner == 0.007
        assert p.insulation_thickness == 0.001
        assert p.turns_per_coil == 10
        assert p.coil_pitch == 5
        assert p.wire_diameter == 0.001
        assert p.slot_fill_factor == 0.45
        assert p.t_lam == 0.00035
        assert p.n_lam == 200
        assert p.z_spacing == 0.0
        assert p.insulation_coating_thickness == 0.00005
        assert p.material_file == ""
        assert p.mesh_yoke == 0.006
        assert p.mesh_slot == 0.003
        assert p.mesh_coil == 0.0015
        assert p.mesh_ins == 0.0007
        assert p.mesh_boundary_layers == 3
        assert p.mesh_curvature == 0.3
        assert p.mesh_transition_layers == 2

    def test_all_enums(self):
        assert SlotShape.RECTANGULAR == 0
        assert SlotShape.TRAPEZOIDAL == 1
        assert SlotShape.ROUND_BOTTOM == 2
        assert SlotShape.SEMI_CLOSED == 3
        assert WindingType.SINGLE_LAYER == 0
        assert WindingType.DOUBLE_LAYER == 1
        assert WindingType.CONCENTRATED == 2
        assert WindingType.DISTRIBUTED == 3
        assert LaminationMaterial.M270_35A == 0
        assert LaminationMaterial.M330_50A == 1
        assert LaminationMaterial.M400_50A == 2
        assert LaminationMaterial.NO20 == 3
        assert LaminationMaterial.CUSTOM == 4

    def test_make_reference_params(self):
        p = make_reference_params()
        assert p.n_slots == 36
        assert p.yoke_height > 0
        assert p.fill_factor > 0

    def test_make_minimal_params(self):
        p = make_minimal_params()
        assert p.n_slots == 12
        assert p.R_outer == 0.12
        assert p.yoke_height > 0
        assert p.fill_factor > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestValidation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_default(self):
        p = validate_and_derive(StatorParams())
        assert p.yoke_height > 0

    def test_rule1_positive_dimensions(self):
        for field_name in ["R_outer", "R_inner", "airgap_length", "slot_depth",
                           "slot_width_outer", "slot_width_inner", "coil_depth",
                           "coil_width_outer", "coil_width_inner",
                           "insulation_thickness", "wire_diameter", "t_lam",
                           "mesh_yoke", "mesh_slot", "mesh_coil", "mesh_ins"]:
            p = StatorParams(**{field_name: 0.0})
            with pytest.raises(ValueError, match="must be > 0"):
                validate_and_derive(p)

    def test_rule2_r_inner_lt_r_outer(self):
        with pytest.raises(ValueError, match="R_inner"):
            validate_and_derive(StatorParams(R_inner=0.30, R_outer=0.25))

    def test_rule3_slot_depth(self):
        with pytest.raises(ValueError, match="slot_depth"):
            validate_and_derive(StatorParams(slot_depth=0.11))

    def test_rule4_n_slots_min(self):
        with pytest.raises(ValueError, match="n_slots"):
            validate_and_derive(StatorParams(n_slots=4))

    def test_rule4_n_slots_even(self):
        with pytest.raises(ValueError, match="even"):
            validate_and_derive(StatorParams(n_slots=37))

    def test_rule5_slot_width_inner(self):
        with pytest.raises(ValueError, match="slot_width_inner"):
            validate_and_derive(StatorParams(slot_width_inner=0.1))

    def test_rule6_semi_closed_opening(self):
        with pytest.raises(ValueError, match="slot_opening"):
            validate_and_derive(StatorParams(
                slot_shape=SlotShape.SEMI_CLOSED,
                slot_opening=0.015,  # > slot_width_inner
            ))

    def test_rule6_semi_closed_opening_depth(self):
        with pytest.raises(ValueError, match="slot_opening_depth"):
            validate_and_derive(StatorParams(
                slot_shape=SlotShape.SEMI_CLOSED,
                slot_opening_depth=0.07,  # > slot_depth
            ))

    def test_rule7_coil_depth(self):
        with pytest.raises(ValueError, match="coil_depth"):
            validate_and_derive(StatorParams(coil_depth=0.058))

    def test_rule8_coil_width_inner(self):
        with pytest.raises(ValueError, match="coil_width_inner"):
            validate_and_derive(StatorParams(coil_width_inner=0.009))

    def test_rule9_n_lam(self):
        with pytest.raises(ValueError, match="n_lam"):
            validate_and_derive(StatorParams(n_lam=0))

    def test_rule10_z_spacing(self):
        with pytest.raises(ValueError, match="z_spacing"):
            validate_and_derive(StatorParams(z_spacing=-0.001))

    def test_rule11_insulation_coating(self):
        with pytest.raises(ValueError, match="insulation_coating_thickness"):
            validate_and_derive(StatorParams(insulation_coating_thickness=-1e-6))

    def test_rule12_custom_material_no_file(self):
        with pytest.raises(ValueError, match="material_file"):
            validate_and_derive(StatorParams(
                material=LaminationMaterial.CUSTOM, material_file="",
            ))

    def test_rule13_mesh_ordering(self):
        with pytest.raises(ValueError, match="mesh sizes"):
            validate_and_derive(StatorParams(mesh_ins=0.01, mesh_coil=0.001))

    def test_rule14_tooth_tip_angle(self):
        with pytest.raises(ValueError, match="tooth_tip_angle"):
            validate_and_derive(StatorParams(tooth_tip_angle=math.pi / 2))

    def test_rule15_fill_factor(self):
        # Craft params where fill_factor > 1:
        # slot_area = 0.5*(swi+swo)*sd = 0.5*(0.010+0.005)*0.06 = 0.00045
        # coil_area = 0.5*(cwi+cwo)*cd = 0.5*(0.007+0.005)*0.056 = 0.000336
        # That gives ff<1. Instead, use slot_width_outer very small:
        # slot_area = 0.5*(0.010+0.004)*0.06 = 0.00042
        # coil_area = 0.5*(0.007+0.008)*0.056 = 0.00042  -- still <1
        # The easiest approach: make slot_width_outer << coil_width_outer
        # but that's geometrically odd. Actually we just need to verify the
        # validation catches it. Use a direct unit test on the check:
        import dataclasses
        p = StatorParams()
        # First validate normally to get a good base
        vp = validate_and_derive(p)
        # Then force fill_factor to >= 1 and check the rule directly
        bad = dataclasses.replace(vp, fill_factor=1.5)
        # The validation function computes fill_factor itself, so we need
        # params that actually produce ff >= 1.
        # Use very narrow slot (small slot_width_outer) with normal coil:
        # slot_area = 0.5*(swi + swo)*sd, coil_area = 0.5*(cwi+cwo)*cd
        # swi=0.010, swo=0.002, sd=0.06 => sa=0.00036
        # cwi=0.007, cwo=0.008, cd=0.05 => ca=0.000375 => ff=1.04
        # But swo=0.002 should still pass other rules.
        with pytest.raises(ValueError, match="fill_factor"):
            validate_and_derive(StatorParams(
                slot_width_outer=0.002,
                # Everything else default -- slot_width_inner=0.010 > swo
                # That's fine, these are independent geometric params
            ))


# ─────────────────────────────────────────────────────────────────────────────
# TestDerivedFields
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedFields:
    def test_slot_pitch(self):
        p = validate_and_derive(StatorParams())
        assert abs(p.slot_pitch - 2.0 * math.pi / 36) < 1e-12

    def test_yoke_height(self):
        p = validate_and_derive(StatorParams())
        expected = 0.25 - 0.15 - 0.06
        assert abs(p.yoke_height - expected) < 1e-12

    def test_tooth_width(self):
        p = validate_and_derive(StatorParams())
        expected = 0.15 * (2.0 * math.pi / 36) - 0.010
        assert abs(p.tooth_width - expected) < 1e-12

    def test_stack_length(self):
        p = validate_and_derive(StatorParams())
        expected = 200 * 0.00035 + 199 * 0.0
        assert abs(p.stack_length - expected) < 1e-12

    def test_fill_factor(self):
        p = validate_and_derive(StatorParams())
        slot_area = 0.5 * (0.010 + 0.012) * 0.06
        coil_area = 0.5 * (0.007 + 0.008) * 0.05
        expected = coil_area / slot_area
        assert abs(p.fill_factor - expected) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# TestSlotShapes
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotShapes:
    def test_rectangular(self):
        p = StatorParams(slot_shape=SlotShape.RECTANGULAR, tooth_tip_angle=0.0)
        v = validate_and_derive(p)
        assert v.fill_factor > 0

    def test_trapezoidal(self):
        p = StatorParams(slot_shape=SlotShape.TRAPEZOIDAL, tooth_tip_angle=0.0)
        v = validate_and_derive(p)
        assert v.fill_factor > 0

    def test_round_bottom(self):
        p = StatorParams(slot_shape=SlotShape.ROUND_BOTTOM, tooth_tip_angle=0.0)
        v = validate_and_derive(p)
        assert v.fill_factor > 0

    def test_semi_closed(self):
        p = StatorParams(slot_shape=SlotShape.SEMI_CLOSED)
        v = validate_and_derive(p)
        assert v.fill_factor > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestWindingTypes
# ─────────────────────────────────────────────────────────────────────────────

class TestWindingTypes:
    def test_single_layer(self):
        v = validate_and_derive(StatorParams(winding_type=WindingType.SINGLE_LAYER))
        assert v.fill_factor > 0

    def test_double_layer(self):
        v = validate_and_derive(StatorParams(winding_type=WindingType.DOUBLE_LAYER))
        assert v.fill_factor > 0

    def test_concentrated(self):
        v = validate_and_derive(StatorParams(winding_type=WindingType.CONCENTRATED))
        assert v.fill_factor > 0

    def test_distributed(self):
        v = validate_and_derive(StatorParams(winding_type=WindingType.DISTRIBUTED))
        assert v.fill_factor > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestMaterials
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterials:
    def test_all_standard_materials(self):
        for mat in [LaminationMaterial.M270_35A, LaminationMaterial.M330_50A,
                    LaminationMaterial.M400_50A, LaminationMaterial.NO20]:
            v = validate_and_derive(StatorParams(material=mat))
            assert v.fill_factor > 0

    def test_custom_with_file(self):
        v = validate_and_derive(StatorParams(
            material=LaminationMaterial.CUSTOM,
            material_file="/path/to/custom.bh",
        ))
        assert v.fill_factor > 0

    def test_custom_without_file_fails(self):
        with pytest.raises(ValueError, match="material_file"):
            validate_and_derive(StatorParams(
                material=LaminationMaterial.CUSTOM, material_file="",
            ))


# ─────────────────────────────────────────────────────────────────────────────
# TestSHA256
# ─────────────────────────────────────────────────────────────────────────────

class TestSHA256:
    def test_empty_string(self):
        h = sha256("")
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_abc(self):
        h = sha256("abc")
        assert h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_deterministic(self):
        assert sha256("test") == sha256("test")

    def test_hex_length(self):
        h = sha256("anything")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ─────────────────────────────────────────────────────────────────────────────
# TestGmshBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestGmshBackend:
    def test_stub_tracks_points(self):
        b = StubGmshBackend()
        t1 = b.add_point(0, 0, 0, 1.0)
        t2 = b.add_point(1, 0, 0, 1.0)
        assert t1 == 1
        assert t2 == 2
        assert b._point_counter == 2

    def test_stub_tracks_lines(self):
        b = StubGmshBackend()
        t = b.add_line(1, 2)
        assert t == 1
        assert b._line_counter == 1

    def test_stub_tracks_surfaces(self):
        b = StubGmshBackend()
        s = b.add_plane_surface([1])
        assert s == 1
        assert len(b._surfaces_2d) == 1

    def test_boolean_cut_returns_objects(self):
        b = StubGmshBackend()
        result = b.boolean_cut([(2, 1), (2, 2)], [(2, 3)])
        assert result == [(2, 1), (2, 2)]

    def test_boolean_fragment_returns_both(self):
        b = StubGmshBackend()
        result = b.boolean_fragment([(2, 1)], [(2, 2)])
        assert result == [(2, 1), (2, 2)]

    def test_phys_group(self):
        b = StubGmshBackend()
        tag = b.add_physical_group(2, [1, 2], "test", 42)
        assert tag == 42
        assert len(b._phys_groups) == 1

    def test_init_finalize(self):
        b = StubGmshBackend()
        b.initialize("model")
        assert b._initialized
        b.finalize()
        assert b._finalized

    def test_make_default_backend(self):
        b = make_default_backend()
        assert isinstance(b, StubGmshBackend)


# ─────────────────────────────────────────────────────────────────────────────
# TestGeometryBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometryBuilder:
    def _build_with_shape(self, shape: SlotShape, winding: WindingType = WindingType.DOUBLE_LAYER):
        p = StatorParams(slot_shape=shape, winding_type=winding, tooth_tip_angle=0.0)
        if shape == SlotShape.SEMI_CLOSED:
            p = StatorParams(slot_shape=shape, winding_type=winding)
        vp = validate_and_derive(p)
        backend = StubGmshBackend()
        builder = GeometryBuilder(backend)
        return builder.build(vp), backend

    def test_rectangular(self):
        geo, _ = self._build_with_shape(SlotShape.RECTANGULAR)
        assert geo.success
        assert geo.n_slots == 36

    def test_trapezoidal(self):
        geo, _ = self._build_with_shape(SlotShape.TRAPEZOIDAL)
        assert geo.success

    def test_round_bottom(self):
        geo, _ = self._build_with_shape(SlotShape.ROUND_BOTTOM)
        assert geo.success

    def test_semi_closed(self):
        geo, _ = self._build_with_shape(SlotShape.SEMI_CLOSED)
        assert geo.success

    def test_single_layer_winding(self):
        geo, _ = self._build_with_shape(SlotShape.RECTANGULAR, WindingType.SINGLE_LAYER)
        assert geo.success
        # Single layer: only upper coil surface should be set
        for sp in geo.slots:
            assert sp.coil_upper_sf >= 0
            assert sp.coil_lower_sf == -1

    def test_double_layer_winding(self):
        geo, _ = self._build_with_shape(SlotShape.SEMI_CLOSED, WindingType.DOUBLE_LAYER)
        assert geo.success
        for sp in geo.slots:
            assert sp.coil_upper_sf >= 0
            assert sp.coil_lower_sf >= 0

    def test_slot_count(self):
        geo, _ = self._build_with_shape(SlotShape.RECTANGULAR)
        assert len(geo.slots) == 36

    def test_yoke_surface_set(self):
        geo, _ = self._build_with_shape(SlotShape.RECTANGULAR)
        assert geo.yoke_surface >= 0

    def test_bore_and_outer_curves(self):
        geo, _ = self._build_with_shape(SlotShape.RECTANGULAR)
        assert geo.bore_curve >= 0
        assert geo.outer_curve >= 0


# ─────────────────────────────────────────────────────────────────────────────
# TestTopologyRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestTopologyRegistry:
    def test_register_surfaces(self):
        reg = TopologyRegistry(12)
        reg.register_surface(RegionType.YOKE, 1)
        reg.register_surface(RegionType.YOKE, 2)
        assert reg.get_surfaces(RegionType.YOKE) == [1, 2]
        assert reg.total_surfaces == 2

    def test_register_boundary(self):
        reg = TopologyRegistry(12)
        reg.register_boundary_curve(RegionType.BOUNDARY_BORE, 10)
        assert reg.get_boundary_curves(RegionType.BOUNDARY_BORE) == [10]

    def test_invalid_boundary_type(self):
        reg = TopologyRegistry(12)
        with pytest.raises(ValueError, match="BOUNDARY"):
            reg.register_boundary_curve(RegionType.YOKE, 1)

    def test_assign_winding(self):
        reg = TopologyRegistry(12)
        for i in range(12):
            reg.register_slot_coil(i, i * 10, i * 10 + 1)
        reg.assign_winding_layout(WindingType.DISTRIBUTED)
        assert reg.winding_assigned
        wa = reg.get_slot_assignment(0)
        assert wa.upper_phase == RegionType.COIL_A_POS

    def test_get_slot_before_assign_fails(self):
        reg = TopologyRegistry(12)
        with pytest.raises(RuntimeError, match="not yet assigned"):
            reg.get_slot_assignment(0)

    def test_slot_idx_out_of_range(self):
        reg = TopologyRegistry(12)
        with pytest.raises(IndexError):
            reg.register_slot_coil(99, 1, 2)

    def test_n_slots_zero(self):
        with pytest.raises(ValueError):
            TopologyRegistry(0)

    def test_thread_safety(self):
        reg = TopologyRegistry(36)
        errors = []

        def worker(start: int):
            try:
                for i in range(start, start + 6):
                    reg.register_surface(RegionType.SLOT_AIR, i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 6,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert reg.total_surfaces == 36


# ─────────────────────────────────────────────────────────────────────────────
# TestMeshGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestMeshGenerator:
    def test_generate_success(self):
        p = validate_and_derive(StatorParams())
        backend = StubGmshBackend()
        builder = GeometryBuilder(backend)
        geo = builder.build(p)
        assert geo.success
        reg = TopologyRegistry(p.n_slots)
        gen = MeshGenerator(backend)
        mesh = gen.generate(p, geo, reg)
        assert mesh.success
        assert mesh.n_nodes > 0
        assert mesh.n_elements_2d > 0

    def test_generate_fails_on_bad_geo(self):
        from stator_pipeline.geometry_builder import GeometryBuildResult
        p = validate_and_derive(StatorParams())
        backend = StubGmshBackend()
        geo = GeometryBuildResult(success=False, error_message="test failure")
        reg = TopologyRegistry(p.n_slots)
        gen = MeshGenerator(backend)
        mesh = gen.generate(p, geo, reg)
        assert not mesh.success
        assert "test failure" in mesh.error_message


# ─────────────────────────────────────────────────────────────────────────────
# TestExportEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestExportEngine:
    def test_compute_stem_deterministic(self):
        p = validate_and_derive(StatorParams())
        s1 = compute_stem(p)
        s2 = compute_stem(p)
        assert s1 == s2
        assert s1.startswith("stator_")
        assert len(s1) == len("stator_") + 8

    def test_compute_stem_different_params(self):
        p1 = validate_and_derive(StatorParams(n_slots=36))
        p2 = validate_and_derive(StatorParams(n_slots=24))
        assert compute_stem(p1) != compute_stem(p2)

    def test_write_json(self):
        p = validate_and_derive(StatorParams())
        backend = StubGmshBackend()
        mesh = MeshResult(success=True, n_nodes=100, n_elements_2d=200)
        engine = ExportEngine(backend)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExportConfig(output_dir=tmpdir, formats=ExportFormat.JSON)
            results = engine.write_all(p, mesh, cfg)
            assert len(results) == 1
            assert results[0].success
            assert os.path.isfile(results[0].path)
            with open(results[0].path) as f:
                data = json.load(f)
            assert "stem" in data
            assert data["params"]["n_slots"] == 36


# ─────────────────────────────────────────────────────────────────────────────
# TestPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_validate_config_success(self):
        result = validate_config(StatorParams())
        assert result["success"] is True
        assert "yoke_height" in result
        assert "fill_factor" in result

    def test_validate_config_failure(self):
        result = validate_config(StatorParams(R_inner=0.30))
        assert result["success"] is False
        assert "error" in result

    def test_generate_single_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_single(StatorParams(), tmpdir, formats="JSON")
            assert result["success"]
            assert "json_path" in result
            assert os.path.isfile(result["json_path"])

    def test_generate_single_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_single(StatorParams(R_inner=0.30), tmpdir)
            assert not result["success"]

    def test_generate_batch_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = [StatorParams(), StatorParams(n_lam=100)]
            results = generate_batch(configs, tmpdir, formats="JSON")
            assert len(results) == 2
            assert all(r["success"] for r in results)

    def test_generate_batch_progress_callback(self):
        calls = []

        def cb(done, total, job_id):
            calls.append((done, total, job_id))

        with tempfile.TemporaryDirectory() as tmpdir:
            configs = [StatorParams()]
            generate_batch(configs, tmpdir, formats="JSON", progress_callback=cb)
            assert len(calls) == 1
            assert calls[0][0] == 1
            assert calls[0][1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestPublicAPI
# ─────────────────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def test_all_names_in_module(self):
        expected = [
            "StatorParams", "StatorConfig", "SlotShape", "WindingType",
            "LaminationMaterial", "ExportFormat",
            "EXPORT_NONE", "EXPORT_MSH", "EXPORT_VTK", "EXPORT_HDF5",
            "EXPORT_JSON", "EXPORT_ALL",
            "validate_and_derive", "validate_config", "sha256",
            "make_reference_params", "make_minimal_params",
            "generate_single", "generate_batch",
            "StatorVisualiser",
        ]
        for name in expected:
            assert hasattr(stator_pipeline, name), f"Missing: {name}"
            assert name in stator_pipeline.__all__, f"Not in __all__: {name}"
