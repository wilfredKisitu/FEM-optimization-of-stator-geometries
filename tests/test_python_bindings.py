"""test_python_bindings.py — pytest suite for the stator_pipeline Python API.

Tests the ctypes bindings against libstator_c_core.so.

Run from project root:
    python -m pytest tests/test_python_bindings.py -v
"""
import json
import math
import os
import sys
import tempfile

import pytest

# Make the package importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import stator_pipeline as sp
from stator_pipeline.pipeline_c import _lib, validate_config, sha256


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def ref_cfg():
    return sp.make_reference_params()


@pytest.fixture()
def min_cfg():
    return sp.make_minimal_params()


# ────────────────────────────────────────────────────────────────────────────
# 1. Library loading
# ────────────────────────────────────────────────────────────────────────────

class TestLibraryLoading:
    def test_lib_loaded(self):
        assert _lib is not None, "libstator_c_core.so failed to load"

    def test_required_symbols(self):
        for sym in [
            "stator_params_validate_and_derive",
            "stator_sha256",
            "stator_params_to_json",
            "stator_make_reference_params",
            "stator_make_minimal_params",
        ]:
            assert hasattr(_lib, sym), f"Missing C symbol: {sym}"


# ────────────────────────────────────────────────────────────────────────────
# 2. Enum constants
# ────────────────────────────────────────────────────────────────────────────

class TestEnumConstants:
    def test_slot_shape_values(self):
        assert sp.SlotShape.RECTANGULAR  == 0
        assert sp.SlotShape.TRAPEZOIDAL  == 1
        assert sp.SlotShape.ROUND_BOTTOM == 2
        assert sp.SlotShape.SEMI_CLOSED  == 3

    def test_winding_type_values(self):
        assert sp.WindingType.SINGLE_LAYER == 0
        assert sp.WindingType.DOUBLE_LAYER == 1
        assert sp.WindingType.CONCENTRATED == 2
        assert sp.WindingType.DISTRIBUTED  == 3

    def test_lamination_material_values(self):
        assert sp.LaminationMaterial.M270_35A == 0
        assert sp.LaminationMaterial.M330_50A == 1
        assert sp.LaminationMaterial.M400_50A == 2
        assert sp.LaminationMaterial.NO20     == 3
        assert sp.LaminationMaterial.CUSTOM   == 4

    def test_export_flags_bitmask(self):
        assert sp.EXPORT_NONE == 0
        assert sp.EXPORT_MSH  == 1
        assert sp.EXPORT_VTK  == 2
        assert sp.EXPORT_HDF5 == 4
        assert sp.EXPORT_JSON == 8
        assert sp.EXPORT_ALL  == (sp.EXPORT_MSH | sp.EXPORT_VTK |
                                   sp.EXPORT_HDF5 | sp.EXPORT_JSON)


# ────────────────────────────────────────────────────────────────────────────
# 3. StatorConfig dataclass
# ────────────────────────────────────────────────────────────────────────────

class TestStatorConfig:
    def test_default_construction(self):
        cfg = sp.StatorConfig()
        assert cfg.n_slots == 36
        assert cfg.R_outer == pytest.approx(0.25)
        assert cfg.R_inner == pytest.approx(0.15)
        assert cfg.slot_shape == sp.SlotShape.SEMI_CLOSED
        assert cfg.winding_type == sp.WindingType.DOUBLE_LAYER

    def test_field_assignment(self):
        cfg = sp.StatorConfig(n_slots=48, R_outer=0.30, R_inner=0.20)
        assert cfg.n_slots == 48
        assert cfg.R_outer == pytest.approx(0.30)
        assert cfg.R_inner == pytest.approx(0.20)

    def test_make_reference_params(self, ref_cfg):
        assert isinstance(ref_cfg, sp.StatorConfig)
        assert ref_cfg.n_slots == 36
        assert ref_cfg.slot_shape == sp.SlotShape.SEMI_CLOSED
        assert ref_cfg.winding_type == sp.WindingType.DOUBLE_LAYER

    def test_make_minimal_params(self, min_cfg):
        assert isinstance(min_cfg, sp.StatorConfig)
        assert min_cfg.n_slots == 12
        assert min_cfg.slot_shape == sp.SlotShape.RECTANGULAR
        assert min_cfg.winding_type == sp.WindingType.SINGLE_LAYER


# ────────────────────────────────────────────────────────────────────────────
# 4. validate_config — success paths
# ────────────────────────────────────────────────────────────────────────────

class TestValidateConfigSuccess:
    def test_reference_params_valid(self, ref_cfg):
        r = sp.validate_config(ref_cfg)
        assert r["success"] is True

    def test_minimal_params_valid(self, min_cfg):
        r = sp.validate_config(min_cfg)
        assert r["success"] is True

    def test_derived_yoke_height(self, ref_cfg):
        r = sp.validate_config(ref_cfg)
        expected = ref_cfg.R_outer - ref_cfg.R_inner - ref_cfg.slot_depth
        assert r["yoke_height"] == pytest.approx(expected, rel=1e-6)

    def test_derived_slot_pitch(self, ref_cfg):
        # C code: slot_pitch = 2*pi / n_slots  (angular pitch in radians)
        r = sp.validate_config(ref_cfg)
        expected = 2.0 * math.pi / ref_cfg.n_slots
        assert r["slot_pitch"] == pytest.approx(expected, rel=1e-6)

    def test_derived_stack_length(self, ref_cfg):
        r = sp.validate_config(ref_cfg)
        expected = ref_cfg.n_lam * ref_cfg.t_lam
        assert r["stack_length"] == pytest.approx(expected, rel=1e-6)

    def test_all_slot_shapes_accepted(self):
        base = sp.make_reference_params()
        for shape in [sp.SlotShape.RECTANGULAR, sp.SlotShape.TRAPEZOIDAL,
                      sp.SlotShape.ROUND_BOTTOM, sp.SlotShape.SEMI_CLOSED]:
            cfg = sp.StatorConfig(**{**base.__dict__, "slot_shape": shape})
            r = sp.validate_config(cfg)
            assert r["success"] is True, f"shape {shape} rejected: {r.get('error')}"

    def test_all_winding_types_accepted(self):
        base = sp.make_reference_params()
        for wt in [sp.WindingType.SINGLE_LAYER, sp.WindingType.DOUBLE_LAYER,
                   sp.WindingType.CONCENTRATED, sp.WindingType.DISTRIBUTED]:
            cfg = sp.StatorConfig(**{**base.__dict__, "winding_type": wt})
            r = sp.validate_config(cfg)
            assert r["success"] is True, f"winding {wt} rejected: {r.get('error')}"

    def test_all_materials_accepted(self):
        base = sp.make_reference_params()
        for mat in [sp.LaminationMaterial.M270_35A, sp.LaminationMaterial.M330_50A,
                    sp.LaminationMaterial.M400_50A, sp.LaminationMaterial.NO20]:
            cfg = sp.StatorConfig(**{**base.__dict__, "material": mat})
            r = sp.validate_config(cfg)
            assert r["success"] is True, f"material {mat} rejected: {r.get('error')}"

    def test_custom_material_requires_file(self):
        base = sp.make_reference_params()
        # Without material_file → error
        cfg_no_file = sp.StatorConfig(**{**base.__dict__,
                                         "material": sp.LaminationMaterial.CUSTOM,
                                         "material_file": ""})
        r = sp.validate_config(cfg_no_file)
        assert r["success"] is False
        # With material_file → accepted
        cfg_with_file = sp.StatorConfig(**{**base.__dict__,
                                           "material": sp.LaminationMaterial.CUSTOM,
                                           "material_file": "/data/m400.csv"})
        r2 = sp.validate_config(cfg_with_file)
        assert r2["success"] is True

    def test_returns_all_derived_fields(self, ref_cfg):
        r = sp.validate_config(ref_cfg)
        for key in ["yoke_height", "tooth_width", "slot_pitch",
                    "stack_length", "fill_factor"]:
            assert key in r, f"Missing derived field: {key}"
            assert isinstance(r[key], float)

    def test_fill_factor_range(self, ref_cfg):
        r = sp.validate_config(ref_cfg)
        assert 0.0 <= r["fill_factor"] <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# 5. validate_config — error paths
# ────────────────────────────────────────────────────────────────────────────

class TestValidateConfigErrors:
    def _bad(self, **overrides):
        base = sp.make_reference_params()
        return sp.StatorConfig(**{**base.__dict__, **overrides})

    def test_r_inner_ge_r_outer(self):
        r = sp.validate_config(self._bad(R_inner=0.30, R_outer=0.25))
        assert r["success"] is False
        assert r["error"]

    def test_zero_slots(self):
        r = sp.validate_config(self._bad(n_slots=0))
        assert r["success"] is False

    def test_negative_slot_depth(self):
        r = sp.validate_config(self._bad(slot_depth=-0.01))
        assert r["success"] is False

    def test_slot_too_deep(self):
        r = sp.validate_config(self._bad(slot_depth=0.20))  # deeper than yoke
        assert r["success"] is False

    def test_zero_laminations(self):
        r = sp.validate_config(self._bad(n_lam=0))
        assert r["success"] is False

    def test_negative_insulation(self):
        r = sp.validate_config(self._bad(insulation_thickness=-0.001))
        assert r["success"] is False

    def test_zero_r_outer(self):
        r = sp.validate_config(self._bad(R_outer=0.0))
        assert r["success"] is False

    def test_error_message_is_string(self):
        r = sp.validate_config(self._bad(n_slots=0))
        assert isinstance(r["error"], str)
        assert len(r["error"]) > 0


# ────────────────────────────────────────────────────────────────────────────
# 6. SHA-256
# ────────────────────────────────────────────────────────────────────────────

class TestSHA256:
    def test_empty_string(self):
        h = sp.sha256("")
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_known_digest(self):
        # SHA-256("abc") — matches Python hashlib.sha256(b"abc").hexdigest()
        h = sp.sha256("abc")
        assert h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_output_is_64_hex_chars(self):
        h = sp.sha256("stator mesh pipeline")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert sp.sha256("hello") == sp.sha256("hello")

    def test_different_inputs(self):
        assert sp.sha256("a") != sp.sha256("b")

    def test_unicode_encoded(self):
        h = sp.sha256("stator\u00e9lectrique")
        assert len(h) == 64


# ────────────────────────────────────────────────────────────────────────────
# 7. generate_single
# ────────────────────────────────────────────────────────────────────────────

class TestGenerateSingle:
    def test_success_returns_dict(self, ref_cfg, tmp_path):
        r = sp.generate_single(ref_cfg, str(tmp_path), formats="JSON")
        assert r["success"] is True

    def test_creates_output_dir(self, ref_cfg, tmp_path):
        outdir = str(tmp_path / "new_subdir")
        sp.generate_single(ref_cfg, outdir, formats="JSON")
        assert os.path.isdir(outdir)

    def test_json_file_written(self, ref_cfg, tmp_path):
        r = sp.generate_single(ref_cfg, str(tmp_path), formats="JSON")
        assert "json_path" in r
        assert os.path.isfile(r["json_path"])

    def test_json_file_content(self, ref_cfg, tmp_path):
        r = sp.generate_single(ref_cfg, str(tmp_path), formats="JSON")
        with open(r["json_path"]) as f:
            meta = json.load(f)
        assert meta["n_slots"] == ref_cfg.n_slots
        assert "yoke_height" in meta
        assert "stack_length" in meta

    def test_no_json_without_flag(self, ref_cfg, tmp_path):
        r = sp.generate_single(ref_cfg, str(tmp_path), formats="MSH")
        assert "json_path" not in r

    def test_stem_is_deterministic(self, ref_cfg, tmp_path):
        r1 = sp.generate_single(ref_cfg, str(tmp_path / "a"), formats="JSON")
        r2 = sp.generate_single(ref_cfg, str(tmp_path / "b"), formats="JSON")
        assert r1["stem"] == r2["stem"]

    def test_different_configs_different_stems(self, ref_cfg, min_cfg, tmp_path):
        r1 = sp.generate_single(ref_cfg, str(tmp_path / "a"), formats="JSON")
        r2 = sp.generate_single(min_cfg, str(tmp_path / "b"), formats="JSON")
        assert r1["stem"] != r2["stem"]

    def test_invalid_config_returns_failure(self, tmp_path):
        bad = sp.StatorConfig(n_slots=0)
        r = sp.generate_single(bad, str(tmp_path))
        assert r["success"] is False
        assert "error" in r


# ────────────────────────────────────────────────────────────────────────────
# 8. generate_batch
# ────────────────────────────────────────────────────────────────────────────

class TestGenerateBatch:
    def test_returns_list(self, ref_cfg, min_cfg, tmp_path):
        results = sp.generate_batch([ref_cfg, min_cfg], str(tmp_path), formats="JSON")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_each_result_has_job_id(self, ref_cfg, tmp_path):
        results = sp.generate_batch([ref_cfg], str(tmp_path), formats="JSON")
        assert results[0]["job_id"] == "batch_0"

    def test_all_succeed(self, ref_cfg, min_cfg, tmp_path):
        results = sp.generate_batch([ref_cfg, min_cfg], str(tmp_path), formats="JSON")
        for r in results:
            assert r["success"] is True

    def test_progress_callback_called(self, ref_cfg, min_cfg, tmp_path):
        calls = []
        def cb(done, total, job_id):
            calls.append((done, total, job_id))
        sp.generate_batch([ref_cfg, min_cfg], str(tmp_path),
                           formats="JSON", progress_callback=cb)
        assert len(calls) == 2
        assert calls[0] == (1, 2, "batch_0")
        assert calls[1] == (2, 2, "batch_1")

    def test_empty_batch(self, tmp_path):
        results = sp.generate_batch([], str(tmp_path))
        assert results == []

    def test_invalid_config_in_batch(self, ref_cfg, tmp_path):
        bad = sp.StatorConfig(n_slots=0)
        results = sp.generate_batch([ref_cfg, bad, ref_cfg], str(tmp_path), formats="JSON")
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    def test_creates_output_dir(self, ref_cfg, tmp_path):
        outdir = str(tmp_path / "batch_out")
        sp.generate_batch([ref_cfg], outdir, formats="JSON")
        assert os.path.isdir(outdir)


# ────────────────────────────────────────────────────────────────────────────
# 9. Public API surface
# ────────────────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def test_all_exports_present(self):
        import stator_pipeline as pkg
        expected = [
            "StatorConfig", "SlotShape", "WindingType", "LaminationMaterial",
            "EXPORT_NONE", "EXPORT_MSH", "EXPORT_VTK", "EXPORT_HDF5",
            "EXPORT_JSON", "EXPORT_ALL",
            "validate_config", "sha256",
            "make_reference_params", "make_minimal_params",
            "generate_single", "generate_batch",
        ]
        for name in expected:
            assert hasattr(pkg, name), f"Missing from public API: {name}"

    def test_all_exported_in_all(self):
        import stator_pipeline as pkg
        for name in pkg.__all__:
            assert hasattr(pkg, name)
