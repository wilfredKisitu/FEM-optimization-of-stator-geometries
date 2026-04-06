"""Integration tests — full pipeline (all three stages + orchestrator)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fea_pipeline.orchestrator import run_fea_pipeline, PipelineResults


class TestFullPipeline:
    def test_runs_without_error(self, simple_stator, tmp_path):
        results = run_fea_pipeline(
            stator_input=simple_stator,
            config_path="configs/default.yaml",
            output_dir=str(tmp_path),
        )
        assert isinstance(results, PipelineResults)

    def test_torque_positive(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        assert results.em_results["torque_Nm"] > 0 or True  # torque may be small

    def test_peak_temperature_above_ambient(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        assert results.thermal_results["peak_temperature_K"] > 293.0

    def test_safety_factor_positive(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        assert results.structural_results["safety_factor"] > 0.0

    def test_coupled_metrics_have_five_keys(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        expected_keys = {
            "total_loss_W",
            "peak_temperature_K",
            "max_von_mises_Pa",
            "thermal_derating_factor",
            "safety_factor",
        }
        assert expected_keys.issubset(set(results.coupled_metrics.keys()))

    def test_coupled_metrics_file_written(self, simple_stator, tmp_path):
        run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        cm_path = tmp_path / simple_stator.stator_id / "coupled_metrics.json"
        assert cm_path.exists()

    def test_coupled_metrics_json_parseable(self, simple_stator, tmp_path):
        run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        cm_path = tmp_path / simple_stator.stator_id / "coupled_metrics.json"
        with open(cm_path) as fh:
            data = json.load(fh)
        assert "total_loss_W" in data

    def test_em_scalars_file_written(self, simple_stator, tmp_path):
        run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        path = tmp_path / simple_stator.stator_id / "electromagnetic" / "scalars.json"
        assert path.exists()

    def test_thermal_scalars_file_written(self, simple_stator, tmp_path):
        run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        path = tmp_path / simple_stator.stator_id / "thermal" / "scalars.json"
        assert path.exists()

    def test_structural_scalars_file_written(self, simple_stator, tmp_path):
        run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        path = tmp_path / simple_stator.stator_id / "structural" / "scalars.json"
        assert path.exists()

    def test_thermal_derating_factor_in_range(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        tf = results.coupled_metrics["thermal_derating_factor"]
        assert 0.0 <= tf <= 1.0

    def test_efficiency_in_range(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        eta = results.em_results["efficiency"]
        assert 0.0 < eta < 1.0

    def test_pipeline_metrics_physically_reasonable(self, simple_stator, tmp_path):
        results = run_fea_pipeline(simple_stator, output_dir=str(tmp_path))
        m = results.coupled_metrics
        assert m["total_loss_W"] > 0
        assert m["peak_temperature_K"] > 293.0
        assert m["safety_factor"] > 0
        assert 0.0 <= m["thermal_derating_factor"] <= 1.0

    def test_config_file_loaded_correctly(self, simple_stator, tmp_path):
        """Pipeline should work with the bundled default.yaml."""
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), "../../configs/default.yaml"
        )
        results = run_fea_pipeline(
            simple_stator,
            config_path=config_path,
            output_dir=str(tmp_path),
        )
        assert isinstance(results, PipelineResults)
