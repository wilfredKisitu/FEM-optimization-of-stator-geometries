"""Integration test — 3-generation GA run with mocked FEA evaluation.

The mock evaluator returns a synthetic ObjectiveVector that varies smoothly
with gene values, simulating a realistic Pareto trade-off between efficiency
and losses.  No stator mesh generation or FEA is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from ga_optimizer.objectives import ObjectiveVector
from ga_optimizer.chromosome import LOWER_BOUNDS, UPPER_BOUNDS


# ---------------------------------------------------------------------------
# Mock evaluator
# ---------------------------------------------------------------------------

def _mock_evaluate_individual(
    genes, generation, individual_index,
    config, cache, fea_config_path, fea_output_dir,
):
    """Synthetic evaluator — no FEA.  Simulates a Pareto trade-off."""
    norm = (genes - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS + 1e-9)
    eff  = 0.80 + 0.15 * norm[0]            # higher OD → higher efficiency
    loss = 2000.0 - 1000.0 * norm[1]        # higher bore_ratio → lower loss
    pd   = 0.5e6 + 1.5e6 * norm[2]         # longer stator → higher power density
    return ObjectiveVector(
        neg_efficiency        = -eff,
        total_loss_W          = loss,
        neg_power_density     = -pd,
        temperature_violation_K = 0.0,
        safety_factor_violation  = 0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAST_CONFIG_PATH = "GA/ga_optimizer/configs/fast_ga.yaml"
FEA_CONFIG_PATH  = "FEA/configs/default.yaml"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_fast_ga_runs_3_generations(mock_eval, tmp_path):
    """GA must complete 3 generations and write pareto_front.json."""
    from ga_optimizer.orchestrator import run_ga

    archive = run_ga(
        config_path=FAST_CONFIG_PATH,
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path),
    )

    assert archive.size >= 1, "Archive must contain at least one solution"

    pf_path = tmp_path / "pareto_front.json"
    assert pf_path.exists(), "pareto_front.json must be written"

    with open(pf_path) as fh:
        pf = json.load(fh)

    assert "solutions" in pf
    assert len(pf["solutions"]) >= 1
    assert pf["pareto_front_size"] == archive.size


@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_all_archive_members_feasible(mock_eval, tmp_path):
    """Every member of the final archive must be feasible."""
    from ga_optimizer.orchestrator import run_ga

    archive = run_ga(
        config_path=FAST_CONFIG_PATH,
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path),
    )

    for member in archive.members:
        assert member.objectives is not None
        assert member.objectives.is_feasible, (
            f"Archive member is infeasible: {member.objectives.to_dict()}"
        )


@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_pareto_front_non_dominated(mock_eval, tmp_path):
    """No member of the final archive should dominate another member."""
    from ga_optimizer.orchestrator import run_ga
    from ga_optimizer.pareto.nsga2 import dominates

    archive = run_ga(
        config_path=FAST_CONFIG_PATH,
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path),
    )

    members = archive.members
    for i, a in enumerate(members):
        for j, b in enumerate(members):
            if i == j:
                continue
            assert not dominates(
                a.objectives.objective_array,
                b.objectives.objective_array,
            ), f"Member {i} dominates member {j} — archive is not clean"


@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_hypervolume_history_non_decreasing(mock_eval, tmp_path):
    """Hypervolume must be non-decreasing (elitist archive guarantee)."""
    from ga_optimizer.orchestrator import run_ga, GAState

    # Intercept GAState to record history
    archive = run_ga(
        config_path=FAST_CONFIG_PATH,
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path),
    )

    hv_hist_path = tmp_path / "hypervolume_history.json"
    if hv_hist_path.exists():
        with open(hv_hist_path) as fh:
            hv_data = json.load(fh)
        hv_hist = hv_data.get("hypervolume_history", [])
        for i in range(1, len(hv_hist)):
            assert hv_hist[i] >= hv_hist[i - 1] - 1e-10, (
                f"HV decreased at gen {i}: {hv_hist[i-1]:.6f} → {hv_hist[i]:.6f}"
            )


@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_checkpoint_and_resume(mock_eval, tmp_path):
    """Run 2 generations, save checkpoint, resume for 1 more gen."""
    from ga_optimizer.orchestrator import run_ga

    # Phase 1: 2 generations
    cfg = yaml.safe_load(open(FAST_CONFIG_PATH))
    cfg["termination"]["max_generations"] = 2
    cfg["checkpoint_every_n_generations"] = 1
    cfg_path_1 = tmp_path / "cfg_2gen.yaml"
    with open(cfg_path_1, "w") as fh:
        yaml.dump(cfg, fh)

    archive1 = run_ga(
        config_path=str(cfg_path_1),
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path / "phase1"),
    )

    # Find checkpoint
    ckpts = sorted((tmp_path / "phase1").glob("checkpoint_gen*.pkl"))
    assert len(ckpts) >= 1, "At least one checkpoint must be written"

    # Phase 2: resume from checkpoint, run 1 more generation
    cfg["termination"]["max_generations"] = 3
    cfg_path_2 = tmp_path / "cfg_3gen.yaml"
    with open(cfg_path_2, "w") as fh:
        yaml.dump(cfg, fh)

    archive2 = run_ga(
        config_path=str(cfg_path_2),
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path / "phase2"),
        resume_from_checkpoint=str(ckpts[-1]),
    )

    assert archive2.size >= archive1.size, (
        "Archive after resume must not shrink (elitist property)"
    )


@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_result_files_written(mock_eval, tmp_path):
    """All expected output files must be present after a GA run."""
    from ga_optimizer.orchestrator import run_ga

    run_ga(
        config_path=FAST_CONFIG_PATH,
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path),
    )

    expected = ["pareto_front.json", "pareto_front.csv", "run_metadata.json"]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing output file: {fname}"


@patch(
    "ga_optimizer.orchestrator.evaluate_individual",
    side_effect=_mock_evaluate_individual,
)
def test_no_nan_in_objectives(mock_eval, tmp_path):
    """All archive members must have finite objective values."""
    from ga_optimizer.orchestrator import run_ga
    import numpy as np

    archive = run_ga(
        config_path=FAST_CONFIG_PATH,
        fea_config_path=FEA_CONFIG_PATH,
        output_dir=str(tmp_path),
    )

    for member in archive.members:
        obj_arr = member.objectives.objective_array
        assert np.all(np.isfinite(obj_arr)), (
            f"Non-finite objectives in archive: {obj_arr}"
        )
