"""orchestrator.py — Main NSGA-II GA loop for stator geometry optimisation.

Implements the standard NSGA-II elitist survival strategy:

  1. Evaluate population (parallel via joblib)
  2. Fast non-dominated sort + crowding distance
  3. Update Pareto archive
  4. Check termination conditions
  5. Generate offspring (tournament → SBX → polynomial mutation)
  6. Evaluate offspring
  7. Combine parent + offspring, re-sort, select next generation
  8. Checkpoint (every N generations)
  9. Go to 2

The function ``run_ga`` is the single public entry point.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .population import initialise_population, Individual, Population
from .evaluation.evaluator import evaluate_individual
from .evaluation.cache import EvaluationCache
from .pareto.nsga2 import fast_non_dominated_sort, crowding_distance_assignment
from .pareto.archive import ParetoArchive
from .operators.selection import crowded_tournament
from .operators.crossover import sbx_crossover
from .operators.mutation import polynomial_mutation
from .io.checkpoint import save_checkpoint, load_checkpoint
from .io.result_writer import write_pareto_results
from .utils.metrics import compute_hypervolume
from .utils.logger import setup_logger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class GAState:
    """Mutable state of one GA run — serialised to checkpoint files."""
    generation: int = 0
    total_evaluations: int = 0
    archive_size_history: list[int] = field(default_factory=list)
    hypervolume_history: list[float] = field(default_factory=list)
    population: Population = field(default_factory=list)
    archive: ParetoArchive = field(default_factory=ParetoArchive)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

def check_termination(state: GAState, config: dict) -> tuple[bool, str]:
    """Evaluate all termination conditions.

    Parameters
    ----------
    state:
        Current :class:`GAState`.
    config:
        Full GA config dict (reads the ``"termination"`` sub-dict).

    Returns
    -------
    tuple[bool, str]
        ``(should_terminate, reason_string)``.
    """
    tc = config["termination"]

    # 1. Max generations
    if state.generation >= tc["max_generations"]:
        return True, f"max_generations ({tc['max_generations']}) reached"

    # 2. Max FEA evaluations
    max_evals = tc.get("max_evaluations", int(1e9))
    if state.total_evaluations >= max_evals:
        return True, f"max_evaluations ({max_evals}) reached"

    # 3. Hypervolume stagnation
    window = tc.get("stagnation_window", 15)
    if (len(state.hypervolume_history) >= window
            and len(state.archive_size_history) >= window):
        recent_hv = state.hypervolume_history[-window:]
        if recent_hv[-1] > 0:
            hv_change = abs(recent_hv[-1] - recent_hv[0]) / max(abs(recent_hv[0]), 1e-12)
            tol = float(tc.get("hypervolume_stagnation_tolerance", 1e-4))
            if hv_change < tol:
                return True, (
                    f"Hypervolume stagnated for {window} generations "
                    f"(change={hv_change:.2e} < tol={tol})"
                )

    # 4. Target hypervolume reached
    target = tc.get("target_hypervolume")
    if target is not None and state.hypervolume_history:
        if state.hypervolume_history[-1] >= float(target):
            return True, f"Target hypervolume {float(target):.4f} reached"

    return False, ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ga(
    config_path: str = "ga_optimizer/configs/default_ga.yaml",
    fea_config_path: str = "FEA/configs/default.yaml",
    output_dir: str = "ga_results/",
    resume_from_checkpoint: str | None = None,
) -> ParetoArchive:
    """Run the full NSGA-II optimisation loop.

    Parameters
    ----------
    config_path:
        Path to the GA YAML configuration file.
    fea_config_path:
        Path to the FEA YAML configuration file (forwarded to ``run_fea_pipeline``).
    output_dir:
        Root directory for checkpoints, FEA evaluations, and final results.
    resume_from_checkpoint:
        Path to a previous checkpoint file.  When provided, the GA resumes
        from the saved generation; the ``config_path`` overrides termination
        settings but not the population state.

    Returns
    -------
    ParetoArchive
        The final archive of all non-dominated feasible solutions.
    """
    # ── Load config ──────────────────────────────────────────────────────
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    setup_logger(config.get("log_level", "INFO"))
    log.info("GA Optimizer starting — config: %s", config_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fea_output_dir = str(out / "fea_evaluations")

    # ── Initialise or resume ─────────────────────────────────────────────
    if resume_from_checkpoint:
        state = load_checkpoint(resume_from_checkpoint)
        log.info("Resumed from checkpoint at generation %d", state.generation)
    else:
        seed = config.get("random_seed", 42)
        state = GAState(rng=np.random.default_rng(seed))
        seed_designs = config.get("seed_designs") or []
        if seed_designs:
            seed_designs = [np.asarray(s, dtype=float) for s in seed_designs]
        state.population = initialise_population(
            pop_size=config["population_size"],
            rng=state.rng,
            seed_designs=seed_designs or None,
        )
        log.info("Population initialised: %d individuals", len(state.population))

    cache    = EvaluationCache()
    pop_size = config["population_size"]
    n_jobs   = config.get("parallel_workers", 1)

    # ── Main GA loop ─────────────────────────────────────────────────────
    while True:
        t0 = time.perf_counter()
        log.info("=== Generation %d ===", state.generation)

        # ── Evaluate unevaluated individuals ─────────────────────────────
        _evaluate_population(
            state.population, state.generation, config,
            cache, fea_config_path, fea_output_dir, n_jobs
        )
        state.total_evaluations += sum(
            1 for ind in state.population
            if ind.evaluated and ind.objectives is not None
            and ind.objectives.is_feasible
        )

        # ── NSGA-II sort ─────────────────────────────────────────────────
        fronts = fast_non_dominated_sort(state.population)
        for front in fronts:
            crowding_distance_assignment(state.population, front)

        # ── Archive update ────────────────────────────────────────────────
        front0 = [state.population[i] for i in fronts[0]]
        new_entries = state.archive.update(front0)
        log.info("Archive: %d solutions (+%d new)", state.archive.size, new_entries)

        # ── Hypervolume ───────────────────────────────────────────────────
        ref_pt = config.get("hypervolume_reference_point", [0.0, 5000.0, 0.0])
        hv = (compute_hypervolume(state.archive.members, ref_pt)
              if state.archive.size > 0 else 0.0)
        state.hypervolume_history.append(hv)
        state.archive_size_history.append(state.archive.size)
        log.info("HV=%.6f  |  elapsed=%.1fs", hv, time.perf_counter() - t0)

        # ── Termination check ─────────────────────────────────────────────
        stop, reason = check_termination(state, config)
        if stop:
            log.info("Termination: %s", reason)
            break

        # ── Generate offspring ────────────────────────────────────────────
        ops = config.get("operators", {})
        eta_c      = float(ops.get("eta_c", 15.0))
        eta_m      = float(ops.get("eta_m", 20.0))
        p_cross    = float(ops.get("p_crossover", 1.0))

        offspring: Population = []
        while len(offspring) < pop_size:
            p1 = crowded_tournament(state.population, state.rng)
            p2 = crowded_tournament(state.population, state.rng)
            c1g, c2g = sbx_crossover(p1.genes, p2.genes, eta_c, state.rng, p_cross)
            c1g = polynomial_mutation(c1g, eta_m, state.rng)
            c2g = polynomial_mutation(c2g, eta_m, state.rng)
            offspring.append(Individual(genes=c1g))
            if len(offspring) < pop_size:
                offspring.append(Individual(genes=c2g))

        # ── Evaluate offspring ────────────────────────────────────────────
        _evaluate_population(
            offspring, state.generation, config,
            cache, fea_config_path, fea_output_dir, n_jobs
        )

        # ── Elitist survivor selection ────────────────────────────────────
        combined = state.population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            crowding_distance_assignment(combined, front)

        next_pop: Population = []
        for front in combined_fronts:
            front_inds = [combined[i] for i in front]
            if len(next_pop) + len(front_inds) <= pop_size:
                next_pop.extend(front_inds)
            else:
                # Fill remaining slots by descending crowding distance
                slots = pop_size - len(next_pop)
                sorted_by_crowd = sorted(
                    front,
                    key=lambda i: combined[i].crowding_distance,
                    reverse=True,
                )
                next_pop.extend(combined[i] for i in sorted_by_crowd[:slots])
                break

        state.population = next_pop
        state.generation += 1

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt_n = config.get("checkpoint_every_n_generations", 10)
        if state.generation % ckpt_n == 0:
            ckpt_path = str(out / f"checkpoint_gen{state.generation:04d}.pkl")
            save_checkpoint(state, ckpt_path)
            log.info("Checkpoint: %s", ckpt_path)

    # ── Final output ──────────────────────────────────────────────────────
    write_pareto_results(
        archive=state.archive,
        output_dir=output_dir,
        config=config,
        state=state,
    )
    log.info(
        "Optimisation complete — Pareto front: %d solutions  "
        "Total FEA evaluations: %d",
        state.archive.size,
        state.total_evaluations,
    )
    return state.archive


# ---------------------------------------------------------------------------
# Parallel evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_population(
    population: Population,
    generation: int,
    config: dict,
    cache: EvaluationCache,
    fea_config_path: str,
    fea_output_dir: str,
    n_jobs: int,
) -> None:
    """Evaluate all unevaluated individuals in *population* (in-place).

    Uses joblib for parallel evaluation.  n_jobs=1 runs serially (for
    debugging and CI).
    """
    unevaluated = [
        (i, ind) for i, ind in enumerate(population) if not ind.evaluated
    ]
    if not unevaluated:
        return

    import joblib

    def _eval(idx, ind):
        obj = evaluate_individual(
            genes=ind.genes,
            generation=generation,
            individual_index=idx,
            config=config,
            cache=cache,
            fea_config_path=fea_config_path,
            fea_output_dir=fea_output_dir,
        )
        return idx, obj

    if n_jobs == 1:
        results = [_eval(i, ind) for i, ind in unevaluated]
    else:
        results = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
            joblib.delayed(_eval)(i, ind) for i, ind in unevaluated
        )

    # Rebuild a mapping from local list index back to population
    local_map = {i: ind for i, ind in unevaluated}
    for local_idx, obj in results:
        ind = local_map[local_idx]
        ind.objectives = obj
        ind.evaluated = True
