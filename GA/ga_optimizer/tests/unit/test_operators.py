"""Unit tests for SBX crossover, polynomial mutation, repair, and selection."""

from __future__ import annotations

import numpy as np
import pytest

from ga_optimizer.chromosome import LOWER_BOUNDS, UPPER_BOUNDS, N_GENES, random_individual
from ga_optimizer.operators.crossover import sbx_crossover
from ga_optimizer.operators.mutation import polynomial_mutation
from ga_optimizer.operators.repair import clamp
from ga_optimizer.operators.selection import crowded_tournament
from ga_optimizer.population import Individual
from ga_optimizer.objectives import ObjectiveVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ind_with_rank(rank: int, cd: float) -> Individual:
    ov = ObjectiveVector(
        neg_efficiency=-0.9, total_loss_W=100.0, neg_power_density=-1e6,
        temperature_violation_K=0.0, safety_factor_violation=0.0,
    )
    return Individual(
        genes=np.zeros(N_GENES), objectives=ov, rank=rank,
        crowding_distance=cd, evaluated=True,
    )


# ---------------------------------------------------------------------------
# Repair / clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_bounds_unchanged(self):
        rng = np.random.default_rng(0)
        genes = random_individual(rng)
        assert np.allclose(clamp(genes), genes)

    def test_below_lower_clamped(self):
        genes = LOWER_BOUNDS - 1.0
        clamped = clamp(genes)
        assert np.all(clamped == LOWER_BOUNDS)

    def test_above_upper_clamped(self):
        genes = UPPER_BOUNDS + 1.0
        clamped = clamp(genes)
        assert np.all(clamped == UPPER_BOUNDS)

    def test_does_not_modify_input(self):
        genes = LOWER_BOUNDS - 0.5
        original = genes.copy()
        clamp(genes)
        assert np.allclose(genes, original)


# ---------------------------------------------------------------------------
# SBX crossover
# ---------------------------------------------------------------------------

class TestSBXCrossover:
    def test_offspring_shape(self):
        rng = np.random.default_rng(0)
        p1 = random_individual(rng)
        p2 = random_individual(rng)
        c1, c2 = sbx_crossover(p1, p2, eta_c=15.0, rng=rng)
        assert c1.shape == (N_GENES,)
        assert c2.shape == (N_GENES,)

    def test_offspring_within_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(500):
            p1 = random_individual(rng)
            p2 = random_individual(rng)
            c1, c2 = sbx_crossover(p1, p2, eta_c=15.0, rng=rng)
            assert np.all(c1 >= LOWER_BOUNDS) and np.all(c1 <= UPPER_BOUNDS)
            assert np.all(c2 >= LOWER_BOUNDS) and np.all(c2 <= UPPER_BOUNDS)

    def test_p_crossover_zero_returns_copies(self):
        """p_crossover=0 must return unmodified copies of parents."""
        rng = np.random.default_rng(7)
        p1 = random_individual(rng)
        p2 = random_individual(rng)
        c1, c2 = sbx_crossover(p1, p2, eta_c=15.0, rng=rng, p_crossover=0.0)
        assert np.allclose(c1, p1)
        assert np.allclose(c2, p2)

    def test_identical_parents_return_copies(self):
        """If parents are identical, children must equal parents."""
        rng = np.random.default_rng(3)
        p = random_individual(rng)
        c1, c2 = sbx_crossover(p, p.copy(), eta_c=15.0, rng=rng)
        assert np.allclose(c1, p)
        assert np.allclose(c2, p)

    def test_different_eta_affects_spread(self):
        """Higher eta_c → offspring stay closer to parents (smaller mean deviation).

        SBX spread factor β_q has larger variance for small eta_c.  We test
        that the mean absolute gene-level deviation from the nearest parent
        is larger for eta_c=2 than for eta_c=20 over many samples.
        """
        rng_base = np.random.default_rng(0)
        p1 = random_individual(rng_base)
        p2 = random_individual(rng_base)
        lo = np.minimum(p1, p2)
        hi = np.maximum(p1, p2)

        rng_low  = np.random.default_rng(1)
        rng_high = np.random.default_rng(2)

        mean_dev_low  = []
        mean_dev_high = []
        for _ in range(2000):
            c1l, _ = sbx_crossover(p1, p2, eta_c=2.0,  rng=rng_low)
            c1h, _ = sbx_crossover(p1, p2, eta_c=20.0, rng=rng_high)
            # Deviation from nearest parent boundary on each gene
            dev_low  = np.maximum(0, lo - c1l) + np.maximum(0, c1l - hi)
            dev_high = np.maximum(0, lo - c1h) + np.maximum(0, c1h - hi)
            mean_dev_low.append(dev_low.mean())
            mean_dev_high.append(dev_high.mean())

        # Low eta_c must produce larger out-of-interval excursions on average
        assert np.mean(mean_dev_low) > np.mean(mean_dev_high), (
            f"eta=2 mean deviation {np.mean(mean_dev_low):.4f} should exceed "
            f"eta=20 mean deviation {np.mean(mean_dev_high):.4f}"
        )


# ---------------------------------------------------------------------------
# Polynomial mutation
# ---------------------------------------------------------------------------

class TestPolynomialMutation:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        genes = random_individual(rng)
        mutant = polynomial_mutation(genes, eta_m=20.0, rng=rng)
        assert mutant.shape == (N_GENES,)

    def test_within_bounds_1000_trials(self):
        rng = np.random.default_rng(1)
        for _ in range(1000):
            genes = random_individual(rng)
            mutant = polynomial_mutation(genes, eta_m=20.0, rng=rng)
            assert np.all(mutant >= LOWER_BOUNDS) and np.all(mutant <= UPPER_BOUNDS)

    def test_p_mutation_1_always_changes(self):
        """p_mutation=1 → all genes mutated → result must differ from input."""
        rng = np.random.default_rng(42)
        genes = random_individual(rng)
        mutant = polynomial_mutation(genes, eta_m=20.0, rng=rng, p_mutation=1.0)
        assert not np.allclose(genes, mutant), (
            "With p_mutation=1.0, at least one gene must change"
        )

    def test_does_not_modify_input(self):
        rng = np.random.default_rng(5)
        genes = random_individual(rng)
        original = genes.copy()
        polynomial_mutation(genes, eta_m=20.0, rng=rng)
        assert np.allclose(genes, original)

    def test_default_mutation_rate(self):
        """Default rate 1/N_GENES → most genes unchanged for a single trial."""
        rng = np.random.default_rng(99)
        unchanged_counts = []
        for _ in range(500):
            genes = random_individual(rng)
            mutant = polynomial_mutation(genes, eta_m=20.0, rng=rng)
            unchanged_counts.append(np.sum(np.isclose(genes, mutant)))
        avg_unchanged = np.mean(unchanged_counts)
        # With 1/12 rate, on average ~11 genes should be unchanged
        assert avg_unchanged > 8


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

class TestTournamentSelection:
    def test_lower_rank_wins(self):
        rng = np.random.default_rng(0)
        winner = _ind_with_rank(0, 1.0)
        loser  = _ind_with_rank(1, 100.0)
        pop = [winner, loser]
        # Run many tournaments; the rank-0 individual should always win
        winners = set()
        for _ in range(100):
            sel = crowded_tournament(pop, rng)
            winners.add(id(sel))
        # Both may be selected (tournament picks 2 random), but rank-0 should dominate
        assert id(winner) in winners

    def test_same_rank_higher_crowding_wins(self):
        rng = np.random.default_rng(0)
        diverse  = _ind_with_rank(0, 100.0)
        crowded  = _ind_with_rank(0, 0.001)
        pop = [diverse, crowded]
        wins = 0
        for _ in range(200):
            sel = crowded_tournament(pop, rng)
            if id(sel) == id(diverse):
                wins += 1
        # The more diverse individual should be selected in the majority of cases
        assert wins > 100

    def test_returns_individual_from_population(self):
        rng = np.random.default_rng(1)
        pop = [_ind_with_rank(i % 3, float(i)) for i in range(20)]
        for _ in range(50):
            sel = crowded_tournament(pop, rng)
            assert sel in pop
