"""Unit tests for EvaluationCache."""

from __future__ import annotations

import numpy as np
import pytest

from ga_optimizer.evaluation.cache import EvaluationCache
from ga_optimizer.objectives import ObjectiveVector, INFEASIBLE_OBJECTIVES


def _make_obj(eff: float = 0.9) -> ObjectiveVector:
    return ObjectiveVector(
        neg_efficiency=-eff,
        total_loss_W=500.0,
        neg_power_density=-1e6,
        temperature_violation_K=0.0,
        safety_factor_violation=0.0,
    )


def _rand_genes(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, 12)


class TestEvaluationCache:
    def test_initial_size_zero(self):
        cache = EvaluationCache()
        assert cache.size == 0

    def test_miss_returns_none(self):
        cache = EvaluationCache()
        result = cache.get(_rand_genes(0))
        assert result is None

    def test_put_and_get(self):
        cache = EvaluationCache()
        genes = _rand_genes(1)
        obj = _make_obj(0.92)
        cache.put(genes, obj)
        retrieved = cache.get(genes)
        assert retrieved is not None
        assert retrieved.neg_efficiency == obj.neg_efficiency

    def test_different_genes_dont_collide(self):
        cache = EvaluationCache()
        g1 = _rand_genes(10)
        g2 = _rand_genes(20)
        cache.put(g1, _make_obj(0.91))
        cache.put(g2, _make_obj(0.85))
        assert cache.get(g1).neg_efficiency == -0.91
        assert cache.get(g2).neg_efficiency == -0.85

    def test_duplicate_put_overwrites(self):
        cache = EvaluationCache()
        genes = _rand_genes(5)
        cache.put(genes, _make_obj(0.8))
        cache.put(genes, _make_obj(0.95))
        assert cache.get(genes).neg_efficiency == -0.95

    def test_size_after_puts(self):
        cache = EvaluationCache()
        for i in range(10):
            cache.put(_rand_genes(i), _make_obj())
        assert cache.size == 10

    def test_hit_rate_tracking(self):
        cache = EvaluationCache()
        genes = _rand_genes(0)
        cache.get(genes)          # miss
        cache.put(genes, _make_obj())
        cache.get(genes)          # hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_max_size_eviction(self):
        cache = EvaluationCache(max_size=4)
        for i in range(10):
            cache.put(_rand_genes(i), _make_obj())
        assert cache.size <= 4

    def test_infeasible_objectives_cached(self):
        cache = EvaluationCache()
        genes = _rand_genes(99)
        cache.put(genes, INFEASIBLE_OBJECTIVES)
        result = cache.get(genes)
        assert result is not None
        assert result.total_loss_W == 1e9

    def test_numpy_dtype_independence(self):
        """Gene vectors of float32 and float64 with same values should match."""
        cache = EvaluationCache()
        genes64 = np.ones(12, dtype=np.float64) * 0.5
        genes32 = np.ones(12, dtype=np.float32) * 0.5
        cache.put(genes64, _make_obj(0.88))
        # float32 → float64 conversion before hash → should match
        result = cache.get(genes32)
        assert result is not None
