"""evaluation/cache.py — Thread-safe hash-keyed evaluation result cache.

Duplicate individuals (identical gene vectors) are common in later generations
when the population converges.  The cache avoids redundant FEA evaluations by
storing results keyed by the SHA-256 hash of the serialised gene vector.

The cache is process-local (not shared across joblib worker processes) and
survives only for the duration of one ``run_ga`` call.  For cross-run caching,
see the checkpoint module.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np

from ..objectives import ObjectiveVector


def _gene_hash(genes: np.ndarray) -> str:
    """Compute a stable SHA-256 hex-digest of a gene vector."""
    # Ensure consistent float64 representation before hashing
    return hashlib.sha256(
        np.asarray(genes, dtype=np.float64).tobytes()
    ).hexdigest()


class EvaluationCache:
    """Thread-safe in-memory cache from gene hash → ObjectiveVector.

    Uses a :class:`threading.Lock` so the cache can be safely shared by
    multiple joblib threads (``backend="threading"``).  For multiprocessing
    workers each process has its own cache instance — caching still saves
    within-process redundant evaluations.

    Parameters
    ----------
    max_size:
        Maximum number of entries.  When exceeded, the oldest 25% of entries
        are evicted (FIFO order).  Default ``None`` means unlimited.
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._store: dict[str, ObjectiveVector] = {}
        self._order: list[str] = []          # insertion-order FIFO tracker
        self._lock  = threading.Lock()
        self._max_size = max_size
        self._hits  = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, genes: np.ndarray) -> Optional[ObjectiveVector]:
        """Look up a cached result.

        Parameters
        ----------
        genes:
            Gene vector to query.

        Returns
        -------
        ObjectiveVector or None
            ``None`` on a cache miss.
        """
        key = _gene_hash(genes)
        with self._lock:
            result = self._store.get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put(self, genes: np.ndarray, obj: ObjectiveVector) -> None:
        """Store an evaluation result.

        Parameters
        ----------
        genes:
            Gene vector that was evaluated.
        obj:
            Resulting :class:`ObjectiveVector`.
        """
        key = _gene_hash(genes)
        with self._lock:
            if key not in self._store:
                self._store[key] = obj
                self._order.append(key)
                self._evict_if_needed()
            else:
                # Update existing entry (e.g. re-evaluation after repair)
                self._store[key] = obj

    def _evict_if_needed(self) -> None:
        """Evict oldest 25% of entries when max_size is exceeded."""
        if self._max_size is not None and len(self._order) > self._max_size:
            n_evict = max(1, self._max_size // 4)
            evict_keys = self._order[:n_evict]
            self._order = self._order[n_evict:]
            for k in evict_keys:
                self._store.pop(k, None)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size":     self.size,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": self.hit_rate,
        }
