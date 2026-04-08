"""io/checkpoint.py — Save and restore full GA state for resumable runs.

Checkpoints are stored as pickle files (``*.pkl``).  HDF5 is listed as an
optional enhancement and requires ``h5py``; when it is unavailable, pickle is
used transparently.

Checkpoint contents:
  - Current generation number and total FEA evaluation count
  - Full population (genes, objectives, ranks, crowding distances)
  - Pareto archive members
  - Hypervolume and archive-size history
  - NumPy RNG state (for fully reproducible resumed runs)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..orchestrator import GAState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(state: "GAState", path: str) -> None:
    """Serialise full GA state to *path*.

    Tries HDF5 first; falls back to pickle if ``h5py`` is not installed.

    Parameters
    ----------
    state:
        :class:`GAState` to serialise.
    path:
        Output file path.  The extension (``.h5`` or ``.pkl``) is overridden
        if the chosen backend does not match.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        import h5py
        _save_hdf5(state, str(p.with_suffix(".h5")))
        log.debug("Checkpoint saved (HDF5): %s", p.with_suffix(".h5"))
    except ImportError:
        pkl_path = p.with_suffix(".pkl")
        _save_pickle(state, str(pkl_path))
        log.debug("Checkpoint saved (pickle): %s", pkl_path)


def _save_hdf5(state: "GAState", path: str) -> None:
    import h5py
    from ..objectives import ObjectiveVector

    with h5py.File(path, "w") as f:
        f.attrs["generation"]        = state.generation
        f.attrs["total_evaluations"] = state.total_evaluations

        # Population
        pg = f.create_group("population")
        genes_arr = np.array([ind.genes for ind in state.population])
        pg.create_dataset("genes", data=genes_arr)

        obj_arr = np.array([
            ind.objectives.objective_array if ind.evaluated and ind.objectives is not None
            else np.full(3, np.nan)
            for ind in state.population
        ])
        pg.create_dataset("objectives", data=obj_arr)

        cv_arr = np.array([
            [ind.objectives.temperature_violation_K,
             ind.objectives.safety_factor_violation]
            if ind.evaluated and ind.objectives is not None
            else [np.nan, np.nan]
            for ind in state.population
        ])
        pg.create_dataset("constraint_violations", data=cv_arr)
        pg.create_dataset("ranks",
            data=np.array([ind.rank for ind in state.population]))
        pg.create_dataset("crowding",
            data=np.array([ind.crowding_distance for ind in state.population]))
        pg.create_dataset("evaluated",
            data=np.array([ind.evaluated for ind in state.population], dtype=bool))

        # Archive
        if state.archive.size > 0:
            ag = f.create_group("archive")
            ag.create_dataset("genes",
                data=np.array([m.genes for m in state.archive.members]))
            ag.create_dataset("objectives",
                data=np.array([m.objectives.objective_array
                                for m in state.archive.members]))
            cv_archive = np.array([
                [m.objectives.temperature_violation_K,
                 m.objectives.safety_factor_violation]
                for m in state.archive.members
            ])
            ag.create_dataset("constraint_violations", data=cv_archive)

        # Histories
        f.create_dataset("hypervolume_history",
            data=np.array(state.hypervolume_history, dtype=float))
        f.create_dataset("archive_size_history",
            data=np.array(state.archive_size_history, dtype=int))

        # RNG state (pickle-serialised into bytes dataset)
        rng_bytes = pickle.dumps(state.rng)
        f.create_dataset("rng_state",
            data=np.frombuffer(rng_bytes, dtype=np.uint8))


def _save_pickle(state: "GAState", path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> "GAState":
    """Restore GA state from *path*.

    Automatically detects HDF5 vs pickle based on the file extension.

    Parameters
    ----------
    path:
        Path to the checkpoint file (``*.h5`` or ``*.pkl``).

    Returns
    -------
    GAState
        Fully restored state ready for ``run_ga`` to continue.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)

    # Try both extensions if the given path doesn't exist but one variant does
    if not p.exists():
        h5_path = p.with_suffix(".h5")
        pkl_path = p.with_suffix(".pkl")
        if h5_path.exists():
            p = h5_path
        elif pkl_path.exists():
            p = pkl_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    if p.suffix == ".h5":
        return _load_hdf5(str(p))
    else:
        return _load_pickle(str(p))


def _load_hdf5(path: str) -> "GAState":
    import h5py
    from ..orchestrator import GAState
    from ..population import Individual
    from ..objectives import ObjectiveVector
    from ..pareto.archive import ParetoArchive

    with h5py.File(path, "r") as f:
        generation        = int(f.attrs["generation"])
        total_evaluations = int(f.attrs["total_evaluations"])

        pg = f["population"]
        genes_arr   = pg["genes"][:]
        obj_arr     = pg["objectives"][:]
        cv_arr      = pg["constraint_violations"][:]
        ranks       = pg["ranks"][:]
        crowding    = pg["crowding"][:]
        evaluated   = pg["evaluated"][:]

        population = []
        for i in range(len(genes_arr)):
            if evaluated[i] and not np.any(np.isnan(obj_arr[i])):
                ov = ObjectiveVector(
                    neg_efficiency=obj_arr[i, 0],
                    total_loss_W=obj_arr[i, 1],
                    neg_power_density=obj_arr[i, 2],
                    temperature_violation_K=cv_arr[i, 0],
                    safety_factor_violation=cv_arr[i, 1],
                )
            else:
                ov = None
            ind = Individual(
                genes=genes_arr[i],
                objectives=ov,
                rank=int(ranks[i]),
                crowding_distance=float(crowding[i]),
                evaluated=bool(evaluated[i]),
            )
            population.append(ind)

        archive = ParetoArchive()
        if "archive" in f:
            ag = f["archive"]
            arc_genes = ag["genes"][:]
            arc_objs  = ag["objectives"][:]
            arc_cv    = ag["constraint_violations"][:]
            for i in range(len(arc_genes)):
                ov = ObjectiveVector(
                    neg_efficiency=arc_objs[i, 0],
                    total_loss_W=arc_objs[i, 1],
                    neg_power_density=arc_objs[i, 2],
                    temperature_violation_K=arc_cv[i, 0],
                    safety_factor_violation=arc_cv[i, 1],
                )
                ind = Individual(genes=arc_genes[i], objectives=ov,
                                 rank=0, evaluated=True)
                archive._members.append(ind)  # bypass update to avoid re-dominance check

        hv_hist      = list(f["hypervolume_history"][:])
        arc_hist     = list(f["archive_size_history"][:].astype(int))
        rng          = pickle.loads(bytes(f["rng_state"][:]))

    state = GAState(
        generation=generation,
        total_evaluations=total_evaluations,
        archive_size_history=arc_hist,
        hypervolume_history=hv_hist,
        population=population,
        archive=archive,
        rng=rng,
    )
    return state


def _load_pickle(path: str) -> "GAState":
    with open(path, "rb") as fh:
        return pickle.load(fh)
