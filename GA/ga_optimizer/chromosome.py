"""chromosome.py — Gene encoding / decoding for the stator GA optimizer.

Each individual is a real-valued vector of 12 genes that parameterise the
stator geometry at a high level.  Integer genes (num_slots, num_poles,
conductors_per_slot) are encoded as floats and rounded during decoding to
the nearest physically valid integer.

The gene vector is decoded into a flat dict of physical parameters that maps
directly onto the fields accepted by StatorMeshInput (FEA) and StatorParams
(mesh generation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GeneDefinition:
    """Metadata for one gene in the chromosome vector."""
    name: str
    lower: float
    upper: float
    dtype: str          # "float" or "int"
    unit: str
    description: str


# ---------------------------------------------------------------------------
# Authoritative gene list — DO NOT REORDER (indices are referenced by number)
# ---------------------------------------------------------------------------

GENE_DEFINITIONS: list[GeneDefinition] = [
    # Index 0
    GeneDefinition("outer_diameter",        0.150, 0.400, "float", "m",
                   "Stator outer diameter"),
    # Index 1
    GeneDefinition("bore_ratio",            0.50,  0.72,  "float", "—",
                   "inner_diameter / outer_diameter; enforces ID < OD"),
    # Index 2
    GeneDefinition("axial_length",          0.050, 0.200, "float", "m",
                   "Active axial stack length"),
    # Index 3
    GeneDefinition("num_slots",             12,    72,    "int",   "—",
                   "Number of stator slots; snapped to multiple of 3*(num_poles/2)"),
    # Index 4
    GeneDefinition("num_poles",             4,     20,    "int",   "—",
                   "Number of rotor poles; must be even"),
    # Index 5
    GeneDefinition("tooth_width_fraction",  0.35,  0.65,  "float", "—",
                   "Tooth width as fraction of slot pitch at bore"),
    # Index 6
    GeneDefinition("yoke_height_fraction",  0.20,  0.55,  "float", "—",
                   "Yoke height as fraction of radial build (OD-ID)/2"),
    # Index 7
    GeneDefinition("slot_depth_fraction",   0.30,  0.65,  "float", "—",
                   "Slot depth as fraction of radial build (OD-ID)/2"),
    # Index 8
    GeneDefinition("conductors_per_slot",   8,     64,    "int",   "—",
                   "Number of conductors per slot; snapped to nearest even integer"),
    # Index 9
    GeneDefinition("fill_factor",           0.35,  0.65,  "float", "—",
                   "Copper fill factor (0–1)"),
    # Index 10
    GeneDefinition("slot_opening_fraction", 0.10,  0.40,  "float", "—",
                   "Slot opening width as fraction of tooth width"),
    # Index 11
    GeneDefinition("axial_length_ratio",    0.50,  3.00,  "float", "—",
                   "axial_length / outer_diameter; aspect-ratio constraint"),
]

N_GENES: int = len(GENE_DEFINITIONS)
LOWER_BOUNDS: np.ndarray = np.array([g.lower for g in GENE_DEFINITIONS])
UPPER_BOUNDS: np.ndarray = np.array([g.upper for g in GENE_DEFINITIONS])


# ---------------------------------------------------------------------------
# Chromosome decoding
# ---------------------------------------------------------------------------

def decode_chromosome(genes: np.ndarray) -> dict:
    """Convert raw gene vector to a StatorMeshInput-compatible parameter dict.

    Handles:
    - Hard-clamping all genes to [lower, upper]
    - Rounding integer genes to nearest valid integer
    - Computing absolute geometric dimensions from fractional genes
    - Enforcing slot/yoke fit constraint

    Parameters
    ----------
    genes:
        (N_GENES,) float array — raw gene vector from the GA.

    Returns
    -------
    dict
        Flat dict of physical parameters suitable for both StatorMeshInput
        (``outer_diameter``, ``inner_diameter``, ``axial_length``, ``num_slots``,
        ``num_poles``, ``tooth_width``, ``yoke_height``, ``slot_depth``,
        ``slot_opening``, ``conductors_per_slot``, ``fill_factor``) and
        the bridge function that builds StatorParams.

    Raises
    ------
    ValueError
        If derived geometry is physically impossible (slot + yoke overflows
        radial build, negative tooth width, etc.).
    """
    g = np.clip(genes, LOWER_BOUNDS, UPPER_BOUNDS)

    # ── Radial dimensions ────────────────────────────────────────────────
    OD         = float(g[0])
    bore_ratio = float(g[1])
    ID         = OD * bore_ratio
    radial_build = (OD - ID) / 2.0          # = R_outer - R_inner [m]

    # ── Axial length ────────────────────────────────────────────────────
    axial = float(g[2])

    # ── Integer genes ────────────────────────────────────────────────────
    # Poles: nearest even integer in [4, 20]
    num_poles_raw = int(round(g[4]))
    num_poles = int(np.clip(num_poles_raw - (num_poles_raw % 2), 4, 20))
    num_poles = max(4, num_poles)

    # Slots: nearest multiple of 3*(num_poles/2) that falls in gene bounds
    q_slots_min = 3 * (num_poles // 2)       # minimum per-phase slot group
    num_slots_raw = int(round(g[3]))
    num_slots = max(q_slots_min,
                    round(num_slots_raw / q_slots_min) * q_slots_min)
    num_slots = int(np.clip(num_slots, int(GENE_DEFINITIONS[3].lower),
                             int(GENE_DEFINITIONS[3].upper)))
    # Re-snap after clipping
    num_slots = max(q_slots_min,
                    round(num_slots / q_slots_min) * q_slots_min)

    # Conductors: nearest even integer in [8, 64]
    cond_raw = int(round(g[8]))
    conductors = int(np.clip(cond_raw - (cond_raw % 2), 8, 64))
    conductors = max(2, conductors)

    # ── Absolute geometry from fractions ────────────────────────────────
    slot_pitch = math.pi * ID / num_slots    # arc pitch at bore [m]
    tooth_w    = float(g[5]) * slot_pitch    # absolute tooth width [m]
    yoke_h     = float(g[6]) * radial_build  # absolute yoke height [m]
    slot_d     = float(g[7]) * radial_build  # absolute slot depth [m]
    slot_op    = float(g[10]) * tooth_w      # absolute slot opening [m]

    fill_factor = float(g[9])

    # ── Feasibility checks ───────────────────────────────────────────────
    if slot_d + yoke_h > radial_build * 0.98:
        raise ValueError(
            f"slot_depth ({slot_d:.4f} m) + yoke_height ({yoke_h:.4f} m) "
            f"exceeds 98% of radial build ({radial_build:.4f} m)"
        )

    slot_w = slot_pitch - tooth_w
    if slot_w <= 1e-4:
        raise ValueError(
            f"Slot width ({slot_w*1e3:.2f} mm) is non-positive; "
            f"tooth_width_fraction too large for this slot pitch."
        )

    if tooth_w <= 0:
        raise ValueError(f"Tooth width {tooth_w:.4f} m must be > 0")

    if slot_op >= slot_w:
        # Clamp slot opening to 90% of slot width — keep it feasible
        slot_op = slot_w * 0.90

    return {
        "outer_diameter":      OD,
        "inner_diameter":      ID,
        "axial_length":        axial,
        "num_slots":           num_slots,
        "num_poles":           num_poles,
        "tooth_width":         tooth_w,
        "yoke_height":         yoke_h,
        "slot_depth":          slot_d,
        "slot_opening":        slot_op,
        "conductors_per_slot": conductors,
        "fill_factor":         fill_factor,
        # Store fractions for StatorParams bridge
        "_bore_ratio":         bore_ratio,
        "_tooth_width_fraction": float(g[5]),
        "_yoke_height_fraction": float(g[6]),
        "_slot_depth_fraction":  float(g[7]),
        "_slot_opening_fraction": float(g[10]),
    }


def random_individual(rng: np.random.Generator) -> np.ndarray:
    """Sample a random gene vector uniformly within all gene bounds.

    Parameters
    ----------
    rng:
        NumPy random Generator instance.

    Returns
    -------
    np.ndarray
        (N_GENES,) float array with genes in [LOWER_BOUNDS, UPPER_BOUNDS].
    """
    return rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
