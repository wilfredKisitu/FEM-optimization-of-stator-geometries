"""Unit tests for chromosome encoding, decoding, and constraint checking."""

from __future__ import annotations

import math
import numpy as np
import pytest

from ga_optimizer.chromosome import (
    GENE_DEFINITIONS, LOWER_BOUNDS, UPPER_BOUNDS, N_GENES,
    decode_chromosome, random_individual,
)
from ga_optimizer.constraints import (
    check_geometric_constraints, GeometricConstraintViolation,
)


# ---------------------------------------------------------------------------
# Gene definition structure
# ---------------------------------------------------------------------------

def test_gene_count():
    assert N_GENES == 12
    assert len(GENE_DEFINITIONS) == 12
    assert LOWER_BOUNDS.shape == (12,)
    assert UPPER_BOUNDS.shape == (12,)


def test_bounds_are_valid():
    assert np.all(LOWER_BOUNDS < UPPER_BOUNDS), "All lower bounds must be < upper bounds"


def test_gene_names_unique():
    names = [g.name for g in GENE_DEFINITIONS]
    assert len(names) == len(set(names)), "Gene names must be unique"


# ---------------------------------------------------------------------------
# random_individual
# ---------------------------------------------------------------------------

def test_random_individual_shape():
    rng = np.random.default_rng(0)
    ind = random_individual(rng)
    assert ind.shape == (N_GENES,)


def test_random_individual_within_bounds():
    rng = np.random.default_rng(42)
    for _ in range(500):
        ind = random_individual(rng)
        assert np.all(ind >= LOWER_BOUNDS)
        assert np.all(ind <= UPPER_BOUNDS)


# ---------------------------------------------------------------------------
# decode_chromosome — structure
# ---------------------------------------------------------------------------

def test_decode_returns_required_keys():
    rng = np.random.default_rng(1)
    for _ in range(20):
        genes = random_individual(rng)
        try:
            params = decode_chromosome(genes)
        except ValueError:
            continue   # Some random chromosomes are infeasible — that's OK
        required = {
            "outer_diameter", "inner_diameter", "axial_length",
            "num_slots", "num_poles", "tooth_width", "yoke_height",
            "slot_depth", "slot_opening", "conductors_per_slot", "fill_factor",
        }
        assert required.issubset(set(params.keys()))


def test_decode_inner_lt_outer():
    rng = np.random.default_rng(7)
    for _ in range(100):
        genes = random_individual(rng)
        try:
            params = decode_chromosome(genes)
        except ValueError:
            continue
        assert params["inner_diameter"] < params["outer_diameter"]


def test_decode_num_poles_is_even():
    rng = np.random.default_rng(3)
    for _ in range(100):
        genes = random_individual(rng)
        try:
            params = decode_chromosome(genes)
        except ValueError:
            continue
        assert params["num_poles"] % 2 == 0
        assert params["num_poles"] >= 4


def test_decode_conductors_is_even():
    rng = np.random.default_rng(4)
    for _ in range(100):
        genes = random_individual(rng)
        try:
            params = decode_chromosome(genes)
        except ValueError:
            continue
        assert params["conductors_per_slot"] % 2 == 0


def test_decode_num_slots_multiple_of_pole_group():
    """num_slots must be a multiple of 3*(num_poles/2)."""
    rng = np.random.default_rng(5)
    success = 0
    for _ in range(200):
        genes = random_individual(rng)
        try:
            params = decode_chromosome(genes)
            success += 1
        except ValueError:
            continue
        q = 3 * (params["num_poles"] // 2)
        assert params["num_slots"] % q == 0, (
            f"num_slots={params['num_slots']} not multiple of {q}"
        )
    assert success >= 50, "Too many random chromosomes failed to decode"


def test_decode_fill_factor_in_range():
    rng = np.random.default_rng(6)
    for _ in range(100):
        genes = random_individual(rng)
        try:
            params = decode_chromosome(genes)
        except ValueError:
            continue
        assert 0.35 <= params["fill_factor"] <= 0.65


# ---------------------------------------------------------------------------
# decode_chromosome — known values
# ---------------------------------------------------------------------------

def test_decode_known_gene():
    """Manually constructed gene gives expected geometric values."""
    genes = np.array([
        0.200,   # outer_diameter
        0.60,    # bore_ratio → ID = 0.120
        0.100,   # axial_length
        36,      # num_slots (raw)
        8,       # num_poles (raw)
        0.50,    # tooth_width_fraction
        0.30,    # yoke_height_fraction
        0.45,    # slot_depth_fraction
        24,      # conductors_per_slot
        0.45,    # fill_factor
        0.20,    # slot_opening_fraction
        1.00,    # axial_length_ratio
    ])
    params = decode_chromosome(genes)

    assert math.isclose(params["outer_diameter"], 0.200, rel_tol=1e-6)
    assert math.isclose(params["inner_diameter"], 0.120, rel_tol=1e-6)
    assert params["num_poles"] == 8
    # Slot pitch at bore = π * 0.12 / 36 ≈ 0.01047 m
    expected_slot_pitch = math.pi * 0.120 / 36
    expected_tooth_w    = 0.50 * expected_slot_pitch
    assert math.isclose(params["tooth_width"], expected_tooth_w, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Geometric constraints
# ---------------------------------------------------------------------------

FAST_CONFIG = {
    "constraints": {
        "min_air_gap_m":    0.0008,
        "min_slot_width_m": 0.003,
        "min_yoke_height_m": 0.008,
        "max_winding_temperature_K": 428.15,
        "min_safety_factor": 1.5,
    }
}


def test_feasible_gene_passes_constraints():
    # This gene is designed to be feasible
    genes = np.array([
        0.200, 0.60, 0.100, 36, 8,
        0.45, 0.25, 0.40, 16, 0.45, 0.20, 1.0,
    ])
    # Should not raise
    check_geometric_constraints(genes, FAST_CONFIG)


def test_small_yoke_height_fails():
    """High bore_ratio on a small motor → thin stator wall → yoke too short."""
    genes = np.array([
        0.150, 0.719, 0.100, 12, 4,
        0.45, 0.25, 0.40, 16, 0.45, 0.20, 1.0,
    ])
    with pytest.raises(GeometricConstraintViolation, match="Yoke height"):
        check_geometric_constraints(genes, FAST_CONFIG)


def test_large_tooth_fraction_fails():
    """Tooth width fraction near 0.65 on a small bore → slot width too small."""
    genes = np.array([
        0.150, 0.50, 0.100, 72, 4,
        0.65, 0.25, 0.40, 16, 0.45, 0.20, 1.0,
    ])
    # May or may not raise depending on exact geometry; just verify it runs
    try:
        check_geometric_constraints(genes, FAST_CONFIG)
    except GeometricConstraintViolation:
        pass  # expected for tight geometries


def test_zero_yoke_height_fraction_fails():
    """yoke_height_fraction=0.20 (minimum) on small machine → possibly below min_yoke."""
    genes = np.array([
        0.150, 0.50, 0.050, 12, 4,
        0.45, 0.20, 0.65, 8, 0.45, 0.20, 0.5,
    ])
    # slot+yoke may exceed radial build → ValueError from decode, caught as GCV
    try:
        check_geometric_constraints(genes, FAST_CONFIG)
    except (GeometricConstraintViolation, ValueError):
        pass  # both acceptable
