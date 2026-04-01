"""Unit conversion registry (all internal values in SI)."""

from __future__ import annotations

MU_0: float = 4.0 * 3.141592653589793e-7   # H/m
PI:   float = 3.141592653589793


def rpm_to_rad_s(rpm: float) -> float:
    return rpm * PI / 30.0


def rad_s_to_rpm(omega: float) -> float:
    return omega * 30.0 / PI


def celsius_to_kelvin(t_c: float) -> float:
    return t_c + 273.15


def kelvin_to_celsius(t_k: float) -> float:
    return t_k - 273.15


def electrical_frequency(speed_rpm: float, n_poles: int) -> float:
    """Electrical frequency [Hz] from mechanical speed and pole count."""
    return speed_rpm * n_poles / 120.0


def skin_depth(freq_Hz: float, conductivity_S_m: float, mu_r: float = 1.0) -> float:
    """Skin depth δ = sqrt(1 / (π f σ μ)) [m]."""
    import math
    denom = PI * freq_Hz * conductivity_S_m * mu_r * MU_0
    if denom <= 0.0:
        return float("inf")
    return math.sqrt(1.0 / denom)
