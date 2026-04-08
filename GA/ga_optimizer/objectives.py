"""objectives.py — Objective vector definition and FEA result extraction.

The GA solves a **three-objective minimisation** problem:

    f1 = -η                  (maximise efficiency → minimise negation)
    f2 = P_loss              (minimise total electromagnetic + thermal losses [W])
    f3 = -(P_out / V)        (maximise power density [W/m³] → minimise negation)

Constraint violations are tracked as separate fields on the ObjectiveVector
so that constrained-dominance and penalty handling are both available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Objective container
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveVector:
    """Three-objective minimisation problem result for one individual.

    All objective values must be finite floats.  Any NaN or Inf marks the
    individual as infeasible regardless of the constraint fields.

    Attributes
    ----------
    neg_efficiency:
        -η ∈ (-1, 0].  Minimising this maximises efficiency.
    total_loss_W:
        Total EM + copper losses [W].
    neg_power_density:
        -(P_mech / V_bounding_cylinder) [W/m³ negated].
    temperature_violation_K:
        max(0, T_peak − T_limit) [K].  Zero means feasible.
    safety_factor_violation:
        max(0, SF_min − SF_actual).  Zero means feasible.
    """

    neg_efficiency: float
    total_loss_W: float
    neg_power_density: float
    temperature_violation_K: float
    safety_factor_violation: float

    @property
    def objective_array(self) -> np.ndarray:
        """The three primary objectives as a (3,) array for NSGA-II comparisons."""
        return np.array([
            self.neg_efficiency,
            self.total_loss_W,
            self.neg_power_density,
        ])

    @property
    def is_feasible(self) -> bool:
        """True iff all constraints are satisfied and all objectives are finite."""
        return (
            self.temperature_violation_K <= 0.0
            and self.safety_factor_violation <= 0.0
            and np.all(np.isfinite(self.objective_array))
        )

    def to_dict(self) -> dict:
        """Serialise to a plain Python dict for JSON export."""
        return {
            "neg_efficiency":             self.neg_efficiency,
            "total_loss_W":               self.total_loss_W,
            "neg_power_density":          self.neg_power_density,
            "temperature_violation_K":    self.temperature_violation_K,
            "safety_factor_violation":    self.safety_factor_violation,
            # Human-readable derivations:
            "efficiency":                 -self.neg_efficiency,
            "power_density_W_m3":         -self.neg_power_density,
            "feasible":                   self.is_feasible,
        }


# Sentinel: returned for any individual that fails geometry, mesh, or FEA.
INFEASIBLE_OBJECTIVES = ObjectiveVector(
    neg_efficiency=1e9,
    total_loss_W=1e9,
    neg_power_density=1e9,
    temperature_violation_K=1e9,
    safety_factor_violation=1e9,
)


# ---------------------------------------------------------------------------
# Objective extraction from FEA results
# ---------------------------------------------------------------------------

def extract_objectives(
    fea_results,          # PipelineResults returned by run_fea_pipeline
    stator_params: dict,  # decoded chromosome dict (from decode_chromosome)
    config: dict,
) -> ObjectiveVector:
    """Map FEA pipeline results to an :class:`ObjectiveVector`.

    Parameters
    ----------
    fea_results:
        :class:`PipelineResults` object returned by ``run_fea_pipeline``.
    stator_params:
        Decoded chromosome dict — must contain ``"outer_diameter"``,
        ``"axial_length"``, and optionally ``"rated_speed_rpm"``.
    config:
        Full GA config dict.  Reads ``config["constraints"]`` for temperature
        and safety-factor limits, and ``config["operating_point"]["speed_rpm"]``
        for mechanical speed.

    Returns
    -------
    ObjectiveVector
        Fully populated objective vector, possibly marking constraint violations.
    """
    em  = fea_results.em_results
    th  = fea_results.thermal_results
    st  = fea_results.structural_results

    speed_rpm = config["operating_point"].get("speed_rpm", 3000.0)
    omega     = speed_rpm * (2.0 * math.pi / 60.0)        # rad/s

    torque   = float(em["torque_Nm"])
    P_mech   = torque * omega                              # [W]
    P_loss   = float(em["total_loss_W"])
    P_in     = P_mech + P_loss
    efficiency = P_mech / max(P_in, 1e-9)

    OD    = float(stator_params["outer_diameter"])
    axial = float(stator_params["axial_length"])
    volume = math.pi * (OD / 2.0) ** 2 * axial            # bounding cylinder [m³]
    power_density = P_mech / max(volume, 1e-9)             # [W/m³]

    T_limit = float(config["constraints"]["max_winding_temperature_K"])
    SF_min  = float(config["constraints"]["min_safety_factor"])

    T_violation  = max(0.0, float(th["peak_temperature_K"]) - T_limit)
    SF_violation = max(0.0, SF_min - float(st["safety_factor"]))

    return ObjectiveVector(
        neg_efficiency        = -efficiency,
        total_loss_W          = P_loss,
        neg_power_density     = -power_density,
        temperature_violation_K = T_violation,
        safety_factor_violation = SF_violation,
    )
