"""params.py — StatorParams dataclass, enums, validation, and derived fields."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum


class SlotShape(IntEnum):
    RECTANGULAR = 0
    TRAPEZOIDAL = 1
    ROUND_BOTTOM = 2
    SEMI_CLOSED = 3


class WindingType(IntEnum):
    SINGLE_LAYER = 0
    DOUBLE_LAYER = 1
    CONCENTRATED = 2
    DISTRIBUTED = 3


class LaminationMaterial(IntEnum):
    M270_35A = 0
    M330_50A = 1
    M400_50A = 2
    NO20 = 3
    CUSTOM = 4


@dataclass
class StatorParams:
    # Radii
    R_outer: float = 0.25
    R_inner: float = 0.15
    airgap_length: float = 0.001

    # Slot geometry
    n_slots: int = 36
    slot_depth: float = 0.06
    slot_width_outer: float = 0.012
    slot_width_inner: float = 0.010
    slot_opening: float = 0.004
    slot_opening_depth: float = 0.003
    tooth_tip_angle: float = 0.1
    slot_shape: SlotShape = SlotShape.SEMI_CLOSED

    # Coil / winding
    coil_depth: float = 0.05
    coil_width_outer: float = 0.008
    coil_width_inner: float = 0.007
    insulation_thickness: float = 0.001
    turns_per_coil: int = 10
    coil_pitch: int = 5
    wire_diameter: float = 0.001
    slot_fill_factor: float = 0.45
    winding_type: WindingType = WindingType.DOUBLE_LAYER

    # Lamination stack
    t_lam: float = 0.00035
    n_lam: int = 200
    z_spacing: float = 0.0
    insulation_coating_thickness: float = 0.00005
    material: LaminationMaterial = LaminationMaterial.M270_35A
    material_file: str = ""

    # Mesh sizing
    mesh_yoke: float = 0.006
    mesh_slot: float = 0.003
    mesh_coil: float = 0.0015
    mesh_ins: float = 0.0007
    mesh_boundary_layers: int = 3
    mesh_curvature: float = 0.3
    mesh_transition_layers: int = 2

    # Derived fields (computed by validate_and_derive)
    yoke_height: float = 0.0
    tooth_width: float = 0.0
    slot_pitch: float = 0.0
    stack_length: float = 0.0
    fill_factor: float = 0.0


def validate_and_derive(p: StatorParams) -> StatorParams:
    """Validate all 15 rules from the C code and compute derived fields.

    Raises ValueError on failure. Returns a new StatorParams with derived
    fields populated.
    """
    # Rule 1: All positive dimensions > 0
    pos_fields = [
        ("R_outer", p.R_outer), ("R_inner", p.R_inner),
        ("airgap_length", p.airgap_length), ("slot_depth", p.slot_depth),
        ("slot_width_outer", p.slot_width_outer), ("slot_width_inner", p.slot_width_inner),
        ("coil_depth", p.coil_depth), ("coil_width_outer", p.coil_width_outer),
        ("coil_width_inner", p.coil_width_inner), ("insulation_thickness", p.insulation_thickness),
        ("wire_diameter", p.wire_diameter), ("t_lam", p.t_lam),
        ("mesh_yoke", p.mesh_yoke), ("mesh_slot", p.mesh_slot),
        ("mesh_coil", p.mesh_coil), ("mesh_ins", p.mesh_ins),
    ]
    for name, val in pos_fields:
        if val <= 0:
            raise ValueError(f"{name} must be > 0 (got {val})")

    # Rule 2: R_inner < R_outer
    if p.R_inner >= p.R_outer:
        raise ValueError(f"R_inner ({p.R_inner}) must be < R_outer ({p.R_outer})")

    # Rule 3: slot_depth < (R_outer - R_inner)
    if p.slot_depth >= (p.R_outer - p.R_inner):
        raise ValueError(
            f"slot_depth ({p.slot_depth}) must be < R_outer - R_inner "
            f"({p.R_outer - p.R_inner})"
        )

    # Rule 4: n_slots >= 6 AND even
    if p.n_slots < 6:
        raise ValueError(f"n_slots ({p.n_slots}) must be >= 6")
    if p.n_slots % 2 != 0:
        raise ValueError(f"n_slots ({p.n_slots}) must be even")

    # Rule 5: slot_width_inner < R_inner * 2*pi/n_slots
    max_sw = p.R_inner * 2.0 * math.pi / p.n_slots
    if p.slot_width_inner >= max_sw:
        raise ValueError(
            f"slot_width_inner ({p.slot_width_inner}) must be < "
            f"R_inner * 2*pi/n_slots ({max_sw})"
        )

    # Rule 6: if SEMI_CLOSED, slot_opening < slot_width_inner AND
    #          slot_opening_depth < slot_depth
    if p.slot_shape == SlotShape.SEMI_CLOSED:
        if p.slot_opening >= p.slot_width_inner:
            raise ValueError(
                f"slot_opening ({p.slot_opening}) must be < "
                f"slot_width_inner ({p.slot_width_inner}) for SEMI_CLOSED"
            )
        if p.slot_opening_depth >= p.slot_depth:
            raise ValueError(
                f"slot_opening_depth ({p.slot_opening_depth}) must be < "
                f"slot_depth ({p.slot_depth}) for SEMI_CLOSED"
            )

    # Rule 7: coil_depth <= slot_depth - slot_opening_depth - 2*insulation_thickness
    max_cd = p.slot_depth - p.slot_opening_depth - 2.0 * p.insulation_thickness
    if p.coil_depth > max_cd:
        raise ValueError(
            f"coil_depth ({p.coil_depth}) must be <= "
            f"slot_depth - slot_opening_depth - 2*insulation_thickness ({max_cd})"
        )

    # Rule 8: coil_width_inner <= slot_width_inner - 2*insulation_thickness
    max_cw = p.slot_width_inner - 2.0 * p.insulation_thickness
    if p.coil_width_inner > max_cw:
        raise ValueError(
            f"coil_width_inner ({p.coil_width_inner}) must be <= "
            f"slot_width_inner - 2*insulation_thickness ({max_cw})"
        )

    # Rule 9: n_lam > 0
    if p.n_lam <= 0:
        raise ValueError(f"n_lam ({p.n_lam}) must be > 0")

    # Rule 10: z_spacing >= 0
    if p.z_spacing < 0:
        raise ValueError(f"z_spacing ({p.z_spacing}) must be >= 0")

    # Rule 11: insulation_coating_thickness >= 0
    if p.insulation_coating_thickness < 0:
        raise ValueError(
            f"insulation_coating_thickness ({p.insulation_coating_thickness}) must be >= 0"
        )

    # Rule 12: if CUSTOM material, material_file must be non-empty
    if p.material == LaminationMaterial.CUSTOM and not p.material_file:
        raise ValueError("material_file must be non-empty for CUSTOM material")

    # Rule 13: mesh_ins <= mesh_coil <= mesh_slot <= mesh_yoke
    if not (p.mesh_ins <= p.mesh_coil <= p.mesh_slot <= p.mesh_yoke):
        raise ValueError(
            f"mesh sizes must satisfy mesh_ins <= mesh_coil <= mesh_slot <= mesh_yoke "
            f"(got {p.mesh_ins}, {p.mesh_coil}, {p.mesh_slot}, {p.mesh_yoke})"
        )

    # Rule 14: 0 <= tooth_tip_angle < pi/4
    if p.tooth_tip_angle < 0 or p.tooth_tip_angle >= math.pi / 4:
        raise ValueError(
            f"tooth_tip_angle ({p.tooth_tip_angle}) must be in [0, pi/4)"
        )

    # Derive fields
    slot_pitch = 2.0 * math.pi / p.n_slots
    yoke_height = p.R_outer - p.R_inner - p.slot_depth
    tooth_width = p.R_inner * slot_pitch - p.slot_width_inner
    stack_length = p.n_lam * p.t_lam + (p.n_lam - 1) * p.z_spacing

    slot_area = 0.5 * (p.slot_width_inner + p.slot_width_outer) * p.slot_depth
    coil_area = 0.5 * (p.coil_width_inner + p.coil_width_outer) * p.coil_depth
    fill_factor = coil_area / slot_area if slot_area > 0 else 0.0

    # Rule 15: fill_factor in (0, 1)
    if fill_factor <= 0 or fill_factor >= 1:
        raise ValueError(
            f"computed fill_factor ({fill_factor}) must be in (0, 1)"
        )

    return replace(
        p,
        slot_pitch=slot_pitch,
        yoke_height=yoke_height,
        tooth_width=tooth_width,
        stack_length=stack_length,
        fill_factor=fill_factor,
    )


def make_reference_params() -> StatorParams:
    """Return a validated 36-slot reference configuration."""
    return validate_and_derive(StatorParams())


def make_minimal_params() -> StatorParams:
    """Return a validated 12-slot minimal configuration."""
    return validate_and_derive(StatorParams(
        R_outer=0.12,
        R_inner=0.07,
        n_slots=12,
        slot_depth=0.03,
        slot_width_outer=0.010,
        slot_width_inner=0.009,
        slot_opening=0.003,
        slot_opening_depth=0.002,
        tooth_tip_angle=0.0,
        slot_shape=SlotShape.RECTANGULAR,
        coil_depth=0.025,
        coil_width_outer=0.007,
        coil_width_inner=0.006,
        insulation_thickness=0.001,
        turns_per_coil=8,
        coil_pitch=3,
        wire_diameter=0.0012,
        slot_fill_factor=0.4,
        winding_type=WindingType.SINGLE_LAYER,
        t_lam=0.00035,
        n_lam=100,
        z_spacing=0.0,
        insulation_coating_thickness=0.00005,
        material=LaminationMaterial.M330_50A,
        mesh_yoke=0.005,
        mesh_slot=0.002,
        mesh_coil=0.001,
        mesh_ins=0.0005,
        mesh_boundary_layers=2,
        mesh_curvature=0.3,
        mesh_transition_layers=2,
    ))
