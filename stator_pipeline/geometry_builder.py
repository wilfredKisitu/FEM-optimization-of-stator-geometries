"""geometry_builder.py — Pure Python geometry builder for stator cross-sections."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .params import StatorParams, SlotShape, WindingType
from .gmsh_backend import GmshBackend

MAX_SLOTS = 256


@dataclass
class SlotProfile:
    slot_idx: int = 0
    angle: float = 0.0
    slot_surface: int = -1
    coil_upper_sf: int = -1
    coil_lower_sf: int = -1
    ins_upper_sf: int = -1
    ins_lower_sf: int = -1
    mouth_curve_bot: int = -1
    mouth_curve_top: int = -1


@dataclass
class GeometryBuildResult:
    success: bool = False
    yoke_surface: int = -1
    bore_curve: int = -1
    outer_curve: int = -1
    n_slots: int = 0
    slots: list[SlotProfile] = field(default_factory=list)
    error_message: str = ""


def _rotate(x: float, y: float, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return x * c - y * s, x * s + y * c


def _slot_angle(k: int, n_slots: int) -> float:
    return 2.0 * math.pi * k / n_slots


def _add_rotated_point(backend: GmshBackend, x: float, y: float, theta: float, mesh_size: float) -> int:
    gx, gy = _rotate(x, y, theta)
    return backend.add_point(gx, gy, 0.0, mesh_size)


class GeometryBuilder:
    def __init__(self, backend: GmshBackend) -> None:
        self.backend = backend

    def build(self, p: StatorParams) -> GeometryBuildResult:
        result = GeometryBuildResult()
        result.slots = [SlotProfile(slot_idx=i, angle=_slot_angle(i, p.n_slots)) for i in range(p.n_slots)]
        result.n_slots = p.n_slots

        # Outer and inner circles
        c_outer = self.backend.add_circle(0.0, 0.0, 0.0, p.R_outer)
        c_inner = self.backend.add_circle(0.0, 0.0, 0.0, p.R_inner)
        result.outer_curve = c_outer
        result.bore_curve = c_inner

        # Yoke annular surface
        cl_outer = self.backend.add_curve_loop([c_outer])
        cl_inner = self.backend.add_curve_loop([-c_inner])
        yoke_sf = self.backend.add_plane_surface([cl_outer, cl_inner])
        result.yoke_surface = yoke_sf

        # Build all slot profiles
        slot_surfaces: list[tuple[int, int]] = []
        for k in range(p.n_slots):
            sp_slot = result.slots[k]
            sp_slot.slot_idx = k
            sp_slot.angle = _slot_angle(k, p.n_slots)

            if p.slot_shape == SlotShape.RECTANGULAR:
                self._build_rectangular(p, k, sp_slot)
            elif p.slot_shape == SlotShape.TRAPEZOIDAL:
                self._build_trapezoidal(p, k, sp_slot)
            elif p.slot_shape == SlotShape.ROUND_BOTTOM:
                self._build_round_bottom(p, k, sp_slot)
            else:  # SEMI_CLOSED
                self._build_semi_closed(p, k, sp_slot)

            self._build_coil(p, sp_slot)
            self._build_insulation(p, sp_slot)

            if sp_slot.slot_surface >= 0:
                slot_surfaces.append((2, sp_slot.slot_surface))

        # Boolean cut: carve slots out of yoke
        if slot_surfaces:
            self.backend.boolean_cut([(2, yoke_sf)], slot_surfaces, remove_tool=False)

        self.backend.synchronize()
        result.success = True
        return result

    def _build_rectangular(self, p: StatorParams, k: int, sp: SlotProfile) -> None:
        th = sp.angle
        hw = p.slot_width_inner * 0.5
        ri = p.R_inner
        ro = ri + p.slot_depth
        pt1 = _add_rotated_point(self.backend, ri,  hw, th, 0.0)
        pt2 = _add_rotated_point(self.backend, ri, -hw, th, 0.0)
        pt3 = _add_rotated_point(self.backend, ro, -hw, th, 0.0)
        pt4 = _add_rotated_point(self.backend, ro,  hw, th, 0.0)
        l1 = self.backend.add_line(pt1, pt2)
        l2 = self.backend.add_line(pt2, pt3)
        l3 = self.backend.add_line(pt3, pt4)
        l4 = self.backend.add_line(pt4, pt1)
        cl = self.backend.add_curve_loop([l1, l2, l3, l4])
        sp.slot_surface = self.backend.add_plane_surface([cl])
        sp.mouth_curve_bot = l1

    def _build_trapezoidal(self, p: StatorParams, k: int, sp: SlotProfile) -> None:
        th = sp.angle
        hwi = p.slot_width_inner * 0.5
        hwo = p.slot_width_outer * 0.5
        ri = p.R_inner
        ro = ri + p.slot_depth
        pt1 = _add_rotated_point(self.backend, ri,  hwi, th, 0.0)
        pt2 = _add_rotated_point(self.backend, ri, -hwi, th, 0.0)
        pt3 = _add_rotated_point(self.backend, ro, -hwo, th, 0.0)
        pt4 = _add_rotated_point(self.backend, ro,  hwo, th, 0.0)
        l1 = self.backend.add_line(pt1, pt2)
        l2 = self.backend.add_line(pt2, pt3)
        l3 = self.backend.add_line(pt3, pt4)
        l4 = self.backend.add_line(pt4, pt1)
        cl = self.backend.add_curve_loop([l1, l2, l3, l4])
        sp.slot_surface = self.backend.add_plane_surface([cl])
        sp.mouth_curve_bot = l1

    def _build_round_bottom(self, p: StatorParams, k: int, sp: SlotProfile) -> None:
        th = sp.angle
        hw = p.slot_width_inner * 0.5
        ri = p.R_inner
        hwo = p.slot_width_outer * 0.5
        r_straight = ri + p.slot_depth - hwo
        pt_tl = _add_rotated_point(self.backend, ri,           hw,  th, 0.0)
        pt_tr = _add_rotated_point(self.backend, ri,          -hw,  th, 0.0)
        pt_rj = _add_rotated_point(self.backend, r_straight,  -hw,  th, 0.0)
        pt_lj = _add_rotated_point(self.backend, r_straight,   hw,  th, 0.0)
        pt_bot = _add_rotated_point(self.backend, r_straight + hwo, 0.0, th, 0.0)
        l_top = self.backend.add_line(pt_tl, pt_tr)
        l_r = self.backend.add_line(pt_tr, pt_rj)
        arc1 = self.backend.add_arc(pt_rj, _add_rotated_point(self.backend, r_straight, 0.0, th, 0.0), pt_bot)
        arc2 = self.backend.add_arc(pt_bot, _add_rotated_point(self.backend, r_straight, 0.0, th, 0.0), pt_lj)
        l_l = self.backend.add_line(pt_lj, pt_tl)
        cl = self.backend.add_curve_loop([l_top, l_r, arc1, arc2, l_l])
        sp.slot_surface = self.backend.add_plane_surface([cl])
        sp.mouth_curve_bot = l_top

    def _build_semi_closed(self, p: StatorParams, k: int, sp: SlotProfile) -> None:
        th = sp.angle
        r0 = p.R_inner
        r1 = r0 + p.slot_opening_depth
        r2 = r1 + p.slot_depth
        hw_mouth = p.slot_opening * 0.5
        hw_shoulder = p.slot_width_inner * 0.5
        hw_bottom = p.slot_width_outer * 0.5
        pt1 = _add_rotated_point(self.backend, r0,  hw_mouth,    th, 0.0)
        pt2 = _add_rotated_point(self.backend, r0, -hw_mouth,    th, 0.0)
        pt3 = _add_rotated_point(self.backend, r1,  hw_shoulder, th, 0.0)
        pt4 = _add_rotated_point(self.backend, r1, -hw_shoulder, th, 0.0)
        pt5 = _add_rotated_point(self.backend, r2,  hw_bottom,   th, 0.0)
        pt6 = _add_rotated_point(self.backend, r2, -hw_bottom,   th, 0.0)
        l1 = self.backend.add_line(pt1, pt2)
        l2 = self.backend.add_line(pt2, pt4)
        l3 = self.backend.add_line(pt4, pt6)
        l4 = self.backend.add_line(pt6, pt5)
        l5 = self.backend.add_line(pt5, pt3)
        l6 = self.backend.add_line(pt3, pt1)
        cl = self.backend.add_curve_loop([l1, l2, l3, l4, l5, l6])
        sp.slot_surface = self.backend.add_plane_surface([cl])
        sp.mouth_curve_bot = l1
        sp.mouth_curve_top = l4

    def _build_coil(self, p: StatorParams, sp: SlotProfile) -> None:
        th = sp.angle
        ins = p.insulation_thickness
        ri = p.R_inner + p.slot_opening_depth + ins
        hwi = p.coil_width_inner * 0.5
        hwo = p.coil_width_outer * 0.5

        if p.winding_type.value == 1:  # DOUBLE_LAYER
            half_depth = (p.coil_depth - 2.0 * ins) / 2.0
            # Upper coil
            ru0, ru1 = ri, ri + half_depth
            u1 = _add_rotated_point(self.backend, ru0,  hwi, th, 0.0)
            u2 = _add_rotated_point(self.backend, ru0, -hwi, th, 0.0)
            u3 = _add_rotated_point(self.backend, ru1, -hwo, th, 0.0)
            u4 = _add_rotated_point(self.backend, ru1,  hwo, th, 0.0)
            ucl = self.backend.add_curve_loop([
                self.backend.add_line(u1, u2),
                self.backend.add_line(u2, u3),
                self.backend.add_line(u3, u4),
                self.backend.add_line(u4, u1),
            ])
            sp.coil_upper_sf = self.backend.add_plane_surface([ucl])
            # Lower coil
            rl0, rl1 = ru1 + 2.0 * ins, ru1 + 2.0 * ins + half_depth
            l1_ = _add_rotated_point(self.backend, rl0,  hwi, th, 0.0)
            l2_ = _add_rotated_point(self.backend, rl0, -hwi, th, 0.0)
            l3_ = _add_rotated_point(self.backend, rl1, -hwo, th, 0.0)
            l4_ = _add_rotated_point(self.backend, rl1,  hwo, th, 0.0)
            lcl = self.backend.add_curve_loop([
                self.backend.add_line(l1_, l2_),
                self.backend.add_line(l2_, l3_),
                self.backend.add_line(l3_, l4_),
                self.backend.add_line(l4_, l1_),
            ])
            sp.coil_lower_sf = self.backend.add_plane_surface([lcl])
        else:
            rc0, rc1 = ri, ri + p.coil_depth
            c1 = _add_rotated_point(self.backend, rc0,  hwi, th, 0.0)
            c2 = _add_rotated_point(self.backend, rc0, -hwi, th, 0.0)
            c3 = _add_rotated_point(self.backend, rc1, -hwo, th, 0.0)
            c4 = _add_rotated_point(self.backend, rc1,  hwo, th, 0.0)
            ccl = self.backend.add_curve_loop([
                self.backend.add_line(c1, c2),
                self.backend.add_line(c2, c3),
                self.backend.add_line(c3, c4),
                self.backend.add_line(c4, c1),
            ])
            sp.coil_upper_sf = self.backend.add_plane_surface([ccl])

    def _make_ins_surface(self, r0: float, r1: float, hwi: float, hwo: float, ins: float, th: float) -> int:
        er0 = r0 - ins
        er1 = r1 + ins
        ehwi = hwi + ins
        ehwo = hwo + ins
        i1 = _add_rotated_point(self.backend, er0,  ehwi, th, 0.0)
        i2 = _add_rotated_point(self.backend, er0, -ehwi, th, 0.0)
        i3 = _add_rotated_point(self.backend, er1, -ehwo, th, 0.0)
        i4 = _add_rotated_point(self.backend, er1,  ehwo, th, 0.0)
        icl = self.backend.add_curve_loop([
            self.backend.add_line(i1, i2),
            self.backend.add_line(i2, i3),
            self.backend.add_line(i3, i4),
            self.backend.add_line(i4, i1),
        ])
        return self.backend.add_plane_surface([icl])

    def _build_insulation(self, p: StatorParams, sp: SlotProfile) -> None:
        th = sp.angle
        ins = p.insulation_thickness
        ri = p.R_inner + p.slot_opening_depth
        hwi = p.coil_width_inner * 0.5
        hwo = p.coil_width_outer * 0.5
        if p.winding_type.value == 1:  # DOUBLE_LAYER
            half_depth = (p.coil_depth - 2.0 * ins) / 2.0
            ru0, ru1 = ri + ins, ri + ins + half_depth
            sp.ins_upper_sf = self._make_ins_surface(ru0, ru1, hwi, hwo, ins, th)
            rl0, rl1 = ru1 + 2.0 * ins, ru1 + 2.0 * ins + half_depth
            sp.ins_lower_sf = self._make_ins_surface(rl0, rl1, hwi, hwo, ins, th)
        else:
            rc0, rc1 = ri + ins, ri + ins + p.coil_depth
            sp.ins_upper_sf = self._make_ins_surface(rc0, rc1, hwi, hwo, ins, th)
