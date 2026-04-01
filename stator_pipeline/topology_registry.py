"""topology_registry.py — Thread-safe registry for stator topology regions."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from .params import WindingType


class RegionType(IntEnum):
    UNKNOWN = 0
    YOKE = 1
    TOOTH = 2
    SLOT_AIR = 3
    SLOT_INS = 4
    COIL_A_POS = 5
    COIL_A_NEG = 6
    COIL_B_POS = 7
    COIL_B_NEG = 8
    COIL_C_POS = 9
    COIL_C_NEG = 10
    BORE_AIR = 11
    BOUNDARY_BORE = 12
    BOUNDARY_OUTER = 13


REGION_CANONICAL_TAG = {r: r.value * 100 for r in RegionType}


@dataclass
class SurfaceRecord:
    type: RegionType
    gmsh_tag: int


@dataclass
class BoundaryRecord:
    type: RegionType
    gmsh_curve: int


@dataclass
class SlotWindingAssignment:
    slot_idx: int
    upper_tag: int
    lower_tag: int
    upper_phase: RegionType
    lower_phase: RegionType


_DISTRIBUTED_PATTERN = [
    RegionType.COIL_A_POS, RegionType.COIL_B_NEG,
    RegionType.COIL_C_POS, RegionType.COIL_A_NEG,
    RegionType.COIL_B_POS, RegionType.COIL_C_NEG,
]
_CONCENTRATED_PATTERN = [
    RegionType.COIL_A_POS, RegionType.COIL_A_NEG,
    RegionType.COIL_B_POS, RegionType.COIL_B_NEG,
    RegionType.COIL_C_POS, RegionType.COIL_C_NEG,
]


class TopologyRegistry:
    def __init__(self, n_slots: int) -> None:
        if n_slots <= 0:
            raise ValueError("n_slots must be > 0")
        self._lock = threading.RLock()
        self.n_slots = n_slots
        self._surface_records: list[SurfaceRecord] = []
        self._boundary_records: list[BoundaryRecord] = []
        self._slot_upper_tags: list[int] = [-1] * n_slots
        self._slot_lower_tags: list[int] = [-1] * n_slots
        self._winding_assignments: list[Optional[SlotWindingAssignment]] = [None] * n_slots
        self._winding_assigned = False

    def register_surface(self, region_type: RegionType, gmsh_tag: int, slot_idx: int = -1) -> None:
        with self._lock:
            self._surface_records.append(SurfaceRecord(type=region_type, gmsh_tag=gmsh_tag))

    def register_slot_coil(self, slot_idx: int, upper_tag: int, lower_tag: int) -> None:
        with self._lock:
            if not (0 <= slot_idx < self.n_slots):
                raise IndexError(f"slot_idx {slot_idx} out of range")
            self._slot_upper_tags[slot_idx] = upper_tag
            self._slot_lower_tags[slot_idx] = lower_tag

    def register_boundary_curve(self, region_type: RegionType, gmsh_curve: int) -> None:
        if region_type not in (RegionType.BOUNDARY_BORE, RegionType.BOUNDARY_OUTER):
            raise ValueError("type must be BOUNDARY_BORE or BOUNDARY_OUTER")
        with self._lock:
            self._boundary_records.append(BoundaryRecord(type=region_type, gmsh_curve=gmsh_curve))

    def assign_winding_layout(self, winding_type: WindingType) -> None:
        with self._lock:
            if all(t < 0 for t in self._slot_upper_tags):
                raise RuntimeError("assign_winding_layout: no coils registered")
            pattern = _CONCENTRATED_PATTERN if winding_type == WindingType.CONCENTRATED else _DISTRIBUTED_PATTERN
            for i in range(self.n_slots):
                upper_phase = pattern[i % 6]
                lower_tag = self._slot_lower_tags[i]
                lower_phase = pattern[i % 6] if lower_tag >= 0 else RegionType.UNKNOWN
                self._winding_assignments[i] = SlotWindingAssignment(
                    slot_idx=i,
                    upper_tag=self._slot_upper_tags[i],
                    lower_tag=lower_tag,
                    upper_phase=upper_phase,
                    lower_phase=lower_phase,
                )
            self._winding_assigned = True

    def get_surfaces(self, region_type: RegionType) -> list[int]:
        with self._lock:
            return [r.gmsh_tag for r in self._surface_records if r.type == region_type]

    def get_boundary_curves(self, region_type: RegionType) -> list[int]:
        with self._lock:
            return [r.gmsh_curve for r in self._boundary_records if r.type == region_type]

    def get_slot_assignment(self, slot_idx: int) -> SlotWindingAssignment:
        if not self._winding_assigned:
            raise RuntimeError("winding not yet assigned")
        if not (0 <= slot_idx < self.n_slots):
            raise IndexError(f"slot_idx {slot_idx} out of range")
        return self._winding_assignments[slot_idx]

    @property
    def total_surfaces(self) -> int:
        with self._lock:
            return len(self._surface_records)

    @property
    def winding_assigned(self) -> bool:
        return self._winding_assigned

    @property
    def winding_assignments(self) -> list[Optional[SlotWindingAssignment]]:
        return self._winding_assignments
