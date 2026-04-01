"""gmsh_backend.py — Abstract GMSH backend and in-memory stub."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

MAX_SLOTS = 256
MAX_PHYS_GROUPS = 512
MAX_TAGS_PER_GROUP = 256


class GmshBackend(ABC):
    """Abstract base for GMSH geometry/mesh backends."""

    @abstractmethod
    def initialize(self, model_name: str) -> None: ...
    @abstractmethod
    def synchronize(self) -> None: ...
    @abstractmethod
    def finalize(self) -> None: ...
    @abstractmethod
    def set_option(self, name: str, value: float) -> None: ...
    @abstractmethod
    def add_point(self, x: float, y: float, z: float, mesh_size: float) -> int: ...
    @abstractmethod
    def add_line(self, start: int, end: int) -> int: ...
    @abstractmethod
    def add_circle(self, cx: float, cy: float, cz: float, r: float) -> int: ...
    @abstractmethod
    def add_arc(self, start: int, centre: int, end: int) -> int: ...
    @abstractmethod
    def add_curve_loop(self, tags: list[int]) -> int: ...
    @abstractmethod
    def add_plane_surface(self, loop_tags: list[int]) -> int: ...
    @abstractmethod
    def boolean_cut(self, objects: list[tuple[int, int]], tools: list[tuple[int, int]], remove_tool: bool = False) -> list[tuple[int, int]]: ...
    @abstractmethod
    def boolean_fragment(self, objects: list[tuple[int, int]], tools: list[tuple[int, int]]) -> list[tuple[int, int]]: ...
    @abstractmethod
    def add_physical_group(self, dim: int, tags: list[int], name: str, tag: int = -1) -> int: ...
    @abstractmethod
    def add_math_eval_field(self, expr: str) -> int: ...
    @abstractmethod
    def add_constant_field(self, value: float, surfaces: list[int]) -> int: ...
    @abstractmethod
    def set_background_field(self, field_tag: int) -> None: ...
    @abstractmethod
    def generate_mesh(self, dim: int) -> None: ...
    @abstractmethod
    def write_mesh(self, filename: str) -> None: ...
    @abstractmethod
    def get_entities_2d(self) -> list[tuple[int, int]]: ...


@dataclass
class PhysGroupRecord:
    dim: int
    tags: list[int]
    name: str
    tag: int


class StubGmshBackend(GmshBackend):
    """In-memory stub that tracks all operations without calling GMSH."""

    def __init__(self) -> None:
        self._initialized = False
        self._finalized = False
        self._sync_count = 0
        self._point_counter = 0
        self._line_counter = 0
        self._curve_loop_counter = 0
        self._surface_counter = 0
        self._field_counter = 0
        self._phys_group_tag_counter = 1000
        self._surfaces_2d: list[tuple[int, int]] = []
        self._phys_groups: list[PhysGroupRecord] = []
        self._background_field: int = -1
        self._mesh_generated = False
        self._last_write_path = ""

    def initialize(self, model_name: str) -> None:
        self._initialized = True

    def synchronize(self) -> None:
        self._sync_count += 1

    def finalize(self) -> None:
        self._finalized = True

    def set_option(self, name: str, value: float) -> None:
        pass

    def add_point(self, x: float, y: float, z: float, mesh_size: float) -> int:
        self._point_counter += 1
        return self._point_counter

    def add_line(self, start: int, end: int) -> int:
        self._line_counter += 1
        return self._line_counter

    def add_circle(self, cx: float, cy: float, cz: float, r: float) -> int:
        self._line_counter += 1
        return self._line_counter

    def add_arc(self, start: int, centre: int, end: int) -> int:
        self._line_counter += 1
        return self._line_counter

    def add_curve_loop(self, tags: list[int]) -> int:
        self._curve_loop_counter += 1
        return self._curve_loop_counter

    def add_plane_surface(self, loop_tags: list[int]) -> int:
        self._surface_counter += 1
        tag = self._surface_counter
        self._surfaces_2d.append((2, tag))
        return tag

    def boolean_cut(self, objects: list[tuple[int, int]], tools: list[tuple[int, int]], remove_tool: bool = False) -> list[tuple[int, int]]:
        return list(objects)

    def boolean_fragment(self, objects: list[tuple[int, int]], tools: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return list(objects) + list(tools)

    def add_physical_group(self, dim: int, tags: list[int], name: str, tag: int = -1) -> int:
        if tag < 0:
            self._phys_group_tag_counter += 1
            tag = self._phys_group_tag_counter
        self._phys_groups.append(PhysGroupRecord(dim=dim, tags=list(tags), name=name, tag=tag))
        return tag

    def add_math_eval_field(self, expr: str) -> int:
        self._field_counter += 1
        return self._field_counter

    def add_constant_field(self, value: float, surfaces: list[int]) -> int:
        self._field_counter += 1
        return self._field_counter

    def set_background_field(self, field_tag: int) -> None:
        self._background_field = field_tag

    def generate_mesh(self, dim: int) -> None:
        self._mesh_generated = True

    def write_mesh(self, filename: str) -> None:
        self._last_write_path = filename

    def get_entities_2d(self) -> list[tuple[int, int]]:
        return list(self._surfaces_2d)


def make_default_backend() -> GmshBackend:
    """Return the default (stub) backend. Replace with real GMSH when available."""
    return StubGmshBackend()
