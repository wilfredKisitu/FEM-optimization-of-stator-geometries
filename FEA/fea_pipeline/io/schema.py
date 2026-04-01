"""I/O schema — Pydantic models for the pipeline input/output contract.

Pydantic is an optional dependency.  When it is not installed the module falls
back to plain dataclasses so that the rest of the pipeline (and the test suite)
can still run without it.
"""

from __future__ import annotations

from typing import Optional

try:
    from pydantic import BaseModel, Field, model_validator

    class StatorMeshInput(BaseModel):
        """Contract for stator geometry received from the mesh construction module."""

        # Identity
        stator_id: str
        geometry_source: str = "parametric"

        # Mesh file
        mesh_file_path: str = ""
        mesh_format: str = "synthetic"          # "gmsh4" | "hdf5_xdmf" | "vtk" | "synthetic"

        # Physical geometry (SI units)
        outer_diameter: float
        inner_diameter: float
        axial_length: float
        num_slots: int
        num_poles: int
        slot_opening: float
        tooth_width: float
        yoke_height: float
        slot_depth: float

        # Winding
        winding_type: str = "distributed"       # "distributed" | "concentrated" | "hairpin"
        num_layers: int = 2
        conductors_per_slot: int = 20
        winding_factor: float = 0.866
        fill_factor: float = 0.45
        wire_diameter: Optional[float] = None

        # Region tags (must match physical groups in mesh file)
        region_tags: dict[str, int] = Field(
            default_factory=lambda: {"stator_core": 1, "winding": 2, "air_gap": 3}
        )

        # Material assignment: region_name → material_id in MATERIAL_DB
        material_map: dict[str, str] = Field(
            default_factory=lambda: {
                "stator_core": "M250-35A",
                "winding": "copper_class_F",
                "air_gap": "air",
            }
        )

        # Operating point
        rated_current_rms: float = 50.0
        rated_speed_rpm: float = 3000.0
        rated_torque: float = 50.0
        dc_bus_voltage: float = 400.0

        # Mesh quality metadata
        min_element_quality: float = 0.6
        max_element_size: float = 0.003
        num_elements: int = 8500
        num_nodes: int = 4320

        # Symmetry (optional)
        symmetry_factor: Optional[int] = None
        periodic_boundary_pairs: Optional[list[tuple[int, int]]] = None

        @model_validator(mode="after")
        def _check_material_map(self):
            missing = set(self.material_map.keys()) - set(self.region_tags.keys())
            if missing:
                raise ValueError(f"material_map references unknown regions: {missing}")
            return self

    class PipelineConfig(BaseModel):
        """Top-level pipeline configuration (loaded from YAML)."""
        pipeline: dict = Field(default_factory=dict)
        electromagnetic: dict = Field(default_factory=dict)
        thermal: dict = Field(default_factory=dict)
        structural: dict = Field(default_factory=dict)
        output: dict = Field(default_factory=dict)

except ImportError:
    # Fallback: plain dataclass
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class StatorMeshInput:  # type: ignore[no-redef]
        stator_id: str
        outer_diameter: float
        inner_diameter: float
        axial_length: float
        num_slots: int
        num_poles: int
        slot_opening: float
        tooth_width: float
        yoke_height: float
        slot_depth: float
        geometry_source: str = "parametric"
        mesh_file_path: str = ""
        mesh_format: str = "synthetic"
        winding_type: str = "distributed"
        num_layers: int = 2
        conductors_per_slot: int = 20
        winding_factor: float = 0.866
        fill_factor: float = 0.45
        wire_diameter: Optional[float] = None
        region_tags: dict = dc_field(
            default_factory=lambda: {"stator_core": 1, "winding": 2, "air_gap": 3}
        )
        material_map: dict = dc_field(
            default_factory=lambda: {
                "stator_core": "M250-35A",
                "winding": "copper_class_F",
                "air_gap": "air",
            }
        )
        rated_current_rms: float = 50.0
        rated_speed_rpm: float = 3000.0
        rated_torque: float = 50.0
        dc_bus_voltage: float = 400.0
        min_element_quality: float = 0.6
        max_element_size: float = 0.003
        num_elements: int = 8500
        num_nodes: int = 4320
        symmetry_factor: Optional[int] = None
        periodic_boundary_pairs: Optional[list] = None

    @dataclass
    class PipelineConfig:  # type: ignore[no-redef]
        pipeline: dict = dc_field(default_factory=dict)
        electromagnetic: dict = dc_field(default_factory=dict)
        thermal: dict = dc_field(default_factory=dict)
        structural: dict = dc_field(default_factory=dict)
        output: dict = dc_field(default_factory=dict)
