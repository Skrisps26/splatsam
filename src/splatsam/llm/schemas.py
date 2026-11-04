"""Versioned wire and experiment contracts. SI units; wxyz quaternions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["synthetic", "recorded", "live", "fallback"]
Failure = Literal[
    "no_grasp_found", "occluder_collision", "knock_over", "slip_or_drop", "instability", "timeout"
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: int = 1


class ObjectInfo(Record):
    id: str
    name: str
    material_guess: str = "plastic"
    approx_size_cm: tuple[float, float, float]
    graspable: bool = True
    count: int = Field(default=1, ge=1, le=20)


class ObjectList(Record):
    objects: list[ObjectInfo]
    source: Mode = "synthetic"


class MaskFrame(Record):
    frame_index: int
    timestamp_s: float
    mask_path: str
    centroid: tuple[float, float]
    area: int
    box: tuple[float, float, float, float]


class MaskTrack(Record):
    id: str
    name: str
    frames: list[MaskFrame]
    image_width: int
    image_height: int
    total_frames: int
    source: Mode = "synthetic"
    stable: bool = True
    diagnostics: list[str] = Field(default_factory=list)


class ObjectSpec(Record):
    id: str
    name: str
    shape: Literal["box", "sphere", "cylinder", "horizontal_cylinder", "mesh"]
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    yaw: float = 0.0
    mass: float = Field(gt=0)
    friction: float = Field(gt=0)
    material: str = "plastic"
    graspable: bool = True
    rgba: tuple[float, float, float, float] = (0.3, 0.5, 0.7, 1)
    height_source: str = "estimate"
    mass_source: str = "estimate"
    friction_source: str = "estimate"
    reconstruction_source: str = "primitive"
    mesh_path: str | None = None
    consistency_iou: float | None = None


class SceneSpec(Record):
    objects: list[ObjectSpec]
    table_size: tuple[float, float] = (1, 1)
    source: Mode = "synthetic"
    seed: int = 42
    scale_source: str = "ground_truth_fixture"
    scale_error: float | None = None
    reference_validation: dict[str, float] = Field(default_factory=dict)


class ScenarioFamily(Record):
    jitter_m: float = Field(default=0.02, ge=0, le=0.04)
    yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    mass_scale: tuple[float, float] = (0.7, 1.5)
    clutter_count: tuple[int, int] = (0, 2)
    occluder_count: tuple[int, int] = (0, 1)
    occluder_height: tuple[float, float] = (0.02, 0.08)

    @model_validator(mode="after")
    def bounds(self):
        for name, lo, hi in [
            ("friction_scale", 0.5, 1.5),
            ("mass_scale", 0.5, 2),
            ("clutter_count", 0, 5),
            ("occluder_count", 0, 3),
            ("occluder_height", 0.005, 0.2),
            ("yaw_range", -3.141592653589793, 3.141592653589793),
        ]:
            a, b = getattr(self, name)
            if not lo <= a <= b <= hi:
                raise ValueError(f"{name} outside [{lo}, {hi}] or reversed")
        return self


class ScenarioSpec(Record):
    seed: int
    target_id: str
    positions: dict[str, tuple[float, float, float]]
    yaws: dict[str, float]
    friction_scale: float
    mass_scale: float
    obstacles: list[ObjectSpec] = Field(default_factory=list)
    degradation: list[str] = Field(default_factory=list)


class EpisodeResult(Record):
    seed: int
    target_id: str
    success: bool
    failure_mode: Failure | None
    start_z: float = 0
    final_z: float = 0
    max_lift_m: float = 0
    min_hold_lift_m: float = 0
    hold_duration_s: float = 0
    end_finger_contact: bool = False
    preclose_displacement_m: float = 0
    preclose_occluder_contact: bool = False
    max_penetration_m: float = 0
    peak_velocity: float = 0
    detail: str = ""
    trace: list[dict[str, float | str]] = Field(default_factory=list)


class PlannerPatch(Record):
    patch: dict[str, float | int]
    reason: str


class CriticLabel(Record):
    label: Failure
    reason: str
