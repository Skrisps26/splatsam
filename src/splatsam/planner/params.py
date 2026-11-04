"""Bounded, hashable rule-based planner configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from math import isfinite
from pathlib import Path
from typing import ClassVar

from splatsam.utils import digest, load_json, save_json


@dataclass(frozen=True)
class PlannerParams:
    grip_safety_factor: float = 1.2
    jaw_margin_m: float = 0.004
    pregrasp_height_m: float = 0.10
    approach_speed: float = 0.10
    lift_speed: float = 0.10
    n_yaw_candidates: int = 5
    clearance_weight: float = 1.0
    center_weight: float = 1.0
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "grip_safety_factor": (0.5, 4),
        "jaw_margin_m": (0, 0.02),
        "pregrasp_height_m": (0.04, 0.20),
        "approach_speed": (0.02, 0.25),
        "lift_speed": (0.03, 0.25),
        "n_yaw_candidates": (1, 16),
        "clearance_weight": (0, 5),
        "center_weight": (0, 5),
    }

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            lo, hi = self.BOUNDS[f.name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"{f.name} must be finite in [{lo},{hi}]")
        if not isinstance(self.n_yaw_candidates, int):
            raise ValueError("n_yaw_candidates must be an integer")  # noqa: TRY004 - all invalid patches use one domain error

    def patch(self, patch: dict[str, float | int]) -> PlannerParams:
        if set(patch) - self.BOUNDS.keys():
            raise ValueError("unknown planner parameter")
        return replace(self, **patch)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        return digest(asdict(self))

    def save(self, path: Path) -> None:
        save_json(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> PlannerParams:
        return cls(**load_json(path))
