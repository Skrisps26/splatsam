"""Seeded, bounded scenario sampling for synthetic physics evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Iterable

import numpy as np

from splatsam.llm.schemas import ObjectSpec, ScenarioFamily, ScenarioSpec, SceneSpec

TRAIN_SEEDS = range(0, 150)
VALIDATION_SEEDS = range(10_000, 10_100)
TEST_SEEDS = range(900_000, 900_200)
_PLACEMENT_ATTEMPTS = 64
_CLEARANCE_M = 0.003


class NoGraspableTargetError(ValueError):
    """The scene cannot produce an honest grasp evaluation scenario."""


@dataclass(frozen=True)
class _Footprint:
    radius: float
    z: float


def _footprint(obj: ObjectSpec) -> _Footprint:
    """A conservative circular XY bound; positions are object centres."""
    x, y, z = obj.size
    if obj.shape == "sphere":
        radius = x / 2
    elif obj.shape == "cylinder":
        radius = x / 2
    else:
        radius = float(np.hypot(x, y) / 2)
    return _Footprint(radius=radius, z=z)


def _inside_table(position: tuple[float, float, float], radius: float, scene: SceneSpec) -> bool:
    half_x, half_y = scene.table_size[0] / 2, scene.table_size[1] / 2
    return abs(position[0]) + radius + _CLEARANCE_M <= half_x and abs(position[1]) + radius + _CLEARANCE_M <= half_y


def _nonoverlapping(
    position: tuple[float, float, float], radius: float, occupied: Iterable[tuple[tuple[float, float, float], float]]
) -> bool:
    return all(np.hypot(position[0] - other[0], position[1] - other[1]) >= radius + other_radius + _CLEARANCE_M
               for other, other_radius in occupied)


def _random_position(
    rng: np.random.Generator,
    scene: SceneSpec,
    radius: float,
    z: float,
    occupied: list[tuple[tuple[float, float, float], float]],
) -> tuple[float, float, float] | None:
    half_x, half_y = scene.table_size[0] / 2 - radius - _CLEARANCE_M, scene.table_size[1] / 2 - radius - _CLEARANCE_M
    if half_x <= 0 or half_y <= 0:
        return None
    for _ in range(_PLACEMENT_ATTEMPTS):
        position = (float(rng.uniform(-half_x, half_x)), float(rng.uniform(-half_y, half_y)), z)
        if _nonoverlapping(position, radius, occupied):
            return position
    return None


def _scene_positions(rng: np.random.Generator, family: ScenarioFamily, scene: SceneSpec) -> tuple[dict[str, tuple[float, float, float]], list[str]]:
    """Jitter each scene object while retaining every object's stable ID."""
    positions: dict[str, tuple[float, float, float]] = {}
    occupied: list[tuple[tuple[float, float, float], float]] = []
    degradation: list[str] = []
    for obj in scene.objects:
        footprint = _footprint(obj)
        chosen: tuple[float, float, float] | None = None
        for _ in range(_PLACEMENT_ATTEMPTS):
            candidate = (
                float(np.clip(obj.position[0] + rng.uniform(-family.jitter_m, family.jitter_m),
                              -scene.table_size[0] / 2 + footprint.radius + _CLEARANCE_M,
                              scene.table_size[0] / 2 - footprint.radius - _CLEARANCE_M)),
                float(np.clip(obj.position[1] + rng.uniform(-family.jitter_m, family.jitter_m),
                              -scene.table_size[1] / 2 + footprint.radius + _CLEARANCE_M,
                              scene.table_size[1] / 2 - footprint.radius - _CLEARANCE_M)),
                obj.position[2],
            )
            if _inside_table(candidate, footprint.radius, scene) and _nonoverlapping(candidate, footprint.radius, occupied):
                chosen = candidate
                break
        if chosen is None:
            chosen = _random_position(rng, scene, footprint.radius, obj.position[2], occupied)
        if chosen is None:
            raise ValueError(f"unable to place scene object {obj.id} without overlap")
        positions[obj.id] = chosen
        occupied.append((chosen, footprint.radius))
    return positions, degradation


def _obstacle(
    *, obstacle_id: str, position: tuple[float, float, float], size: tuple[float, float, float], yaw: float,
    kind: str,
) -> ObjectSpec:
    return ObjectSpec(
        id=obstacle_id,
        name=kind,
        shape="box",
        size=size,
        position=position,
        yaw=yaw,
        mass=0.05,
        friction=0.6,
        material="plastic",
        graspable=False,
        rgba=(0.35, 0.35, 0.35, 1.0),
        height_source="scenario",
        mass_source="scenario",
        friction_source="scenario",
    )


def _place_occluder(
    rng: np.random.Generator,
    scene: SceneSpec,
    target_position: tuple[float, float, float],
    target_radius: float,
    height: float,
    occupied: list[tuple[tuple[float, float, float], float]],
    obstacle_id: str,
) -> ObjectSpec | None:
    width = float(rng.uniform(0.035, 0.065))
    depth = float(rng.uniform(0.035, 0.065))
    radius = float(np.hypot(width, depth) / 2)
    # A near-target obstacle changes approach clearance instead of merely adding distant clutter.
    for _ in range(_PLACEMENT_ATTEMPTS):
        angle = float(rng.uniform(-pi, pi))
        distance = target_radius + radius + _CLEARANCE_M + float(rng.uniform(0.002, 0.018))
        position = (target_position[0] + distance * cos(angle), target_position[1] + distance * sin(angle), height / 2)
        if _inside_table(position, radius, scene) and _nonoverlapping(position, radius, occupied):
            return _obstacle(obstacle_id=obstacle_id, position=position, size=(width, depth, height), yaw=angle, kind="occluder")
    return None


def sample_scenario(family: ScenarioFamily, seed: int, scene: SceneSpec) -> ScenarioSpec:
    """Generate a deterministic, initially non-overlapping scenario.

    Hidden mass and friction multipliers belong only to this scenario, never to
    the planner-facing scene estimates.
    """
    graspable = [obj for obj in scene.objects if obj.graspable]
    if not graspable:
        raise NoGraspableTargetError("scene has no graspable target; evaluation is blocked")
    if len({obj.id for obj in scene.objects}) != len(scene.objects):
        raise ValueError("scene object IDs must be unique")

    rng = np.random.default_rng(seed)
    positions, degradation = _scene_positions(rng, family, scene)
    yaws = {obj.id: float(obj.yaw + rng.uniform(*family.yaw_range)) for obj in scene.objects}
    target = graspable[int(rng.integers(len(graspable)))]
    occupied = [(positions[obj.id], _footprint(obj).radius) for obj in scene.objects]
    obstacles: list[ObjectSpec] = []

    requested_clutter = int(rng.integers(family.clutter_count[0], family.clutter_count[1] + 1))
    for index in range(requested_clutter):
        size = (float(rng.uniform(0.03, 0.065)), float(rng.uniform(0.03, 0.065)), float(rng.uniform(0.02, 0.055)))
        radius = float(np.hypot(size[0], size[1]) / 2)
        position = _random_position(rng, scene, radius, size[2] / 2, occupied)
        if position is None:
            degradation.append(f"clutter placement degraded: requested {requested_clutter}, placed {len(obstacles)}")
            break
        obstacle = _obstacle(obstacle_id=f"clutter-{seed}-{index}", position=position, size=size,
                             yaw=float(rng.uniform(-pi, pi)), kind="clutter")
        obstacles.append(obstacle)
        occupied.append((position, radius))

    requested_occluders = int(rng.integers(family.occluder_count[0], family.occluder_count[1] + 1))
    target_radius = _footprint(target).radius
    placed_occluders = 0
    for index in range(requested_occluders):
        height = float(rng.uniform(*family.occluder_height))
        obstacle = _place_occluder(rng, scene, positions[target.id], target_radius, height, occupied,
                                   f"occluder-{seed}-{index}")
        if obstacle is None:
            degradation.append(f"occluder placement degraded: requested {requested_occluders}, placed {placed_occluders}")
            break
        obstacles.append(obstacle)
        occupied.append((obstacle.position, _footprint(obstacle).radius))
        placed_occluders += 1

    return ScenarioSpec(
        seed=seed,
        target_id=target.id,
        positions=positions,
        yaws=yaws,
        friction_scale=float(rng.uniform(*family.friction_scale)),
        mass_scale=float(rng.uniform(*family.mass_scale)),
        obstacles=obstacles,
        degradation=degradation,
    )


def split_scenarios(family: ScenarioFamily, scene: SceneSpec, split: str) -> list[ScenarioSpec]:
    """Return public train/validation scenarios; held-out seeds are seal-private."""
    if split == "train":
        seeds = TRAIN_SEEDS
    elif split == "validation":
        seeds = VALIDATION_SEEDS
    else:
        raise ValueError("only train and validation splits are public")
    return [sample_scenario(family, seed, scene) for seed in seeds]
