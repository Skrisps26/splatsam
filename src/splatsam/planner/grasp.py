"""Deterministic top-down grasp candidates for the floating parallel-jaw hand."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Mapping

from splatsam.llm.schemas import ObjectSpec, ScenarioSpec, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.sim.scene import CLOSED_GAP_M, OPEN_GAP_M


@dataclass(frozen=True)
class Grasp:
    """A feasible top-down grasp expressed in the hand target frame.

    ``yaw`` rotates hand local-x in world XY; the real jaws close along local-y.
    Thus a long marker aligned to local-x is squeezed perpendicular to its long
    axis. ``force_per_finger_n`` is based only on nominal reconstructed physics.
    """

    target_id: str
    position: tuple[float, float, float]
    yaw: float
    jaw_width_m: float
    force_per_finger_n: float
    score: float


def smoothstep_peak_acceleration(distance_m: float, speed_m_s: float) -> float:
    """Analytic peak acceleration for a cubic smoothstep position profile."""
    if not isfinite(distance_m) or not isfinite(speed_m_s) or distance_m < 0 or speed_m_s <= 0:
        raise ValueError("distance must be finite/non-negative and speed positive")
    if distance_m == 0:
        return 0.0
    duration = distance_m / speed_m_s
    return 6.0 * distance_m / (duration * duration)


def grip_force_per_finger(object_mass_kg: float, friction: float, params: PlannerParams) -> float:
    """Return the bounded motor force from nominal mass/friction estimates."""
    if not isfinite(object_mass_kg) or not isfinite(friction) or object_mass_kg <= 0 or friction <= 0:
        raise ValueError("mass and friction must be finite and positive")
    lift_accel = smoothstep_peak_acceleration(0.10, params.lift_speed)
    force = params.grip_safety_factor * object_mass_kg * (9.81 + lift_accel) / (2.0 * friction)
    return min(30.0, max(0.0, force))


def _finite_xyz(values: tuple[float, float, float] | object) -> bool:
    try:
        return len(values) == 3 and all(isfinite(float(value)) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _position(spec: ObjectSpec, observed: Mapping[str, tuple[float, float, float]] | None) -> tuple[float, float, float]:
    candidate = spec.position if observed is None else observed.get(spec.id, spec.position)
    if not _finite_xyz(candidate):
        raise ValueError(f"{spec.id}: non-finite observed position")
    return tuple(float(value) for value in candidate)  # type: ignore[return-value]


def _axis_yaws(spec: ObjectSpec, count: int) -> list[float]:
    """Return OBB principal-axis candidates with deterministic angle offsets."""
    if spec.shape in {"box", "horizontal_cylinder"}:
        principal = spec.yaw
        bases = (principal, principal + pi / 2)
    else:
        # A sphere/vertical cylinder has no unique planar principal axis.
        bases = (0.0,)
    offsets = [0.0]
    # Bounded nearby offsets capture imperfect reconstruction without random search.
    for index in range(1, count):
        magnitude = (index + 1) // 2 * (pi / max(8, count * 2))
        offsets.append(magnitude if index % 2 else -magnitude)
    yaws: list[float] = []
    for base in bases:
        for offset in offsets:
            yaw = base + offset
            # pi symmetry is sufficient for a parallel gripper.
            normalized = ((yaw + pi / 2) % pi) - pi / 2
            if not any(abs(normalized - existing) < 1e-12 for existing in yaws):
                yaws.append(normalized)
    return yaws


def _projected_width(spec: ObjectSpec, hand_yaw: float) -> float:
    """Full target extent along the hand's jaw-closing local-y direction."""
    relative = hand_yaw - spec.yaw
    closure_x = -sin(hand_yaw)
    closure_y = cos(hand_yaw)
    if spec.shape == "box":
        return abs(cos(relative)) * spec.size[1] + abs(sin(relative)) * spec.size[0]
    if spec.shape == "horizontal_cylinder":
        # Capsule footprint: segment projection plus circular end-cap diameter.
        return abs(closure_x * cos(spec.yaw) + closure_y * sin(spec.yaw)) * spec.size[0] + spec.size[1]
    if spec.shape == "cylinder":
        return spec.size[0]
    if spec.shape == "sphere":
        return spec.size[0]
    return float("inf")


def _footprint_radius(spec: ObjectSpec) -> float:
    if spec.shape == "box":
        return 0.5 * (spec.size[0] ** 2 + spec.size[1] ** 2) ** 0.5
    if spec.shape == "horizontal_cylinder":
        return 0.5 * spec.size[0] + 0.5 * spec.size[1]
    return spec.size[0] / 2


def _candidate_offsets() -> tuple[tuple[float, float], ...]:
    # Centre first gives a deterministic, stable tie break. Small offsets make
    # clearance scoring useful when the reconstructed centre is slightly wrong.
    return ((0.0, 0.0), (0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006))


def _clearance(
    candidate_xy: tuple[float, float],
    target: ObjectSpec,
    obstacles: list[ObjectSpec],
    observed: Mapping[str, tuple[float, float, float]] | None,
) -> float:
    if not obstacles:
        return 0.25
    target_radius = _footprint_radius(target)
    nearest = float("inf")
    for obstacle in obstacles:
        try:
            ox, oy, _ = _position(obstacle, observed)
        except ValueError:
            continue
        dx = candidate_xy[0] - ox
        dy = candidate_xy[1] - oy
        clearance = (dx * dx + dy * dy) ** 0.5 - target_radius - _footprint_radius(obstacle)
        nearest = min(nearest, clearance)
    return nearest if isfinite(nearest) else 0.25


def plan_grasp(
    scene: SceneSpec,
    scenario: ScenarioSpec,
    params: PlannerParams,
    observed_positions: Mapping[str, tuple[float, float, float]] | None = None,
) -> Grasp | None:
    """Choose the best finite, jaw-feasible candidate for ``scenario.target_id``.

    Only scene estimates are used for the force calculation. Scenario's hidden
    mass/friction multipliers are intentionally ignored by this planner.
    """
    objects = {object_spec.id: object_spec for object_spec in scene.objects}
    target = objects.get(scenario.target_id)
    if target is None or not target.graspable or target.shape == "mesh":
        return None
    try:
        target_position = _position(target, observed_positions)
    except ValueError:
        return None
    if not all(isfinite(value) and value > 0 for value in (*target.size, target.mass, target.friction)):
        return None
    if max(target.size) < .015 or min(target.size[:2]) > .08:
        return None

    obstacle_specs = [
        item for item in [*scene.objects, *scenario.obstacles] if item.id != target.id
    ]
    force = grip_force_per_finger(target.mass, target.friction, params)
    candidates: list[Grasp] = []
    for yaw in _axis_yaws(target, params.n_yaw_candidates):
        width = _projected_width(target, yaw)
        # A smaller-than-closed-gap target cannot be held by the finite fingers;
        # margins apply on both sides of the target.
        if not isfinite(width) or width <= CLOSED_GAP_M or width + 2 * params.jaw_margin_m > OPEN_GAP_M:
            continue
        for offset_x, offset_y in _candidate_offsets():
            world_dx = cos(yaw) * offset_x - sin(yaw) * offset_y
            world_dy = sin(yaw) * offset_x + cos(yaw) * offset_y
            xy = (target_position[0] + world_dx, target_position[1] + world_dy)
            clearance = _clearance(xy, target, obstacle_specs, observed_positions)
            center_error = (offset_x * offset_x + offset_y * offset_y) ** 0.5
            score = params.clearance_weight * clearance - params.center_weight * center_error
            if not all(isfinite(value) for value in (*xy, yaw, width, force, score)):
                continue
            candidates.append(
                Grasp(
                    target_id=target.id,
                    position=(xy[0], xy[1], target_position[2]),
                    yaw=yaw,
                    jaw_width_m=width,
                    force_per_finger_n=force,
                    score=score,
                )
            )
    if not candidates:
        return None
    # Higher score wins; stable geometric fields make exact ties independent of
    # Python object/list ordering.
    return min(
        candidates,
        key=lambda item: (-item.score, abs(item.yaw), item.yaw, item.position[0], item.position[1]),
    )
