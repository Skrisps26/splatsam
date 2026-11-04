"""Deterministic MuJoCo grasp episodes with physical, contact-based scoring."""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from math import ceil, cos, isfinite, sin
from multiprocessing import get_context

import numpy as np

from splatsam.llm.schemas import EpisodeResult, ScenarioSpec, SceneSpec
from splatsam.planner.grasp import Grasp, plan_grasp
from splatsam.planner.params import PlannerParams
from splatsam.sim.scene import DT, FINGERTIP_CLEARANCE_M, body_name, build_model, geom_name
from splatsam.utils import digest

import mujoco  # isort: skip


SETTLE_SECONDS = 0.5
CLOSE_SECONDS = 0.5
LIFT_DISTANCE_M = 0.10
HOLD_SECONDS = 2.0
MAX_SIM_SECONDS = 30.0
PENETRATION_LIMIT_M = 0.005
LINEAR_SPEED_LIMIT_M_S = 5.0
ANGULAR_SPEED_LIMIT_RAD_S = 100.0
TRACE_INTERVAL_STEPS = 20


def hold_scored_success(
    min_hold_lift_m: float,
    observed_hold_steps: int,
    end_finger_contact: bool,
    *,
    timestep: float = DT,
) -> bool:
    """Apply the inclusive .05 m / continuous two-second hold boundary."""
    required_steps = round(HOLD_SECONDS / timestep)
    return (
        isfinite(min_hold_lift_m)
        and min_hold_lift_m >= 0.05
        and observed_hold_steps == required_steps
        and end_finger_contact
    )


# Each spawned process owns these model objects. Mutable arrays are restored before
# every use, while MjData is always newly allocated for an episode.
_MODEL_CACHE: dict[str, tuple[mujoco.MjModel, np.ndarray, np.ndarray]] = {}


@dataclass
class _Metrics:
    target_geom: int
    target_body: int
    hand_geoms: set[int]
    finger_geoms: set[int]
    obstacle_geoms: set[int]
    trace_enabled: bool
    trace: list[dict[str, float | str]] = field(default_factory=list)
    max_penetration: float = 0.0
    peak_velocity: float = 0.0
    preclose_occluder_contact: bool = False
    max_preclose_displacement: float = 0.0
    invalid: bool = False
    timed_out: bool = False
    end_finger_contact: bool = False
    _settled_xyz: np.ndarray | None = None


def _cached_model(scene: SceneSpec, scenario: ScenarioSpec) -> mujoco.MjModel:
    key = digest(
        {
            "scene": scene.model_dump(mode="json"),
            "scenario": scenario.model_dump(mode="json"),
            "physics": "parallel-jaw-v1",
        }
    )
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        model = build_model(scene, scenario)
        cached = (model, model.body_mass.copy(), model.geom_friction.copy())
        if len(_MODEL_CACHE) >= 8:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = cached
    model, original_mass, original_friction = cached
    # A caller or future fitting pass must not leak mutable model physics between
    # scenarios that happen to share a worker cache entry.
    model.body_mass[:] = original_mass
    model.geom_friction[:] = original_friction
    model.opt.timestep = DT
    return model


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = mujoco.mj_name2id(model, object_type, name)
    if identifier < 0:
        raise ValueError(f"MuJoCo object missing from model: {name}")
    return identifier


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array((cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)), dtype=float)


def _set_hand_target(data: mujoco.MjData, position: np.ndarray, yaw: float) -> None:
    data.mocap_pos[0, :] = position
    data.mocap_quat[0, :] = _yaw_quat(yaw)


def _target_position(data: mujoco.MjData, target_body: int) -> np.ndarray:
    return np.array(data.xpos[target_body], dtype=float)


def _free_body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> tuple[float, float]:
    joint_address = int(model.body_jntadr[body])
    if joint_address < 0 or int(model.body_jntnum[body]) == 0:
        return 0.0, 0.0
    dof_address = int(model.jnt_dofadr[joint_address])
    if int(model.jnt_type[joint_address]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return 0.0, 0.0
    return (
        float(np.linalg.norm(data.qvel[dof_address : dof_address + 3])),
        float(np.linalg.norm(data.qvel[dof_address + 3 : dof_address + 6])),
    )


def _contacts(data: mujoco.MjData) -> Iterable[mujoco.MjContact]:
    for index in range(data.ncon):
        yield data.contact[index]


def _observe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metrics: _Metrics,
    *,
    phase: str,
    preclose: bool,
    hand_yaw: float,
    force_trace: bool = False,
) -> None:
    """Record physical evidence after an actual MuJoCo step."""
    finger_target_contact = False
    for contact in _contacts(data):
        metrics.max_penetration = max(metrics.max_penetration, max(0.0, -float(contact.dist)))
        pair = {int(contact.geom1), int(contact.geom2)}
        if preclose and pair & metrics.hand_geoms and pair & metrics.obstacle_geoms:
            metrics.preclose_occluder_contact = True
        if pair & metrics.finger_geoms and metrics.target_geom in pair:
            finger_target_contact = True
    # The criterion is an end-of-hold contact, not merely a contact that happened
    # while the jaws were closing.
    metrics.end_finger_contact = finger_target_contact

    linear_speed, angular_speed = _free_body_velocity(model, data, metrics.target_body)
    metrics.peak_velocity = max(metrics.peak_velocity, linear_speed)
    if (
        metrics.max_penetration > PENETRATION_LIMIT_M
        or linear_speed > LINEAR_SPEED_LIMIT_M_S
        or angular_speed > ANGULAR_SPEED_LIMIT_RAD_S
        or not np.all(np.isfinite(data.qpos))
        or not np.all(np.isfinite(data.qvel))
        or not np.all(np.isfinite(data.qacc))
    ):
        metrics.invalid = True
    if data.time > MAX_SIM_SECONDS:
        metrics.timed_out = True

    if preclose and metrics._settled_xyz is not None:
        metrics.max_preclose_displacement = max(
            metrics.max_preclose_displacement,
            float(np.linalg.norm(_target_position(data, metrics.target_body) - metrics._settled_xyz)),
        )
    if metrics.trace_enabled and (
        force_trace or round(data.time / model.opt.timestep) % TRACE_INTERVAL_STEPS == 0
    ):
        hand = data.mocap_pos[0]
        target = _target_position(data, metrics.target_body)
        metrics.trace.append(
            {
                "time": float(data.time),
                "z": float(target[2]),
                "phase": phase,
                "hand_x": float(hand[0]),
                "hand_y": float(hand[1]),
                "hand_z": float(hand[2]),
                "hand_yaw": float(hand_yaw),
                "qpos_json": json.dumps(data.qpos.tolist(), separators=(",", ":")),
            }
        )


def _step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metrics: _Metrics,
    *,
    phase: str,
    preclose: bool,
    hand_yaw: float,
    force_trace: bool = False,
) -> bool:
    mujoco.mj_step(model, data)
    _observe(model, data, metrics, phase=phase, preclose=preclose, hand_yaw=hand_yaw, force_trace=force_trace)
    return not metrics.invalid and not metrics.timed_out


def _smoothstep(fraction: float) -> float:
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metrics: _Metrics,
    *,
    start: np.ndarray,
    end: np.ndarray,
    yaw: float,
    speed: float,
    phase: str,
    preclose: bool,
) -> bool:
    distance = float(np.linalg.norm(end - start))
    duration = max(model.opt.timestep, distance / speed)
    steps = max(1, ceil(duration / model.opt.timestep))
    for index in range(steps):
        fraction = _smoothstep((index + 1) / steps)
        _set_hand_target(data, start + (end - start) * fraction, yaw)
        if not _step(model, data, metrics, phase=phase, preclose=preclose, hand_yaw=yaw):
            return False
    return True


def _hold_steps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metrics: _Metrics,
    *,
    count: int,
    yaw: float,
    phase: str,
    preclose: bool,
) -> bool:
    for index in range(count):
        if not _step(
            model,
            data,
            metrics,
            phase=phase,
            preclose=preclose,
            hand_yaw=yaw,
            force_trace=index == count - 1,
        ):
            return False
    return True


def _result(
    scenario: ScenarioSpec,
    metrics: _Metrics,
    *,
    start_z: float,
    final_z: float,
    max_lift: float,
    min_hold_lift: float,
    hold_duration: float,
    success: bool,
    failure: str | None,
    detail: str = "",
) -> EpisodeResult:
    return EpisodeResult(
        seed=scenario.seed,
        target_id=scenario.target_id,
        success=success,
        failure_mode=failure,  # type: ignore[arg-type]
        start_z=start_z,
        final_z=final_z,
        max_lift_m=max_lift,
        min_hold_lift_m=min_hold_lift,
        hold_duration_s=hold_duration,
        end_finger_contact=metrics.end_finger_contact,
        preclose_displacement_m=metrics.max_preclose_displacement,
        preclose_occluder_contact=metrics.preclose_occluder_contact,
        max_penetration_m=metrics.max_penetration,
        peak_velocity=metrics.peak_velocity,
        detail=detail,
        trace=metrics.trace,
    )


def _physical_failure(metrics: _Metrics, grasp: Grasp | None) -> tuple[str | None, str]:
    """Return the mandated precedence order for completed/invalid episodes."""
    if metrics.invalid:
        return "instability", "physical threshold exceeded"
    if metrics.timed_out:
        return "timeout", "simulation exceeded 30 seconds"
    if metrics.preclose_occluder_contact:
        return "occluder_collision", "contact with non-target before close"
    if metrics.max_preclose_displacement > 0.03:
        return "knock_over", "target displaced more than .03 m before close"
    if grasp is None:
        return "no_grasp_found", "no finite jaw-feasible candidate"
    return None, ""


def run_episode(
    scene: SceneSpec,
    scenario: ScenarioSpec,
    params: PlannerParams,
    *,
    record: bool = False,
) -> EpisodeResult:
    """Run one reset, settle, approach, close, lift and stationary-hold episode.

    ``record=True`` stores sparse replay samples containing target ``z``, phase,
    simulation time, and commanded mocap hand ``x/y/z/yaw``.
    """
    model = _cached_model(scene, scenario)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0
    _set_hand_target(data, np.array((0.0, 0.0, 0.25), dtype=float), 0.0)
    mujoco.mj_forward(model, data)

    target_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(scenario.target_id))
    target_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name(scenario.target_id))
    finger_geoms = {
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"),
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"),
    }
    hand_geoms = {*finger_geoms, _id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")}
    table_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    obstacle_geoms = {
        index
        for index in range(model.ngeom)
        if index not in hand_geoms and index not in {target_geom, table_geom}
    }
    metrics = _Metrics(
        target_geom=target_geom,
        target_body=target_body,
        hand_geoms=hand_geoms,
        finger_geoms=finger_geoms,
        obstacle_geoms=obstacle_geoms,
        trace_enabled=record,
    )

    # A fixed half-second establishes a physical start state before perception/
    # planning. The hand remains safely above the table throughout settlement.
    _hold_steps(
        model,
        data,
        metrics,
        count=round(SETTLE_SECONDS / model.opt.timestep),
        yaw=0.0,
        phase="settle",
        preclose=True,
    )
    settled = _target_position(data, target_body)
    metrics._settled_xyz = settled.copy()
    start_z = float(settled[2])
    observed = {
        object_spec.id: tuple(
            _target_position(data, _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(object_spec.id)))
        )
        for object_spec in [*scene.objects, *scenario.obstacles]
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(object_spec.id)) >= 0
    }
    grasp = plan_grasp(scene, scenario, params, observed)
    failure, detail = _physical_failure(metrics, grasp)
    if failure is not None:
        current_z = float(_target_position(data, target_body)[2])
        return _result(
            scenario,
            metrics,
            start_z=start_z,
            final_z=current_z,
            max_lift=max(0.0, current_z - start_z),
            min_hold_lift=0.0,
            hold_duration=0.0,
            success=False,
            failure=failure,
            detail=detail,
        )
    assert grasp is not None

    low = np.array((grasp.position[0], grasp.position[1], FINGERTIP_CLEARANCE_M), dtype=float)
    pregrasp = low + np.array((0.0, 0.0, params.pregrasp_height_m), dtype=float)
    initial_hand = np.array(data.mocap_pos[0], dtype=float)
    if not _move(
        model,
        data,
        metrics,
        start=initial_hand,
        end=pregrasp,
        yaw=grasp.yaw,
        speed=params.approach_speed,
        phase="pregrasp",
        preclose=True,
    ) or not _move(
        model,
        data,
        metrics,
        start=pregrasp,
        end=low,
        yaw=grasp.yaw,
        speed=params.approach_speed,
        phase="approach",
        preclose=True,
    ):
        failure, detail = _physical_failure(metrics, grasp)
    else:
        data.ctrl[:] = grasp.force_per_finger_n
        if not _hold_steps(
            model,
            data,
            metrics,
            count=round(CLOSE_SECONDS / model.opt.timestep),
            yaw=grasp.yaw,
            phase="close",
            preclose=False,
        ) or not _move(
            model,
            data,
            metrics,
            start=low,
            end=low + np.array((0.0, 0.0, LIFT_DISTANCE_M), dtype=float),
            yaw=grasp.yaw,
            speed=params.lift_speed,
            phase="lift",
            preclose=False,
        ):
            failure, detail = _physical_failure(metrics, grasp)

    target_now = _target_position(data, target_body)
    max_lift = max(0.0, float(target_now[2] - start_z))
    hold_lifts: list[float] = []
    if failure is None:
        # Command position is constant for precisely 1000 dt=.002 steps.
        data.ctrl[:] = grasp.force_per_finger_n
        hold_count = round(HOLD_SECONDS / model.opt.timestep)
        for index in range(hold_count):
            if not _step(
                model,
                data,
                metrics,
                phase="hold",
                preclose=False,
                hand_yaw=grasp.yaw,
                force_trace=index == hold_count - 1,
            ):
                failure, detail = _physical_failure(metrics, grasp)
                break
            lift = float(_target_position(data, target_body)[2] - start_z)
            hold_lifts.append(lift)
            max_lift = max(max_lift, lift)

    final_z = float(_target_position(data, target_body)[2])
    min_hold_lift = min(hold_lifts) if hold_lifts else 0.0
    held_steps = 0
    for lift in hold_lifts:
        if lift < 0.05:
            break
        held_steps += 1
    scored_success = hold_scored_success(
        min_hold_lift, len(hold_lifts), metrics.end_finger_contact, timestep=model.opt.timestep
    )
    hold_duration = held_steps * model.opt.timestep
    if failure is None:
        if scored_success:
            return _result(
                scenario,
                metrics,
                start_z=start_z,
                final_z=final_z,
                max_lift=max_lift,
                min_hold_lift=min_hold_lift,
                hold_duration=hold_duration,
                success=True,
                failure=None,
            )
        detail = "never_lifted" if max_lift < 0.05 else "hold_lift_or_finger_contact_failed"
        failure = "slip_or_drop"

    # Mandatory error precedence also applies to a partially completed trajectory.
    precedence_failure, precedence_detail = _physical_failure(metrics, grasp)
    if precedence_failure is not None:
        failure, detail = precedence_failure, precedence_detail
    return _result(
        scenario,
        metrics,
        start_z=start_z,
        final_z=final_z,
        max_lift=max_lift,
        min_hold_lift=min_hold_lift,
        hold_duration=hold_duration,
        success=False,
        failure=failure,
        detail=detail,
    )


def _episode_worker(arguments: tuple[SceneSpec, ScenarioSpec, PlannerParams]) -> EpisodeResult:
    scene, scenario, params = arguments
    return run_episode(scene, scenario, params, record=False)


def evaluate(
    scene: SceneSpec,
    scenarios: list[ScenarioSpec],
    params: PlannerParams,
    workers: int = 4,
) -> list[EpisodeResult]:
    """Evaluate scenarios in stable input order using spawned isolated workers."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    jobs = [(scene, scenario, params) for scenario in scenarios]
    if workers == 1 or len(jobs) < 2:
        return [_episode_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        # executor.map preserves submission order irrespective of worker timing.
        return list(executor.map(_episode_worker, jobs))
