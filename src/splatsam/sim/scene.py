"""Native MuJoCo scenes for the floating parallel-jaw evaluation hand."""
from __future__ import annotations

from math import cos, sin, sqrt
import re
from typing import Iterable

# Rendering config must be applied before MuJoCo is imported anywhere in this process.
from splatsam import rendering as _rendering  # noqa: F401
from splatsam.llm.schemas import ObjectSpec, ScenarioSpec, SceneSpec

import mujoco


DT = 0.002
TABLE_TOP_Z = 0.0
FINGERTIP_CLEARANCE_M = 0.003
OPEN_GAP_M = 0.10
CLOSED_GAP_M = 0.01


def _safe_id(identifier: str) -> str:
    """Turn a stable external identifier into a predictable XML name suffix."""
    suffix = re.sub(r"[^A-Za-z0-9_]", "_", identifier)
    if not suffix:
        raise ValueError("object id must contain an XML-name character")
    return suffix


def body_name(identifier: str) -> str:
    return f"object_{_safe_id(identifier)}"


def geom_name(identifier: str) -> str:
    return f"geom_{_safe_id(identifier)}"


def _yaw_quat(yaw: float) -> str:
    return f"{cos(yaw / 2):.12g} 0 0 {sin(yaw / 2):.12g}"


def _number_triplet(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _geom_xml(spec: ObjectSpec, *, mass: float, friction: float) -> str:
    """Emit one geom in the object's local frame using full external dimensions."""
    if any(value <= 0 for value in spec.size):
        raise ValueError(f"{spec.id}: non-positive primitive size")
    if mass <= 0 or friction <= 0:
        raise ValueError(f"{spec.id}: non-positive physical property")

    shape = spec.shape
    if shape == "box":
        geom_type = "box"
        size = tuple(value / 2 for value in spec.size)
        extra = ""
    elif shape == "sphere":
        geom_type = "sphere"
        size = (spec.size[0] / 2,)
        extra = ""
    elif shape == "cylinder":
        geom_type = "cylinder"
        size = (spec.size[0] / 2, spec.size[2] / 2)
        extra = ""
    elif shape == "horizontal_cylinder":
        # MuJoCo cylinders are local-z. Rotate local z to local x before body yaw.
        geom_type = "cylinder"
        size = (spec.size[1] / 2, spec.size[0] / 2)
        root_half = sqrt(0.5)
        extra = f' quat="{root_half:.12g} 0 {root_half:.12g} 0"'
    else:
        raise ValueError(f"{spec.id}: mesh scenes require a collision primitive")

    rgba = _number_triplet(spec.rgba)
    return (
        f'<geom name="{geom_name(spec.id)}" type="{geom_type}" size="{_number_triplet(size)}" '
        f'mass="{mass:.12g}" friction="{friction:.12g} .005 .0001" rgba="{rgba}"{extra}/>'
    )


def _object_xml(spec: ObjectSpec, scenario: ScenarioSpec | None) -> str:
    position = spec.position
    yaw = spec.yaw
    mass_scale = 1.0
    friction_scale = 1.0
    if scenario is not None:
        position = scenario.positions.get(spec.id, position)
        yaw = scenario.yaws.get(spec.id, yaw)
        mass_scale = scenario.mass_scale
        friction_scale = scenario.friction_scale
    if len(position) != 3:
        raise ValueError(f"{spec.id}: position must be xyz")

    # Non-graspable paper/obstacles are fixtures. Graspable scene objects are free.
    fixed = not spec.graspable
    joint = "" if fixed else "<freejoint/>"
    return (
        f'<body name="{body_name(spec.id)}" pos="{_number_triplet(position)}" quat="{_yaw_quat(yaw)}">'
        f"{joint}{_geom_xml(spec, mass=spec.mass * mass_scale, friction=spec.friction * friction_scale)}"
        "</body>"
    )


def _hand_xml() -> str:
    """A free hand welded to a mocap body; fingers remain real slide-joint bodies."""
    return """
    <body name="hand_target" mocap="true" pos="0 0 .25" quat="1 0 0 0"/>
    <body name="hand" pos="0 0 .25" quat="1 0 0 0">
      <freejoint/>
      <geom name="palm" type="box" pos="0 0 .085" size=".035 .08 .012" mass=".20" rgba=".18 .18 .18 1"/>
      <body name="left_finger" pos="0 .06 .035">
        <joint name="left_slide" type="slide" axis="0 -1 0" limited="true" range="0 .045" damping="1"/>
        <geom name="left_finger_geom" type="box" size=".014 .01 .035" mass=".03" friction="1 .005 .0001" rgba=".12 .12 .12 1"/>
      </body>
      <body name="right_finger" pos="0 -.06 .035">
        <joint name="right_slide" type="slide" axis="0 1 0" limited="true" range="0 .045" damping="1"/>
        <geom name="right_finger_geom" type="box" size=".014 .01 .035" mass=".03" friction="1 .005 .0001" rgba=".12 .12 .12 1"/>
      </body>
    </body>
    """


def build_model(scene: SceneSpec, scenario: ScenarioSpec | None = None) -> mujoco.MjModel:
    """Build a complete CPU-only model for *scene* and hidden scenario physics.

    Object dimensions in ``SceneSpec`` remain full metres. Scenario mass/friction
    multipliers are intentionally applied only here, never exposed to the planner.
    """
    specs = list(scene.objects)
    if scenario is not None:
        existing = {item.id for item in specs}
        for obstacle in scenario.obstacles:
            if obstacle.id in existing:
                raise ValueError(f"duplicate scene/scenario object id {obstacle.id!r}")
            specs.append(obstacle.model_copy(update={"graspable": False}))
    ids = [item.id for item in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("scene has duplicate object ids")

    table_x, table_y = scene.table_size
    if table_x <= 0 or table_y <= 0:
        raise ValueError("table dimensions must be positive")
    objects_xml = "\n".join(_object_xml(item, scenario) for item in specs)
    xml = f"""
<mujoco model="splatsam_parallel_jaw">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="Euler"/>
  <size njmax="2000" nconmax="1000"/>
  <default>
    <geom condim="3" solref=".003 1" solimp=".95 .99 .001"/>
    <joint limited="true"/>
  </default>
  <worldbody>
    <geom name="table" type="box" pos="0 0 -.02" size="{table_x / 2:.12g} {table_y / 2:.12g} .02" friction="1 .005 .0001" rgba=".35 .27 .18 1"/>
    {_hand_xml()}
    {objects_xml}
  </worldbody>
  <equality>
    <weld name="hand_weld" body1="hand_target" body2="hand" solref=".002 1" solimp=".99 .999 .001"/>
  </equality>
  <actuator>
    <motor name="left_motor" joint="left_slide" gear="1" ctrllimited="true" ctrlrange="0 30" forcelimited="true" forcerange="0 30"/>
    <motor name="right_motor" joint="right_slide" gear="1" ctrllimited="true" ctrlrange="0 30" forcelimited="true" forcerange="0 30"/>
  </actuator>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def handcrafted_scene(seed: int = 42) -> SceneSpec:
    """Return the compact three-object fixture used by physics and pipeline smoke runs."""
    return SceneSpec(
        seed=seed,
        source="synthetic",
        table_size=(1.0, 1.0),
        objects=[
            ObjectSpec(
                id="remote_box",
                name="small plastic remote",
                shape="box",
                size=(0.100, 0.035, 0.025),
                position=(-0.16, -0.08, 0.0125),
                yaw=0.12,
                mass=0.055,
                friction=0.50,
                material="plastic",
                rgba=(0.18, 0.30, 0.70, 1.0),
            ),
            ObjectSpec(
                id="rubber_ball",
                name="rubber ball",
                shape="sphere",
                size=(0.040, 0.040, 0.040),
                position=(0.10, -0.08, 0.020),
                mass=0.038,
                friction=1.00,
                material="rubber",
                rgba=(0.85, 0.20, 0.16, 1.0),
            ),
            ObjectSpec(
                id="plastic_marker",
                name="thin horizontal marker",
                shape="horizontal_cylinder",
                size=(0.100, 0.016, 0.016),
                position=(0.00, 0.15, 0.008),
                yaw=-0.35,
                mass=0.018,
                friction=0.50,
                material="plastic",
                rgba=(0.95, 0.72, 0.10, 1.0),
            ),
        ],
    )
