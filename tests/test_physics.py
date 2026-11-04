from __future__ import annotations

from splatsam.llm.schemas import ObjectSpec, ScenarioSpec, SceneSpec
from splatsam.planner.grasp import plan_grasp
from splatsam.planner.params import PlannerParams
from splatsam.sim.episode import hold_scored_success, run_episode
from splatsam.sim.scene import handcrafted_scene


def _scenario(
    scene: SceneSpec, target_id: str, *, seed: int = 7, friction_scale: float = 1.0
) -> ScenarioSpec:
    return ScenarioSpec(
        seed=seed,
        target_id=target_id,
        positions={item.id: item.position for item in scene.objects},
        yaws={item.id: item.yaw for item in scene.objects},
        friction_scale=friction_scale,
        mass_scale=1.0,
    )


def _grip_fixture() -> tuple[SceneSpec, ScenarioSpec]:
    target = ObjectSpec(
        id="test_box",
        name="hidden-low-friction block",
        shape="box",
        size=(0.080, 0.040, 0.030),
        position=(0.0, 0.0, 0.015),
        mass=0.100,
        friction=1.00,
    )
    scene = SceneSpec(objects=[target], table_size=(0.6, 0.6), seed=17)
    # The planner intentionally sees nominal mu=1; simulator physics sees .5.
    return scene, _scenario(scene, target.id, seed=17, friction_scale=0.5)


def test_hold_scoring_keeps_inclusive_lift_boundary():
    assert hold_scored_success(0.050, 1000, True)
    assert not hold_scored_success(0.049999, 1000, True)
    assert not hold_scored_success(0.050, 999, True)
    assert not hold_scored_success(0.050, 1000, False)


def test_physical_weak_grip_drops_while_strong_grip_holds():
    scene, scenario = _grip_fixture()
    weak = run_episode(scene, scenario, PlannerParams(grip_safety_factor=0.5))
    strong = run_episode(scene, scenario, PlannerParams(grip_safety_factor=4.0))
    assert not weak.success
    assert weak.failure_mode == "slip_or_drop"
    assert strong.success
    assert strong.end_finger_contact
    assert strong.min_hold_lift_m >= 0.05


def test_same_seed_reinitializes_to_identical_physical_result():
    scene, scenario = _grip_fixture()
    params = PlannerParams(grip_safety_factor=4.0)
    first = run_episode(scene, scenario, params)
    second = run_episode(scene, scenario, params)
    assert first.model_dump(exclude={"trace"}) == second.model_dump(exclude={"trace"})


def test_planner_rejects_objects_outside_finite_jaw_range():
    too_small = ObjectSpec(
        id="too_small",
        name="thin pin",
        shape="box",
        size=(0.008, 0.008, 0.010),
        position=(0.0, 0.0, 0.005),
        mass=0.01,
        friction=0.5,
    )
    too_large = ObjectSpec(
        id="too_large",
        name="wide box",
        shape="box",
        size=(0.150, 0.120, 0.030),
        position=(0.15, 0.0, 0.015),
        mass=0.1,
        friction=0.5,
    )
    scene = SceneSpec(objects=[too_small, too_large])
    for target_id in (too_small.id, too_large.id):
        assert plan_grasp(scene, _scenario(scene, target_id), PlannerParams()) is None


def test_handcrafted_scene_objects_have_plannable_dimensions():
    scene = handcrafted_scene()
    for object_spec in scene.objects:
        grasp = plan_grasp(scene, _scenario(scene, object_spec.id), PlannerParams())
        assert grasp is not None
        assert 0.01 < grasp.jaw_width_m < 0.10
