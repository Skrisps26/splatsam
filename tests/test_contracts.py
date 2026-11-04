import pytest
from pydantic import ValidationError
from splatsam.planner import PlannerParams
from splatsam.llm.schemas import ScenarioFamily

def test_params_roundtrip_and_bounds(tmp_path):
    p=PlannerParams(); path=tmp_path/'params.json'; p.save(path)
    assert PlannerParams.load(path)==p
    assert p.patch({'grip_safety_factor':2}).hash != p.hash
    for patch in [{'n_yaw_candidates':True},{'n_yaw_candidates':2.2},{'jaw_margin_m':-.1},{'lift_speed':float('nan')},{'force':10}]:
        with pytest.raises((ValueError,TypeError)):
            p.patch(patch)

def test_family_enforces_simulation_domain():
    for kw in [{'friction_scale':(.4,1)}, {'mass_scale':(2,1)}, {'jitter_m':.041}, {'clutter_count':(0,6)}]:
        with pytest.raises(ValidationError):
            ScenarioFamily(**kw)
