import pytest

from splatsam.llm.schemas import (
    EpisodeResult,
    ObjectSpec,
    PlannerPatch,
    ScenarioFamily,
    ScenarioSpec,
    SceneSpec,
)
from splatsam.loop import engine
from splatsam.loop.acceptance import acceptance
from splatsam.planner import PlannerParams
from splatsam.utils import load_json


def rows(successes):
    return [
        EpisodeResult(
            seed=i,
            target_id="box",
            success=i < successes,
            failure_mode=None if i < successes else "slip_or_drop",
        )
        for i in range(100)
    ]


def test_paired_gain_and_bootstrap_gates():
    assert not acceptance(rows(50), rows(51))["accepted"]
    assert not acceptance(rows(50), rows(50))["accepted"]
    assert acceptance(rows(50), rows(70))["accepted"]
    with pytest.raises(ValueError):
        acceptance(rows(50), list(reversed(rows(70))))


def test_mode_regression_guard_even_when_total_gain_positive():
    old = rows(40)
    new = rows(70)
    # Two modes protected by >=15 incumbent episodes. Improve total but worsen
    # one by 10pp, which must reject rather than hide regression in aggregate.
    for i in range(40, 60):
        old[i] = old[i].model_copy(update={"failure_mode": "knock_over"})
    for i in range(70, 100):
        new[i] = new[i].model_copy(update={"failure_mode": "knock_over"})
    decision = acceptance(old, new)
    assert not decision["accepted"] and decision["protected_regressions"]["knock_over"] == 0.10


def _scene_and_family():
    scene = SceneSpec(
        objects=[
            ObjectSpec(
                id="box",
                name="box",
                shape="box",
                size=(0.04, 0.04, 0.02),
                position=(0, 0, 0.01),
                mass=0.1,
                friction=0.5,
            )
        ]
    )
    return scene, ScenarioFamily()


def _scenarios(split):
    count, first_seed = (1, 0) if split == "train" else (100, 10_000)
    return [
        ScenarioSpec(
            seed=first_seed + index,
            target_id="box",
            positions={"box": (0, 0, 0.01)},
            yaws={"box": 0},
            friction_scale=1,
            mass_scale=1,
        )
        for index in range(count)
    ]


def _install_deterministic_evaluator(monkeypatch):
    monkeypatch.setattr(engine, "split_scenarios", lambda family, scene, split: _scenarios(split))

    def evaluate(scene, scenarios, params, *, workers):
        succeeds = params.grip_safety_factor >= 1.3
        return [
            EpisodeResult(
                seed=scenario.seed,
                target_id=scenario.target_id,
                success=succeeds,
                failure_mode=None if succeeds else "slip_or_drop",
            )
            for scenario in scenarios
        ]

    monkeypatch.setattr(engine, "evaluate", evaluate)


def _rejected_history(count):
    return [
        {"round": index + 1, "accepted": False, "reason": "preserved rejected fallback"}
        for index in range(count)
    ]


def test_continuation_persists_completed_round_before_unavailable_proposal(monkeypatch, tmp_path):
    _install_deterministic_evaluator(monkeypatch)
    scene, family = _scene_and_family()
    preserved = _rejected_history(6)

    class Proposer:
        calls = 0

        def generate(self, schema, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return PlannerPatch(
                    patch={"grip_safety_factor": 1.3},
                    reason="increase force margin",
                )
            raise engine.ProposalUnavailable("provider returned a fallback proposal")

    result = engine.improve(
        scene,
        family,
        PlannerParams(),
        Proposer(),
        tmp_path,
        workers=1,
        max_rounds=8,
        critic=False,
        prior_history=preserved,
        prior_validation_evaluations=600,
        prior_fresh_validation_simulations=400,
        prior_consecutive_rejections=0,
        max_additional_rounds=5,
    )

    rounds = load_json(tmp_path / "rounds.json")
    summary = load_json(tmp_path / "loop.json")
    assert rounds[:6] == preserved
    assert [row["round"] for row in rounds] == list(range(1, 8))
    assert rounds[-1]["accepted"] is True
    assert len(rounds) <= 8
    assert result == summary
    assert summary["history"] == rounds
    assert summary["final_params"] == load_json(tmp_path / "final_params.json")
    assert summary["final_params"]["grip_safety_factor"] == 1.3
    assert summary["validation_evaluations"] == 800
    assert summary["fresh_validation_simulations"] == 600
    assert summary["status"] == "blocked"
    assert summary["blocked_reason"] == {
        "stage": "proposal",
        "provenance": "unavailable",
        "error_type": "ProposalUnavailable",
        "reason": "provider returned a fallback proposal",
    }


def test_continuation_honors_total_round_cap_with_preserved_history(monkeypatch, tmp_path):
    _install_deterministic_evaluator(monkeypatch)
    scene, family = _scene_and_family()
    preserved = _rejected_history(6)

    class Proposer:
        def generate(self, schema, **kwargs):
            return PlannerPatch(patch={}, reason="no-op proposal")

    result = engine.improve(
        scene,
        family,
        PlannerParams(),
        Proposer(),
        tmp_path,
        workers=1,
        max_rounds=7,
        critic=False,
        prior_history=preserved,
        prior_validation_evaluations=0,
        prior_fresh_validation_simulations=0,
        prior_consecutive_rejections=0,
        max_additional_rounds=5,
    )

    assert [row["round"] for row in result["history"]] == list(range(1, 8))
    assert result["history"][:6] == preserved
    assert result["validation_evaluations"] == 200
    assert result["fresh_validation_simulations"] == 100
    assert result["status"] == "ok"


def test_continuation_stops_after_three_new_rejections(monkeypatch, tmp_path):
    _install_deterministic_evaluator(monkeypatch)
    scene, family = _scene_and_family()
    preserved = _rejected_history(3)

    class Proposer:
        calls = 0

        def generate(self, schema, **kwargs):
            self.calls += 1
            if self.calls > 3:
                raise AssertionError("new rejection stop did not prevent another proposal")
            return PlannerPatch(patch={}, reason="no-op proposal")

    result = engine.improve(
        scene,
        family,
        PlannerParams(),
        Proposer(),
        tmp_path,
        workers=1,
        max_rounds=8,
        critic=False,
        prior_history=preserved,
        prior_validation_evaluations=5,
        prior_fresh_validation_simulations=4,
        prior_consecutive_rejections=0,
        max_additional_rounds=5,
    )

    assert [row["round"] for row in result["history"]] == list(range(1, 7))
    assert result["history"][:3] == preserved
    assert all(row["accepted"] is False for row in result["history"][3:])
    assert result["validation_evaluations"] == 605
    assert result["fresh_validation_simulations"] == 104


def test_critic_render_failure_does_not_block_proposal(monkeypatch, tmp_path):
    _install_deterministic_evaluator(monkeypatch)
    scene, family = _scene_and_family()

    def unavailable_renderer(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr("splatsam.sim.replay.episode_frames", unavailable_renderer)

    class Proposer:
        def generate(self, schema, **kwargs):
            assert schema is PlannerPatch
            assert kwargs["stage"] == "proposal"
            return PlannerPatch(
                patch={"grip_safety_factor": 1.3},
                reason="increase force margin",
            )

    result = engine.improve(
        scene,
        family,
        PlannerParams(),
        Proposer(),
        tmp_path,
        workers=1,
        max_rounds=1,
        critic=True,
    )

    assert result["history"][0]["critic"] == [
        {
            "seed": 0,
            "label": "slip_or_drop",
            "reason": "critic unavailable: renderer RuntimeError",
        }
    ]
