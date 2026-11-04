from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from splatsam.curriculum.sampler import split_scenarios
from splatsam.eval.seal import SealError, run_final, seal_test
from splatsam.eval.statistics import compare, summarize
from splatsam.llm.schemas import EpisodeResult, ObjectSpec, ScenarioFamily, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.utils import load_json, save_json


def _scene() -> SceneSpec:
    return SceneSpec(
        objects=[
            ObjectSpec(
                id="target",
                name="target",
                shape="box",
                size=(0.08, 0.04, 0.03),
                position=(-0.12, 0, 0.015),
                mass=0.1,
                friction=0.5,
            ),
            ObjectSpec(
                id="other",
                name="other",
                shape="sphere",
                size=(0.04, 0.04, 0.04),
                position=(0.12, 0, 0.02),
                mass=0.03,
                friction=0.8,
            ),
        ]
    )


def _result(seed: int, success: bool) -> EpisodeResult:
    return EpisodeResult(
        seed=seed, target_id="target", success=success, failure_mode=None if success else "slip_or_drop"
    )


def test_public_splits_are_disjoint_and_holdout_is_private():
    scene, family = _scene(), ScenarioFamily()
    train = split_scenarios(family, scene, "train")
    validation = split_scenarios(family, scene, "validation")
    assert [item.seed for item in train] == list(range(150))
    assert [item.seed for item in validation] == list(range(10_000, 10_100))
    assert not {item.seed for item in train} & {item.seed for item in validation}
    with pytest.raises(ValueError, match="train and validation"):
        split_scenarios(family, scene, "test")


def test_seal_detects_tampered_frozen_specs(tmp_path):
    scene, family, baseline = _scene(), ScenarioFamily(), PlannerParams()
    root = tmp_path / "synthetic"
    manifest = seal_test(scene, family, baseline, root)
    assert manifest["scenario_count"] == 200
    frozen = root / "splits" / "test.json"
    payload = load_json(frozen)
    payload["scenarios"][0]["mass_scale"] = 2.0
    save_json(frozen, payload)
    with pytest.raises(SealError, match="altered"):
        seal_test(scene, family, baseline, root)


def test_started_and_existing_result_refuse_final_test(tmp_path):
    scene, family, baseline = _scene(), ScenarioFamily(), PlannerParams()
    root = tmp_path / "synthetic"
    seal_test(scene, family, baseline, root)
    started = root / "results" / "FINAL_TEST.started"
    started.parent.mkdir(parents=True)
    started.write_text("already claimed\n")
    with pytest.raises(SealError, match="already started"):
        run_final(scene, family, baseline, baseline, root, tmp_path / "run", workers=1)

    started.unlink()
    save_json(root / "results" / "final_test.json", {"existing": True})
    with pytest.raises(SealError, match="already exists"):
        run_final(scene, family, baseline, baseline, root, tmp_path / "run", workers=1)


def test_final_runs_paired_evaluations_once_with_fake_evaluator(tmp_path, monkeypatch):
    scene, family, baseline = _scene(), ScenarioFamily(), PlannerParams()
    final = baseline.patch({"grip_safety_factor": 1.3})
    root = tmp_path / "synthetic"
    seal_test(scene, family, baseline, root)
    calls: list[str] = []

    episode = types.ModuleType("splatsam.sim.episode")
    package = types.ModuleType("splatsam.sim")
    package.__path__ = []

    def evaluate(_scene, scenarios, params, workers=4):
        calls.append(params.hash)
        success = params == final
        return [
            EpisodeResult(
                seed=spec.seed,
                target_id=spec.target_id,
                success=success,
                failure_mode=None if success else "slip_or_drop",
            )
            for spec in scenarios
        ]

    episode.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "splatsam.sim", package)
    monkeypatch.setitem(sys.modules, "splatsam.sim.episode", episode)
    outcome = run_final(scene, family, baseline, final, root, tmp_path / "run", workers=1)
    assert calls == [baseline.hash, final.hash]
    assert outcome["comparison"]["gain"] == 1.0
    assert (root / "results" / "final_test.json").is_file()
    with pytest.raises(SealError, match="already exists"):
        run_final(scene, family, baseline, final, root, tmp_path / "run", workers=1)


def test_concurrent_final_calls_allow_only_one_claim(tmp_path, monkeypatch):
    scene, family, baseline = _scene(), ScenarioFamily(), PlannerParams()
    root = tmp_path / "synthetic"
    seal_test(scene, family, baseline, root)
    entered, release = threading.Event(), threading.Event()
    episode = types.ModuleType("splatsam.sim.episode")
    package = types.ModuleType("splatsam.sim")
    package.__path__ = []

    def evaluate(_scene, scenarios, params, workers=4):
        entered.set()
        release.wait(timeout=2)
        return [
            EpisodeResult(
                seed=spec.seed, target_id=spec.target_id, success=False, failure_mode="slip_or_drop"
            )
            for spec in scenarios
        ]

    episode.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "splatsam.sim", package)
    monkeypatch.setitem(sys.modules, "splatsam.sim.episode", episode)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(run_final, scene, family, baseline, baseline, root, tmp_path / "run", 1)
        assert entered.wait(timeout=2)
        with pytest.raises(SealError, match="already started"):
            run_final(scene, family, baseline, baseline, root, tmp_path / "run", workers=1)
        release.set()
        first.result()


def test_wilson_summary_and_paired_statistics():
    baseline = [_result(0, False), _result(1, False), _result(2, True), _result(3, False)]
    final = [_result(0, True), _result(1, False), _result(2, True), _result(3, True)]
    summary = summarize(baseline)
    assert summary["n"] == 4
    assert summary["successes"] == 1
    assert summary["failure_modes"]["slip_or_drop"] == 3
    assert summary["ci95"][0] < 0.25 < summary["ci95"][1]
    paired = compare(baseline, final)
    assert paired["gain"] == 0.5
    assert paired["gain_ci95"][0] <= paired["gain"] <= paired["gain_ci95"][1]


def test_paired_identical_difference_has_exact_interval():
    baseline = [_result(0, False), _result(1, False)]
    final = [_result(0, True), _result(1, True)]
    assert compare(baseline, final)["gain_ci95"] == [1.0, 1.0]
