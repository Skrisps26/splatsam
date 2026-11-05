from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import export_attempt_preview as preview
from splatsam.llm.schemas import EpisodeResult


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _live_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "saved-live"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "input.json", {"source": "live"})
    _write_json(
        run_dir / "scene.json",
        {
            "objects": [
                {
                    "id": "target",
                    "name": "target",
                    "shape": "box",
                    "size": [0.04, 0.04, 0.04],
                    "position": [0.0, 0.0, 0.02],
                    "mass": 0.1,
                    "friction": 0.5,
                }
            ],
            "table_size": [1.0, 1.0],
            "source": "live",
        },
    )
    _write_json(run_dir / "family.json", {})
    _write_json(
        run_dir / "baseline_params.json",
        {
            "grip_safety_factor": 1.2,
            "jaw_margin_m": 0.004,
            "pregrasp_height_m": 0.1,
            "approach_speed": 0.1,
            "lift_speed": 0.1,
            "n_yaw_candidates": 5,
            "clearance_weight": 1.0,
            "center_weight": 1.0,
        },
    )
    return run_dir


def test_preview_cache_invalidates_when_saved_planner_changes(monkeypatch, tmp_path: Path) -> None:
    run_dir = _live_run(tmp_path)
    evaluations = []

    def evaluate(scene, scenarios, params, *, workers):
        evaluations.append(params.hash)
        return [
            EpisodeResult(
                seed=scenario.seed,
                target_id=scenario.target_id,
                success=scenario.seed == 0,
                failure_mode=None if scenario.seed == 0 else "no_grasp_found",
            )
            for scenario in scenarios
        ]

    monkeypatch.setattr(preview, "ROOT", tmp_path)
    monkeypatch.setattr(preview, "evaluate", evaluate)
    protected = run_dir / "report.json"
    protected.write_text('{"held_out":null}', encoding="utf-8")

    assert preview.export_attempt_preview(run_dir, count=2)["cached"] is False
    assert preview.export_attempt_preview(run_dir, count=2)["cached"] is True
    assert len(evaluations) == 1

    params_path = run_dir / "baseline_params.json"
    params = json.loads(params_path.read_text())
    params["grip_safety_factor"] = 1.8
    _write_json(params_path, params)
    assert preview.export_attempt_preview(run_dir, count=2)["cached"] is False
    assert len(evaluations) == 2
    assert evaluations[0] != evaluations[1]
    assert protected.read_text() == '{"held_out":null}'


def test_export_attempt_preview_rejects_count_outside_public_train_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="count"):
        preview.export_attempt_preview(_live_run(tmp_path), count=151)


def test_preview_rejects_sample_scene_with_live_input_label(monkeypatch, tmp_path: Path) -> None:
    run_dir = _live_run(tmp_path)
    scene_path = run_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["source"] = "synthetic"
    _write_json(scene_path, scene)
    monkeypatch.setattr(preview, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="live scene"):
        preview.export_attempt_preview(run_dir, count=2)
