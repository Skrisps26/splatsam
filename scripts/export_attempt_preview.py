#!/usr/bin/env python3
"""Export recorded, preview-only fixed-planner attempts for a saved live run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Supports both ``python -m scripts...`` and direct script execution.
    from scripts.export_ui_events import _event
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI execution.
    from export_ui_events import _event

from splatsam.curriculum.sampler import TRAIN_SEEDS, sample_scenario
from splatsam.llm.schemas import ScenarioFamily, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.sim.episode import evaluate
from splatsam.utils import ROOT, file_digest, load_json, save_json

_DESCRIPTION = (
    "Recorded replay of fixed-planner attempts from the saved live scene; "
    "preview-only and excluded from evaluation."
)
_DESTINATION_NAME = "attempt_preview"


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"path is outside {root}: {resolved}") from error
    return resolved


def _live_run(run_dir: Path) -> Path:
    runs = (ROOT / "runs").resolve(strict=True)
    resolved = _inside(run_dir, runs)
    if not resolved.is_dir():
        raise ValueError(f"run directory does not exist: {resolved}")
    input_artifact = load_json(resolved / "input.json")
    if not isinstance(input_artifact, dict) or input_artifact.get("source") != "live":
        raise ValueError("attempt preview requires input.json source='live'")
    return resolved


def _validate_count(count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(TRAIN_SEEDS):
        raise ValueError("count must be an integer in [1, 150]")
    return count


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return workers


def _input_hashes(run_dir: Path) -> dict[str, str]:
    names = ("scene.json", "family.json", "baseline_params.json")
    return {name: file_digest(run_dir / name) for name in names}


def _manifest(count: int, source_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "preview_only": True,
        "count": count,
        "source_hashes": source_hashes,
        "description": _DESCRIPTION,
    }


def _event_seconds(index: int, count: int) -> float:
    if count == 1:
        return 22.0
    return 22.0 + (41.0 * index / (count - 1))


def _events(results: list[Any]) -> list[dict[str, Any]]:
    count = len(results)
    events = [
        _event(
            "stage_started",
            21.5,
            {
                "stage": "tune",
                "caption": "Recorded fixed-planner attempts",
                "preview_only": True,
                "excluded_from_evaluation": True,
            },
        )
    ]
    for index, result in enumerate(results):
        events.append(
            _event(
                "episode_finished",
                _event_seconds(index, count),
                {
                    "seed": result.seed,
                    "success": result.success,
                    "failure_mode": result.failure_mode,
                    "phase": "preview",
                    "preview_only": True,
                    "excluded_from_evaluation": True,
                },
            )
        )
    events.append(
        _event(
            "stage_finished",
            64.0,
            {
                "stage": "tune",
                "available": True,
                "status": "done",
                "caption": "Recorded fixed-planner attempts",
                "preview_only": True,
                "excluded_from_evaluation": True,
            },
        )
    )
    return events


def _read_events(path: Path) -> list[dict[str, Any]] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def _cached_summary(events_path: Path, count: int) -> dict[str, int] | None:
    rows = _read_events(events_path)
    if rows is None or len(rows) != count + 2:
        return None
    started, *episodes, finished = rows
    started_payload = started.get("payload")
    finished_payload = finished.get("payload")
    if (
        started.get("type") != "stage_started"
        or not isinstance(started_payload, dict)
        or started_payload.get("stage") != "tune"
        or started_payload.get("preview_only") is not True
        or started_payload.get("excluded_from_evaluation") is not True
        or finished.get("type") != "stage_finished"
        or not isinstance(finished_payload, dict)
        or finished_payload.get("stage") != "tune"
        or finished_payload.get("available") is not True
        or finished_payload.get("status") != "done"
        or finished_payload.get("preview_only") is not True
        or finished_payload.get("excluded_from_evaluation") is not True
    ):
        return None
    successes = 0
    for seed, row in enumerate(episodes):
        payload = row.get("payload")
        if (
            row.get("type") != "episode_finished"
            or not isinstance(payload, dict)
            or payload.get("seed") != seed
            or not isinstance(payload.get("success"), bool)
            or payload.get("failure_mode") is not None and not isinstance(payload.get("failure_mode"), str)
            or payload.get("phase") != "preview"
            or payload.get("preview_only") is not True
            or payload.get("excluded_from_evaluation") is not True
        ):
            return None
        successes += int(payload["success"])
    return {"successes": successes, "failures": count - successes}


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    encoded = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for event in events
    )
    path.write_text(encoded, encoding="utf-8")


def export_attempt_preview(run_dir: Path, *, count: int = 64, workers: int = 4) -> dict[str, int | bool]:
    """Evaluate public train seeds and export a cached, preview-only event replay."""
    count = _validate_count(count)
    workers = _validate_workers(workers)
    run_dir = _live_run(run_dir)
    destination = run_dir / "presentation" / _DESTINATION_NAME
    source_hashes = _input_hashes(run_dir)
    manifest = _manifest(count, source_hashes)
    manifest_path = destination / "manifest.json"
    events_path = destination / "events.jsonl"

    if destination.is_dir() and manifest_path.is_file():
        try:
            cached_manifest = load_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            cached_manifest = None
        if cached_manifest == manifest:
            cached = _cached_summary(events_path, count)
            if cached is not None:
                return {"count": count, **cached, "cached": True}

    scene = SceneSpec.model_validate(load_json(run_dir / "scene.json"))
    if scene.source != "live":
        raise ValueError("attempt preview requires a live scene, not a relabeled example")
    family = ScenarioFamily.model_validate(load_json(run_dir / "family.json"))
    params = PlannerParams.load(run_dir / "baseline_params.json")
    seeds = list(range(count))
    assert all(seed in TRAIN_SEEDS for seed in seeds), "attempt preview seeds must remain in the public train split"
    scenarios = [sample_scenario(family, seed, scene) for seed in seeds]
    results = evaluate(scene, scenarios, params, workers=workers)
    if len(results) != count or [result.seed for result in results] != seeds:
        raise RuntimeError("episode evaluator did not return one result per preview seed in input order")

    destination.mkdir(parents=True, exist_ok=True)
    _write_events(events_path, _events(results))
    save_json(manifest_path, manifest)
    successes = sum(result.success for result in results)
    return {"count": count, "successes": successes, "failures": count - successes, "cached": False}


def _cli_count(value: str) -> int:
    try:
        return _validate_count(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _cli_workers(value: str) -> int:
    try:
        return _validate_workers(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="private live run below runs/")
    parser.add_argument("--count", type=_cli_count, default=64, help="public train attempts to replay (1..150)")
    parser.add_argument("--workers", type=_cli_workers, default=4, help="CPU evaluator worker count")
    args = parser.parse_args()
    try:
        summary = export_attempt_preview(args.run_dir, count=args.count, workers=args.workers)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    cache_status = "cached" if summary["cached"] else "evaluated"
    print(
        f"{cache_status}: {summary['count']} preview attempts; "
        f"{summary['successes']} succeeded, {summary['failures']} failed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
