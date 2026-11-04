#!/usr/bin/env python3
"""Export an append-only, offline-only presentation event stream for the dashboard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STAGES = ("understand", "track", "scale", "build", "scenarios", "tune", "sealed")
STAGE_ARTIFACTS = {
    "understand": "objects.json",
    "track": "tracks.json",
    "scale": "reference.json",
    "build": "scene.json",
    "scenarios": "family.json",
    "tune": "rounds.json",
    "sealed": "final_test.json",
}
STAGE_SCHEDULE = {
    "understand": (0.0, 4.0),
    "track": (4.5, 8.5),
    "scale": (9.0, 13.0),
    "build": (13.5, 17.0),
    "scenarios": (17.5, 21.0),
    "tune": (21.5, 53.0),
    "sealed": (53.5, 75.0),
}
PRESENTATION_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _safe_relative(path: str) -> bool:
    candidate = Path(path)
    return bool(path) and not candidate.is_absolute() and ".." not in candidate.parts


def _safe_source(root: Path, relative: str) -> Path | None:
    if not _safe_relative(relative):
        return None
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _load_json(root: Path, relative: str) -> Any | None:
    path = _safe_source(root, relative)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_jsonl(root: Path, relative: str) -> list[dict[str, Any]]:
    path = _safe_source(root, relative)
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, UnicodeDecodeError):
        return []
    return rows


class _BuiltinOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"disallowed pickle global: {module}.{name}")


def _cached_outcomes(root: Path) -> list[dict[str, Any]]:
    path = _safe_source(root, "episode_cache/cache.db")
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        values = connection.execute("SELECT key, value FROM Cache WHERE mode = 4 ORDER BY rowid")
        for cache_key, raw_value in values:
            if not isinstance(cache_key, str) or not isinstance(raw_value, bytes):
                continue
            try:
                outcome = _BuiltinOnlyUnpickler(io.BytesIO(raw_value)).load()
            except (EOFError, pickle.UnpicklingError):
                continue
            if not (
                isinstance(outcome, dict)
                and isinstance(outcome.get("seed"), int)
                and not isinstance(outcome.get("seed"), bool)
                and isinstance(outcome.get("success"), bool)
                and (isinstance(outcome.get("failure_mode"), str) or outcome.get("failure_mode") is None)
            ):
                continue
            payload = copy.deepcopy(outcome)
            payload.pop("trace", None)
            payload.update(
                {
                    "phase": "tune",
                    "source": "episode_cache",
                    "variant": "recorded_evaluation",
                    "round_association": "unavailable",
                    "cache_key": cache_key,
                    "artifact_path": "episode_cache/cache.db",
                }
            )
            rows.append(payload)
    except (OSError, sqlite3.DatabaseError):
        return []
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    return rows


def _timestamp(seconds: float) -> str:
    value = PRESENTATION_EPOCH + timedelta(seconds=seconds)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _event(event_type: str, seconds: float, payload: dict[str, Any]) -> dict[str, Any]:
    timestamp = _timestamp(seconds)
    identity = json.dumps(
        {"timestamp": timestamp, "type": event_type, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return {
        "event_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "timestamp": timestamp,
        "type": event_type,
        "payload": payload,
    }


def _stage_events(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    captions = {
        "understand": "Reading the recorded synthetic scene snapshot",
        "track": "Replaying persistent-object tracking evidence",
        "scale": "Applying the recorded reference-scale result",
        "build": "Building the recorded primitive scene",
        "scenarios": "Loading the frozen synthetic scenario family",
        "tune": "Replaying validation-gated planner rounds",
        "sealed": "Replaying the sealed synthetic final measurement",
    }
    for stage in STAGES:
        artifact_path = STAGE_ARTIFACTS[stage]
        start, finish = STAGE_SCHEDULE[stage]
        available = _safe_source(root, artifact_path) is not None
        if not available:
            events.append(
                _event(
                    "stage_finished",
                    finish,
                    {
                        "stage": stage,
                        "artifact_path": artifact_path,
                        "available": False,
                        "status": "absent",
                        "caption": f"{stage} artifact is absent from this replay snapshot",
                    },
                )
            )
            continue
        events.extend(
            (
                _event(
                    "stage_started",
                    start,
                    {
                        "stage": stage,
                        "artifact_path": artifact_path,
                        "caption": captions[stage],
                        "timeline_provenance": "presentation_paced_from_snapshot",
                    },
                ),
                _event(
                    "stage_finished",
                    finish,
                    {
                        "stage": stage,
                        "artifact_path": artifact_path,
                        "available": True,
                        "status": "done",
                        "caption": captions[stage],
                        "timeline_provenance": "presentation_paced_from_snapshot",
                    },
                ),
            )
        )
    return events


def _object_events(root: Path) -> list[dict[str, Any]]:
    snapshot = _load_json(root, "objects.json")
    objects = snapshot.get("objects") if isinstance(snapshot, dict) else None
    if not isinstance(objects, list):
        return []
    events: list[dict[str, Any]] = []
    count = len(objects)
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            continue
        payload = copy.deepcopy(item)
        payload["artifact_path"] = "objects.json"
        seconds = 0.75 + ((index + 1) / (count + 1)) * 2.5
        events.append(_event("object_detected", seconds, payload))
    return events


def _scene_event(root: Path) -> list[dict[str, Any]]:
    scene = _load_json(root, "scene.json")
    if not isinstance(scene, dict):
        return []
    payload = copy.deepcopy(scene)
    payload["artifact_path"] = "scene.json"
    return [_event("scene_built", 16.5, payload)]


def _round_events(root: Path) -> list[dict[str, Any]]:
    rounds = _load_json(root, "rounds.json")
    if not isinstance(rounds, list):
        return []
    events: list[dict[str, Any]] = []
    count = len(rounds)
    for index, row in enumerate(rounds):
        if not isinstance(row, dict):
            continue
        payload = copy.deepcopy(row)
        payload["artifact_path"] = "rounds.json"
        seconds = 24.0 + ((index + 1) / (count + 1)) * 26.0
        events.append(_event("round_finished", seconds, payload))
    return events


def _llm_events(root: Path) -> list[dict[str, Any]]:
    calls = _load_jsonl(root, "trajectory.jsonl")
    indexed = [(index, row) for index, row in enumerate(calls) if isinstance(row.get("model"), str)]
    family = [(index, row) for index, row in indexed if row.get("stage") == "family"]
    other = [(index, row) for index, row in indexed if row.get("stage") != "family"]
    events: list[dict[str, Any]] = []
    for group, start, span in ((family, 18.0, 2.0), (other, 22.0, 28.0)):
        for position, (_, row) in enumerate(group):
            payload = {
                "model": row.get("model"),
                "stage": row.get("stage"),
                "usage": copy.deepcopy(row.get("usage")),
                "cost_usd": copy.deepcopy(row.get("cost_usd")),
                "provenance": row.get("provenance"),
                "artifact_path": "trajectory.jsonl",
            }
            seconds = start + ((position + 1) / (len(group) + 1)) * span
            events.append(_event("llm_call", seconds, payload))
    return events




def _tune_episode_events(root: Path) -> list[dict[str, Any]]:
    outcomes = _cached_outcomes(root)
    events: list[dict[str, Any]] = []
    for index, payload in enumerate(outcomes):
        seconds = 22.25 + ((index + 1) / (len(outcomes) + 1)) * 29.75
        events.append(_event("episode_finished", seconds, payload))
    return events


def _final_events(root: Path) -> list[dict[str, Any]]:
    final_test = _load_json(root, "final_test.json")
    if not isinstance(final_test, dict):
        return []
    events: list[dict[str, Any]] = []
    episode_rows: list[tuple[str, dict[str, Any]]] = []
    for phase in ("baseline", "final"):
        block = final_test.get(phase)
        results = block.get("results") if isinstance(block, dict) else None
        if not isinstance(results, list):
            continue
        for result in results:
            if isinstance(result, dict):
                payload = copy.deepcopy(result)
                payload.pop("trace", None)
                payload.update(
                    {
                        "phase": phase,
                        "source": "final_test.json",
                        "variant": phase,
                        "artifact_path": "final_test.json",
                    }
                )
                episode_rows.append((phase, payload))
    for index, (_, payload) in enumerate(episode_rows):
        seconds = 55.0 + ((index + 1) / (len(episode_rows) + 1)) * 14.0
        events.append(_event("episode_finished", seconds, payload))

    comparison = final_test.get("comparison")
    seal_hash = final_test.get("seal_hash")
    if isinstance(comparison, dict) and isinstance(seal_hash, str):
        report = _load_json(root, "report.json")
        costs = report.get("costs") if isinstance(report, dict) and isinstance(report.get("costs"), dict) else None
        payload: dict[str, Any] = {
            "comparison": copy.deepcopy(comparison),
            "seal_hash": seal_hash,
            "artifact_path": "final_test.json",
        }
        if costs is not None:
            payload["cost_usd"] = copy.deepcopy(costs)
        events.append(_event("final_test_finished", 70.0, payload))
    return events


def _presentation_manifest(run_dir: Path, destination: Path) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for stage, relative in STAGE_ARTIFACTS.items():
        artifacts[stage] = {
            "artifact_path": relative,
            "available_in_source": _safe_source(run_dir, relative) is not None,
            "available_in_bundle": _safe_source(destination, relative) is not None,
        }
    return {
        "schema_version": 1,
        "origin": "synthetic",
        "provenance": "snapshot-derived replay; no original stage timing was recorded",
        "timestamp_basis": "presentation_paced_from_snapshot",
        "replay_duration_s": 75,
        "events_file": "events.jsonl",
        "source_run": run_dir.name,
        "artifacts": artifacts,
    }


def export_events(run_dir: Path, destination: Path) -> int:
    run_dir = run_dir.resolve()
    destination = destination.resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    destination.mkdir(parents=True, exist_ok=True)

    generated = (
        _stage_events(run_dir)
        + _object_events(run_dir)
        + _scene_event(run_dir)
        + _llm_events(run_dir)
        + _tune_episode_events(run_dir)
        + _round_events(run_dir)
        + _final_events(run_dir)
    )
    generated.sort(key=lambda event: (event["timestamp"], event["event_id"]))

    events_path = destination / "events.jsonl"
    existing_ids: set[str] = set()
    if events_path.exists():
        try:
            with events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and isinstance(row.get("event_id"), str):
                        existing_ids.add(row["event_id"])
        except (OSError, UnicodeDecodeError):
            pass
    additions = [event for event in generated if event["event_id"] not in existing_ids]
    if additions:
        with events_path.open("a", encoding="utf-8") as handle:
            for event in additions:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
                handle.write("\n")

    manifest_path = destination / "presentation.json"
    manifest = _presentation_manifest(run_dir, destination)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != encoded:
        manifest_path.write_text(encoded, encoding="utf-8")
    return len(additions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/synth-demo"))
    parser.add_argument("--destination", type=Path, default=Path("demo_data"))
    args = parser.parse_args()
    additions = export_events(args.run_dir, args.destination)
    print(f"Appended {additions} presentation events; no pipeline, API, or physics execution occurred.")


if __name__ == "__main__":
    main()
