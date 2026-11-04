"""Offline event-log projection for the dashboard replay controls."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

STAGE_KEYS = ("understand", "track", "scale", "build", "scenarios", "tune", "sealed")
_EVENT_TYPES = {
    "stage_started",
    "stage_finished",
    "object_detected",
    "scene_built",
    "episode_finished",
    "round_finished",
    "llm_call",
    "final_test_finished",
}


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _safe_payload(value: object, key: str | None = None) -> bool:
    if key is not None and (key == "path" or key.endswith("_path")):
        return value is None or _safe_relative_path(value)
    if isinstance(value, dict):
        return all(isinstance(child_key, str) and _safe_payload(child, child_key) for child_key, child in value.items())
    if isinstance(value, list):
        return all(_safe_payload(child) for child in value)
    return value is None or isinstance(value, (str, int, float, bool))


def _valid_event(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("event_id"), str) or not value["event_id"]:
        return False
    if _parse_timestamp(value.get("timestamp")) is None:
        return False
    if value.get("type") not in _EVENT_TYPES:
        return False
    return isinstance(value.get("payload"), dict) and _safe_payload(value["payload"])


def load_events(run_dir: Path) -> list[dict]:
    """Load well-formed replay events, ignoring partial lines and unsafe artifact paths."""
    path = run_dir / "events.jsonl"
    try:
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    loaded: list[tuple[datetime, int, dict]] = []
    event_ids: set[str] = set()
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not _valid_event(event) or event["event_id"] in event_ids:
            continue
        timestamp = _parse_timestamp(event["timestamp"])
        if timestamp is None:
            continue
        event_ids.add(event["event_id"])
        loaded.append((timestamp, index, event))
    loaded.sort(key=lambda item: (item[0], item[1]))
    return [event for _, _, event in loaded]


def event_seconds(events: list[dict]) -> list[float]:
    """Return each valid timestamp as seconds relative to the first valid event."""
    timestamps = [_parse_timestamp(event.get("timestamp")) for event in events if isinstance(event, dict)]
    valid = [timestamp for timestamp in timestamps if timestamp is not None]
    if not valid:
        return []
    first = valid[0]
    return [(timestamp - first).total_seconds() for timestamp in valid]


def _empty_state() -> dict[str, Any]:
    return {
        "stages": {
            stage: {"status": "pending", "duration_s": None}
            for stage in STAGE_KEYS
        },
        "objects": [],
        "rounds": [],
        "episodes": [],
        "llm_calls": [],
        "final": None,
        "scene": None,
        "caption": None,
        "cost_usd": None,
    }


def project_state(events: list[dict], elapsed_s: float) -> dict:
    """Project recorded event payloads at a replay clock position without recomputing metrics."""
    state = _empty_state()
    if not isinstance(elapsed_s, (int, float)) or isinstance(elapsed_s, bool):
        return state
    valid: list[tuple[datetime, dict]] = []
    seen: set[str] = set()
    for event in events:
        if not _valid_event(event) or event["event_id"] in seen:
            continue
        timestamp = _parse_timestamp(event["timestamp"])
        if timestamp is None:
            continue
        seen.add(event["event_id"])
        valid.append((timestamp, event))
    if not valid:
        return state
    valid.sort(key=lambda item: item[0])
    first = valid[0][0]
    starts: dict[str, float] = {}

    for timestamp, event in valid:
        event_second = (timestamp - first).total_seconds()
        if event_second > float(elapsed_s):
            break
        payload = event["payload"]
        caption = payload.get("caption")
        if isinstance(caption, str):
            state["caption"] = caption
        event_type = event["type"]
        if event_type == "stage_started":
            stage = payload.get("stage")
            if stage in state["stages"]:
                state["stages"][stage] = {"status": "running", "duration_s": 0.0}
                starts[stage] = event_second
        elif event_type == "stage_finished":
            stage = payload.get("stage")
            if stage in state["stages"] and payload.get("available") is True and payload.get("status") == "done":
                started = starts.get(stage, event_second)
                state["stages"][stage] = {
                    "status": "done",
                    "duration_s": max(0.0, event_second - started),
                }
        elif event_type == "object_detected":
            state["objects"].append(copy.deepcopy(payload))
        elif event_type == "scene_built":
            state["scene"] = copy.deepcopy(payload)
        elif event_type == "round_finished":
            state["rounds"].append(copy.deepcopy(payload))
        elif event_type == "episode_finished":
            state["episodes"].append(copy.deepcopy(payload))
        elif event_type == "llm_call":
            state["llm_calls"].append(copy.deepcopy(payload))
        elif event_type == "final_test_finished":
            state["final"] = copy.deepcopy(payload)
            if isinstance(payload.get("cost_usd"), dict):
                state["cost_usd"] = copy.deepcopy(payload["cost_usd"])

    for stage, started in starts.items():
        if state["stages"][stage]["status"] == "running":
            state["stages"][stage]["duration_s"] = max(0.0, float(elapsed_s) - started)
    return state


__all__ = ["load_events", "event_seconds", "project_state"]
