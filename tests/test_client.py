from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from splatsam.llm.client import LiveDispatchUnavailable, ModelClient, _file_or_value_digest, _parse_model_response, normalized_json_schema
from splatsam.llm.mock import MockClient
from splatsam.llm.schemas import CriticLabel, ObjectList, PlannerPatch, ScenarioFamily


class FakeBudget:
    def __init__(self) -> None:
        self.reservations: list[tuple[str, int, int, int, int]] = []
        self.settlements: list[tuple[str, object]] = []

    def reserve(self, model: str, input_tokens: int = 0, output_tokens: int = 0, frames: int = 0, images: int = 0) -> str:
        self.reservations.append((model, input_tokens, output_tokens, frames, images))
        return f"reservation-{len(self.reservations)}"

    def settle(self, reservation_id: str, cost: Decimal | float | str | None = None) -> None:
        self.settlements.append((reservation_id, cost))


def test_normalized_schema_requires_default_fields_and_forbids_extras() -> None:
    schema = normalized_json_schema(ScenarioFamily)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_planner_patch_wire_schema_closes_parameter_names() -> None:
    schema = normalized_json_schema(PlannerPatch)
    patch = schema["properties"]["patch"]
    assert patch["additionalProperties"] is False
    assert set(patch["required"]) == set(patch["properties"])
    assert patch["properties"]["n_yaw_candidates"]["type"] == ["integer", "null"]



def test_planner_patch_response_strips_required_wire_nulls() -> None:
    result = _parse_model_response(
        PlannerPatch,
        '{"schema_version":1,"patch":{"grip_safety_factor":1.4,"jaw_margin_m":null,"pregrasp_height_m":null,"approach_speed":null,"lift_speed":null,"n_yaw_candidates":null,"clearance_weight":null,"center_weight":null},"reason":"fixture"}',
    )
    assert result.patch == {"grip_safety_factor": 1.4}

def test_offline_client_returns_typed_fallback_and_redacts_prompt(tmp_path: Path) -> None:
    client = ModelClient(tmp_path / "runs" / "offline", budget=FakeBudget())
    result = client.generate(
        ObjectList,
        model="muse-wonder-1.0",
        instructions="private instruction",
        input="a private prompt",
        stage="objects",
    )
    assert result.source == "fallback"
    assert client.last_provenance == "fallback"
    row = json.loads((tmp_path / "runs" / "offline" / "trajectory.jsonl").read_text())
    serialized = json.dumps(row)
    assert "private instruction" not in serialized
    assert "a private prompt" not in serialized


def test_mock_fixture_and_bounded_proposal(tmp_path: Path) -> None:
    client = MockClient(
        tmp_path / "runs" / "mock",
        budget=FakeBudget(),
        fixture_responses={CriticLabel: {"label": "slip_or_drop", "reason": "fixture"}},
    )
    assert client.generate(CriticLabel, model="muse-spark-1.3", instructions="x", input="y", stage="critic").label == "slip_or_drop"
    patch = client.generate(PlannerPatch, model="muse-wonder-1.0", instructions="x", input="y", stage="proposal")
    assert patch.patch == {"grip_safety_factor": 1.3}
    assert client.last_provenance == "mock"


def test_request_fingerprint_never_contains_data_uri_or_text() -> None:
    fingerprint = _file_or_value_digest({"prompt": "do not log me", "image_url": "data:image/png;base64,aGVsbG8="})
    encoded = json.dumps(fingerprint)
    assert "do not log me" not in encoded
    assert "aGVsbG8=" not in encoded
    assert fingerprint["image_url"]["bytes"] == 5


def test_live_only_sam_helper_rejects_default_offline_mode(tmp_path: Path) -> None:
    client = ModelClient(tmp_path / "runs" / "offline", budget=FakeBudget())
    with pytest.raises(LiveDispatchUnavailable):
        client.sam_request(tmp_path / "missing.mp4", "segment object")
