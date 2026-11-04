from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from splatsam.llm.client import (
    LiveDispatchUnavailable,
    ModelClient,
    _file_or_value_digest,
)
from splatsam.llm.schemas import ObjectList, PlannerPatch, ScenarioFamily


class FakeBudget:
    def __init__(self) -> None:
        self.reservations: list[tuple[str, int, int, int, int]] = []
        self.settlements: list[tuple[str, object]] = []

    def reserve(
        self, model: str, input_tokens: int = 0, output_tokens: int = 0, frames: int = 0, images: int = 0
    ) -> str:
        self.reservations.append((model, input_tokens, output_tokens, frames, images))
        return f"reservation-{len(self.reservations)}"

    def settle(self, reservation_id: str, cost: Decimal | float | str | None = None) -> None:
        self.settlements.append((reservation_id, cost))


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


def test_request_fingerprint_never_contains_data_uri_or_text() -> None:
    fingerprint = _file_or_value_digest(
        {"prompt": "do not log me", "image_url": "data:image/png;base64,aGVsbG8="}
    )
    encoded = json.dumps(fingerprint)
    assert "do not log me" not in encoded
    assert "aGVsbG8=" not in encoded
    assert fingerprint["image_url"]["bytes"] == 5


def test_live_only_sam_helper_rejects_default_offline_mode(tmp_path: Path) -> None:
    client = ModelClient(tmp_path / "runs" / "offline", budget=FakeBudget())
    with pytest.raises(LiveDispatchUnavailable):
        client.sam_request(tmp_path / "missing.mp4", "segment object")


def test_successful_generation_is_cached_but_invalid_response_is_not(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("MODEL_API_KEY", "test-not-a-real-key")
    budget = FakeBudget()
    client = ModelClient(tmp_path / "run", budget=budget, live=True)
    monkeypatch.setattr(client, "_preflight_tokens", lambda payload: 100)
    calls = []

    def create(**payload):
        calls.append(payload)
        return SimpleNamespace(
            status="completed",
            output_text='{"patch":{"grip_safety_factor":2},"reason":"bounded"}',
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )

    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    kwargs = {
        "model": "muse-wonder-1.0",
        "instructions": "bounded edits",
        "input": "counts",
        "stage": "proposal",
    }
    first = client.generate(PlannerPatch, **kwargs)
    second = client.generate(PlannerPatch, **kwargs)
    assert first == second and len(calls) == 1
    assert len(budget.reservations) == len(budget.settlements) == 1
    assert client.last_provenance == "cache"
    # An invalid provider completion is charged, corrected once, then falls back;
    # subsequent requests must try again rather than treating that fallback as live cache.
    calls.clear()

    def invalid(**payload):
        calls.append(payload)
        return SimpleNamespace(status="completed", output_text="not JSON", usage=None)

    client._client = SimpleNamespace(responses=SimpleNamespace(create=invalid))
    for _ in range(2):
        client.generate(ScenarioFamily, **(kwargs | {"input": "different", "stage": "family"}))
    assert len(calls) == 4
    assert client.last_provenance == "fallback"
