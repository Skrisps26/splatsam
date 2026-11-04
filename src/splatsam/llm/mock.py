"""Offline typed model fixtures for synthetic and unit-test execution."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from splatsam.budget import BudgetTracker
from splatsam.llm.client import ModelClient, _default_for

T = TypeVar("T", bound=BaseModel)


class MockClient(ModelClient):
    """Never-dispatching client with typed fixture queues and bounded defaults."""

    def __init__(
        self,
        run_dir: Path,
        budget: BudgetTracker | None = None,
        cache_dir: Path | None = None,
        live: bool = False,
        fixture_responses: dict[str | type[BaseModel], Any] | None = None,
    ) -> None:
        super().__init__(run_dir=run_dir, budget=budget, cache_dir=cache_dir, live=False)
        self._fixtures: dict[str, deque[Any]] = {}
        for key, values in (fixture_responses or {}).items():
            name = key if isinstance(key, str) else key.__name__
            self._fixtures[name] = deque(values if isinstance(values, list) else [values])

    def generate(self, schema: type[T], *, model: str, instructions: str, input: list | str, stage: str) -> T:
        queue = self._fixtures.get(stage) or self._fixtures.get(schema.__name__)
        if queue:
            value = queue.popleft()
            try:
                result = value if isinstance(value, schema) else schema.model_validate(value)
            except ValidationError:
                result = _default_for(schema)
                provenance = "mock_fallback"
            else:
                provenance = "mock_fixture"
        else:
            result = _default_for(schema)
            provenance = "mock"
        self._trajectory(
            provenance=provenance,
            stage=stage,
            model=model,
            request={"instructions": instructions, "input": input, "schema": schema.__name__},
            response=result,
        )
        return result
