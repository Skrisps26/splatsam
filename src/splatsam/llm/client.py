"""Budgeted, cacheable Meta Responses transport with safe local evidence."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

import httpx
from diskcache import Cache
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from splatsam.budget import BudgetTracker
from splatsam.llm.schemas import CriticLabel, ObjectList, PlannerPatch, ScenarioFamily
from splatsam.utils import canonical, config, file_digest

META_BASE_URL = "https://api.meta.ai/v1"
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_FALLBACK_TOKENS = 1_048_576
REQUEST_TIMEOUT = httpx.Timeout(timeout=120.0, read=30.0)
T = TypeVar("T", bound=BaseModel)


class LiveDispatchUnavailable(RuntimeError):
    """Raised when a caller explicitly requested a live-only operation."""


class StructuredOutputError(ValueError):
    """A response completed but did not contain a valid structured object."""


def normalized_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Make Pydantic's JSON Schema acceptable to strict Responses output mode."""
    if schema is PlannerPatch:
        planner_fields = {
            "grip_safety_factor": "number",
            "jaw_margin_m": "number",
            "pregrasp_height_m": "number",
            "approach_speed": "number",
            "lift_speed": "number",
            "n_yaw_candidates": "integer",
            "clearance_weight": "number",
            "center_weight": "number",
        }
        patch_properties = {name: {"type": [kind, "null"]} for name, kind in planner_fields.items()}
        return {
            "type": "object",
            "properties": {
                "schema_version": {"type": "integer"},
                "patch": {
                    "type": "object",
                    "properties": patch_properties,
                    "required": list(patch_properties),
                    "additionalProperties": False,
                },
                "reason": {"type": "string"},
            },
            "required": ["schema_version", "patch", "reason"],
            "additionalProperties": False,
        }
    result = copy.deepcopy(schema.model_json_schema())

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        value.pop("default", None)
        prefix_items = value.pop("prefixItems", None)
        if isinstance(prefix_items, list) and prefix_items:
            value["items"] = prefix_items[0]
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = list(properties)
            value["additionalProperties"] = False
        for child in value.values():
            visit(child)

    visit(result)
    if result.get("type") != "object":
        raise ValueError("Responses structured output schemas must have an object root")
    return result


def _safe_decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    value = getattr(usage, name, None) if usage is not None else None
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    return int(value or 0)


def _status_code(error: BaseException) -> int | None:
    code = getattr(error, "status_code", None)
    if code is None:
        response = getattr(error, "response", None)
        code = getattr(response, "status_code", None)
    return int(code) if isinstance(code, int) else None


def _retry_wait(retry_state: Any) -> float:
    delay = wait_exponential_jitter(initial=1, max=8)(retry_state)
    error = retry_state.outcome.exception()
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    try:
        retry_after = float(headers.get("retry-after", 0))
    except (TypeError, ValueError):
        retry_after = 0
    if retry_after > 120:
        raise LiveDispatchUnavailable("Retry-After exceeds bounded request window")
    return max(delay, retry_after)

def _retryable(error: BaseException) -> bool:
    code = _status_code(error)
    return code == 429 or (code is not None and 500 <= code < 600)


def _file_or_value_digest(value: Any) -> Any:
    """Fingerprint request content without retaining prompts or media bytes."""
    if isinstance(value, Path):
        return {"path_name": value.name, "sha256": file_digest(value)}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, str):
        if value.startswith("data:"):
            header, _, encoded = value.partition(",")
            try:
                payload = base64.b64decode(encoded, validate=True)
                return {"data_media_type": header.split(";", 1)[0], "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            except ValueError:
                pass
        return {"text_sha256": hashlib.sha256(value.encode()).hexdigest(), "characters": len(value)}
    if isinstance(value, BaseModel):
        return _file_or_value_digest(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _file_or_value_digest(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_file_or_value_digest(item) for item in value]
    return value


def _default_for(schema: type[T]) -> T:
    """Typed bounded offline outputs; they are marked fallback, never live evidence."""
    if schema is ObjectList:
        return schema.model_validate(
            {
                "objects": [
                    {
                        "id": "mock-object-1",
                        "name": "mock object",
                        "material_guess": "plastic",
                        "approx_size_cm": [10.0, 4.0, 2.0],
                        "graspable": True,
                    }
                ],
                "source": "fallback",
            }
        )  # type: ignore[return-value]
    if schema is ScenarioFamily:
        return schema.model_validate({})  # type: ignore[return-value]
    if schema is PlannerPatch:
        return schema.model_validate(
            {"patch": {"grip_safety_factor": 1.3}, "reason": "mock bounded proposal; no live model dispatch"}
        )  # type: ignore[return-value]
    if schema is CriticLabel:
        return schema.model_validate(
            {"label": "no_grasp_found", "reason": "mock fallback; no live model dispatch"}
        )  # type: ignore[return-value]
    try:
        return schema.model_validate({})
    except ValidationError as error:
        raise LiveDispatchUnavailable(f"No typed offline default for {schema.__name__}") from error


def _parse_model_response(schema: type[T], raw: str) -> T:
    """Translate strict wire-only null patch fields back to the domain model."""
    if schema is PlannerPatch:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise StructuredOutputError("response was not JSON") from error
        if isinstance(value, dict) and isinstance(value.get("patch"), dict):
            value["patch"] = {key: item for key, item in value["patch"].items() if item is not None}
        return schema.model_validate(value)  # type: ignore[return-value]
    return schema.model_validate_json(raw)


class ModelClient:
    """Meta Responses client that cannot make a network request by default."""

    def __init__(
        self,
        run_dir: Path,
        budget: BudgetTracker | None = None,
        cache_dir: Path | None = None,
        live: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.live = live
        self.budget = budget or BudgetTracker(self.run_dir)
        self.cache = Cache(str(cache_dir or self.run_dir / "model_cache"))
        self._client: OpenAI | None = None
        self.last_provenance: str | None = None

    @property
    def can_dispatch(self) -> bool:
        return self.live and bool(os.environ.get("MODEL_API_KEY"))

    def _openai(self) -> OpenAI:
        if not self.can_dispatch:
            raise LiveDispatchUnavailable("live=True and MODEL_API_KEY are both required for Meta dispatch")
        if self._client is None:
            self._client = OpenAI(
                base_url=META_BASE_URL,
                api_key=os.environ["MODEL_API_KEY"],
                max_retries=0,
                timeout=REQUEST_TIMEOUT,
            )
        return self._client

    def _trajectory(self, *, provenance: str, stage: str, model: str, request: Any, response: Any = None, usage: dict[str, int] | None = None, cost: Decimal | None = None, reserved_cost: Decimal | None = None, latency_s: float | None = None, error: BaseException | None = None) -> None:
        self.last_provenance = provenance
        row: dict[str, Any] = {
            "provenance": provenance,
            "stage": stage,
            "model": model,
            "request": _file_or_value_digest(request),
            "usage": usage or {},
            "cost_usd": str(cost or Decimal("0")),
            "reserved_cost_usd": str(reserved_cost or Decimal("0")),
        }
        if response is not None:
            row["response"] = _file_or_value_digest(response)
        if latency_s is not None:
            row["latency_s"] = round(latency_s, 6)
        if error is not None:
            row["error"] = {"type": type(error).__name__, "status": _status_code(error)}
        with (self.run_dir / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(row) + "\n")

    @staticmethod
    def _price(model: str, input_tokens: int, output_tokens: int) -> Decimal:
        prices = config().get("prices", {})
        rate = prices.get(model) or prices.get("muse-wonder-1.0", {"input": 4, "output": 20})
        return (_safe_decimal(rate["input"]) * input_tokens + _safe_decimal(rate["output"]) * output_tokens) / Decimal(1_000_000)

    def _preflight_tokens(self, payload: dict[str, Any]) -> int:
        if not self.can_dispatch:
            return MAX_INPUT_FALLBACK_TOKENS
        key = hashlib.sha256(canonical(_file_or_value_digest(payload)).encode()).hexdigest()
        cached = self.cache.get(f"input_tokens:{key}")
        if isinstance(cached, int) and cached >= 0:
            return cached
        headers = {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}
        started = time.monotonic()
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.post(f"{META_BASE_URL}/responses/input_tokens", headers=headers, json=payload)
                response.raise_for_status()
                tokens = response.json().get("input_tokens")
                if not isinstance(tokens, int) or tokens < 0:
                    return MAX_INPUT_FALLBACK_TOKENS
                self.cache.set(f"input_tokens:{key}", tokens)
                self._trajectory(provenance="live_preflight", stage="token_count", model=payload["model"], request=payload, response={"input_tokens": tokens}, latency_s=time.monotonic()-started)
                return tokens
        except httpx.HTTPError as error:
            self._trajectory(provenance="preflight_error", stage="token_count", model=payload["model"], request=payload, error=error, latency_s=time.monotonic()-started)
            return MAX_INPUT_FALLBACK_TOKENS

    def _dispatch_json(self, *, payload: dict[str, Any], schema: type[T], model: str, stage: str, attempt_kind: str) -> T:
        input_tokens = self._preflight_tokens(payload)
        reserved_cost = self._price(model, input_tokens, MAX_OUTPUT_TOKENS)
        reservation = self.budget.reserve(model, input_tokens=input_tokens, output_tokens=MAX_OUTPUT_TOKENS)
        started = time.monotonic()
        reservation_closed = False
        charged_cost = reserved_cost
        try:
            response = self._openai().responses.create(**payload)
            raw = getattr(response, "output_text", "")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("completed response contained no text")
            parsed = _parse_model_response(schema, raw)
            usage = {
                "preflight_input_tokens": input_tokens,
                "input_tokens": _usage_value(response, "input_tokens"),
                "output_tokens": _usage_value(response, "output_tokens"),
            }
            cost = self._price(model, usage["input_tokens"], usage["output_tokens"])
            if getattr(response, "usage", None) is None:
                cost = reserved_cost
            charged_cost = cost
            reservation_closed = True
            self.budget.settle(reservation, cost)
            self._trajectory(provenance="live", stage=stage, model=model, request=payload, response=parsed, usage=usage, cost=cost, reserved_cost=reserved_cost, latency_s=time.monotonic() - started)
            return parsed
        except Exception as error:
            if not reservation_closed:
                self.budget.settle(reservation)
            self._trajectory(provenance="live_error", stage=stage, model=model, request=payload, usage={"preflight_input_tokens": input_tokens}, cost=charged_cost, reserved_cost=reserved_cost, latency_s=time.monotonic() - started, error=error)
            raise

    def _live_json(self, *, payload: dict[str, Any], schema: type[T], model: str, stage: str, attempt_kind: str) -> T:
        retrying = Retrying(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(3),
            wait=_retry_wait,
            reraise=True,
        )
        return retrying(self._dispatch_json, payload=payload, schema=schema, model=model, stage=stage, attempt_kind=attempt_kind)

    def generate(self, schema: type[T], *, model: str, instructions: str, input: list | str, stage: str) -> T:
        output_schema = normalized_json_schema(schema)
        payload: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "text": {"format": {"type": "json_schema", "name": schema.__name__.lower(), "schema": output_schema, "strict": True}},
        }
        key = hashlib.sha256(canonical(_file_or_value_digest(payload)).encode()).hexdigest()
        cached = self.cache.get(f"response:{key}")
        if cached is not None:
            try:
                result = schema.model_validate(cached)
            except ValidationError:
                self.cache.delete(f"response:{key}")
            else:
                self._trajectory(provenance="cache", stage=stage, model=model, request=payload, response=result)
                return result
        if not self.can_dispatch:
            result = _default_for(schema)
            self._trajectory(provenance="fallback", stage=stage, model=model, request=payload, response=result)
            return result
        live_success = False
        try:
            result = self._live_json(payload=payload, schema=schema, model=model, stage=stage, attempt_kind="initial")
            live_success = True
        except (ValidationError, StructuredOutputError):
            correction = dict(payload)
            correction["instructions"] = f"{instructions}\nReturn valid JSON matching the supplied schema exactly; no prose."
            try:
                result = self._live_json(payload=correction, schema=schema, model=model, stage=stage, attempt_kind="correction")
                live_success = True
            except Exception as error:
                result = _default_for(schema)
                self._trajectory(provenance="fallback", stage=stage, model=model, request=payload, response=result, error=error)
        except Exception as error:
            result = _default_for(schema)
            self._trajectory(provenance="fallback", stage=stage, model=model, request=payload, response=result, error=error)
        if live_success:
            self.cache.set(f"response:{key}", result.model_dump(mode="json"))
        return result

    def upload_video(self, path: Path) -> str:
        """Upload an authorized local video and cache its ID by immutable digest."""
        path = Path(path).resolve()
        if not self.can_dispatch:
            raise LiveDispatchUnavailable("video upload requires explicit live mode and MODEL_API_KEY")
        digest = file_digest(path)
        key = f"upload:{digest}"
        cached = self.cache.get(key)
        if isinstance(cached, str):
            return cached
        started = time.monotonic()
        try:
            with path.open("rb") as stream:
                uploaded = self._openai().files.create(file=(path.name, stream), purpose="user_data")
            file_id = getattr(uploaded, "id", None)
            if not isinstance(file_id, str) or not file_id:
                raise ValueError("Meta Files upload returned no file ID")
            self.cache.set(key, file_id)
            self._trajectory(provenance="live", stage="sam_upload", model="sam-3.1", request={"video": path}, response={"file_id": file_id}, latency_s=time.monotonic() - started)
            return file_id
        except Exception as error:
            self._trajectory(provenance="live_error", stage="sam_upload", model="sam-3.1", request={"video": path}, latency_s=time.monotonic() - started, error=error)
            raise

    @staticmethod
    def _video_frames(path: Path) -> int:
        import imageio.v3 as iio

        return sum(1 for _ in iio.imiter(path))

    @staticmethod
    def _box_text(box_prompt: Any) -> str:
        if box_prompt is None:
            return ""
        if isinstance(box_prompt, str):
            return box_prompt
        from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt

        size = tuple(box_prompt["size"])
        left, top, right, bottom = box_prompt["box"]
        frame = int(box_prompt.get("frame_index", 0))
        object_id = int(box_prompt.get("object_id", 1))
        return build_object_prompt(
            size=(int(size[0]), int(size[1])),
            objects=[PromptObject(id=object_id, prompts=[FramePrompt(frame=frame, box=PromptBox(int(left), int(top), int(right), int(bottom)))])],
        )

    def sam_request(self, path: Path, phrase: str, box_prompt: Any = None) -> dict[str, Any]:
        """Stream SAM output into the public parser and return raw completed evidence."""
        if not self.can_dispatch:
            raise LiveDispatchUnavailable("SAM requires explicit live mode and MODEL_API_KEY")
        path = Path(path).resolve()
        frames = max(50, self._video_frames(path))
        prompt = self._box_text(box_prompt) if box_prompt is not None else phrase.strip()
        sam_key = hashlib.sha256(canonical(_file_or_value_digest({"video": path, "phrase": phrase, "box_prompt": box_prompt})).encode()).hexdigest()
        cached = self.cache.get(f"sam:{sam_key}")
        if isinstance(cached, dict) and isinstance(cached.get("output_text"), str):
            from meta_sam_parser import CompletedOutcome, video_segmentation_format

            parser = video_segmentation_format().create_parser()
            parser.push(cached["output_text"])
            result = dict(cached)
            result["parsed"] = parser.finish(CompletedOutcome()).result
            self._trajectory(provenance="cache", stage="sam", model="sam-3.1", request={"video": path, "phrase": phrase, "box_prompt": box_prompt}, response={"output_sha256": hashlib.sha256(cached["output_text"].encode()).hexdigest()}, usage=cached.get("usage", {}))
            return result
        file_id = self.upload_video(path)

        def request_once() -> dict[str, Any]:
            nonlocal file_id
            reserved_cost = _safe_decimal(config()["prices"]["sam-3.1"]["per_frame"]) * frames
            reservation = self.budget.reserve("sam-3.1", frames=frames)
            started = time.monotonic()
            reservation_closed = False
            charged_cost = reserved_cost
            payload = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": file_id}]}]}
            try:
                from meta_sam_parser import CompletedOutcome, video_segmentation_format

                parser = video_segmentation_format().create_parser()
                chunks: list[str] = []
                completed = False
                usage: dict[str, int] = {}
                with self._openai().responses.stream(**payload) as stream:
                    for event in stream:
                        if time.monotonic() - started > 120:
                            raise TimeoutError("SAM overall request deadline exceeded")
                        if getattr(event, "type", None) == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if isinstance(delta, str):
                                chunks.append(delta)
                                parser.push(delta)
                        if getattr(event, "type", None) == "response.completed":
                            completed = getattr(getattr(event, "response", None), "status", None) == "completed"
                            response = getattr(event, "response", None)
                            usage = {
                                "input_tokens": _usage_value(response, "input_tokens"),
                                "output_tokens": _usage_value(response, "output_tokens"),
                                "video_frames_processed": _usage_value(response, "video_frames_processed"),
                            }
                if not completed:
                    raise ValueError("SAM stream did not end in completed status")
                parsed = parser.finish(CompletedOutcome()).result
                raw = "".join(chunks)
                if not raw:
                    raise ValueError("completed SAM response contained no text")
                billed_frames = max(50, usage["video_frames_processed"] or frames)
                cost = _safe_decimal(config()["prices"]["sam-3.1"]["per_frame"]) * billed_frames
                charged_cost = cost
                reservation_closed = True
                self.budget.settle(reservation, cost)
                result = {"output_text": raw, "parsed": parsed, "usage": usage, "frames_billed": billed_frames}
                self.cache.set(f"sam:{sam_key}", {"output_text": raw, "usage": usage, "frames_billed": billed_frames})
                self._trajectory(provenance="live", stage="sam", model="sam-3.1", request=payload | {"video": path}, response={"output_sha256": hashlib.sha256(raw.encode()).hexdigest()}, usage=usage, cost=cost, reserved_cost=reserved_cost, latency_s=time.monotonic() - started)
                return result
            except Exception as error:
                if not reservation_closed:
                    self.budget.settle(reservation)
                self._trajectory(provenance="live_error", stage="sam", model="sam-3.1", request=payload | {"video": path}, cost=charged_cost, reserved_cost=reserved_cost, latency_s=time.monotonic() - started, error=error)
                if _status_code(error) == 404:
                    self.cache.delete(f"upload:{file_digest(path)}")
                    file_id = self.upload_video(path)
                raise

        retrying = Retrying(retry=retry_if_exception(_retryable), stop=stop_after_attempt(3), wait=_retry_wait, reraise=True)
        try:
            return retrying(request_once)
        except Exception as error:
            if _status_code(error) != 404:
                raise
            # A cached Files ID can expire; retry exactly once with the fresh ID.
            return retrying(request_once)
