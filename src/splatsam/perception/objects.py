"""Semantic object identification from explicitly authorized RGB frames."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import TYPE_CHECKING

import imageio.v2 as imageio

from splatsam.llm.schemas import ObjectList
from splatsam.perception.frames import FrameSample

if TYPE_CHECKING:
    from splatsam.llm.client import ModelClient


_OBJECT_INSTRUCTIONS = """Identify only visible workspace objects on the support surface.
Return stable IDs, a concise segmentation noun phrase, material estimate, full approximate
dimensions in centimetres (length, width, vertical height), a graspable flag, and instance
count. Prefer one class entry with count for repeated objects, not overlapping duplicate
entries. Do not identify printed stickers, surface patterns or shadows as separate objects.
Do not infer hidden positions or use external scene knowledge. Every listed object must
be visible in at least one frame."""


def _image_data_uri(frame: FrameSample) -> str:
    """Encode an already-authorized local RGB frame for Responses input."""
    buffer = BytesIO()
    imageio.imwrite(buffer, frame.image, format="png")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _input_for_frames(frames: list[FrameSample]) -> list[dict[str, object]]:
    if not frames:
        raise ValueError("object identification requires at least one frame")
    # Three spread-out views are sufficient semantic evidence while bounding context.
    positions = sorted({0, len(frames) // 2, len(frames) - 1})
    content: list[dict[str, str]] = [
        {"type": "input_text", "text": "Identify objects in these authorized video frames."}
    ]
    for position in positions:
        frame = frames[position]
        content.append(
            {
                "type": "input_text",
                "text": f"source frame {frame.index} at {frame.timestamp_s:.6f} seconds",
            }
        )
        content.append({"type": "input_image", "image_url": _image_data_uri(frame)})
    return [{"type": "message", "role": "user", "content": content}]


def _record_fixture(client: object, fixture: ObjectList, frame_count: int) -> None:
    """Record a synthetic semantic fixture without making a model/network request."""
    trajectory = getattr(client, "_trajectory", None)
    if callable(trajectory):
        trajectory(
            provenance="fixture",
            stage="objects",
            model="fixture",
            request={"authorized_frame_count": frame_count, "schema": "ObjectList"},
            response=fixture,
        )
    if hasattr(client, "last_provenance"):
        client.last_provenance = "fixture"


def identify_objects(
    frames: list[FrameSample],
    client: ModelClient,
    *,
    fixture: ObjectList | None = None,
    reference: dict | None = None,
) -> ObjectList:
    """Return semantic priors, never geometry or positions from hidden scene truth.

    A fixture is deliberately logged as a zero-cost local model artifact.  It is
    for synthetic/mocked execution only; normal calls send only the supplied RGB
    frame bytes through the typed Wonder transport.
    """
    model_input = _input_for_frames(frames)
    if fixture is not None:
        result = ObjectList.model_validate(fixture)
        _record_fixture(client, result, len(frames))
        return result
    instructions = _OBJECT_INSTRUCTIONS
    if reference is not None:
        instructions += (
            "\nThe configured calibration reference is "
            + json.dumps(reference)
            + '. If visible, include it once with id="reference", use its supplied name '
            "as the segmentation phrase, mark graspable=false, and use the supplied "
            "metric dimensions as priors. Do not invent a sheet of paper if absent."
        )
    result = client.generate(
        ObjectList,
        model="muse-wonder-1.0",
        instructions=instructions,
        input=model_input,
        stage="objects",
    )
    return ObjectList.model_validate(result)
