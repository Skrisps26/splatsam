#!/usr/bin/env python
"""Opt-in, bounded live Meta format spike. It is deliberately not a Make target."""
from __future__ import annotations

import argparse
import base64
import json
import os
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from splatsam.budget import BudgetTracker
from splatsam.llm.client import ModelClient, MAX_OUTPUT_TOKENS, normalized_json_schema
from splatsam.llm.schemas import ObjectList, ScenarioFamily
from splatsam.utils import save_json, file_digest


def _synthetic_media(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (8, 10), (24, 21), (30, 100, 200), thickness=-1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode synthetic spike image")
    video = root / "synthetic-spike.mp4"
    iio.imwrite(video, np.stack((image, np.roll(image, 1, axis=1))), fps=2)
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), video


def _json_safe(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _require_live(client: ModelClient, what: str) -> None:
    if client.last_provenance not in {"live", "cache"}:
        raise RuntimeError(f"{what} did not produce observed live evidence ({client.last_provenance})")


def _append_observed_notes(project_root: Path, fixture: Path, snapshot: dict) -> None:
    notes = project_root / "docs" / "api_notes.md"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entry = (
        "\n## Live-confirmed spike\n\n"
        f"- UTC: `{timestamp}`\n"
        "- Observed successful Wonder JSON-schema, Spark image, Files upload, and streamed SAM requests. "
        f"Sanitized fixture: `{fixture.relative_to(project_root)}`.\n"
        f"- Ledger snapshot recorded at completion: `{json.dumps(snapshot, sort_keys=True)}`. "
        "This note records format evidence only; it is not a quality or real-video claim.\n"
        f"\nWorking wire requests (synthetic media only, bearer header omitted):\n\n```json\n{json.dumps(json.loads(fixture.read_text())['requests'], indent=2)}\n```\n"
    )
    prior = notes.read_text(encoding="utf-8")
    prior = prior.replace(
        "No live calls have been made in this checkout. `scripts/spike_api.py` is opt-in and records an appended entry here only after all three synthetic requests and the SAM upload/stream complete with observed responses. A missing key exits blocked without dispatch or spend.",
        "`scripts/spike_api.py` is opt-in. Successful synthetic confirmations are recorded below; a missing key exits blocked without dispatch or spend.",
    )
    notes.write_text(prior + entry, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-usd", type=float, default=10.0)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/spike-api"))
    args = parser.parse_args()
    if not 0 < args.max_usd <= 10:
        parser.error("--max-usd must be positive and no greater than the Phase A $10 cap")
    if args.max_usd != 10:
        parser.error("this Phase A spike is fixed to the configured $10 cap")
    if not os.environ.get("MODEL_API_KEY"):
        print("BLOCKED: MODEL_API_KEY is absent; no API call, upload, or spend was attempted.")
        return 0

    project_root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    budget = BudgetTracker(project_root / "runs" / "meta_budget.json")
    client = ModelClient(run_dir, budget=budget, live=True)
    media_dir = run_dir / "synthetic_media"
    image_data_uri, video = _synthetic_media(media_dir)
    try:
        wonder = client.generate(
            ScenarioFamily,
            model="muse-wonder-1.0",
            instructions="Return a bounded scenario family for a synthetic tabletop scene.",
            input="Synthetic tabletop format check only.",
            stage="spike_wonder",
        )
        _require_live(client, "Wonder request")
        spark = client.generate(
            ObjectList,
            model="muse-spark-1.3",
            instructions="List the single synthetic colored rectangle as an object.",
            input=[{"role": "user", "content": [{"type": "input_text", "text": "synthetic image only"}, {"type": "input_image", "image_url": image_data_uri}]}],
            stage="spike_spark",
        )
        _require_live(client, "Spark request")
        sam = client.sam_request(video, "colored rectangle")
        _require_live(client, "SAM request")
    except Exception as error:
        print(f"BLOCKED/FAILED: {type(error).__name__}: {error}")
        return 1

    snapshot = budget.snapshot()
    total = float(snapshot.get("total_usd", 0))
    if total > args.max_usd:
        print(f"BLOCKED: observed ledger total ${total:.6f} exceeds requested spike limit ${args.max_usd:.2f}.")
        return 1
    fixture = run_dir / "observed_fixture.json"
    safe_snapshot = _json_safe(snapshot)
    requests = {
        "wonder": {"model":"muse-wonder-1.0","instructions":"Return a bounded scenario family for a synthetic tabletop scene.","input":"Synthetic tabletop format check only.","max_output_tokens":MAX_OUTPUT_TOKENS,"text":{"format":{"type":"json_schema","name":"scenariofamily","schema":normalized_json_schema(ScenarioFamily),"strict":True}}},
        "spark": {"model":"muse-spark-1.3","instructions":"List the single synthetic colored rectangle as an object.","input":[{"role":"user","content":[{"type":"input_text","text":"synthetic image only"},{"type":"input_image","image_url":image_data_uri}]}],"max_output_tokens":MAX_OUTPUT_TOKENS,"text":{"format":{"type":"json_schema","name":"objectlist","schema":normalized_json_schema(ObjectList),"strict":True}}},
        "files": {"method":"POST","url":"https://api.meta.ai/v1/files","purpose":"user_data","file_artifact":str(video.relative_to(project_root)),"file_sha256":file_digest(video)},
        "sam": {"model":"sam-3.1","stream":True,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"colored rectangle"},{"type":"input_video","file_id":client.cache.get("upload:"+file_digest(video))}]}]},
    }
    save_json(
        fixture,
        {
            "source": "live_observed_synthetic_spike",
            "requests": requests,
            "wonder": wonder.model_dump(mode="json"),
            "spark": spark.model_dump(mode="json"),
            "sam": {"output_text": sam["output_text"], "usage": sam["usage"], "frames_billed": sam["frames_billed"]},
            "ledger": safe_snapshot,
        },
    )
    _append_observed_notes(project_root, fixture, safe_snapshot)
    print(f"LIVE-CONFIRMED: sanitized fixture saved to {fixture}; ledger total ${total:.6f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
