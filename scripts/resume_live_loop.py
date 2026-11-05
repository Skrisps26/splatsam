#!/usr/bin/env python3
"""Resume the authorized live B2 validation loop without touching held-out outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from splatsam.authorization import require_live
from splatsam.budget import BudgetTracker
from splatsam.llm.client import ModelClient
from splatsam.llm.schemas import ScenarioFamily, SceneSpec
from splatsam.loop.engine import ProposalUnavailable, improve
from splatsam.planner import PlannerParams
from splatsam.report import generate_report
from splatsam.utils import ROOT, canonical, digest, load_json, save_json

_PROTOCOL_VERSION = 1


class _LiveProposalClient:
    """Permit normal cached/live calls while rejecting typed fallback proposals."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def generate(self, *args, **kwargs):
        if kwargs.get("stage") in {"critic", "proposal"}:
            _require_final_pending()
        result = self._client.generate(*args, **kwargs)
        if kwargs.get("stage") in {"critic", "proposal"}:
            _require_final_pending()
        if kwargs.get("stage") == "proposal" and self._client.last_provenance not in {"live", "cache"}:
            raise ProposalUnavailable(
                "live proposal dispatch did not produce a cached or completed live result"
            )
        return result


def _require_run_dir(path: Path) -> Path:
    run_dir = path.expanduser().resolve()
    runs_root = (ROOT / "runs").resolve()
    if not run_dir.is_relative_to(runs_root) or not run_dir.is_dir():
        raise ValueError("--run-dir must name an existing private run below runs/")
    return run_dir


def _require_final_pending() -> None:
    results = ROOT / "results"
    if (results / "final_test.json").exists():
        raise RuntimeError("real final test result already exists; continuation is forbidden")
    if (results / "FINAL_TEST.started").exists():
        raise RuntimeError("real final test was already started; continuation is forbidden")


def _verify_frozen_inputs(
    scene: SceneSpec, family: ScenarioFamily, baseline: PlannerParams
) -> dict[str, Any]:
    """Check seal inputs and lock without reading the private held-out scenarios."""
    from splatsam.eval.seal import _input_hashes

    split_dir = ROOT / "data" / "splits"
    test_path = split_dir / "test.json"
    manifest_path = split_dir / "SEALED.manifest.json"
    lock_path = split_dir / "SEALED.lock"
    if not (test_path.is_file() and manifest_path.is_file() and lock_path.is_file()):
        raise RuntimeError("sealed real inputs are incomplete")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("sealed manifest has an invalid format")
    if lock_path.read_text().strip() != digest(manifest):
        raise RuntimeError("SEALED.lock does not match the frozen manifest")
    if manifest.get("schema_version") != 1 or manifest.get("mode") != "real":
        raise RuntimeError("sealed manifest is not the expected real B2 seal")
    if manifest.get("hashes") != _input_hashes(scene, family, baseline):
        raise RuntimeError("frozen scene, family, baseline, scoring, config, or dependencies changed")
    return manifest


def _load_loop_state(
    run_dir: Path,
) -> tuple[SceneSpec, ScenarioFamily, PlannerParams, PlannerParams, list[dict], dict]:
    required = (
        "input.json",
        "scene.json",
        "family.json",
        "baseline_params.json",
        "final_params.json",
        "rounds.json",
        "loop.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"B2 continuation artifacts missing: {', '.join(missing)}")
    input_record = load_json(run_dir / "input.json")
    if input_record.get("source") != "live" or not isinstance(input_record.get("video_path"), str):
        raise RuntimeError("run is not a recorded live B2 run")
    video = Path(input_record["video_path"])
    require_live("B2", video)
    scene = SceneSpec.model_validate(load_json(run_dir / "scene.json"))
    if scene.source != "live":
        raise RuntimeError("scene is not live; refusing to resume it as B2")
    family = ScenarioFamily.model_validate(load_json(run_dir / "family.json"))
    baseline = PlannerParams.load(run_dir / "baseline_params.json")
    current = PlannerParams.load(run_dir / "final_params.json")
    rounds = load_json(run_dir / "rounds.json")
    loop = load_json(run_dir / "loop.json")
    if not isinstance(rounds, list) or not all(isinstance(row, dict) for row in rounds):
        raise RuntimeError("rounds history is invalid")
    if not isinstance(loop, dict) or loop.get("history") != rounds:
        raise RuntimeError("loop summary and rounds history disagree")
    if loop.get("final_params") != current.to_dict():
        raise RuntimeError("final parameters disagree with loop summary")
    if [row.get("round") for row in rounds] != list(range(1, len(rounds) + 1)):
        raise RuntimeError("round history is not a contiguous immutable sequence")
    for name in ("validation_evaluations", "fresh_validation_simulations"):
        if isinstance(loop.get(name), bool) or not isinstance(loop.get(name), int) or loop[name] < 0:
            raise RuntimeError(f"loop {name} is invalid")
    return scene, family, baseline, current, rounds, loop


def _trailing_rejections(history: list[dict], first_new_round: int) -> int:
    rejected = 0
    for row in reversed(history[first_new_round:]):
        if row.get("accepted") is True:
            break
        rejected += 1
    return rejected


def _continuation_protocol(
    run_dir: Path, history: list[dict], max_total_rounds: int
) -> tuple[dict[str, Any], int, int]:
    path = run_dir / "live_loop_continuation.json"
    if path.exists():
        protocol = load_json(path)
        if not isinstance(protocol, dict):
            raise RuntimeError("private continuation protocol is invalid")
        initial_rounds = protocol.get("initial_rounds")
        initial_hash = protocol.get("initial_history_sha256")
        recorded_max = protocol.get("max_total_rounds")
        if (
            protocol.get("schema_version") != _PROTOCOL_VERSION
            or not isinstance(initial_rounds, int)
            or not 0 <= initial_rounds <= len(history)
            or not isinstance(initial_hash, str)
            or initial_hash != digest(history[:initial_rounds])
            or not isinstance(recorded_max, int)
            or not 1 <= recorded_max <= 8
        ):
            raise RuntimeError("private continuation protocol does not match immutable history")
        max_total_rounds = min(max_total_rounds, recorded_max)
    else:
        initial_rounds = len(history)
        if initial_rounds != 3 or any(row.get("accepted") is not False for row in history):
            raise RuntimeError("continuation requires the three preserved rejected B2 fallback rounds")
        protocol = {
            "schema_version": _PROTOCOL_VERSION,
            "protocol": "authorized_B2_validation_continuation",
            "initial_rounds": initial_rounds,
            "initial_history_sha256": digest(history),
            "max_total_rounds": max_total_rounds,
            "max_additional_rounds": 5,
            "new_rejection_stop": 3,
            "held_out_access": "not_read",
            "status": "started",
        }
        save_json(path, protocol)
    if len(history) > max_total_rounds:
        raise RuntimeError("existing history exceeds the requested total round bound")
    return protocol, initial_rounds, max_total_rounds


def _costs(run_dir: Path, budget: BudgetTracker) -> dict[str, float]:
    existing = load_json(run_dir / "report.json") if (run_dir / "report.json").is_file() else {}
    costs = existing.get("costs") if isinstance(existing, dict) else None
    result = dict(costs) if isinstance(costs, dict) else {"gpu_usd": 0.0}
    result["meta_usd"] = float(budget.total_usd)
    result.setdefault("gpu_usd", 0.0)
    return result


def resume(run_dir: Path, *, workers: int, max_total_rounds: int) -> dict:
    if not 1 <= max_total_rounds <= 8:
        raise ValueError("--max-total-rounds must be 1..8")
    run_dir = _require_run_dir(run_dir)
    _require_final_pending()
    scene, family, baseline, current, history, prior_loop = _load_loop_state(run_dir)
    manifest = _verify_frozen_inputs(scene, family, baseline)
    if manifest.get("hashes", {}).get("baseline") != baseline.hash:
        raise RuntimeError("sealed baseline hash does not match saved B2 baseline")
    protocol, initial_rounds, max_total_rounds = _continuation_protocol(run_dir, history, max_total_rounds)
    manifest_hash = digest(manifest)
    recorded_manifest_hash = protocol.get("frozen_manifest_sha256")
    if recorded_manifest_hash is not None and recorded_manifest_hash != manifest_hash:
        raise RuntimeError("private continuation protocol does not match the frozen manifest")
    protocol.setdefault("frozen_manifest_sha256", manifest_hash)
    protocol.setdefault("initial_validation_evaluations", prior_loop["validation_evaluations"])
    protocol.setdefault("initial_fresh_validation_simulations", prior_loop["fresh_validation_simulations"])
    save_json(run_dir / "live_loop_continuation.json", protocol)
    prior_rejections = _trailing_rejections(history, initial_rounds)
    if prior_rejections > 3:
        raise RuntimeError("continuation history already exceeds its rejection stop")
    budget = BudgetTracker(ROOT / "runs" / "meta_budget.json", phase="B")
    client = ModelClient(run_dir, budget=budget, cache_dir=ROOT / ".cache" / "meta", live=True)
    if not client.can_dispatch:
        raise RuntimeError("MODEL_API_KEY is required; live continuation will not use a fallback proposal")
    loop = improve(
        scene,
        family,
        current,
        _LiveProposalClient(client),
        run_dir,
        workers=workers,
        max_rounds=max_total_rounds,
        prior_history=history,
        prior_validation_evaluations=prior_loop["validation_evaluations"],
        prior_fresh_validation_simulations=prior_loop["fresh_validation_simulations"],
        prior_consecutive_rejections=prior_rejections,
        max_additional_rounds=min(5, max_total_rounds - len(history)),
    )
    continuation_status = loop.get("status", "ok")
    if continuation_status not in {"ok", "blocked"}:
        raise RuntimeError("continuation loop returned an invalid status")
    blocked_reason = loop.get("blocked_reason")
    if continuation_status == "blocked" and not isinstance(blocked_reason, dict):
        raise RuntimeError("blocked continuation loop has no reason")
    protocol.update(
        {
            "status": continuation_status,
            "total_rounds": len(loop["history"]),
            "new_rounds": len(loop["history"]) - initial_rounds,
            "new_consecutive_rejections": _trailing_rejections(loop["history"], initial_rounds),
        }
    )
    if continuation_status == "blocked":
        protocol["blocked_reason"] = blocked_reason
    else:
        protocol.pop("blocked_reason", None)
    save_json(run_dir / "live_loop_continuation.json", protocol)
    previous_report = load_json(run_dir / "report.json") if (run_dir / "report.json").is_file() else {}
    stages = previous_report.get("stages", []) if isinstance(previous_report, dict) else []
    if not isinstance(stages, list):
        stages = []
    stages = [
        stage
        for stage in stages
        if not isinstance(stage, dict) or stage.get("stage") != "validation-loop-continuation"
    ]
    continuation_stage: dict[str, Any] = {
        "stage": "validation-loop-continuation",
        "status": continuation_status,
    }
    if continuation_status == "blocked":
        continuation_stage["reason"] = blocked_reason
    stages.append(continuation_stage)
    generate_report(
        run_dir,
        scene,
        None,
        loop["history"],
        _costs(run_dir, budget),
        validation_evaluations=loop["validation_evaluations"],
        fresh_validation_simulations=loop["fresh_validation_simulations"],
        stages=stages,
    )
    return loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-total-rounds", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        result = resume(args.run_dir, workers=args.workers, max_total_rounds=args.max_total_rounds)
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    output = {
        "rounds": len(result["history"]),
        "final_params": result["final_params"],
        "status": result.get("status", "ok"),
    }
    if result.get("status") == "blocked":
        output["blocked_reason"] = result["blocked_reason"]
    print(canonical(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
