#!/usr/bin/env python
"""One-shot sealed final evaluation; root mode is inferred, never flag-selected."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from splatsam.eval.seal import run_final
from splatsam.llm.schemas import ScenarioFamily, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.utils import ROOT, load_json


def _load_inputs(run_dir: Path) -> tuple[SceneSpec, ScenarioFamily, PlannerParams, PlannerParams]:
    required = {
        "scene": run_dir / "scene.json",
        "family": run_dir / "family.json",
        "baseline": run_dir / "baseline_params.json",
        "final": run_dir / "final_params.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen run inputs missing: {', '.join(missing)}")
    return (
        SceneSpec.model_validate(load_json(required["scene"])),
        ScenarioFamily.model_validate(load_json(required["family"])),
        PlannerParams.load(required["baseline"]),
        PlannerParams.load(required["final"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="sealed data root (data_synth/... or authorized data/...)")
    parser.add_argument("--run-dir", required=True, type=Path, help="directory holding frozen scene/family/parameter JSON")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")

    scene, family, baseline, final = _load_inputs(args.run_dir)
    result = run_final(scene, family, baseline, final, args.root, args.run_dir, workers=args.workers)
    root = args.root.resolve()
    real_data = (ROOT / "data").resolve()
    result_root = ROOT / "results" if root == real_data or root.is_relative_to(real_data) else root / "results"
    print(json.dumps({"result": str(result_root / "final_test.json"), "seal_hash": result["seal_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
