#!/usr/bin/env python3
"""Publish only an explicitly synthetic completed run; never copy private caches."""

import argparse
import shutil
from pathlib import Path

from splatsam.utils import ROOT, file_digest, load_json, save_json

FILES = [
    "report.json",
    "report.md",
    "scene.json",
    "family.json",
    "rounds.json",
    "loop.json",
    "accuracy.json",
    "baseline_params.json",
    "final_params.json",
    "final_test.json",
    "source.png",
    "sim.png",
    "overlay.mp4",
    "grasp.mp4",
    "replay_result.json",
    "trajectory.jsonl",
]


def build(run_dir: Path, destination: Path):
    report = load_json(run_dir / "report.json")
    if report.get("mode") != "synthetic" or not (run_dir / "completed.json").exists():
        raise ValueError(
            "Only completed synthetic runs may be published automatically; real raw media is private"
        )
    completed = load_json(run_dir / "completed.json")
    for path, expected in completed["artifacts"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"completed artifact altered: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in FILES:
        source = run_dir / name
        if not source.is_file():
            raise ValueError(f"required demo artifact missing: {name}")
        shutil.copyfile(source, destination / name)
        files[name] = file_digest(destination / name)
    consistency = run_dir / "consistency"
    for source in sorted(consistency.iterdir()):
        if source.is_file() and source.suffix in {".png", ".json"}:
            name = "consistency/" + source.name
            (destination / "consistency").mkdir(exist_ok=True)
            shutil.copyfile(source, destination / name)
            files[name] = file_digest(destination / name)
    root = Path(completed["root"])
    for name in ["test.json", "SEALED.manifest.json", "SEALED.lock"]:
        source = root / "splits" / name
        relative = "seal/" + name
        (destination / "seal").mkdir(exist_ok=True)
        shutil.copyfile(source, destination / relative)
        files[relative] = file_digest(destination / relative)
    save_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "mode": "synthetic",
            "inference_disabled": True,
            "files": files,
            "source_run": run_dir.name,
            "description": "Actual synthetic CPU simulation. Illustrative, not real-video results.",
        },
    )
    evidence = ROOT / "runs/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "trajectory.jsonl", evidence / "trajectory.jsonl")
    print(f"Bundled {len(files)} verified synthetic artifacts in {destination}; no key/GPU required.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/synth-demo")
    p.add_argument("--destination", type=Path, default=ROOT / "demo_data")
    a = p.parse_args()
    build(a.run_dir, a.destination)
