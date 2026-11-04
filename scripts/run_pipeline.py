#!/usr/bin/env python3
"""Video-to-report orchestration. Phase A is synthetic; live actions fail closed."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from splatsam.authorization import require_live
from splatsam.budget import BudgetTracker
from splatsam.llm.schemas import ObjectList, PlannerPatch, ScenarioFamily, SceneSpec
from splatsam.planner import PlannerParams
from splatsam.utils import ROOT, config, digest, file_digest, load_json, save_json


def print_cost(stage: str, phase="A") -> dict:
    budget = BudgetTracker(ROOT / "runs/meta_budget.json", phase=phase)
    gpu = (
        load_json(ROOT / "runs/gpu_ledger.json")
        if (ROOT / "runs/gpu_ledger.json").exists()
        else {"total_usd": 0, "resources": []}
    )
    costs = {"meta_usd": float(budget.total_usd), "gpu_usd": float(gpu.get("total_usd", 0))}
    print(f"[{stage}] Meta USD {costs['meta_usd']:.4f}; GPU USD {costs['gpu_usd']:.4f}", flush=True)
    return costs


def calibrate(scene, workers: int, initial=None):
    from splatsam.curriculum.sampler import sample_scenario
    from splatsam.sim.episode import evaluate

    presets = [
        initial or ScenarioFamily(),
        ScenarioFamily(
            friction_scale=(0.5, 1.2),
            mass_scale=(1, 2),
            jitter_m=0.04,
            clutter_count=(1, 3),
            occluder_count=(0, 2),
        ),
        ScenarioFamily(
            friction_scale=(1, 1.5),
            mass_scale=(0.5, 1),
            jitter_m=0.01,
            clutter_count=(0, 0),
            occluder_count=(0, 0),
        ),
    ]
    pilots = []
    for index, family in enumerate(presets):
        scenarios = [sample_scenario(family, s, scene) for s in range(5000, 5100)]
        start = time.perf_counter()
        results = evaluate(scene, scenarios, PlannerParams(), workers=workers)
        rate = float(np.mean([r.success for r in results]))
        elapsed = time.perf_counter() - start
        pilots.append(
            {
                "preset": index,
                "success_rate": rate,
                "n": 100,
                "wall_seconds": elapsed,
                "family": family.model_dump(mode="json"),
            }
        )
        print(f"Pilot {index}: {rate:.1%}, 100 episodes in {elapsed:.3f}s", flush=True)
        if 0.25 <= rate <= 0.75:
            return family, pilots
    best = min(pilots, key=lambda p: abs(p["success_rate"] - 0.5))
    raise RuntimeError(
        f"Pilot difficulty gate not met: best={best['success_rate']:.1%}; inspect physics before freezing"
    )


def handcrafted(args):
    from splatsam.curriculum.sampler import split_scenarios
    from splatsam.eval.seal import seal_test
    from splatsam.eval.statistics import summarize
    from splatsam.sim.episode import evaluate, run_episode
    from splatsam.sim.scene import handcrafted_scene

    scene = handcrafted_scene(args.seed)
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    family, pilots = calibrate(scene, args.workers)
    save_json(run_dir / "pilot.json", pilots)
    baseline = PlannerParams()
    seal_test(scene, family, baseline, Path(args.root).resolve())
    save_json(run_dir / "scene.json", scene)
    save_json(run_dir / "family.json", family)
    baseline.save(run_dir / "baseline_params.json")
    baseline.save(run_dir / "final_params.json")
    scenarios = split_scenarios(family, scene, "train")[: args.benchmark]
    start = time.perf_counter()
    results = evaluate(scene, scenarios, baseline, workers=args.workers)
    elapsed = time.perf_counter() - start
    assert run_episode(scene, scenarios[0], baseline) == results[0], "serial/parallel determinism failed"
    summary = {
        "mode": "synthetic",
        "stage": "A2",
        "baseline_train": summarize(results),
        "wall_seconds": elapsed,
        "episodes": len(results),
        "pilot": pilots[-1],
        "deterministic": True,
        "costs": print_cost("A2"),
    }
    save_json(run_dir / "baseline_train.json", [r.model_dump(mode="json") for r in results])
    save_json(run_dir / "a2_summary.json", summary)
    print(summary, flush=True)
    if not 0.25 <= sum(r.success for r in results) / len(results) <= 0.75:
        raise RuntimeError("Frozen baseline outside A2 25–75% gate")
    if elapsed >= 60:
        raise RuntimeError("A2 throughput gate exceeded 60s")


def synthetic_accuracy(scene, reference, tracks, truth_path):
    """Truth is consulted AFTER reconstruction, only to measure synthetic errors."""
    truth = SceneSpec.model_validate(load_json(Path(truth_path)))
    recovered = {o.id: o for o in scene.objects}
    errors = {}
    for obj in truth.objects:
        if not obj.graspable:
            continue
        candidate = recovered[obj.id]
        error = np.abs(np.sort(candidate.size[:2]) - np.sort(obj.size[:2])) / np.sort(obj.size[:2])
        errors[obj.id] = error.tolist()
        if max(error) > 0.10:
            raise RuntimeError(f"{obj.id}: recovered size error {error}, must be <=10%")
    anchors = [o for o in truth.objects if o.graspable][:2]
    observed = []
    for anchor in anchors:
        track = next(t for t in tracks if t.id == anchor.id)
        frame = min(track.frames, key=lambda f: abs(f.frame_index - reference.working_frame_index))
        observed.append(reference.pixel_to_metric(np.array([frame.centroid]))[0])
    expected = np.linalg.norm(np.subtract(anchors[0].position[:2], anchors[1].position[:2]))
    scale_error = abs(np.linalg.norm(observed[0] - observed[1]) - expected) / expected
    if scale_error >= 0.05:
        raise RuntimeError(f"independent scale error {scale_error}, must be <5%")
    return float(scale_error), errors


def perceive(
    video,
    client,
    run_dir,
    *,
    fixture=None,
    tracks_path=None,
    measured_path=None,
    reference_config=None,
    source="synthetic",
    seed=42,
):
    from splatsam.perception.frames import sample_frames
    from splatsam.perception.objects import identify_objects
    from splatsam.perception.reference import find_reference
    from splatsam.perception.sam import MockSamClient, SamClient, save_overlays
    from splatsam.recon.primitives import reconstruct
    from splatsam.sim.fit import fit_scene, render_scene

    frames = sample_frames(video, out_dir=run_dir / "frames")
    reference_settings = config()["reference"] if reference_config is None else reference_config
    objects = identify_objects(frames, client, fixture=fixture, reference=reference_settings)
    sam = MockSamClient(tracks_path) if tracks_path else SamClient(client, run_dir)
    tracks = sam.track(video, objects)
    save_json(run_dir / "objects.json", objects)
    save_json(run_dir / "tracks.json", [t.model_dump(mode="json") for t in tracks])
    save_overlays(video, tracks, run_dir)
    ref = find_reference(
        tracks,
        frames,
        manual_px_per_cm=reference_settings.get("manual_px_per_cm"),
        reference_name=reference_settings.get("name", "sheet of white paper"),
        width_mm=reference_settings.get("width_mm", 210),
        height_mm=reference_settings.get("height_mm", 297),
        elevation_mm=reference_settings.get("elevation_mm", 0),
        material=reference_settings.get("material", "paper"),
    )
    scene = reconstruct(tracks, objects, ref, measured_path=measured_path, source=source, seed=seed)
    if not any(o.graspable for o in scene.objects):
        raise RuntimeError("No reliable graspable target; scene evaluation blocked")
    scene = fit_scene(scene, tracks, ref, run_dir / "consistency")
    selected = min(frames, key=lambda f: abs(f.index - ref.working_frame_index))
    cv2.imwrite(str(run_dir / "source.png"), cv2.cvtColor(selected.image, cv2.COLOR_RGB2BGR))
    render_scene(scene, run_dir / "sim.png")
    save_json(
        run_dir / "reference.json",
        {
            "homography": ref.homography.tolist(),
            "bounds": ref.bounds,
            "meters_per_pixel": ref.meters_per_pixel,
            "working_frame_index": ref.working_frame_index,
            "source": ref.scale_source,
            "reference_id": ref.reference_id,
            "reference_height_m": ref.reference_height_m,
            "reference_material": ref.reference_material,
            "validation": ref.validation,
            "diagnostics": ref.diagnostics,
        },
    )
    return scene, ref, tracks


def prepare_evaluation(scene, client, root, run_dir, workers):
    from splatsam.eval.seal import seal_test
    from splatsam.loop.engine import improve

    family = client.generate(
        ScenarioFamily,
        model="muse-wonder-1.0",
        instructions="Design ONE bounded scenario family. Respect friction .5..1.5, mass .5..2, jitter <=.04m. Do not design held-out outcomes.",
        input="Scene primitive geometry: " + str([(o.shape, o.size) for o in scene.objects if o.graspable]),
        stage="family",
    )
    family, pilots = calibrate(scene, workers, initial=family)
    save_json(run_dir / "pilot.json", pilots)
    baseline = PlannerParams()
    save_json(run_dir / "scene.json", scene)
    save_json(run_dir / "family.json", family)
    baseline.save(run_dir / "baseline_params.json")
    seal_test(scene, family, baseline, root)
    loop = improve(scene, family, baseline, client, run_dir, workers=workers)
    costs = print_cost("baseline and validation loop", phase="B" if scene.source == "live" else "A")
    from splatsam.report import generate_report

    generate_report(
        run_dir,
        scene,
        None,
        loop["history"],
        costs,
        validation_evaluations=loop["validation_evaluations"],
        fresh_validation_simulations=loop["fresh_validation_simulations"],
        stages=[
            {"stage": "perception", "status": "ok"},
            {"stage": "validation-loop", "status": "ok"},
            {"stage": "sealed-test", "status": "awaiting_final_confirmation"},
        ],
    )
    return family, baseline, PlannerParams(**loop["final_params"]), loop


def finish(scene, family, baseline, final, root, run_dir, workers, loop):
    from splatsam.curriculum.sampler import split_scenarios
    from splatsam.eval.seal import run_final
    from splatsam.report import generate_report
    from splatsam.sim.episode import run_episode
    from splatsam.sim.replay import save_replay

    existing = Path(root) / "results/final_test.json"
    if scene.source != "live" and existing.exists():
        result = load_json(existing)
        expected = {
            "scene": digest(scene),
            "family": digest(family),
            "baseline": baseline.hash,
            "final": final.hash,
        }
        if any(result["hashes"].get(k) != v for k, v in expected.items()):
            raise RuntimeError("stored synthetic final result does not match frozen inputs")
        print(
            "Reusing completed synthetic sealed outcomes for artifact generation; no evaluations repeated.",
            flush=True,
        )
    else:
        result = run_final(scene, family, baseline, final, root, run_dir, workers=workers)
    candidates = split_scenarios(family, scene, "train")[:12]
    selected = candidates[0]
    for candidate in candidates:
        if run_episode(scene, candidate, final).success:
            selected = candidate
            break
    replay = save_replay(scene, selected, final, run_dir / "grasp.mp4")
    save_json(run_dir / "replay_result.json", replay)
    save_json(run_dir / "replay_scenario.json", selected)
    save_json(run_dir / "final_test.json", result)
    report = generate_report(
        run_dir,
        scene,
        result,
        loop["history"],
        print_cost("final report", phase="B" if scene.source == "live" else "A"),
        validation_evaluations=loop["validation_evaluations"],
        fresh_validation_simulations=loop["fresh_validation_simulations"],
        stages=[
            {"stage": "video-to-scene", "status": "ok"},
            {"stage": "validation-loop", "status": "ok"},
            {"stage": "sealed-test", "status": "ok"},
        ],
    )
    print("Held-out simulation:", result["comparison"], flush=True)
    return report


def _fingerprint(args):
    relevant = [
        "sim/scene.py",
        "sim/episode.py",
        "planner/grasp.py",
        "planner/params.py",
        "curriculum/sampler.py",
        "recon/primitives.py",
        "sim/fit.py",
        "synth/scene_gen.py",
        "perception/reference.py",
    ]
    return digest(
        {
            "seed": args.seed,
            "config": config(),
            "source": {p: file_digest(ROOT / "src/splatsam" / p) for p in relevant},
            "versions": {
                n: importlib.metadata.version(n) for n in ["mujoco", "numpy", "scipy", "statsmodels"]
            },
        }
    )


def synthetic(args):
    from splatsam.llm.mock import MockClient
    from splatsam.synth.scene_gen import generate

    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root).resolve()
    if not root.is_relative_to((ROOT / "data_synth").resolve()):
        raise ValueError("synthetic CLI root must be under data_synth")
    fingerprint = _fingerprint(args)
    marker = run_dir / "completed.json"
    if marker.exists():
        prior = load_json(marker)
        if prior["fingerprint"] == fingerprint:
            for name, expected in prior["artifacts"].items():
                if file_digest(Path(name)) != expected:
                    raise RuntimeError(f"completed artifact altered: {name}")
            print("Verified cached synthetic run reused; no sealed evaluation repeated.", flush=True)
            print_cost("cache")
            return
        root = root / "revisions" / fingerprint[:12]
    namespace_marker = root / "input_fingerprint.json"
    if (root / "splits/SEALED.lock").exists() and (
        not namespace_marker.exists() or load_json(namespace_marker).get("fingerprint") != fingerprint
    ):
        root = root / "revisions" / fingerprint[:12]
    save_json(root / "input_fingerprint.json", {"fingerprint": fingerprint})
    save_json(run_dir / "active_root.json", {"root": str(root), "fingerprint": fingerprint})
    client = MockClient(
        run_dir,
        fixture_responses={
            PlannerPatch: [
                {"patch": {"grip_safety_factor": v}, "reason": "mock bounded safety-factor proposal"}
                for v in [1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 1.2]
            ]
        },
    )
    bundle = generate(root, args.seed)
    save_json(
        run_dir / "input.json",
        {"source": "synthetic", "bundle": bundle, "video_sha256": file_digest(Path(bundle["video_path"]))},
    )
    fixture = ObjectList.model_validate(load_json(Path(bundle["object_list_path"])))
    scene, reference, tracks = perceive(
        Path(bundle["video_path"]),
        client,
        run_dir,
        fixture=fixture,
        tracks_path=Path(bundle["tracks_path"]),
        source="synthetic",
        seed=args.seed,
    )
    scale_error, size_errors = synthetic_accuracy(scene, reference, tracks, bundle["scene_path"])
    scene = scene.model_copy(update={"scale_error": scale_error})
    save_json(run_dir / "accuracy.json", {"scale_error": scale_error, "size_errors": size_errors})
    print("Synthetic scale error", scale_error, "size errors", size_errors, flush=True)
    print_cost("perception and reconstruction")
    family, baseline, final, loop = prepare_evaluation(scene, client, root, run_dir, args.workers)
    finish(scene, family, baseline, final, root, run_dir, args.workers, loop)
    immutable = [
        run_dir / p
        for p in [
            "scene.json",
            "family.json",
            "baseline_params.json",
            "final_params.json",
            "report.json",
            "rounds.json",
            "final_test.json",
        ]
    ] + [
        root / "splits/SEALED.lock",
        root / "splits/SEALED.manifest.json",
        root / "splits/test.json",
        root / "results/final_test.json",
    ]
    save_json(
        marker,
        {
            "fingerprint": fingerprint,
            "root": str(root),
            "artifacts": {str(p): file_digest(p) for p in immutable},
        },
    )


def cached(args):
    source = ROOT / "demo_data"
    manifest = load_json(source / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_digest(source / name) != expected:
            raise RuntimeError(f"cached demo integrity failed: {name}")
    destination = ROOT / "runs" / args.run_id
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print("Offline synthetic bundle loaded. API/GPU clients disabled.", flush=True)


def live(args):
    from splatsam.llm.client import ModelClient

    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.action is None:
        raise ValueError("live requires explicit --action B1/B2/B3/B4")
    video = args.video
    if video is None and (run_dir / "input.json").exists():
        video = Path(load_json(run_dir / "input.json")["video_path"])
    if video is None:
        raise ValueError("real input video is not ready")
    require_live(args.action, video)
    client = ModelClient(
        run_dir,
        budget=BudgetTracker(ROOT / "runs/meta_budget.json", phase="B"),
        cache_dir=ROOT / ".cache/meta",
        live=True,
    )
    if args.action == "B1":
        if not client.can_dispatch:
            raise RuntimeError("MODEL_API_KEY absent; live SAM verification unavailable")
        evidence = ROOT / "runs/spike-api/observed_fixture.json"
        if not evidence.exists() or load_json(evidence).get("source") != "live_observed_synthetic_spike":
            raise RuntimeError(
                "Run the bounded synthetic scripts/spike_api.py format verification before sending real video"
            )
        scene, _, _ = perceive(
            video,
            client,
            run_dir,
            measured_path=ROOT / "data/measured.yaml",
            reference_config=config().get("live_reference", config()["reference"]),
            source="live",
            seed=args.seed,
        )
        save_json(run_dir / "scene.json", scene)
        save_json(
            run_dir / "input.json",
            {"source": "live", "video_path": str(video.resolve()), "video_sha256": file_digest(video)},
        )
        from splatsam.report import generate_report

        generate_report(
            run_dir,
            scene,
            None,
            [],
            print_cost("B1", phase="B"),
            stages=[{"stage": "perception", "status": "awaiting_user_mask_approval"}],
        )
        print("B1 complete: show overlay.mp4 and WAIT for user mask approval. No seal created.", flush=True)
    elif args.action == "B2":
        scene = SceneSpec.model_validate(load_json(run_dir / "scene.json"))
        prepare_evaluation(scene, client, ROOT / "data", run_dir, args.workers)
        print(
            "B2 complete. No real final test ran. Wait for separate B3 spend or B4 confirmation.", flush=True
        )
    elif args.action == "B3":
        save_json(
            run_dir / "gpu_upgrade_status.json",
            {
                "status": "skipped",
                "reason": "RunPod MCP lifecycle route unavailable",
                "spend_usd": 0,
                "reconstruction": "primitive",
            },
        )
        print(
            "Optional GPU skipped: no RunPod MCP lifecycle route mounted. Primitive scene and evaluations retained.",
            flush=True,
        )
    else:
        scene = SceneSpec.model_validate(load_json(run_dir / "scene.json"))
        family = ScenarioFamily.model_validate(load_json(run_dir / "family.json"))
        finish(
            scene,
            family,
            PlannerParams.load(run_dir / "baseline_params.json"),
            PlannerParams.load(run_dir / "final_params.json"),
            ROOT / "data",
            run_dir,
            args.workers,
            load_json(run_dir / "loop.json"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "handcrafted", "cached", "live"], default="synthetic")
    parser.add_argument("--root", default="data_synth")
    parser.add_argument("--run-id", default="synth-demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark", type=int, default=100)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--action", choices=["B1", "B2", "B3", "B4"])
    args = parser.parse_args()
    if not 1 <= args.benchmark <= 150:
        parser.error("benchmark must be 1..150")
    if Path(args.run_id).name != args.run_id:
        parser.error("run-id must be one directory name")
    try:
        {"synthetic": synthetic, "handcrafted": handcrafted, "cached": cached, "live": live}[args.mode](args)
        return 0
    except Exception as error:  # noqa: BLE001 - top-level diagnostic artifact must survive every stage failure
        # Diagnostics survive failure. No synthetic substitution for a real scene,
        # no automatic seal retry, no fake numeric results.
        run_dir = ROOT / "runs" / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "stage": args.mode,
            "status": "blocked",
            "error_type": type(error).__name__,
            "reason": str(error),
        }
        save_json(run_dir / "pipeline_status.json", status)
        if not (run_dir / "report.json").exists():
            from splatsam.report import generate_report

            generate_report(
                run_dir,
                SceneSpec(objects=[], source="fallback"),
                None,
                [],
                print_cost("blocked"),
                stages=[status],
            )
        print("BLOCKED:", type(error).__name__, str(error), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
