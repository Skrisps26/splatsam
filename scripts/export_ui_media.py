#!/usr/bin/env python3
"""Export UI-only presentation media from an already completed synthetic replay.

This tool only loads saved masks, scene specifications, and recorded qpos samples.
It never imports or calls episode evaluation and never advances MuJoCo simulation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from diskcache import Cache

from splatsam import rendering
from splatsam.curriculum.sampler import split_scenarios
from splatsam.llm.schemas import ScenarioFamily, ScenarioSpec, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.sim.scene import build_model
from splatsam.utils import ROOT, digest, load_json, save_json

import mujoco  # isort: skip

WIDTH = 480
HEIGHT = 360
FPS = 12
COLORS_BGR = [
    (114, 188, 255),
    (120, 231, 150),
    (235, 150, 255),
    (86, 216, 255),
    (170, 140, 255),
    (120, 240, 230),
]


def _relative(path: Path, bundle: Path) -> str:
    return path.relative_to(bundle).as_posix()


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def _resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 44), (12, 18, 29), -1)
    cv2.putText(output, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (238, 243, 250), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(
            output, subtitle, (12, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (172, 191, 211), 1, cv2.LINE_AA
        )
    return output


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise ValueError(f"no frames for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(path),
        frames,
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )


def _frame_for_track(track: dict[str, Any], frame_index: int) -> dict[str, Any]:
    return min(track["frames"], key=lambda frame: abs(int(frame["frame_index"]) - frame_index))


def _mask(frame: dict[str, Any]) -> np.ndarray:
    mask = cv2.imread(frame["mask_path"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(frame["mask_path"])
    return mask > 0


def _source_frame(run_dir: Path, reference: dict[str, Any], bundle: Path) -> tuple[np.ndarray, int]:
    frame_index = int(reference.get("working_frame_index", 0))
    candidate = run_dir / "frames" / f"frame_{frame_index:06d}.png"
    source = candidate if candidate.is_file() else bundle / "source.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    return image, frame_index


def _overlay_sequence(
    source: np.ndarray, tracks: list[dict[str, Any]], frame_index: int, destination: Path
) -> list[Path]:
    overlay = source.copy()
    outputs = []
    for index, track in enumerate(tracks):
        frame = _frame_for_track(track, frame_index)
        mask = _mask(frame)
        if mask.shape != overlay.shape[:2]:
            raise ValueError(f"mask dimensions for {track['id']} do not match source frame")
        color = np.array(COLORS_BGR[index % len(COLORS_BGR)], dtype=np.uint8)
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.44 + color * 0.56).astype(np.uint8)
        x, y = (round(value) for value in frame["centroid"])
        caption = f"{track['name']}  |  {track['id']}"
        cv2.putText(
            overlay, caption, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            caption,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            tuple(int(v) for v in color),
            1,
            cv2.LINE_AA,
        )
        out = destination / f"mask_reveal_{index + 1:02d}_{track['id']}.png"
        _write_image(
            out,
            _label(
                _resize(overlay),
                "Persistent mask tracks",
                f"Reveal {index + 1}/{len(tracks)} • {track['id']}",
            ),
        )
        outputs.append(out)
    return outputs


def _reference_outline(
    source: np.ndarray, paper_track: dict[str, Any], frame_index: int, reference: dict[str, Any]
) -> np.ndarray:
    output = source.copy()
    mask = _mask(_frame_for_track(paper_track, frame_index)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("saved paper mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    if len(polygon) != 4:
        polygon = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(output, [polygon], True, (90, 240, 180), 3, cv2.LINE_AA)
    meters_per_pixel = float(reference["meters_per_pixel"])
    bar_px = round(0.10 / meters_per_pixel)
    origin = (45, output.shape[0] - 52)
    cv2.line(output, origin, (origin[0] + bar_px, origin[1]), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(
        output,
        "10 cm",
        (origin[0], origin[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return _label(
        _resize(output),
        "Reference outline • scale locked",
        f"Synthetic A4 reference sheet • {meters_per_pixel * 1000:.1f} mm / px",
    )


def _footprints(tracks: list[dict[str, Any]], frame_index: int, reference: dict[str, Any]) -> np.ndarray:
    min_x, min_y, max_x, max_y = (float(value) for value in reference["bounds"])
    meters_per_pixel = float(reference["meters_per_pixel"])
    width = max(1, math.ceil((max_x - min_x) / meters_per_pixel))
    height = max(1, math.ceil((max_y - min_y) / meters_per_pixel))
    transform = np.array(
        [
            [1.0 / meters_per_pixel, 0.0, -min_x / meters_per_pixel],
            [0.0, 1.0 / meters_per_pixel, -min_y / meters_per_pixel],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ) @ np.asarray(reference["homography"], dtype=np.float64)
    canvas = np.full((height, width, 3), (22, 31, 42), dtype=np.uint8)
    for index, track in enumerate(tracks):
        observed = _mask(_frame_for_track(track, frame_index)).astype(np.uint8) * 255
        footprint = cv2.warpPerspective(observed, transform, (width, height), flags=cv2.INTER_NEAREST)
        color = np.array(COLORS_BGR[index % len(COLORS_BGR)], dtype=np.uint8)
        canvas[footprint > 0] = color
        moments = cv2.moments(footprint)
        if moments["m00"]:
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            cv2.putText(
                canvas, track["id"], center, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 248, 250), 1, cv2.LINE_AA
            )
    bar = round(0.10 / meters_per_pixel)
    cv2.line(canvas, (18, height - 20), (18 + bar, height - 20), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(
        canvas, "10 cm", (18, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
    )
    return _label(
        _resize(canvas), "Bird's-eye observed footprints", "Warped saved masks on the locked reference raster"
    )


def _camera(azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (0.0, 0.0, 0.02)
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    return camera


def _scene_media(scene: SceneSpec, scenario: ScenarioSpec) -> tuple[np.ndarray, list[np.ndarray]]:
    model = build_model(scene, scenario)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    still_camera = _camera(42, -28, 0.88)
    orbit_frames = []
    with rendering.renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=still_camera)
        still = renderer.render().copy()
        for azimuth in np.linspace(8, 368, 48, endpoint=False):
            renderer.update_scene(data, camera=_camera(float(azimuth), -27, 0.88))
            orbit_frames.append(
                _label(
                    renderer.render().copy(),
                    "Reconstructed primitive scene",
                    "Static SceneSpec render • no simulation steps",
                )
            )
    return _label(
        still, "Reconstructed primitive scene", "Upright SceneSpec render • synthetic geometry"
    ), orbit_frames


def _recorded_replay(scene: SceneSpec, scenario: ScenarioSpec, replay: dict[str, Any]) -> list[np.ndarray]:
    trace = replay.get("trace", [])
    if not trace:
        return []
    model = build_model(scene, scenario)
    data = mujoco.MjData(model)
    target = scenario.positions[scenario.target_id]
    camera = _camera(90, -60, 0.65)
    camera.lookat[:] = (target[0], target[1], 0.04)
    indices = np.linspace(0, len(trace) - 1, min(120, len(trace)), dtype=int)
    frames = []
    with rendering.renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for index in indices:
            sample = trace[int(index)]
            data.qpos[:] = json.loads(sample["qpos_json"])
            data.mocap_pos[0] = [sample["hand_x"], sample["hand_y"], sample["hand_z"]]
            yaw = float(sample["hand_yaw"])
            data.mocap_quat[0] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(
                _label(
                    renderer.render().copy(),
                    "Recorded final replay • seed 0",
                    f"Saved qpos trace • {sample['phase']} • final parameters",
                )
            )
    return frames


def _cached_pair(cache_dir: Path) -> tuple[dict[str, Any] | None, str]:
    if not cache_dir.exists():
        return None, "episode cache is absent"
    cache = Cache(str(cache_dir))
    try:
        results = []
        for value in cache:
            candidate = cache.get(value)
            if isinstance(candidate, dict) and {"seed", "success", "trace"} <= candidate.keys():
                results.append(candidate)
    finally:
        cache.close()
    traced = [result for result in results if result.get("trace")]
    if not traced:
        return None, "cached EpisodeResults contain no recorded trace samples"
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for result in traced:
        by_seed.setdefault(int(result["seed"]), []).append(result)
    for seed, values in sorted(by_seed.items()):
        failures = [value for value in values if not value.get("success")]
        successes = [value for value in values if value.get("success")]
        if failures and successes:
            return {"seed": seed, "baseline": failures[0], "final": successes[0]}, ""
    return None, "no same-seed failure/success trace pair is present in the episode cache"


def _training_pair_from_cache(
    cache_dir: Path, scene: SceneSpec, family: ScenarioFamily, baseline: PlannerParams, final: PlannerParams
) -> tuple[ScenarioSpec | None, str]:
    """Select only a cached train seed with a baseline failure and final success."""
    if not cache_dir.exists():
        return None, "episode cache is absent"
    cache = Cache(str(cache_dir))
    try:
        for scenario in split_scenarios(family, scene, "train"):

            def cached(params: PlannerParams, scenario=scenario) -> dict[str, Any] | None:
                key = digest(
                    {
                        "scene": scene.model_dump(mode="json"),
                        "scenario": scenario.model_dump(mode="json"),
                        "params": params.to_dict(),
                        "physics": "parallel-jaw-v1",
                    }
                )
                result = cache.get(key)
                return result if isinstance(result, dict) else None

            baseline_result = cached(baseline)
            final_result = cached(final)
            if (
                baseline_result
                and final_result
                and not baseline_result.get("success")
                and final_result.get("success")
            ):
                return scenario, ""
    finally:
        cache.close()
    return None, "no cached train seed has baseline failure and final success"


def _export_training_pair(
    destination: Path,
    cache_dir: Path,
    scene: SceneSpec,
    family: ScenarioFamily,
    baseline: PlannerParams,
    final: PlannerParams,
    rounds: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Opt-in only: make illustrative train-seed recordings outside reported metrics."""
    scenario, reason = _training_pair_from_cache(cache_dir, scene, family, baseline, final)
    if scenario is None:
        return None, reason
    # Import lazily so the default exporter cannot execute a physical episode.
    from splatsam.sim.replay import save_replay

    baseline_temp = destination / ".paired_baseline.tmp.mp4"
    final_temp = destination / ".paired_final.tmp.mp4"
    baseline_path = destination / "paired_training_baseline.mp4"
    final_path = destination / "paired_training_final.mp4"
    try:
        baseline_result = save_replay(scene, scenario, baseline, baseline_temp)
        final_result = save_replay(scene, scenario, final, final_temp)
        if baseline_result.success or not final_result.success:
            raise RuntimeError("new recording no longer matches selected cached baseline/final outcomes")
        baseline_temp.replace(baseline_path)
        final_temp.replace(final_path)
    finally:
        baseline_temp.unlink(missing_ok=True)
        final_temp.unlink(missing_ok=True)

    current = baseline
    round_clips: dict[str, dict[str, Any]] = {}
    for row in rounds:
        if not row.get("accepted"):
            continue
        candidate = current.patch(row["patch"])
        if candidate.hash != row["candidate_hash"]:
            raise ValueError(f"round {row['round']} does not apply sequentially from the saved baseline")
        temporary = destination / f".round_{row['round']}.tmp.mp4"
        output = destination / f"paired_training_round_{row['round']}.mp4"
        try:
            result = save_replay(scene, scenario, candidate, temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        round_clips[str(row["round"])] = {
            "path": output,
            "seed": scenario.seed,
            "params_hash": candidate.hash,
            "params": candidate.to_dict(),
            "success": result.success,
            "provenance": "New illustrative synthetic train-seed recording; excluded from report metrics and sealed evaluation.",
        }
        current = candidate
    return {
        "seed": scenario.seed,
        "baseline": baseline_path,
        "final": final_path,
        "round_clips": round_clips,
        "provenance": (
            "New illustrative synthetic train-seed recordings selected from cached outcomes; "
            "excluded from report metrics and sealed evaluation."
        ),
    }, ""


def export(run_dir: Path, bundle: Path, *, paired_training_replay: bool = False) -> dict[str, Any]:
    if not (run_dir / "completed.json").is_file():
        raise ValueError("media export only accepts a completed run directory")
    report = load_json(run_dir / "report.json")
    if report.get("mode") != "synthetic":
        raise ValueError("only synthetic media may be written to the publishable demo bundle")
    destination = bundle / "presentation"
    destination.mkdir(parents=True, exist_ok=True)
    reference = load_json(run_dir / "reference.json")
    tracks = load_json(run_dir / "tracks.json")
    scene = SceneSpec.model_validate(load_json(bundle / "scene.json"))
    scenario = ScenarioSpec.model_validate(load_json(run_dir / "replay_scenario.json"))
    replay = load_json(bundle / "replay_result.json")
    source, frame_index = _source_frame(run_dir, reference, bundle)
    input_artifact = load_json(run_dir / "input.json")
    source_video = Path(input_artifact["bundle"]["video_path"])
    if input_artifact.get("source") != "synthetic" or not source_video.is_file():
        raise ValueError("only an existing synthetic source video may be published")
    published_source_video = destination / "source.mp4"
    shutil.copyfile(source_video, published_source_video)

    original = destination / "original_frame.png"
    _write_image(
        original,
        _label(
            _resize(source), "Original synthetic input frame", f"Recorded frame {frame_index} • no live video"
        ),
    )
    reveals = _overlay_sequence(source, tracks, frame_index, destination)
    paper = next(track for track in tracks if track["id"] == "paper")
    reference_path = destination / "reference_scale.png"
    _write_image(reference_path, _reference_outline(source, paper, frame_index, reference))
    footprint_path = destination / "observed_footprints.png"
    _write_image(footprint_path, _footprints(tracks, frame_index, reference))
    scene_still, orbit_frames = _scene_media(scene, scenario)
    scene_path = destination / "scene_upright.png"
    _write_image(scene_path, scene_still)
    orbit_path = destination / "scene_orbit.mp4"
    _write_video(orbit_path, orbit_frames)
    replay_path = destination / "recorded_final_replay.mp4"
    recorded_frames = _recorded_replay(scene, scenario, replay)
    if recorded_frames:
        _write_video(replay_path, recorded_frames)

    _cached_pair_value, cached_pair_reason = _cached_pair(run_dir / "episode_cache")
    training_pair = None
    training_pair_reason = ""
    if paired_training_replay:
        family = ScenarioFamily.model_validate(load_json(run_dir / "family.json"))
        baseline = PlannerParams.load(run_dir / "baseline_params.json")
        final = PlannerParams.load(run_dir / "final_params.json")
        training_pair, training_pair_reason = _export_training_pair(
            destination,
            run_dir / "episode_cache",
            scene,
            family,
            baseline,
            final,
            load_json(run_dir / "rounds.json"),
        )
    missing = []
    if training_pair is None:
        reason = training_pair_reason if paired_training_replay else cached_pair_reason
        missing.append({"asset": "paired_baseline_final", "reason": reason})
    if not recorded_frames:
        missing.append({"asset": "recorded_final_replay", "reason": "replay_result.json has no qpos trace"})

    assets = {
        "original_frame": {
            "path": _relative(original, bundle),
            "kind": "image",
            "provenance": "saved synthetic frame",
        },
        "source_video": {
            "path": _relative(published_source_video, bundle),
            "kind": "video",
            "provenance": "unsegmented saved synthetic input video; copied without re-encoding",
        },
        "mask_reveals": {
            "path": [_relative(path, bundle) for path in reveals],
            "kind": "image_sequence",
            "provenance": "saved persistent masks",
        },
        "reference_scale": {
            "path": _relative(reference_path, bundle),
            "kind": "image",
            "provenance": "saved paper mask and locked synthetic A4 homography",
        },
        "observed_footprints": {
            "path": _relative(footprint_path, bundle),
            "kind": "image",
            "provenance": "saved observed masks warped through saved homography",
        },
        "scene_upright": {
            "path": _relative(scene_path, bundle),
            "kind": "image",
            "provenance": "SceneSpec renderer with mj_forward only",
        },
        "scene_orbit": {
            "path": _relative(orbit_path, bundle),
            "kind": "video",
            "provenance": "SceneSpec renderer with mj_forward only",
        },
        "sim_fallback": {
            "path": "sim.png",
            "kind": "image",
            "provenance": "existing bundled synthetic renderer fallback",
        },
        "recorded_final_replay": {
            "path": _relative(replay_path, bundle),
            "kind": "video",
            "provenance": "saved seed-0 final qpos trace",
        }
        if recorded_frames
        else None,
        "paired_training_baseline": {
            "path": _relative(training_pair["baseline"], bundle),
            "kind": "video",
            "provenance": training_pair["provenance"],
        }
        if training_pair
        else None,
        "paired_training_final": {
            "path": _relative(training_pair["final"], bundle),
            "kind": "video",
            "provenance": training_pair["provenance"],
        }
        if training_pair
        else None,
        "grasp_fallback": {
            "path": "grasp.mp4",
            "kind": "video",
            "provenance": "existing bundled replay fallback",
        },
    }
    assets = {name: asset for name, asset in assets.items() if asset is not None}
    reveal_frames = [
        {
            "stage": "understand",
            "kind": "image",
            "path": _relative(original, bundle),
            "label": "Original synthetic input frame",
        },
        *[
            {
                "stage": "track",
                "kind": "image",
                "path": _relative(path, bundle),
                "label": f"Persistent mask reveal {index + 1}: {tracks[index]['name']} ({tracks[index]['id']})",
            }
            for index, path in enumerate(reveals)
        ],
        {
            "stage": "scale",
            "kind": "image",
            "path": _relative(reference_path, bundle),
            "label": "Synthetic A4 reference outline and locked scale",
        },
        {
            "stage": "build",
            "kind": "image",
            "path": _relative(footprint_path, bundle),
            "label": "Bird's-eye observed footprints",
        },
        {
            "stage": "build",
            "kind": "image",
            "path": _relative(scene_path, bundle),
            "label": "Upright reconstructed primitive scene",
        },
        {
            "stage": "build",
            "kind": "video",
            "path": _relative(orbit_path, bundle),
            "label": "Static reconstructed-scene orbit",
        },
    ]
    if recorded_frames:
        reveal_frames.append(
            {
                "stage": "sealed",
                "kind": "video",
                "path": _relative(replay_path, bundle),
                "label": "Recorded final replay, seed 0",
            }
        )
    if training_pair:
        for number, clip in training_pair["round_clips"].items():
            reveal_frames.append(
                {
                    "stage": "tune",
                    "kind": "video",
                    "path": _relative(clip["path"], bundle),
                    "label": f"Illustrative training seed {clip['seed']} • accepted round {number}",
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "synthetic",
        "description": "Presentation-only media derived from saved synthetic artifacts. No real-video claim or new simulation run.",
        "assets": assets,
        "reveal_frames": reveal_frames,
        "provenance": {
            "input": "saved synthetic source frame and persistent mock-perception masks",
            "reference": "saved synthetic A4 paper homography; reconstructed presentation timing is not measured runtime",
            "scene": "existing SceneSpec and saved replay scenario rendered with mujoco.mj_forward only",
            "replay": "saved replay_result qpos trace; no episode execution or evaluation",
        },
        "missing": missing,
    }
    if training_pair:
        manifest["paired"] = {
            "status": "available",
            "seed": training_pair["seed"],
            "baseline": {
                "path": _relative(training_pair["baseline"], bundle),
                "params_hash": PlannerParams.load(run_dir / "baseline_params.json").hash,
            },
            "final": {
                "path": _relative(training_pair["final"], bundle),
                "params_hash": PlannerParams.load(run_dir / "final_params.json").hash,
            },
            "provenance": training_pair["provenance"],
        }
        manifest["round_clips"] = {
            number: {**clip, "path": _relative(clip["path"], bundle)}
            for number, clip in training_pair["round_clips"].items()
        }
    else:
        manifest["paired"] = {
            "status": "missing",
            "reason": training_pair_reason if paired_training_replay else cached_pair_reason,
            "fallback": "grasp.mp4",
        }
    save_json(destination / "media.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "synth-demo")
    parser.add_argument("--bundle", type=Path, default=ROOT / "demo_data")
    parser.add_argument(
        "--paired-training-replay",
        action="store_true",
        help="opt in to new illustrative train-seed recordings; never used for report metrics or sealed evaluation",
    )
    args = parser.parse_args()
    manifest = export(
        args.run_dir.resolve(), args.bundle.resolve(), paired_training_replay=args.paired_training_replay
    )
    print(f"Exported {len(manifest['assets'])} presentation asset groups to {args.bundle / 'presentation'}")


if __name__ == "__main__":
    main()
