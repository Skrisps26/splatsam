#!/usr/bin/env python3
"""Export private presentation media from an existing partial live run.

The export consumes saved B1/B2 artifacts only.  It validates the authorized local
video before copying it into the same ignored run directory; it never calls a model,
runs a final evaluation, or substitutes synthetic media.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

from export_ui_events import ensure_snapshot_compatible, export_events
from export_ui_media import (
    _footprints,
    _label,
    _overlay_sequence,
    _reference_outline,
    _relative,
    _resize,
    _scene_media,
    _source_frame,
    _write_image,
    _write_video,
)

from splatsam.llm.schemas import ScenarioFamily, ScenarioSpec, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.utils import ROOT, digest, file_digest, load_json, save_json


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"path is outside {root}: {resolved}") from error
    return resolved


def _private_run(path: Path) -> Path:
    runs = (ROOT / "runs").resolve(strict=True)
    run_dir = _inside(path, runs)
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    return run_dir


def _presentation_dir(run_dir: Path) -> Path:
    destination = run_dir / "presentation"
    destination.mkdir(parents=True, exist_ok=True)
    return _inside(destination, run_dir)


def _authorized_video(run_dir: Path) -> Path:
    input_artifact = load_json(run_dir / "input.json")
    if not isinstance(input_artifact, dict) or input_artifact.get("source") != "live":
        raise ValueError("real presentation export requires input.json source='live'")
    raw_path = input_artifact.get("video_path")
    expected = input_artifact.get("video_sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise TypeError("input.json video_path and video_sha256 must be strings")
    if not Path(raw_path).is_absolute():
        raise ValueError("input.json video_path must be an absolute path below data/videos")
    videos = (ROOT / "data" / "videos").resolve(strict=True)
    video = _inside(Path(raw_path), videos)
    if not video.is_file() or file_digest(video) != expected:
        raise ValueError("authorized live video is missing or does not match input.json video_sha256")
    return video


def _require_saved_track_masks(run_dir: Path, tracks: list[dict[str, Any]]) -> None:
    for track in tracks:
        if not isinstance(track, dict) or not isinstance(track.get("id"), str):
            raise TypeError("tracks.json contains a track with a non-string id")
        frames = track.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"track {track['id']} has no saved masks")
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("mask_path"), str):
                raise TypeError(f"track {track['id']} has a mask path that is not a string")
            _inside(Path(frame["mask_path"]), run_dir)


def _saved_scene_scenario(scene: SceneSpec) -> ScenarioSpec:
    if not scene.objects:
        raise ValueError("scene has no reconstructed objects")
    target = next((obj.id for obj in scene.objects if obj.graspable), scene.objects[0].id)
    return ScenarioSpec(
        seed=scene.seed,
        target_id=target,
        positions={obj.id: obj.position for obj in scene.objects},
        yaws={obj.id: obj.yaw for obj in scene.objects},
        friction_scale=1.0,
        mass_scale=1.0,
    )


def _reference_metadata(
    reference: dict[str, Any], scene: SceneSpec, tracks: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_id = reference.get("reference_id")
    if not isinstance(reference_id, str):
        raise TypeError("reference.json reference_id must be a string")
    reference_object = next(
        (obj for obj in scene.objects if obj.id == reference_id and not obj.graspable), None
    )
    reference_track = next((track for track in tracks if track.get("id") == reference_id), None)
    if reference_object is None or reference_track is None:
        raise ValueError("saved non-graspable reference is not represented by both scene and track artifacts")
    validation = reference.get("validation")
    validation = validation if isinstance(validation, dict) else {}
    width_m = validation.get("reference_width_m")
    height_m = validation.get("reference_height_m")
    if not isinstance(width_m, (int, float)) or not isinstance(height_m, (int, float)):
        raise TypeError("reference dimensions must be numeric")
    source = reference.get("source")
    if not isinstance(source, str):
        raise TypeError("reference.json source must be a string")
    return (
        {
            "name": reference_object.name,
            "width_m": float(width_m),
            "height_m": float(height_m),
            "source": source,
        },
        reference_track,
    )


def _validate_train_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9_999:
        raise ValueError("selected-planner recording seed must be an integer in the train range [0, 9999]")
    return seed


def _cli_train_seed(value: str) -> int:
    try:
        return _validate_train_seed(int(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _missing_selected_planner_replay(seed: int, reason: str) -> dict[str, Any]:
    return {
        "path": None,
        "kind": "video",
        "seed": seed,
        "params_hash": None,
        "provenance": reason,
        "caption": "Selected-planner example unavailable; no evaluation metric is shown.",
        "excluded_from_evaluation": True,
        "status": "missing",
    }


def _selected_planner_replay(run_dir: Path, destination: Path, scene: SceneSpec, seed: int) -> dict[str, Any]:
    """Record one opt-in, unsealed training scenario from frozen saved inputs."""
    family_path = run_dir / "family.json"
    params_path = run_dir / "final_params.json"
    missing = [path.name for path in (family_path, params_path) if not path.is_file()]
    if missing:
        return _missing_selected_planner_replay(
            seed, f"selected-planner example unavailable: saved {', '.join(missing)} is absent"
        )

    family = ScenarioFamily.model_validate(load_json(family_path))
    params = PlannerParams.load(params_path)
    input_hashes = {
        "scene_sha256": file_digest(run_dir / "scene.json"),
        "family_sha256": file_digest(family_path),
        "params_sha256": params.hash,
        "seed_sha256": digest({"seed": seed}),
    }
    recording_key = digest(input_hashes)
    output = destination / "selected_planner_replay.mp4"
    state_path = destination / "selected_planner_replay.json"
    if output.is_file() and state_path.is_file():
        try:
            state = load_json(state_path)
        except (OSError, ValueError):
            state = None
        asset = state.get("asset") if isinstance(state, dict) else None
        if (
            isinstance(state, dict)
            and state.get("recording_key") == recording_key
            and state.get("output_sha256") == file_digest(output)
            and isinstance(asset, dict)
            and asset.get("path") == _relative(output, run_dir)
            and asset.get("kind") == "video"
            and asset.get("seed") == seed
            and asset.get("params_hash") == params.hash
            and asset.get("excluded_from_evaluation") is True
            and asset.get("status") == "ready"
        ):
            return asset

    # Keep the physical recording opt-in: the ordinary real-media export only renders saved static artifacts.
    from splatsam.curriculum.sampler import sample_scenario
    from splatsam.sim.replay import save_replay

    scenario = sample_scenario(family, seed, scene)
    recorded = destination / ".selected_planner_replay.recording.mp4"
    faststart = destination / ".selected_planner_replay.faststart.mp4"
    try:
        save_replay(scene, scenario, params, recorded)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(recorded),
                "-map",
                "0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-metadata",
                "title=Selected-planner example",
                "-metadata",
                "comment=Illustrative video-derived scene simulation excluded from evaluation",
                str(faststart),
            ],
            check=True,
        )
        faststart.replace(output)
    finally:
        recorded.unlink(missing_ok=True)
        faststart.unlink(missing_ok=True)

    asset = {
        "path": _relative(output, run_dir),
        "kind": "video",
        "seed": seed,
        "params_hash": params.hash,
        "provenance": (
            "Opt-in selected-planner example simulated from the frozen video-derived scene and "
            "saved selected planner parameters; not real robot footage and excluded from evaluation."
        ),
        "caption": "Selected-planner example • illustrative simulation excluded from evaluation",
        "excluded_from_evaluation": True,
        "status": "ready",
    }
    save_json(
        state_path,
        {
            "schema_version": 1,
            "recording_key": recording_key,
            "input_hashes": input_hashes,
            "output_sha256": file_digest(output),
            "asset": asset,
        },
    )
    return asset


def export(run_dir: Path, *, record_grasp: bool = False, seed: int = 0) -> dict[str, Any]:
    seed = _validate_train_seed(seed)
    run_dir = _private_run(run_dir)
    video = _authorized_video(run_dir)
    ensure_snapshot_compatible(run_dir, run_dir)
    destination = _presentation_dir(run_dir)
    reference = load_json(run_dir / "reference.json")
    tracks = load_json(run_dir / "tracks.json")
    if not isinstance(reference, dict) or not isinstance(tracks, list):
        raise TypeError("reference.json and tracks.json must contain object/list JSON artifacts")
    _require_saved_track_masks(run_dir, tracks)
    scene = SceneSpec.model_validate(load_json(run_dir / "scene.json"))
    if scene.source != "live":
        raise ValueError("real presentation export requires a live SceneSpec")
    reference_metadata, reference_track = _reference_metadata(reference, scene, tracks)
    source, frame_index = _source_frame(run_dir, reference, run_dir)

    copied_video = destination / "source.mp4"
    shutil.copyfile(video, copied_video)
    original = destination / "original_frame.png"
    _write_image(
        original,
        _label(
            _resize(source),
            "Uploaded-video working frame",
            f"Recorded frame {frame_index} • saved live artifact",
        ),
    )
    reveals = _overlay_sequence(source, tracks, frame_index, destination)
    reference_path = destination / "reference_scale.png"
    _write_image(
        reference_path,
        _reference_outline(
            source,
            reference_track,
            frame_index,
            reference,
            title=f"Reference outline • {reference_metadata['name']}",
            subtitle=(
                f"{reference_metadata['width_m']:.3f} m × {reference_metadata['height_m']:.3f} m"
                f" • {reference_metadata['source']}"
            ),
        ),
    )
    footprint_path = destination / "observed_footprints.png"
    _write_image(footprint_path, _footprints(tracks, frame_index, reference))
    scenario = _saved_scene_scenario(scene)
    scene_path = destination / "scene_upright.png"
    orbit_path = destination / "scene_orbit.mp4"
    scene_still, orbit_frames = _scene_media(
        scene,
        scenario,
        title="Video-derived reconstructed scene",
        subtitle="Saved live SceneSpec • static simulator render, not a real robot",
    )
    _write_image(scene_path, scene_still)
    _write_video(orbit_path, orbit_frames)

    selected_planner_replay = _missing_selected_planner_replay(
        seed,
        "selected-planner example was not recorded; pass --record-grasp to create an illustrative excluded clip",
    )
    if record_grasp:
        selected_planner_replay = _selected_planner_replay(run_dir, destination, scene, seed)

    assets = {
        "original_frame": {
            "path": _relative(original, run_dir),
            "kind": "image",
            "provenance": "saved live working frame",
        },
        "source_video": {
            "path": _relative(copied_video, run_dir),
            "kind": "video",
            "provenance": "authorized uploaded video copied only into this private run",
        },
        "mask_reveals": {
            "path": [_relative(path, run_dir) for path in reveals],
            "kind": "image_sequence",
            "provenance": "saved persistent live mask tracks",
        },
        "reference_scale": {
            "path": _relative(reference_path, run_dir),
            "kind": "image",
            "provenance": "saved live reference mask and homography",
        },
        "observed_footprints": {
            "path": _relative(footprint_path, run_dir),
            "kind": "image",
            "provenance": "saved live masks warped through the saved reference homography",
        },
        "scene_upright": {
            "path": _relative(scene_path, run_dir),
            "kind": "image",
            "provenance": "saved live SceneSpec rendered statically with MuJoCo",
        },
        "scene_orbit": {
            "path": _relative(orbit_path, run_dir),
            "kind": "video",
            "provenance": "static orbit of the video-derived reconstructed scene",
        },
        "selected_planner_replay": selected_planner_replay,
    }
    reveal_frames = [
        {
            "stage": "understand",
            "kind": "image",
            "path": _relative(original, run_dir),
            "label": "Uploaded-video working frame",
        },
        *[
            {
                "stage": "track",
                "kind": "image",
                "path": _relative(path, run_dir),
                "label": f"Persistent mask reveal {index + 1}: {track['name']} ({track['id']})",
            }
            for index, (path, track) in enumerate(zip(reveals, tracks, strict=True))
        ],
        {
            "stage": "scale",
            "kind": "image",
            "path": _relative(reference_path, run_dir),
            "label": f"Reference outline: {reference_metadata['name']} ({reference_metadata['width_m']:.3f} m × {reference_metadata['height_m']:.3f} m)",
        },
        {
            "stage": "build",
            "kind": "image",
            "path": _relative(footprint_path, run_dir),
            "label": "Metric footprints from saved masks",
        },
        {
            "stage": "build",
            "kind": "image",
            "path": _relative(scene_path, run_dir),
            "label": "Upright video-derived reconstructed scene",
        },
        {
            "stage": "build",
            "kind": "video",
            "path": _relative(orbit_path, run_dir),
            "label": "Static reconstructed-scene orbit (not real robot footage)",
        },
    ]
    if selected_planner_replay["status"] == "ready":
        reveal_frames.append(
            {
                "stage": "build",
                "kind": "video",
                "path": selected_planner_replay["path"],
                "label": selected_planner_replay["caption"],
            }
        )
    missing = (
        []
        if (run_dir / "final_test.json").is_file()
        else [{"asset": "final_test", "reason": "pending explicit final-test authorization"}]
    )
    if selected_planner_replay["status"] == "missing":
        missing.append({"asset": "selected_planner_replay", "reason": selected_planner_replay["provenance"]})

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "live",
        "source_mode": "live",
        "description": "Private recorded reconstruction from saved artifacts of the authorized uploaded video.",
        "assets": assets,
        "reveal_frames": reveal_frames,
        "reference": reference_metadata,
        "paired": {
            "status": "missing",
            "reason": "no matching baseline/selected-planner training videos have been recorded",
        },
        "round_clips": {},
        "missing": missing,
        "provenance": {
            "input": "authorized uploaded video and saved persistent live masks",
            "reference": "saved live reference homography; presentation timing is not measured runtime",
            "scene": "saved video-derived SceneSpec rendered statically with MuJoCo; not a real robot",
            "final_test": "read only from a saved final_test.json; absent held-out results remain pending",
        },
    }
    save_json(destination / "media.json", manifest)
    export_events(run_dir, run_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="private live run below runs/")
    parser.add_argument(
        "--record-grasp",
        action="store_true",
        help="record one illustrative selected-planner train scenario; excluded from every evaluation",
    )
    parser.add_argument(
        "--seed", type=_cli_train_seed, default=0, help="illustrative train seed in [0, 9999] (default: 0)"
    )
    args = parser.parse_args()
    try:
        manifest = export(args.run_dir, record_grasp=args.record_grasp, seed=args.seed)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Exported {len(manifest['assets'])} private live presentation asset groups to {args.run_dir / 'presentation'}"
    )


if __name__ == "__main__":
    main()
