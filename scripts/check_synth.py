#!/usr/bin/env python3
"""Generate video and independently measure mask/homography accuracy."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from splatsam.llm.schemas import ObjectList, SceneSpec
from splatsam.utils import load_json, save_json


def check(root: Path, seed=42):
    from splatsam.perception.frames import sample_frames
    from splatsam.perception.reference import find_reference
    from splatsam.perception.sam import MockSamClient, save_overlays, stability_gate
    from splatsam.synth.scene_gen import generate

    bundle = generate(root, seed=seed)
    objects = ObjectList.model_validate(load_json(Path(bundle["object_list_path"])))
    truth = SceneSpec.model_validate(load_json(Path(bundle["scene_path"])))
    frames = sample_frames(Path(bundle["video_path"]), out_dir=root / "sampled")
    tracks = MockSamClient(Path(bundle["tracks_path"])).track(Path(bundle["video_path"]), objects)
    for track in tracks:
        ok, diagnostics = stability_gate(track)
        if not ok:
            raise AssertionError(f"{track.id}: {diagnostics}")
    reference = find_reference(tracks, frames)
    recovered = {}
    centers = {}
    errors = {}
    for obj in truth.objects:
        if not obj.graspable:
            continue
        track = next(t for t in tracks if t.id == obj.id)
        sample = min(track.frames, key=lambda f: abs(f.frame_index - reference.working_frame_index))
        mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        warped = reference.warp_mask(mask)
        contours, _ = cv2.findContours(
            (warped > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        _center, size, _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        measured = np.sort(np.array(size) * reference.meters_per_pixel)
        expected = np.sort(np.array(obj.size[:2]))
        relative = np.abs(measured - expected) / expected
        recovered[obj.id] = measured.tolist()
        errors[obj.id] = relative.tolist()
        centers[obj.id] = reference.pixel_to_metric(np.array([sample.centroid], dtype=float))[0]
        if np.max(relative) > 0.10:
            raise AssertionError(f"{obj.id} size error {relative}: recovered {measured}, expected {expected}")
    remotes = [o for o in truth.objects if o.id.startswith("remote")]
    if len(remotes) < 2:
        raise AssertionError("two independent scale anchors required")
    a, b = remotes[:2]
    expected = np.linalg.norm(np.array(a.position[:2]) - np.array(b.position[:2]))
    measured = np.linalg.norm(centers[a.id] - centers[b.id])
    scale_error = abs(measured - expected) / expected
    if scale_error >= 0.05:
        raise AssertionError(f"independent scale error {scale_error}")
    overlay = save_overlays(Path(bundle["video_path"]), tracks, root / "overlays")
    result = {
        "scale_error": float(scale_error),
        "size_errors": errors,
        "recovered_sizes_m": recovered,
        "scale_source": reference.scale_source,
        "reference_validation": reference.validation,
        "working_frame_index": reference.working_frame_index,
        "overlay": str(overlay),
        "bundle": bundle,
    }
    save_json(root / "checks.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_synth/perception-check"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    check(args.root, args.seed)
