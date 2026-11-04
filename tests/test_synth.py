from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from splatsam.llm.schemas import MaskTrack, SceneSpec
from splatsam.synth import generate

EXPECTED_IDS = ("paper", "remote_1", "remote_2", "ball", "marker_1", "marker_2")


def test_synthetic_metadata_is_deterministic_and_truth_is_separate(tmp_path: Path):
    first = generate(tmp_path, seed=17)
    artifact_paths = {key: Path(value) for key, value in first.items()}

    metadata_before = {key: path.read_bytes() for key, path in artifact_paths.items() if key != "video_path"}
    second = generate(tmp_path, seed=17)
    assert first == second
    for key, original in metadata_before.items():
        assert Path(second[key]).read_bytes() == original


def test_synthetic_tracks_align_to_saved_rgb_and_binary_masks(tmp_path: Path):
    artifacts = generate(tmp_path, seed=42)
    tracks = [
        MaskTrack.model_validate(value) for value in json.loads(Path(artifacts["tracks_path"]).read_text())
    ]
    truth = SceneSpec.model_validate_json(Path(artifacts["scene_path"]).read_text())
    colors = {o.id: np.array(o.rgba[:3]) for o in truth.objects}
    assert [track.id for track in tracks] == list(EXPECTED_IDS)

    for track in tracks:
        for frame in track.frames:
            rgb = iio.imread(tmp_path / "synthetic" / "frames" / f"frame_{frame.frame_index:06d}.png")
            mask = iio.imread(frame.mask_path)
            assert rgb.shape[:2] == mask.shape == (track.image_height, track.image_width)
            assert set(np.unique(mask)).issubset({0, 255})
            ys, xs = np.nonzero(mask)
            assert len(xs) == frame.area
            assert frame.centroid == (float(xs.mean()), float(ys.mean()))
            assert frame.box == (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            # The pixels at the recorded centroid are rendered object/sheet
            # pixels, not an accidental all-zero or hand-only segmentation.
            assert mask[round(frame.centroid[1]), round(frame.centroid[0])] == 255
            color = rgb[round(frame.centroid[1]), round(frame.centroid[0])].astype(float)
            expected = colors[track.id]
            assert np.dot(color, expected) / (np.linalg.norm(color) * np.linalg.norm(expected)) > 0.9
