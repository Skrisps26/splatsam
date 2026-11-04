from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from splatsam.llm.schemas import MaskFrame, MaskTrack, ObjectInfo, ObjectList
from splatsam.perception.reference import ReferenceResult
from splatsam.recon import reconstruct


_SIZE = (240, 320)


def _reference() -> ReferenceResult:
    height, width = _SIZE
    return ReferenceResult(
        homography=np.eye(3),
        image_to_raster=np.eye(3),
        meters_per_pixel=0.001,
        bounds=(0.0, 0.0, width * 0.001, height * 0.001),
        working_frame_index=7,
        scale_source="paper_homography",
        validation={"aspect_ratio_error": 0.01},
        _raster_shape=_SIZE,
    )


def _track(tmp_path: Path, identifier: str, mask: np.ndarray, *, index: int = 7) -> MaskTrack:
    path = tmp_path / f"{identifier}-{index}.png"
    imageio.imwrite(path, (mask.astype(np.uint8) * 255))
    ys, xs = np.nonzero(mask)
    frame = MaskFrame(
        frame_index=index,
        timestamp_s=float(index),
        mask_path=str(path),
        centroid=(float(xs.mean()), float(ys.mean())),
        area=int(len(xs)),
        box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
    )
    return MaskTrack(id=identifier, name=identifier.replace("_", " "), frames=[frame], image_width=_SIZE[1], image_height=_SIZE[0], total_frames=8)


def _rectangle(x: int, y: int, width: int, height: int) -> np.ndarray:
    mask = np.zeros(_SIZE, dtype=np.uint8)
    mask[y : y + height, x : x + width] = 1
    return mask


def _circle(x: int, y: int, radius: int) -> np.ndarray:
    mask = np.zeros(_SIZE, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 1, -1)
    return mask


def test_analytic_masks_classify_primitives_and_dimension_gates(tmp_path: Path) -> None:
    tracks = [
        _track(tmp_path, "box", _rectangle(10, 20, 60, 30)),
        _track(tmp_path, "sphere", _circle(130, 50, 20)),
        _track(tmp_path, "cylinder", _circle(200, 50, 20)),
        _track(tmp_path, "marker", _rectangle(20, 130, 90, 12)),
        _track(tmp_path, "tiny", _rectangle(180, 150, 10, 10)),
    ]
    objects = ObjectList(
        objects=[
            ObjectInfo(id="box", name="plastic box", material_guess="plastic", approx_size_cm=(6, 3, 2)),
            ObjectInfo(id="sphere", name="rubber ball", material_guess="rubber", approx_size_cm=(4, 4, 4)),
            ObjectInfo(id="cylinder", name="upright cylinder", material_guess="plastic", approx_size_cm=(4, 4, 8)),
            ObjectInfo(id="marker", name="thin marker", material_guess="plastic", approx_size_cm=(9, 1.2, 1.6)),
            ObjectInfo(id="tiny", name="tiny plastic square", material_guess="plastic", approx_size_cm=(1, 1, 1)),
        ]
    )

    scene = reconstruct(tracks, objects, _reference(), seed=99)
    result = {item.id: item for item in scene.objects}
    assert {item.id: item.shape for item in scene.objects} == {
        "box": "box",
        "sphere": "sphere",
        "cylinder": "cylinder",
        "marker": "horizontal_cylinder",
        "tiny": "box",
    }
    assert result["sphere"].size[0] == result["sphere"].size[1]
    assert result["marker"].size[0] > 0.08
    assert np.isclose(result["marker"].size[1], result["marker"].size[2], atol=0.002)
    assert not result["tiny"].graspable
    assert "clutter_dimension_gate" in result["tiny"].reconstruction_source
    assert scene.scale_source == "paper_homography"
    assert scene.scale_error is None
    assert scene.reference_validation["aspect_ratio_error"] == 0.01
    assert scene.reference_validation["scene_origin_x"] == 0.16
    assert scene.reference_validation["scene_origin_y"] == 0.12
    assert result["box"].rgba == reconstruct(tracks, objects, _reference(), seed=1).objects[0].rgba


def test_measured_values_override_independently_and_invalid_values_fall_back(tmp_path: Path) -> None:
    box = _track(tmp_path, "box", _rectangle(30, 30, 50, 30))
    invalid = _track(tmp_path, "invalid", _rectangle(160, 30, 50, 30))
    objects = ObjectList(
        objects=[
            ObjectInfo(id="box", name="plastic box", material_guess="plastic", approx_size_cm=(5, 3, 2)),
            ObjectInfo(id="invalid", name="rubber box", material_guess="rubber", approx_size_cm=(5, 3, 2)),
        ]
    )
    measured = tmp_path / "measured.yaml"
    measured.write_text(
        """objects:
  box:
    height_m: 0.021
    size_m: [9.0, 8.0, 0.050]
    mass_kg: 0.123
    friction: 0.77
  invalid:
    height_m: -1
    size_m: [1, 2]
    mass_kg: nope
    friction: .nan
"""
    )

    result = {item.id: item for item in reconstruct([box, invalid], objects, _reference(), measured_path=measured).objects}
    valid = result["box"]
    fallback = result["invalid"]
    assert valid.size[2] == 0.021  # explicit height takes priority over size_m.
    assert valid.size[0] < 0.06 and valid.size[1] < 0.04  # planar dimensions stay mask-derived.
    assert valid.mass == 0.123 and valid.friction == 0.77
    assert (valid.height_source, valid.mass_source, valid.friction_source) == ("measured", "measured", "measured")
    assert fallback.size[2] == 0.02
    assert fallback.mass_source == "estimate" and fallback.friction_source == "estimate"
    assert "invalid_measured_height_m" in fallback.reconstruction_source
    assert "invalid_measured_size_m" in fallback.reconstruction_source
    assert "invalid_measured_mass_kg" in fallback.reconstruction_source
    assert "invalid_measured_friction" in fallback.reconstruction_source


def test_reference_alias_is_static_thin_paper_and_nearest_source_mask_is_used(tmp_path: Path) -> None:
    old_mask = _rectangle(10, 10, 80, 30)
    near_mask = _rectangle(200, 100, 30, 30)
    track = _track(tmp_path, "reference:page", old_mask, index=0)
    path = tmp_path / "reference-near.png"
    imageio.imwrite(path, near_mask * 255)
    ys, xs = np.nonzero(near_mask)
    track = track.model_copy(
        update={
            "frames": track.frames
            + [
                MaskFrame(
                    frame_index=8,
                    timestamp_s=8.0,
                    mask_path=str(path),
                    centroid=(float(xs.mean()), float(ys.mean())),
                    area=int(len(xs)),
                    box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                )
            ]
        }
    )
    scene = reconstruct(
        [track],
        ObjectList(objects=[ObjectInfo(id="reference", name="paper", approx_size_cm=(21, 29.7, 3), graspable=True)]),
        _reference(),
    )
    paper = scene.objects[0]
    assert paper.shape == "box"
    assert not paper.graspable
    assert paper.size[2] == 0.001
    # Reference frame 7 chooses source index 8 rather than index 0, so the
    # result has the square's mask-derived footprint rather than the old box.
    assert abs(paper.size[0] - paper.size[1]) < 0.002
