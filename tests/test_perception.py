from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from splatsam.llm.schemas import MaskFrame, MaskTrack
from splatsam.perception.frames import FrameSample
from splatsam.perception.reference import ReferenceError, find_reference
from splatsam.perception.sam import associate_instances, stability_gate


def _mask_frame(path: Path, mask: np.ndarray, *, index: int = 0) -> MaskFrame:
    imageio.imwrite(path, (mask * 255).astype(np.uint8))
    ys, xs = np.nonzero(mask)
    return MaskFrame(
        frame_index=index,
        timestamp_s=float(index),
        mask_path=str(path),
        centroid=(float(xs.mean()), float(ys.mean())),
        area=int(mask.sum()),
        box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
    )


def _track(identifier: str, name: str, frames: list[MaskFrame], *, width: int = 400, height: int = 400, total: int = 1) -> MaskTrack:
    return MaskTrack(id=identifier, name=name, frames=frames, image_width=width, image_height=height, total_frames=total)


def test_reference_recovers_known_object_footprint_not_just_sheet(tmp_path: Path) -> None:
    paper = np.zeros((400, 400), dtype=np.uint8)
    # An analytic A4 rectangle: 210 by 297 image pixels at one mm/pixel.
    paper[40:337, 40:250] = 1
    remote = np.zeros_like(paper)
    remote[70:115, 275:355] = 1  # independently known .080 x .045 m footprint
    paper_frame = _mask_frame(tmp_path / "paper.png", paper)
    remote_frame = _mask_frame(tmp_path / "remote.png", remote)
    tracks = [_track("paper", "white paper reference", [paper_frame]), _track("remote", "remote", [remote_frame])]
    sample = FrameSample(0, 0.0, np.zeros((400, 400, 3), dtype=np.uint8), 100.0)

    reference = find_reference(tracks, [sample])
    metric = reference.pixel_to_metric(np.asarray([[275, 70], [355, 115]], dtype=float))
    dimensions = np.abs(metric[1] - metric[0])
    assert reference.scale_source == "paper_homography"
    assert np.allclose(dimensions, [0.080, 0.045], atol=0.002)
    assert reference.validation["aspect_ratio_error"] < 0.05
    # The metric raster must include the remote footprint, which is outside paper.
    warped_remote = reference.warp_mask(remote)
    assert warped_remote.sum() > 0


def test_reference_rejects_border_sheet_then_uses_manual_or_blocks(tmp_path: Path) -> None:
    paper = np.zeros((100, 100), dtype=np.uint8)
    paper[:, :40] = 1  # touches frame border and cannot be a fully visible reference.
    frame = _mask_frame(tmp_path / "border_paper.png", paper)
    tracks = [_track("paper", "paper", [frame], width=100, height=100)]
    sample = FrameSample(0, 0.0, np.zeros((100, 100, 3), dtype=np.uint8), 10.0)

    with pytest.raises(ReferenceError, match="manual_px_per_cm"):
        find_reference(tracks, [sample])
    manual = find_reference(tracks, [sample], manual_px_per_cm=100.0)
    assert manual.scale_source == "manual_px_per_cm"
    assert "paper_touches_border:0" in manual.diagnostics


def test_stability_gate_accepts_boundaries_and_rejects_strict_violations() -> None:
    paths = [f"mask-{index}.png" for index in range(3)]
    frames = [
        MaskFrame(frame_index=index, timestamp_s=float(index), mask_path=paths[index], centroid=(25.0 * index, 10.0), area=100 * (3**index), box=(0, 0, 1, 1))
        for index in range(3)
    ]
    boundary = _track("boundary", "object", frames, width=100, total=5)
    stable, diagnostics = stability_gate(boundary)
    assert stable and not diagnostics  # visibility .60, jump .25 width, area ratio 3 are allowed.
    failing = boundary.model_copy(update={"frames": [frames[0], frames[1].model_copy(update={"area": 301}), frames[2]]})
    stable, diagnostics = stability_gate(failing)
    assert not stable
    assert any(item.startswith("area_ratio") for item in diagnostics)
    low_visibility = boundary.model_copy(update={"total_frames": 6})
    stable, diagnostics = stability_gate(low_visibility)
    assert not stable
    assert any(item.startswith("visibility_below") for item in diagnostics)
    jump_frames = [frames[0], frames[1].model_copy(update={"centroid": (25.01, 10.0)}), frames[2]]
    stable, diagnostics = stability_gate(boundary.model_copy(update={"frames": jump_frames}))
    assert not stable
    assert any(item.startswith("centroid_jump") for item in diagnostics)


def test_associate_instances_keeps_identity_when_source_order_crosses() -> None:
    association = associate_instances(
        [
            [{"centroid": (80, 30), "area": 20, "source_id": "b"}, {"centroid": (20, 30), "area": 20, "source_id": "a"}],
            [{"centroid": (60, 30), "area": 20, "source_id": "b"}, {"centroid": (40, 30), "area": 20, "source_id": "a"}],
            [{"centroid": (60, 30), "area": 20, "source_id": "a"}, {"centroid": (40, 30), "area": 20, "source_id": "b"}],
        ],
        image_width=100,
        phrase="marker",
    )
    assert [item.id for item in association] == ["marker-1", "marker-2"]
    assert [item.source_id for item in association[0].observations] == ["a", "a", "a"]
    assert [item.source_id for item in association[1].observations] == ["b", "b", "b"]
