"""Independent metric masks catch camera-axis errors in physical silhouette fits."""

from math import radians

import cv2
import numpy as np
import pytest

from splatsam.llm.schemas import MaskFrame, MaskTrack, ObjectSpec, SceneSpec
from splatsam.perception.reference import ReferenceResult
from splatsam.sim.fit import _iou, _render_isolated_mask, fit_scene


@pytest.mark.parametrize("offset", [(0.0, 0.0), (-0.020, 0.018)])
def test_rotated_box_matches_independently_rasterized_metric_footprint(tmp_path, offset):
    reference = ReferenceResult(
        homography=np.eye(3),
        image_to_raster=np.eye(3),
        meters_per_pixel=0.001,
        bounds=(0.0, -0.150, 0.200, 0.050),
        working_frame_index=7,
        scale_source="paper_homography",
        _raster_shape=(200, 200),
    )
    true_yaw = radians(10)
    center = (100 + offset[0] * 1000, 100 + offset[1] * 1000)
    observed = np.zeros((200, 200), np.uint8)
    corners = cv2.boxPoints((center, (150.0, 30.0), 10.0))
    cv2.fillConvexPoly(observed, np.rint(corners).astype(np.int32), 1)
    path = tmp_path / "independent_mask.png"
    cv2.imwrite(str(path), observed * 255)
    y, x = np.nonzero(observed)
    track = MaskTrack(
        id="remote",
        name="remote",
        image_width=200,
        image_height=200,
        total_frames=1,
        frames=[
            MaskFrame(
                frame_index=7,
                timestamp_s=0.5,
                mask_path=str(path),
                centroid=(float(x.mean()), float(y.mean())),
                area=len(x),
                box=(float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)),
            )
        ],
    )
    obj = ObjectSpec(
        id="remote",
        name="nonsquare box",
        shape="box",
        size=(0.150, 0.030, 0.025),
        position=(offset[0], offset[1], 0.0125),
        yaw=0,
        mass=0.1,
        friction=0.5,
    )
    scene = SceneSpec(objects=[obj], reference_validation={"scene_origin_x": 0.100, "scene_origin_y": -0.050})
    initial = _render_isolated_mask(scene, obj, reference)
    fitted = fit_scene(scene, [track], reference, tmp_path / "fit").objects[0]
    assert fitted.consistency_iou > 0.90
    assert fitted.consistency_iou > _iou(observed, initial) + 0.10
    assert abs(fitted.yaw - true_yaw) < radians(5)
