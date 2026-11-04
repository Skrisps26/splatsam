"""Segmentation-backed metric-footprint consistency fitting.

The fit intentionally validates only visible footprints.  It never treats a 2-D
silhouette as evidence for object height, mass, friction, or collision shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import radians
from pathlib import Path

import cv2
import imageio.v3 as iio

import mujoco  # isort: skip
import numpy as np

from splatsam import rendering
from splatsam.llm.schemas import MaskFrame, MaskTrack, ObjectSpec, SceneSpec
from splatsam.perception.reference import ReferenceResult
from splatsam.sim.scene import build_model, geom_name
from splatsam.utils import save_json

FIT_THRESHOLD = 0.60
_YAW_DELTAS = (0.0, -5.0, 5.0, -10.0, 10.0)
_HAND_GEOMS = ("palm", "left_finger_geom", "right_finger_geom")


@dataclass(frozen=True, slots=True)
class _FitCandidate:
    yaw: float
    mask: np.ndarray
    iou: float


def _raster_shape(reference: ReferenceResult) -> tuple[int, int]:
    """Return the exact common raster dimensions, including the reference padding."""
    height, width = reference._raster_shape
    if height > 0 and width > 0:
        return height, width
    xmin, ymin, xmax, ymax = reference.bounds
    meters_per_pixel = reference.meters_per_pixel
    return (
        max(1, int(np.ceil((ymax - ymin) / meters_per_pixel))),
        max(1, int(np.ceil((xmax - xmin) / meters_per_pixel))),
    )


def _scene_origin(scene: SceneSpec) -> tuple[float, float]:
    """Recover the metric-raster origin removed when reconstruction recentred XY."""
    validation = scene.reference_validation
    return float(validation.get("scene_origin_x", 0.0)), float(validation.get("scene_origin_y", 0.0))


def _original_coordinates(scene: SceneSpec, object_spec: ObjectSpec) -> ObjectSpec:
    """Map reference raster coordinates into the right-handed MuJoCo top view.

    The common metric raster uses image axes (right, down). A top-down MuJoCo
    camera with its screen-up aligned to positive world Y uses (right, up), so
    this isolated render reflects Y and yaw after restoring the scene origin.
    The public SceneSpec remains in its reference-raster convention.
    """
    origin_x, origin_y = _scene_origin(scene)
    x, y, z = object_spec.position
    return object_spec.model_copy(
        update={"position": (x + origin_x, -(y + origin_y), z), "yaw": -object_spec.yaw}
    )


def _isolated_model(
    scene: SceneSpec, object_spec: ObjectSpec
) -> tuple[mujoco.MjModel, int, mujoco.MjvOption]:
    """Build a real MuJoCo model that renders precisely one object's geom."""
    original = _original_coordinates(scene, object_spec)
    model = build_model(scene.model_copy(update={"objects": [original]}))
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name(original.id))
    if target_geom < 0:
        raise RuntimeError(f"silhouette model lacks geom for {original.id!r}")

    # The hand is well above the table and can otherwise cover a central target.
    # Hiding all non-target geoms also makes the resulting segmentation genuinely
    # isolated rather than merely selecting target pixels after occlusion.
    model.geom_group[:] = 5
    model.geom_group[target_geom] = 0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[5] = 0
    return model, target_geom, option


def _orthographic_camera(model: mujoco.MjModel, reference: ReferenceResult) -> mujoco.MjvCamera:
    """Construct the official free-camera orthographic top-down metric view.

    MuJoCo's XML/API reference defines orthographic camera ``fovy`` as vertical
    model-space extent rather than an angle.  The free camera uses the model
    visual global setting, while ``MjvCamera`` supplies the top-down pose.
    """
    xmin, ymin, xmax, ymax = reference.bounds
    vertical_extent = max(ymax - ymin, reference.meters_per_pixel)
    model.vis.global_.orthographic = 1
    model.vis.global_.fovy = float(vertical_extent)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = ((xmin + xmax) / 2, -(ymin + ymax) / 2, 0.0)
    camera.distance = max(vertical_extent * 2.0, 1.0)
    camera.azimuth = 90.0
    camera.elevation = -90.0
    return camera


def _segmentation_mask(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geom: int,
    reference: ReferenceResult,
    option: mujoco.MjvOption,
) -> np.ndarray:
    """Render a target silhouette through MuJoCo's geom-ID segmentation pass."""
    height, width = _raster_shape(reference)
    camera = _orthographic_camera(model, reference)
    mujoco.mj_forward(model, data)
    renderer = rendering.renderer(model, height=height, width=width)
    try:
        renderer.enable_segmentation_rendering()
        try:
            renderer.update_scene(data, camera=camera, scene_option=option)
            segmentation = renderer.render().copy()
        finally:
            renderer.disable_segmentation_rendering()
    finally:
        renderer.close()
    geom_pixels = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return ((segmentation[..., 0] == target_geom) & geom_pixels).astype(np.uint8)


def _render_isolated_mask(
    scene: SceneSpec, object_spec: ObjectSpec, reference: ReferenceResult
) -> np.ndarray:
    """Render one physical primitive into the same full metric grid as observations."""
    model, target_geom, option = _isolated_model(scene, object_spec)
    return _segmentation_mask(model, mujoco.MjData(model), target_geom, reference, option)


def _load_mask(frame: MaskFrame) -> np.ndarray:
    image = np.asarray(iio.imread(frame.mask_path))
    if image.ndim == 3:
        image = image[..., :3].max(axis=2)
    if image.ndim != 2:
        raise ValueError(f"mask {frame.mask_path!r} is not a two-dimensional image")
    return (image > 0).astype(np.uint8)


def _track_for_object(object_spec: ObjectSpec, tracks: Iterable[MaskTrack]) -> MaskTrack | None:
    """Match stable raw IDs first, then preserve the multi-instance base-ID convention."""
    stable_tracks = [track for track in tracks if track.stable and track.frames]
    exact = next((track for track in stable_tracks if track.id == object_spec.id), None)
    if exact is not None:
        return exact
    base = object_spec.id.split(":", 1)[0]
    return next((track for track in stable_tracks if track.id.split(":", 1)[0] == base), None)


def _observed_mask(
    object_spec: ObjectSpec, tracks: Iterable[MaskTrack], reference: ReferenceResult
) -> tuple[np.ndarray | None, str]:
    track = _track_for_object(object_spec, tracks)
    if track is None:
        return None, "no_stable_track"
    frame = next((item for item in track.frames if item.frame_index == reference.working_frame_index), None)
    if frame is None:
        return None, "track_missing_working_frame"
    return reference.warp_mask(_load_mask(frame)), "observed"


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 1.0 if not np.any(left) and not np.any(right) else 0.0
    return float(np.count_nonzero(left & right) / union)


def _yaw_candidates(object_spec: ObjectSpec) -> tuple[float, ...]:
    deltas = list(_YAW_DELTAS)
    if object_spec.shape == "box":
        deltas.extend((-90.0, 90.0))
    return tuple(object_spec.yaw + radians(delta) for delta in deltas)


def _comparison_image(observed: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    """Encode observed-only red, rendered-only green, and overlap yellow."""
    image = np.zeros((*observed.shape, 3), dtype=np.uint8)
    image[..., 0] = observed * 255
    image[..., 1] = rendered * 255
    return image


def _artifact_prefix(identifier: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in identifier)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), (mask.astype(np.uint8) * 255)):
        raise RuntimeError(f"failed writing consistency artifact {path}")


def _save_comparison(path: Path, observed: np.ndarray, rendered: np.ndarray) -> None:
    if not cv2.imwrite(str(path), _comparison_image(observed, rendered)):
        raise RuntimeError(f"failed writing consistency artifact {path}")


def fit_scene(
    scene: SceneSpec,
    tracks: list[MaskTrack],
    reference: ReferenceResult,
    out_dir: Path,
) -> SceneSpec:
    """Fit primitive yaw against actual MuJoCo segmentation silhouettes.

    The input scene is not mutated.  A low IoU records a failed footprint check;
    no remote provider or fabricated alternate shape is attempted.
    """
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fitted: list[ObjectSpec] = []
    records: list[dict[str, object]] = []

    for object_spec in scene.objects:
        observed, observation_status = _observed_mask(object_spec, tracks, reference)
        prefix = _artifact_prefix(object_spec.id)
        if observed is None:
            blank = np.zeros(_raster_shape(reference), dtype=np.uint8)
            _save_mask(destination / f"{prefix}_observedmask.png", blank)
            _save_mask(destination / f"{prefix}_renderedbest.png", blank)
            _save_comparison(destination / f"{prefix}_comparison.png", blank, blank)
            fitted.append(object_spec.model_copy(update={"consistency_iou": None}))
            records.append(
                {
                    "id": object_spec.id,
                    "status": observation_status,
                    "iou": None,
                    "source": object_spec.reconstruction_source,
                    "low_confidence": True,
                    "alternative_attempted": False,
                    "alternative_reason": "no observed stable mask",
                }
            )
            continue

        best: _FitCandidate | None = None
        for yaw in _yaw_candidates(object_spec):
            candidate_spec = object_spec.model_copy(update={"yaw": yaw})
            mask = _render_isolated_mask(scene, candidate_spec, reference)
            candidate = _FitCandidate(yaw=yaw, mask=mask, iou=_iou(observed, mask))
            if best is None or candidate.iou > best.iou:
                best = candidate
        assert best is not None

        low_confidence = best.iou < FIT_THRESHOLD
        fitted.append(object_spec.model_copy(update={"yaw": best.yaw, "consistency_iou": best.iou}))
        _save_mask(destination / f"{prefix}_observedmask.png", observed)
        _save_mask(destination / f"{prefix}_renderedbest.png", best.mask)
        _save_comparison(destination / f"{prefix}_comparison.png", observed, best.mask)
        records.append(
            {
                "id": object_spec.id,
                "status": "low_confidence" if low_confidence else "ok",
                "iou": best.iou,
                "source": object_spec.reconstruction_source,
                "fitted_yaw_rad": best.yaw,
                "low_confidence": low_confidence,
                "alternative_attempted": False,
                "alternative_reason": (
                    "no other real reconstruction candidate is available" if low_confidence else "not_needed"
                ),
            }
        )

    result = scene.model_copy(update={"objects": fitted})
    save_json(
        destination / "consistency.json",
        {
            "reference_bounds_m": list(reference.bounds),
            "meters_per_pixel": reference.meters_per_pixel,
            "scene_origin_m": list(_scene_origin(scene)),
            "objects": records,
        },
    )
    return result


def render_scene(scene: SceneSpec, path: Path) -> None:
    """Render a real oblique MuJoCo PNG for the dashboard, with hand group hidden."""
    model = build_model(scene)
    model.vis.headlight.ambient[:] = (0.4, 0.4, 0.4)
    model.vis.headlight.diffuse[:] = (0.8, 0.8, 0.8)
    data = mujoco.MjData(model)
    for name in _HAND_GEOMS:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            model.geom_group[geom_id] = 5
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[5] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.0)
    camera.distance = max(scene.table_size) * 1.45
    camera.azimuth = 45.0
    camera.elevation = -34.0
    mujoco.mj_forward(model, data)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    renderer = rendering.renderer(model, height=720, width=960)
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        image = renderer.render().copy()
    finally:
        renderer.close()
    if not cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed writing scene render {target}")
