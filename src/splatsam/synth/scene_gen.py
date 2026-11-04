"""Generate deterministic MuJoCo RGB, segmentation, and truth-separated fixtures.

The artifact set deliberately keeps metric truth in ``ground_truth.json`` and
``scene.json``.  The SAM fixture sees only per-frame binary masks and semantic
size priors in ``object_list.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np

# Configure the CPU renderer before importing MuJoCo.
from splatsam import rendering
from splatsam.llm.schemas import MaskFrame, MaskTrack, ObjectInfo, ObjectList, ObjectSpec, SceneSpec
from splatsam.sim.scene import body_name, build_model
from splatsam.utils import save_json

import mujoco  # isort: skip


FPS = 15
VIDEO_SECONDS = 12
VIDEO_FRAMES = FPS * VIDEO_SECONDS
SAMPLE_COUNT = 12
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960
_MASK_IDS = ("paper", "remote_1", "remote_2", "ball", "marker_1", "marker_2")
_HAND_GEOMS = ("palm", "left_finger_geom", "right_finger_geom")


def _colour(base: tuple[float, float, float], rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Apply a small seed-controlled colour variation without reducing contrast."""
    rgb = np.clip(np.asarray(base) + rng.uniform(-0.025, 0.025, 3), 0.05, 0.95)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _scene(seed: int) -> SceneSpec:
    """Create a deliberately separated 1 m table scene in metres."""
    rng = np.random.default_rng(seed)

    def position(x: float, y: float, z: float) -> tuple[float, float, float]:
        # Small bounded translation keeps the scene visibly seed-dependent while
        # retaining generous inter-object clearance.
        dx, dy = rng.uniform(-0.008, 0.008, size=2)
        return (float(x + dx), float(y + dy), z)

    def yaw(center: float, span: float = 0.18) -> float:
        return float(center + rng.uniform(-span, span))

    return SceneSpec(
        seed=seed,
        source="synthetic",
        table_size=(1.0, 1.0),
        scale_source="ground_truth_fixture",
        objects=[
            ObjectSpec(
                id="paper",
                name="white A4 reference sheet",
                shape="box",
                size=(0.210, 0.297, 0.001),
                position=position(-0.21, 0.22, 0.0005),
                yaw=yaw(0.04, 0.06),
                mass=0.015,
                friction=0.50,
                material="paper",
                graspable=False,
                rgba=(0.97, 0.97, 0.97, 1.0),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            ),
            ObjectSpec(
                id="remote_1",
                name="blue plastic remote",
                shape="box",
                size=(0.160, 0.045, 0.025),
                position=position(-0.23, -0.23, 0.0125),
                yaw=yaw(-0.24),
                mass=0.189,
                friction=0.50,
                material="plastic",
                rgba=_colour((0.10, 0.31, 0.78), rng),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            ),
            ObjectSpec(
                id="remote_2",
                name="teal plastic remote",
                shape="box",
                size=(0.160, 0.045, 0.025),
                position=position(0.22, -0.22, 0.0125),
                yaw=yaw(0.30),
                mass=0.189,
                friction=0.50,
                material="plastic",
                rgba=_colour((0.05, 0.56, 0.58), rng),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            ),
            ObjectSpec(
                id="ball",
                name="red rubber ball",
                shape="sphere",
                size=(0.050, 0.050, 0.050),
                position=position(0.22, 0.17, 0.025),
                yaw=0.0,
                mass=0.072,
                friction=1.00,
                material="rubber",
                rgba=_colour((0.86, 0.12, 0.10), rng),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            ),
            ObjectSpec(
                id="marker_1",
                name="orange plastic marker",
                shape="horizontal_cylinder",
                size=(0.140, 0.016, 0.016),
                position=position(-0.20, -0.025, 0.008),
                yaw=yaw(0.26),
                mass=0.030,
                friction=0.50,
                material="plastic",
                rgba=_colour((0.96, 0.43, 0.06), rng),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            ),
            ObjectSpec(
                id="marker_2",
                name="purple plastic marker",
                shape="horizontal_cylinder",
                size=(0.140, 0.016, 0.016),
                position=position(0.065, 0.005, 0.008),
                yaw=yaw(-0.34),
                mass=0.030,
                friction=0.50,
                material="plastic",
                rgba=_colour((0.48, 0.16, 0.75), rng),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            ),
        ],
    )


def vertical_cylinder_fixture(seed: int = 42) -> SceneSpec:
    """Return the separate, non-main-scene fixture used for cylinder classification."""
    rng = np.random.default_rng(seed)
    return SceneSpec(
        seed=seed,
        source="synthetic",
        table_size=(1.0, 1.0),
        scale_source="ground_truth_fixture",
        objects=[
            ObjectSpec(
                id="vertical_cylinder",
                name="upright plastic cylinder",
                shape="cylinder",
                size=(0.050, 0.050, 0.080),
                position=(float(rng.uniform(-0.01, 0.01)), float(rng.uniform(-0.01, 0.01)), 0.040),
                yaw=float(rng.uniform(-0.1, 0.1)),
                mass=0.082,
                friction=0.50,
                material="plastic",
                graspable=True,
                rgba=_colour((0.18, 0.62, 0.30), rng),
                height_source="synthetic_truth",
                mass_source="synthetic_truth",
                friction_source="synthetic_truth",
            )
        ],
    )


def _hide_hand(model: mujoco.MjModel) -> mujoco.MjvOption:
    """Move hand geoms to group 5 and hide that group in every render pass."""
    for name in _HAND_GEOMS:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise RuntimeError(f"render model lacks hand geom {name!r}")
        model.geom_group[geom_id] = 5
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[5] = 0
    return option


def _camera(step: int, seed: int) -> tuple[mujoco.MjvCamera, dict[str, Any]]:
    """Build one of twelve deterministic, held camera poses for the source video."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 701]))
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.0)
    camera.azimuth = float(90.0 + rng.uniform(-2.5, 2.5))
    # Keep the first eight sampled views effectively top-down for reference
    # recovery; the remaining held poses test mild perspective robustness.
    topdown = step < 8
    camera.elevation = float((-89.0 if topdown else -68.0) + rng.uniform(-0.5, 0.5))
    camera.distance = float((3.0 if topdown else 3.2) + rng.uniform(-0.015, 0.015))
    return camera, {
        "view": "topdown" if topdown else "oblique",
        "lookat_m": [0.0, 0.0, 0.0],
        "azimuth_deg": camera.azimuth,
        "elevation_deg": camera.elevation,
        "distance_m": camera.distance,
        "vertical_fov_deg": 23.0,
    }


def _render(
    renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera, option: mujoco.MjvOption
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB and raw MuJoCo ``(geom_id, object_type)`` segmentation."""
    renderer.update_scene(data, camera=camera, scene_option=option)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(data, camera=camera, scene_option=option)
        segmentation = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    if segmentation.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 2):
        raise RuntimeError(f"unexpected segmentation shape {segmentation.shape}")
    return rgb, segmentation


def _masks_from_segmentation(
    model: mujoco.MjModel, segmentation: np.ndarray, ids: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Union mjOBJ_GEOM pixels by body, retaining an explicit -1 background."""
    geom_pixels = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_ids = segmentation[..., 0]
    masks: dict[str, np.ndarray] = {}
    for identifier in ids:
        target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name(identifier))
        if target_body < 0:
            raise RuntimeError(f"render model lacks object body {identifier!r}")
        valid = geom_pixels & (geom_ids >= 0)
        mask = np.zeros(segmentation.shape[:2], dtype=np.uint8)
        selected_geoms = np.unique(geom_ids[valid])
        for geom_id in selected_geoms:
            if model.geom_bodyid[int(geom_id)] == target_body:
                mask[geom_ids == geom_id] = 255
        if not np.any(mask):
            raise RuntimeError(f"{identifier} is not visible in generated view")
        masks[identifier] = mask
    return masks


def _frame_record(identifier: str, frame_index: int, mask_path: Path, mask: np.ndarray) -> MaskFrame:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"empty mask for {identifier} at frame {frame_index}")
    return MaskFrame(
        frame_index=frame_index,
        timestamp_s=frame_index / FPS,
        mask_path=str(mask_path.resolve()),
        centroid=(float(xs.mean()), float(ys.mean())),
        area=len(xs),
        # Half-open pixel bounds are consistent with later box-prompt conversion.
        box=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
    )


def _semantic_objects() -> ObjectList:
    """Return visual/semantic priors only: no true position, yaw, or scale transform."""
    return ObjectList(
        source="synthetic",
        objects=[
            ObjectInfo(
                id="paper",
                name="white reference sheet",
                material_guess="paper",
                approx_size_cm=(21.0, 29.7, 0.1),
                graspable=False,
            ),
            ObjectInfo(
                id="remote_1",
                name="plastic remote",
                material_guess="plastic",
                approx_size_cm=(15.5, 4.8, 2.5),
            ),
            ObjectInfo(
                id="remote_2",
                name="plastic remote",
                material_guess="plastic",
                approx_size_cm=(15.5, 4.8, 2.5),
            ),
            ObjectInfo(
                id="ball", name="rubber ball", material_guess="rubber", approx_size_cm=(5.1, 5.1, 5.1)
            ),
            ObjectInfo(
                id="marker_1",
                name="plastic marker",
                material_guess="plastic",
                approx_size_cm=(13.8, 1.7, 1.6),
            ),
            ObjectInfo(
                id="marker_2",
                name="plastic marker",
                material_guess="plastic",
                approx_size_cm=(13.8, 1.7, 1.6),
            ),
        ],
    )


def _ground_truth(scene: SceneSpec, sampled: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep independent scene truth out of the semantic and mask inputs."""
    objects = {item.id: item for item in scene.objects}
    remote_1 = np.asarray(objects["remote_1"].position[:2])
    remote_2 = np.asarray(objects["remote_2"].position[:2])
    ball = np.asarray(objects["ball"].position[:2])
    return {
        "schema_version": 1,
        "source": "synthetic_ground_truth_only",
        "seed": scene.seed,
        "units": {"length": "m", "time": "s"},
        "video": {"fps": FPS, "frame_count": VIDEO_FRAMES, "duration_s": VIDEO_SECONDS},
        "table_size_m": list(scene.table_size),
        "paper_size_m": [0.210, 0.297],
        "objects": [item.model_dump(mode="json") for item in scene.objects],
        "sampled_frames": sampled,
        "independent_planar_distances_m": {
            "remote_centers": float(np.linalg.norm(remote_1 - remote_2)),
            "remote_1_to_ball": float(np.linalg.norm(remote_1 - ball)),
        },
    }


def generate(root: Path, seed: int = 42) -> dict[str, str]:
    """Generate the A4 reference scene under ``root / 'synthetic'``.

    The 180-frame, 15 fps video is made from twelve held real MuJoCo render
    poses.  Saved frame/mask records use the exact source indices returned by
    ``numpy.linspace(0, 179, 12)`` so video sampling and mask tracks align.
    """
    root = Path(root).resolve()
    output = root / "synthetic"
    frames_dir = output / "frames"
    masks_dir = output / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene = _scene(seed)
    model = build_model(scene)
    # A distant camera with matched framing limits elevated-silhouette parallax;
    # paper calibration alone cannot remove the object's height contribution.
    model.vis.global_.fovy = 23.0
    option = _hide_hand(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sample_indices = np.linspace(0, VIDEO_FRAMES - 1, SAMPLE_COUNT, dtype=int)

    rgb_steps: list[np.ndarray] = []
    tracks = {identifier: [] for identifier in _MASK_IDS}
    sampled_truth: list[dict[str, Any]] = []
    with rendering.renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH) as renderer:
        for step, frame_index in enumerate(sample_indices):
            camera, camera_truth = _camera(step, seed)
            rgb, segmentation = _render(renderer, data, camera, option)
            masks = _masks_from_segmentation(model, segmentation, _MASK_IDS)
            rgb_steps.append(rgb)
            frame_path = frames_dir / f"frame_{frame_index:06d}.png"
            iio.imwrite(frame_path, rgb)
            sampled_truth.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_s": frame_index / FPS,
                    "camera": camera_truth,
                }
            )
            for identifier, mask in masks.items():
                mask_path = masks_dir / identifier / f"frame_{frame_index:06d}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(mask_path, mask)
                tracks[identifier].append(_frame_record(identifier, int(frame_index), mask_path, mask))

    video_path = output / "synthetic_scene.mp4"
    with imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8) as writer:
        for source_index in range(VIDEO_FRAMES):
            writer.append_data(rgb_steps[min(source_index // FPS, SAMPLE_COUNT - 1)])

    mask_tracks = [
        MaskTrack(
            id=identifier,
            name=next(item.name for item in scene.objects if item.id == identifier),
            frames=tracks[identifier],
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            total_frames=SAMPLE_COUNT,
            source="synthetic",
            stable=True,
        )
        for identifier in _MASK_IDS
    ]
    scene_path = output / "scene.json"
    tracks_path = output / "tracks.json"
    object_list_path = output / "object_list.json"
    ground_truth_path = output / "ground_truth.json"
    save_json(scene_path, scene)
    save_json(tracks_path, [track.model_dump(mode="json") for track in mask_tracks])
    save_json(object_list_path, _semantic_objects())
    save_json(ground_truth_path, _ground_truth(scene, sampled_truth))
    return {
        "video_path": str(video_path),
        "scene_path": str(scene_path),
        "tracks_path": str(tracks_path),
        "object_list_path": str(object_list_path),
        "ground_truth_path": str(ground_truth_path),
    }
