"""Truth-independent primitive reconstruction from metric mask footprints."""

from __future__ import annotations

import colorsys
import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

from splatsam.llm.schemas import MaskTrack, ObjectInfo, ObjectList, ObjectSpec, SceneSpec
from splatsam.perception.reference import ReferenceResult
from splatsam.utils import config

MATERIAL_ESTIMATES: dict[str, tuple[float, float]] = {
    "plastic": (1050.0, 0.5),
    "rubber": (1100.0, 1.0),
    "paper": (800.0, 0.5),
}
_DEFAULT_DENSITY_KG_M3 = 1000.0
_DEFAULT_FRICTION = 0.5
_REFERENCE_ALIASES = ("paper", "sheet", "reference")


def _positive_number(value: Any) -> float | None:
    """Return a finite positive scalar, rejecting booleans and coercion surprises."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _measurement_objects(path: Path | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load the deliberately small measured-physics override format without failing a scene."""
    if path is None:
        return {}, []
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}, ["invalid_measured_file"]
    if not isinstance(loaded, dict) or not isinstance(loaded.get("objects"), dict):
        return {}, ["invalid_measured_objects"]
    objects: dict[str, dict[str, Any]] = {}
    for identifier, values in loaded["objects"].items():
        if isinstance(values, dict):
            objects[str(identifier)] = values
        else:
            objects[str(identifier)] = {"__invalid_entry": True}
    return objects, []


def _semantic_for(track: MaskTrack, objects: ObjectList) -> ObjectInfo | None:
    """Match semantic priors without changing a persistent track's raw identifier."""
    by_id = {item.id: item for item in objects.objects}
    if track.id in by_id:
        return by_id[track.id]
    base_id = track.id.split(":", 1)[0]
    if base_id in by_id:
        return by_id[base_id]

    def normalized(value: str) -> str:
        return "".join(character for character in value.lower() if character.isalnum())

    track_name = normalized(track.name)
    for item in objects.objects:
        item_name = normalized(item.name)
        if track_name and item_name and (track_name in item_name or item_name in track_name):
            return item
    return None


def _is_reference(track: MaskTrack, semantic: ObjectInfo | None) -> bool:
    values = (track.id, track.name, semantic.id if semantic else "", semantic.name if semantic else "")
    return any(alias in value.lower() for value in values for alias in _REFERENCE_ALIASES)


def _read_mask(path: str) -> np.ndarray:
    image = np.asarray(imageio.imread(Path(path)))
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("mask must be a 2-D raster")
    return (image > 0).astype(np.uint8)


def _working_mask(track: MaskTrack, reference: ReferenceResult) -> np.ndarray | None:
    """Use an actual source-frame mask nearest to the reference frame index."""
    if not track.frames:
        return None
    frame = min(
        track.frames,
        key=lambda item: (abs(item.frame_index - reference.working_frame_index), item.frame_index),
    )
    try:
        mask = _read_mask(frame.mask_path)
        warped = reference.warp_mask(mask.astype(np.uint8))
    except (OSError, ValueError):
        return None
    return warped if np.any(warped) else None


def _contour_geometry(
    mask: np.ndarray, meters_per_pixel: float
) -> tuple[float, float, float, float, float, float] | None:
    """Return centre, long/short footprint axes, yaw, and circularity from the largest contour."""
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    # Simplify sub-pixel stair steps before circularity, retaining the original
    # contour for the footprint rectangle. Warped pixel edges inflate perimeter.
    smooth = cv2.approxPolyDP(contour, 0.75, True)
    area = float(cv2.contourArea(smooth))
    perimeter = float(cv2.arcLength(smooth, True))
    if area <= 0 or perimeter <= 0:
        return None
    rectangle = cv2.minAreaRect(contour)
    width_px, height_px = rectangle[1]
    if width_px <= 0 or height_px <= 0:
        return None
    box = cv2.boxPoints(rectangle)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    edge = edges[int(np.argmax(lengths))]
    yaw = math.atan2(float(edge[1]), float(edge[0]))
    # A rectangle is unchanged by a half turn; this makes yaw deterministic.
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    long_axis, short_axis = sorted((float(width_px), float(height_px)), reverse=True)
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    return (
        float(rectangle[0][0]),
        float(rectangle[0][1]),
        long_axis * meters_per_pixel,
        short_axis * meters_per_pixel,
        yaw,
        circularity,
    )


def _colour(identifier: str) -> tuple[float, float, float, float]:
    """Stable visible colours must derive from IDs, never synthetic hidden colours."""
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65536.0
    saturation = 0.50 + digest[2] / 2550.0
    value = 0.62 + digest[3] / 1020.0
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (red, green, blue, 1.0)


def _volume(shape: str, size: tuple[float, float, float]) -> float:
    if shape == "box":
        return size[0] * size[1] * size[2]
    if shape == "sphere":
        return 4.0 * math.pi * (size[0] / 2.0) ** 3 / 3.0
    if shape == "cylinder":
        return math.pi * (size[0] / 2.0) ** 2 * size[2]
    if shape == "horizontal_cylinder":
        return math.pi * (size[1] / 2.0) ** 2 * size[0]
    raise ValueError(f"unsupported primitive shape {shape!r}")


def _measurement(
    values: dict[str, Any] | None, global_diagnostics: list[str]
) -> tuple[float | None, float | None, float | None, list[str]]:
    """Validate individual overrides, preserving valid siblings when one is malformed."""
    diagnostics = list(global_diagnostics)
    if not values:
        return None, None, None, diagnostics

    if values.get("__invalid_entry"):
        diagnostics.append("invalid_measured_object")
    size_height: float | None = None
    if "size_m" in values:
        candidate = values["size_m"]
        if isinstance(candidate, (list, tuple)) and len(candidate) == 3:
            components = [_positive_number(value) for value in candidate]
            if all(value is not None for value in components):
                # Footprint dimensions remain measured from the mask; only the
                # physical vertical component can inform this top-down pipeline.
                size_height = float(components[2])
            else:
                diagnostics.append("invalid_measured_size_m")
        else:
            diagnostics.append("invalid_measured_size_m")

    explicit_height: float | None = None
    if "height_m" in values:
        explicit_height = _positive_number(values["height_m"])
        if explicit_height is None:
            diagnostics.append("invalid_measured_height_m")
    height = explicit_height if explicit_height is not None else size_height

    mass: float | None = None
    if "mass_kg" in values:
        mass = _positive_number(values["mass_kg"])
        if mass is None:
            diagnostics.append("invalid_measured_mass_kg")

    friction: float | None = None
    if "friction" in values:
        friction = _positive_number(values["friction"])
        if friction is None:
            diagnostics.append("invalid_measured_friction")
    return height, mass, friction, diagnostics


def _scene_table_size(specs: list[ObjectSpec]) -> tuple[float, float]:
    if not specs:
        return (1.0, 1.0)

    def extents(item: ObjectSpec) -> tuple[float, float]:
        if item.shape in {"sphere", "cylinder"}:
            radius = item.size[0] / 2.0
            return radius, radius
        half_x, half_y = item.size[0] / 2.0, item.size[1] / 2.0
        cosine, sine = abs(math.cos(item.yaw)), abs(math.sin(item.yaw))
        return cosine * half_x + sine * half_y, sine * half_x + cosine * half_y

    x_min = min(item.position[0] - extents(item)[0] for item in specs)
    x_max = max(item.position[0] + extents(item)[0] for item in specs)
    y_min = min(item.position[1] - extents(item)[1] for item in specs)
    y_max = max(item.position[1] + extents(item)[1] for item in specs)
    return (max(1.0, x_max - x_min + 0.16), max(1.0, y_max - y_min + 0.16))


def reconstruct(
    tracks: list[MaskTrack],
    objects: ObjectList,
    reference: ReferenceResult,
    *,
    measured_path: Path | None = None,
    source: str = "synthetic",
    seed: int = 42,
) -> SceneSpec:
    """Reconstruct native primitive specs from stable tracks and a metric reference.

    Geometry comes only from masks and the reference transform. Semantic entries
    supply material and height estimates, while optional measured values can only
    replace physical estimates after finite-positive validation.
    """
    measured, file_diagnostics = _measurement_objects(measured_path)
    material_settings = config().get("materials", {})
    origin_x = (reference.bounds[0] + reference.bounds[2]) / 2.0
    origin_y = (reference.bounds[1] + reference.bounds[3]) / 2.0
    specs: list[ObjectSpec] = []

    for track in tracks:
        warped = _working_mask(track, reference)
        if warped is None:
            continue
        geometry = _contour_geometry(warped, reference.meters_per_pixel)
        if geometry is None:
            continue
        center_x_px, center_y_px, long_axis, short_axis, yaw, circularity = geometry
        semantic = _semantic_for(track, objects)
        is_reference = _is_reference(track, semantic)
        material = "paper" if is_reference else (semantic.material_guess if semantic else "plastic").lower()
        if is_reference:
            semantic_height = 0.001
        else:
            semantic_height = (
                _positive_number(semantic.approx_size_cm[2] / 100.0) if semantic is not None else None
            )
            semantic_height = semantic_height if semantic_height is not None else 0.02
        entry = measured.get(track.id)
        if entry is None:
            entry = measured.get(track.id.split(":", 1)[0])
        measured_height, measured_mass, measured_friction, diagnostics = _measurement(entry, file_diagnostics)
        height = measured_height if measured_height is not None else semantic_height
        height_source = "measured" if measured_height is not None else "estimate"

        diameter = 2.0 * math.sqrt(float(np.count_nonzero(warped)) * reference.meters_per_pixel**2 / math.pi)
        aspect = long_axis / short_axis
        if is_reference:
            shape = "box"
            size = (long_axis, short_axis, height)
        elif circularity > 0.85:
            if 0.75 <= height / diameter <= 1.25:
                shape = "sphere"
            else:
                shape = "cylinder"
            chosen_diameter = (
                measured_height if shape == "sphere" and measured_height is not None else diameter
            )
            size = (chosen_diameter, chosen_diameter, chosen_diameter if shape == "sphere" else height)
            yaw = 0.0
        elif aspect > 3.0 and max(short_axis, height) <= 0.03:
            shape = "horizontal_cylinder"
            # Native horizontal cylinders have a circular cross-section: their
            # vertical extent is the footprint diameter, not a semantic prior.
            cross_section = measured_height if measured_height is not None else short_axis
            size = (long_axis, cross_section, cross_section)
            height_source = "measured" if measured_height is not None else "estimate"
        else:
            shape = "box"
            size = (long_axis, short_axis, height)

        density, estimated_friction = MATERIAL_ESTIMATES.get(
            material, (_DEFAULT_DENSITY_KG_M3, _DEFAULT_FRICTION)
        )
        configured = material_settings.get(material, {})
        density = _positive_number(configured.get("density")) or density
        estimated_friction = _positive_number(configured.get("friction")) or estimated_friction
        mass = measured_mass if measured_mass is not None else density * _volume(shape, size)
        friction = measured_friction if measured_friction is not None else estimated_friction
        mass_source = "measured" if measured_mass is not None else "estimate"
        friction_source = "measured" if measured_friction is not None else "estimate"
        graspable = False if is_reference else (semantic.graspable if semantic is not None else True)
        if not track.stable:
            graspable = False
            diagnostics.append("unstable_track_clutter")
        if all(value < 0.015 for value in size) or min(size[0], size[1]) > 0.08:
            graspable = False
            diagnostics.append("clutter_dimension_gate")

        reconstruction_source = "primitive"
        if diagnostics:
            reconstruction_source += ";" + ";".join(sorted(set(diagnostics)))
        center_x = reference.bounds[0] + center_x_px * reference.meters_per_pixel - origin_x
        center_y = reference.bounds[1] + center_y_px * reference.meters_per_pixel - origin_y
        specs.append(
            ObjectSpec(
                id=track.id,
                name=semantic.name if semantic is not None else track.name,
                shape=shape,
                size=tuple(float(value) for value in size),
                position=(float(center_x), float(center_y), float(size[2] / 2.0)),
                yaw=float(yaw),
                mass=float(mass),
                friction=float(friction),
                material=material,
                graspable=graspable,
                rgba=_colour(track.id),
                height_source=height_source,
                mass_source=mass_source,
                friction_source=friction_source,
                reconstruction_source=reconstruction_source,
            )
        )

    validation = dict(reference.validation)
    validation["scene_origin_x"] = float(origin_x)
    validation["scene_origin_y"] = float(origin_y)
    return SceneSpec(
        objects=specs,
        table_size=_scene_table_size(specs),
        source=source,
        seed=seed,
        scale_source=reference.scale_source,
        scale_error=None,
        reference_validation=validation,
    )
