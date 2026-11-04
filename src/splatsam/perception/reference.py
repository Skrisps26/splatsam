"""Paper-reference homography and metric raster construction for perception masks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as imageio
import numpy as np

from splatsam.llm.schemas import MaskFrame, MaskTrack
from splatsam.perception.frames import FrameSample


SHEET_SHORT_M = 0.210
SHEET_LONG_M = 0.297
METERS_PER_PIXEL = 0.001
_REFERENCE_WORDS = ("reference", "sheet", "paper")


class ReferenceError(RuntimeError):
    """Raised when metric scale is unavailable, so callers can mark a stage blocked."""


@dataclass(slots=True)
class ReferenceResult:
    """A pixel-to-table transform plus a common millimetre metric raster."""

    homography: np.ndarray
    image_to_raster: np.ndarray
    meters_per_pixel: float = METERS_PER_PIXEL
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    working_frame_index: int = 0
    scale_source: str = "paper_homography"
    validation: dict[str, float] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    _raster_shape: tuple[int, int] = field(default=(0, 0), repr=False)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        """Warp an image-pixel mask to this full shared uint8 metric raster."""
        height, width = self._raster_shape
        if height <= 0 or width <= 0:
            raise ReferenceError("metric raster has no dimensions")
        image = np.asarray(mask)
        if image.ndim != 2:
            raise ValueError("mask must be a 2-D raster")
        warped = cv2.warpPerspective(
            image.astype(np.uint8),
            self.image_to_raster,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 0).astype(np.uint8)

    def pixel_to_metric(self, points: np.ndarray) -> np.ndarray:
        """Transform Nx2 image pixels into table metres."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an Nx2 array")
        return cv2.perspectiveTransform(points.reshape(1, -1, 2), self.homography).reshape(-1, 2)


def _read_mask(path: str) -> np.ndarray:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    image = np.asarray(imageio.imread(candidate))
    if image.ndim == 3:
        image = image[..., 0]
    return image > 0


def _reference_track(tracks: Iterable[MaskTrack]) -> MaskTrack | None:
    matches = [
        track
        for track in tracks
        if any(word in track.name.lower() or word in track.id.lower() for word in _REFERENCE_WORDS)
    ]
    # Prefer the explicit `reference` name when present over generic paper.
    return next((track for track in matches if "reference" in track.name.lower() or "reference" in track.id.lower()), matches[0] if matches else None)


def _frame_masks(tracks: Iterable[MaskTrack], frame_index: int) -> list[tuple[MaskTrack, MaskFrame, np.ndarray]]:
    result: list[tuple[MaskTrack, MaskFrame, np.ndarray]] = []
    for track in tracks:
        for frame in track.frames:
            if frame.frame_index == frame_index:
                result.append((track, frame, _read_mask(frame.mask_path)))
    return result


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Return clockwise TL/TR/BR/BL ordering with a stable top-left start."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    ordered = points[order]
    # Angle sorting may be CW or CCW depending on image y direction.  Normalize
    # with polygon orientation then rotate to the visual top-left corner.
    a, b = ordered[1] - ordered[0], ordered[2] - ordered[1]
    cross = a[0] * b[1] - a[1] * b[0]
    if cross < 0:
        ordered = ordered[::-1]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def _paper_quad(mask: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    contour_items = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_items[-2]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    corners = approximation.reshape(-1, 2) if len(approximation) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    if len(corners) != 4:
        return None
    corners = _order_corners(corners)
    side_a = float(np.linalg.norm(corners[1] - corners[0]))
    side_b = float(np.linalg.norm(corners[2] - corners[1]))
    if min(side_a, side_b) <= 0:
        return None
    quad_area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if quad_area <= 0:
        return None
    ratio_error = abs(max(side_a, side_b) / min(side_a, side_b) / (SHEET_LONG_M / SHEET_SHORT_M) - 1.0)
    area_error = abs(float(cv2.contourArea(contour)) / quad_area - 1.0)
    return corners, ratio_error, area_error


def _touches_border(mask: np.ndarray) -> bool:
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return True
    return bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))


def _metric_destination(corners: np.ndarray) -> np.ndarray:
    edge_a = float(np.linalg.norm(corners[1] - corners[0]))
    edge_b = float(np.linalg.norm(corners[2] - corners[1]))
    first, second = (SHEET_LONG_M, SHEET_SHORT_M) if edge_a >= edge_b else (SHEET_SHORT_M, SHEET_LONG_M)
    return np.asarray([[0.0, 0.0], [first, 0.0], [first, second], [0.0, second]], dtype=np.float32)


def _raster_transform(
    homography: np.ndarray,
    masks: list[tuple[MaskTrack, MaskFrame, np.ndarray]],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """Cover all selected-frame object footprints, not merely the paper."""
    pixel_points: list[np.ndarray] = []
    for _, _, mask in masks:
        ys, xs = np.nonzero(mask)
        if len(xs):
            pixel_points.append(np.column_stack((xs, ys)).astype(np.float32))
    if not pixel_points:
        height, width = image_shape
        pixel_points.append(np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32))
    metric_points = np.concatenate([cv2.perspectiveTransform(points[None, ...], homography)[0] for points in pixel_points])
    minimum = metric_points.min(axis=0) - 2 * METERS_PER_PIXEL
    maximum = metric_points.max(axis=0) + 2 * METERS_PER_PIXEL
    width = max(1, int(np.ceil((maximum[0] - minimum[0]) / METERS_PER_PIXEL)))
    height = max(1, int(np.ceil((maximum[1] - minimum[1]) / METERS_PER_PIXEL)))
    # This affine prefix turns image -> metres into image -> raster pixels.
    raster_prefix = np.asarray(
        [[1 / METERS_PER_PIXEL, 0, -minimum[0] / METERS_PER_PIXEL], [0, 1 / METERS_PER_PIXEL, -minimum[1] / METERS_PER_PIXEL], [0, 0, 1]],
        dtype=np.float64,
    )
    bounds = (float(minimum[0]), float(minimum[1]), float(minimum[0] + width * METERS_PER_PIXEL), float(minimum[1] + height * METERS_PER_PIXEL))
    return raster_prefix @ homography, bounds, (height, width)


def _manual_result(frames: list[FrameSample], tracks: list[MaskTrack], pixels_per_cm: float) -> ReferenceResult:
    if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        raise ReferenceError("manual pixels/cm must be finite and positive")
    frame = max(frames, key=lambda candidate: candidate.blur_score)
    metres_per_image_pixel = 0.01 / pixels_per_cm
    homography = np.asarray(
        [[metres_per_image_pixel, 0.0, 0.0], [0.0, metres_per_image_pixel, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    masks = _frame_masks(tracks, frame.index)
    image_to_raster, bounds, shape = _raster_transform(homography, masks, frame.image.shape[:2])
    return ReferenceResult(
        homography=homography,
        image_to_raster=image_to_raster,
        bounds=bounds,
        working_frame_index=frame.index,
        scale_source="manual_px_per_cm",
        validation={"manual_pixels_per_cm": float(pixels_per_cm)},
        diagnostics=["manual scale fallback; no paper-homography self-consistency validation"],
        _raster_shape=shape,
    )


def find_reference(
    tracks: list[MaskTrack], frames: list[FrameSample], *, manual_px_per_cm: float | None = None
) -> ReferenceResult:
    """Find a valid paper reference frame or fail closed without a manual scale."""
    if not frames:
        raise ReferenceError("reference stage needs sampled frames")
    paper = _reference_track(tracks)
    diagnostics: list[str] = []
    if paper is not None:
        by_index = {frame.index: frame for frame in frames}
        candidates: list[tuple[FrameSample, np.ndarray, float]] = []
        for mask_frame in paper.frames:
            sample = by_index.get(mask_frame.frame_index)
            if sample is None:
                continue
            mask = _read_mask(mask_frame.mask_path)
            if _touches_border(mask):
                diagnostics.append(f"paper_touches_border:{sample.index}")
                continue
            all_masks = _frame_masks(tracks, sample.index)
            object_area = float(sum(np.count_nonzero(item[2]) for item in all_masks))
            candidates.append((sample, mask, object_area))
        candidates.sort(key=lambda candidate: (-candidate[0].blur_score, -candidate[2], candidate[0].index))
        for sample, mask, _ in candidates:
            extraction = _paper_quad(mask)
            if extraction is None:
                diagnostics.append(f"paper_corners_unavailable:{sample.index}")
                continue
            corners, ratio_error, area_error = extraction
            if ratio_error > 0.05 or area_error > 0.10:
                diagnostics.append(f"paper_validation_failed:{sample.index}:ratio={ratio_error:.3f}:area={area_error:.3f}")
                continue
            destination = _metric_destination(corners)
            homography, _ = cv2.findHomography(corners.astype(np.float32), destination, method=0)
            if homography is None:
                diagnostics.append(f"homography_failed:{sample.index}")
                continue
            selected_masks = _frame_masks(tracks, sample.index)
            image_to_raster, bounds, shape = _raster_transform(homography, selected_masks, sample.image.shape[:2])
            return ReferenceResult(
                homography=homography,
                image_to_raster=image_to_raster,
                bounds=bounds,
                working_frame_index=sample.index,
                scale_source="paper_homography",
                validation={"aspect_ratio_error": ratio_error, "paper_area_error": area_error},
                diagnostics=diagnostics + ["paper ratio/area checks are self-consistency checks, not independent scale truth"],
                _raster_shape=shape,
            )
    else:
        diagnostics.append("reference_paper_track_missing")
    if manual_px_per_cm is not None:
        result = _manual_result(frames, tracks, manual_px_per_cm)
        result.diagnostics = diagnostics + result.diagnostics
        return result
    detail = "; ".join(diagnostics) if diagnostics else "no usable reference paper frame"
    raise ReferenceError(f"reference stage blocked: {detail}; provide manual_px_per_cm")
