"""Video frame sampling for perception, with source-frame timing preserved."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One RGB source frame selected from a video."""

    index: int
    timestamp_s: float
    image: np.ndarray
    blur_score: float


def video_info(video_path: Path) -> tuple[int, float]:
    """Return decoded frame count and frames per second, rejecting unreadable media."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"video has invalid frame metadata: {video_path}")
    return count, fps


def sample_frames(video_path: Path, n: int = 12, out_dir: Path | None = None) -> list[FrameSample]:
    """Sample evenly spaced RGB frames using their actual source indexes.

    ``imageio`` performs the decoding; OpenCV supplies robust metadata and the
    Laplacian-variance sharpness measurement.  The optional image directory is
    only an inspection artifact and does not affect the in-memory samples.
    """
    video_path = Path(video_path)
    if n <= 0:
        raise ValueError("n must be positive")
    count, fps = video_info(video_path)
    indexes = np.unique(np.linspace(0, count - 1, min(n, count), dtype=np.int64))
    wanted = {int(index) for index in indexes}
    decoded: dict[int, np.ndarray] = {}
    for index, image in enumerate(imageio.get_reader(video_path)):
        if index not in wanted:
            continue
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"frame {index} does not have RGB channels")
        decoded[index] = np.ascontiguousarray(rgb[..., :3])
        if len(decoded) == len(wanted):
            break
    missing = sorted(wanted.difference(decoded))
    if missing:
        raise ValueError(f"decoder did not return selected source frames: {missing}")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    for index in indexes:
        source_index = int(index)
        rgb = decoded[source_index]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sample = FrameSample(
            index=source_index,
            timestamp_s=source_index / fps,
            image=rgb,
            blur_score=blur,
        )
        samples.append(sample)
        if out_dir is not None:
            imageio.imwrite(out_dir / f"frame_{source_index:06d}.png", rgb)
    return samples
