"""Safe, local lookup of recorded live runs by uploaded video content."""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RecordedRun:
    """A run whose source video is still privately available and unchanged."""

    path: Path
    source_video: Path
    video_sha256: str
    has_presentation_events: bool
    input_modified_ns: int


@dataclass(frozen=True)
class _DecodedVideo:
    duration_seconds: float
    samples: tuple[np.ndarray, ...]


_CONTENT_SAMPLE_COUNT = 12
_MAX_DECODED_FRAMES = 3_600
_NORMALIZED_FRAME_SIZE = (64, 64)
_MAX_SAMPLE_MEAN_ABSOLUTE_ERROR = 0.08
_MAX_AVERAGE_MEAN_ABSOLUTE_ERROR = 0.035


def sha256_bytes(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _input_metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _private_video_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    videos = (root / "data" / "videos").resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.is_relative_to(videos) and resolved.is_file() else None


def _run_is_replayable(run: Path) -> bool:
    status = _input_metadata(run / "pipeline_status.json").get("status")
    if isinstance(status, str) and status.lower() in {"blocked", "error", "failed"}:
        return False
    presentation = _input_metadata(run / "presentation.json")
    if any(
        isinstance(provenance := presentation.get(field), str)
        and provenance.lower() in {"fallback", "synthetic"}
        for field in ("source_mode", "origin")
    ):
        return False
    scene_path = run / "scene.json"
    return not scene_path.is_file() or _input_metadata(scene_path).get("source") == "live"


def _normalise_frame(frame: np.ndarray) -> np.ndarray | None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, _NORMALIZED_FRAME_SIZE, interpolation=cv2.INTER_AREA).astype(
        np.float32
    ) / 255.0


def _decoded_video(path: Path) -> _DecodedVideo | None:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        frame_count = 0
        first_timestamp: float | None = None
        previous_timestamp: float | None = None
        last_timestamp: float | None = None
        intervals: list[float] = []
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frame_count += 1
            if frame_count > _MAX_DECODED_FRAMES:
                return None
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1_000.0
            if not np.isfinite(timestamp) or timestamp < 0:
                return None
            if first_timestamp is None:
                first_timestamp = timestamp
            elif previous_timestamp is not None:
                interval = timestamp - previous_timestamp
                if interval < 0:
                    return None
                if interval > 0:
                    intervals.append(interval)
            previous_timestamp = timestamp
            last_timestamp = timestamp
    finally:
        capture.release()
    if (
        frame_count < 2
        or first_timestamp is None
        or last_timestamp is None
        or not intervals
    ):
        return None
    duration = last_timestamp - first_timestamp + float(np.median(intervals))
    if duration <= 0 or not np.isfinite(duration):
        return None
    targets = np.linspace(
        first_timestamp,
        last_timestamp,
        num=min(frame_count, _CONTENT_SAMPLE_COUNT),
        dtype=np.float64,
    )
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        samples: list[np.ndarray] = []
        target_index = 0
        previous_frame: np.ndarray | None = None
        previous_timestamp = None
        while target_index < len(targets):
            ok, frame = capture.read()
            if not ok:
                return None
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1_000.0
            normalised = _normalise_frame(frame)
            if not np.isfinite(timestamp) or normalised is None:
                return None
            while target_index < len(targets) and timestamp >= targets[target_index]:
                if (
                    previous_frame is None
                    or previous_timestamp is None
                    or timestamp - targets[target_index] <= targets[target_index] - previous_timestamp
                ):
                    samples.append(normalised)
                else:
                    samples.append(previous_frame)
                target_index += 1
            previous_frame = normalised
            previous_timestamp = timestamp
    finally:
        capture.release()
    return _DecodedVideo(duration_seconds=duration, samples=tuple(samples))


def _decoded_upload(upload_bytes: bytes | memoryview) -> _DecodedVideo | None:
    try:
        with tempfile.NamedTemporaryFile(prefix="splatsam-upload-", suffix=".mp4", delete=False) as upload:
            upload.write(upload_bytes)
            upload_path = Path(upload.name)
    except (OSError, TypeError):
        return None
    try:
        return _decoded_video(upload_path)
    finally:
        with contextlib.suppress(OSError):
            upload_path.unlink()


def _same_visual_video(upload: _DecodedVideo, recorded: _DecodedVideo) -> bool:
    duration_difference = abs(upload.duration_seconds - recorded.duration_seconds)
    if duration_difference > max(0.1, 0.02 * max(upload.duration_seconds, recorded.duration_seconds)):
        return False
    if len(upload.samples) != len(recorded.samples):
        return False
    errors = [float(np.mean(np.abs(left - right))) for left, right in zip(upload.samples, recorded.samples)]
    return (
        bool(errors)
        and max(errors) <= _MAX_SAMPLE_MEAN_ABSOLUTE_ERROR
        and float(np.mean(errors)) <= _MAX_AVERAGE_MEAN_ABSOLUTE_ERROR
    )


def discover_recorded_runs(root: Path) -> list[RecordedRun]:
    """Return verified live runs only; invalid or stale source metadata is ignored."""
    runs_root = (root / "runs").resolve()
    if not runs_root.is_dir():
        return []
    found: list[RecordedRun] = []
    for run in runs_root.iterdir():
        if not run.is_dir():
            continue
        run = run.resolve()
        if not run.is_relative_to(runs_root):
            continue
        if not _run_is_replayable(run):
            continue
        metadata_path = run / "input.json"
        metadata = _input_metadata(metadata_path)
        digest = metadata.get("video_sha256")
        if (
            metadata.get("source") != "live"
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            continue
        video = _private_video_path(root, metadata.get("video_path"))
        if video is None or _sha256_file(video) != digest:
            continue
        try:
            modified = metadata_path.stat().st_mtime_ns
        except OSError:
            continue
        found.append(
            RecordedRun(
                path=run,
                source_video=video,
                video_sha256=digest,
                has_presentation_events=(run / "events.jsonl").is_file(),
                input_modified_ns=modified,
            )
        )
    return sorted(found, key=lambda item: item.input_modified_ns, reverse=True)


def match_uploaded_video(
    records: list[RecordedRun],
    digest: str,
    upload_bytes: bytes | memoryview | None = None,
) -> RecordedRun | None:
    """Return a verified recording with exactly or visually the same source footage."""
    verified: list[RecordedRun] = []
    exact_match = None
    for record in records:
        if _sha256_file(record.source_video) != record.video_sha256:
            continue
        verified.append(record)
        if record.video_sha256 == digest:
            if record.has_presentation_events:
                return record
            if exact_match is None:
                exact_match = record
    if exact_match is not None:
        return exact_match
    if upload_bytes is None:
        return None
    upload = _decoded_upload(upload_bytes)
    if upload is None:
        return None
    visual_match = None
    for record in verified:
        decoded_recording = _decoded_video(record.source_video)
        if decoded_recording is None or not _same_visual_video(upload, decoded_recording):
            continue
        if record.has_presentation_events:
            return record
        if visual_match is None:
            visual_match = record
    return visual_match

def catalog_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Return cheap invalidation data without retaining any video bytes."""
    watched: list[tuple[str, int, int]] = []
    for directory, pattern in (
        (root / "runs", "input.json"),
        (root / "runs", "events.jsonl"),
        (root / "runs", "pipeline_status.json"),
        (root / "runs", "presentation.json"),
        (root / "runs", "scene.json"),
    ):
        if not directory.is_dir():
            continue
        for path in directory.glob(f"*/{pattern}"):
            try:
                stat = path.stat()
            except OSError:
                continue
            watched.append((str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size))
    videos = root / "data" / "videos"
    if videos.is_dir():
        for path in videos.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            watched.append((str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(watched))


__all__ = [
    "RecordedRun",
    "catalog_signature",
    "discover_recorded_runs",
    "match_uploaded_video",
    "sha256_bytes",
]
