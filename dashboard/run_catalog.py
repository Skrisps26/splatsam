"""Safe, local lookup of recorded live runs by uploaded video digest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordedRun:
    """A run whose source video is still privately available and unchanged."""

    path: Path
    source_video: Path
    video_sha256: str
    has_presentation_events: bool
    input_modified_ns: int


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


def match_uploaded_video(records: list[RecordedRun], digest: str) -> RecordedRun | None:
    partial = None
    for record in records:
        if record.video_sha256 != digest or _sha256_file(record.source_video) != record.video_sha256:
            continue
        if record.has_presentation_events:
            return record
        if partial is None:
            partial = record
    return partial


def catalog_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Return cheap invalidation data without retaining any video bytes."""
    watched: list[tuple[str, int, int]] = []
    for directory, pattern in ((root / "runs", "input.json"), (root / "runs", "events.jsonl")):
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
