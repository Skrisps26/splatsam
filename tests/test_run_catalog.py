import json
from pathlib import Path

import cv2
import numpy as np

from dashboard.run_catalog import RecordedRun, discover_recorded_runs, match_uploaded_video, sha256_bytes


def _write_video(path: Path, *, unrelated: bool = False, scale: int = 1) -> None:
    width, height = 96 * scale, 64 * scale
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (width, height)
    )
    assert writer.isOpened()
    try:
        for frame_index in range(48):
            if unrelated:
                frame = np.full((64, 96, 3), 230, dtype=np.uint8)
                cv2.rectangle(frame, (0, 0), (95, 63), (15, 15, 15), 3)
            else:
                gradient = np.linspace(20, 160, 96, dtype=np.uint8)
                frame = np.repeat(gradient[None, :, None], 64, axis=0)
                frame = np.repeat(frame, 3, axis=2)
                center = ((12 + frame_index * 3) % 80 + 8, 32)
                cv2.circle(frame, center, 7, (30, 220, 80), -1)
                cv2.putText(
                    frame,
                    str(frame_index),
                    (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (240, 60, 30),
                    1,
                )
            if scale > 1:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
            writer.write(frame)
    finally:
        writer.release()


def _recorded_run(source_video: Path, *, events: bool = True) -> RecordedRun:
    return RecordedRun(
        path=source_video.parent / "recorded-run",
        source_video=source_video,
        video_sha256=sha256_bytes(source_video.read_bytes()),
        has_presentation_events=events,
        input_modified_ns=0,
    )


def test_exact_match_never_falls_back_to_an_unrelated_recording(tmp_path):
    source_video = tmp_path / "recorded.mp4"
    source_video.write_bytes(b"verified recording")
    record = _recorded_run(source_video)

    assert match_uploaded_video([record], record.video_sha256) is record
    assert match_uploaded_video([record], sha256_bytes(b"unrelated upload")) is None


def test_reencoded_upload_matches_only_the_same_decoded_footage(tmp_path):
    source_video = tmp_path / "recorded.mp4"
    reencoded_upload = tmp_path / "upload.mp4"
    unrelated_upload = tmp_path / "unrelated.mp4"
    _write_video(source_video)
    _write_video(reencoded_upload, scale=2)
    _write_video(unrelated_upload, unrelated=True, scale=2)
    record = _recorded_run(source_video)
    upload_bytes = reencoded_upload.read_bytes()

    assert sha256_bytes(upload_bytes) != record.video_sha256
    assert match_uploaded_video([record], sha256_bytes(upload_bytes), upload_bytes) is record

    unrelated_bytes = unrelated_upload.read_bytes()
    assert match_uploaded_video([record], sha256_bytes(unrelated_bytes), unrelated_bytes) is None


def test_discovery_excludes_runs_with_terminal_pipeline_failures(tmp_path):
    videos = tmp_path / "data" / "videos"
    videos.mkdir(parents=True)
    video = videos / "desk.mp4"
    video.write_bytes(b"verified recording")
    digest = sha256_bytes(video.read_bytes())
    runs = tmp_path / "runs"
    for name, status in (
        ("ready", None),
        ("blocked", "blocked"),
        ("failed", "failed"),
        ("error", "error"),
    ):
        run = runs / name
        run.mkdir(parents=True)
        (run / "input.json").write_text(
            json.dumps(
                {
                    "source": "live",
                    "video_path": "data/videos/desk.mp4",
                    "video_sha256": digest,
                }
            ),
            encoding="utf-8",
        )
        if status is not None:
            (run / "pipeline_status.json").write_text(
                json.dumps({"status": status}), encoding="utf-8"
            )

    assert [record.path.name for record in discover_recorded_runs(tmp_path)] == ["ready"]


def test_discovery_rejects_synthetic_presentation_and_scene_provenance(tmp_path):
    videos = tmp_path / "data" / "videos"
    videos.mkdir(parents=True)
    video = videos / "desk.mp4"
    video.write_bytes(b"verified recording")
    digest = sha256_bytes(video.read_bytes())
    for name, presentation, scene in (
        ("ready", None, None),
        ("synthetic-presentation", {"source_mode": "synthetic"}, None),
        ("synthetic-origin", {"origin": "synthetic"}, None),
        ("synthetic-scene", None, {"source": "synthetic"}),
    ):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "input.json").write_text(
            json.dumps(
                {
                    "source": "live",
                    "video_path": "data/videos/desk.mp4",
                    "video_sha256": digest,
                }
            ),
            encoding="utf-8",
        )
        if presentation is not None:
            (run / "presentation.json").write_text(json.dumps(presentation), encoding="utf-8")
        if scene is not None:
            (run / "scene.json").write_text(json.dumps(scene), encoding="utf-8")

    assert [record.path.name for record in discover_recorded_runs(tmp_path)] == ["ready"]
