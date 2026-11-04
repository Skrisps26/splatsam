"""Persistent-mask tracking adapters for parsed SAM video responses and fixtures."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.optimize import linear_sum_assignment

from splatsam.llm.schemas import MaskFrame, MaskTrack, ObjectInfo, ObjectList
from splatsam.perception.frames import video_info
from splatsam.utils import load_json

if TYPE_CHECKING:
    from splatsam.llm.client import ModelClient


MAX_JUMP_FRACTION = 0.25
MIN_VISIBILITY = 0.60


@dataclass(frozen=True, slots=True)
class InstanceObservation:
    """A per-frame candidate used by :func:`associate_instances`."""

    frame_index: int
    centroid: tuple[float, float]
    area: int
    source_id: str = ""
    mask: np.ndarray | None = None
    box: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class AssociatedInstance:
    """Motion-associated observations under one first-frame identity."""

    id: str
    observations: list[InstanceObservation] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def _as_observation(
    value: InstanceObservation | dict[str, Any] | Sequence[Any], frame_index: int
) -> InstanceObservation:
    if isinstance(value, InstanceObservation):
        return value
    if isinstance(value, dict):
        centroid = value["centroid"]
        return InstanceObservation(
            frame_index=int(value.get("frame_index", frame_index)),
            centroid=(float(centroid[0]), float(centroid[1])),
            area=int(value["area"]),
            source_id=str(value.get("source_id", "")),
            mask=value.get("mask"),
            box=tuple(value["box"]) if value.get("box") is not None else None,
        )
    if len(value) < 2:
        raise ValueError("instance observation tuple needs centroid and area")
    centroid, area = value[0], value[1]
    return InstanceObservation(
        frame_index=frame_index,
        centroid=(float(centroid[0]), float(centroid[1])),
        area=int(area),
        source_id=str(value[2]) if len(value) > 2 else "",
    )


def _mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    union = int(np.count_nonzero(np.logical_or(left, right)))
    return 0.0 if union == 0 else float(np.count_nonzero(np.logical_and(left, right)) / union)


def associate_instances(
    detections_by_frame: Sequence[Sequence[InstanceObservation | dict[str, Any] | Sequence[Any]]],
    image_width: int,
    *,
    phrase: str = "object",
    max_jump_fraction: float = MAX_JUMP_FRACTION,
) -> list[AssociatedInstance]:
    """Associate same-class instances by predicted motion and mask IoU.

    Identities are assigned once from first-frame position (left-to-right, then
    top-to-bottom).  Later source ordering is ignored.  A constant-velocity
    prediction prevents identity swaps at a crossing; pairs beyond the jump gate
    are not assigned.  The returned observations retain their parser/source IDs.
    """
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    gate = image_width * max_jump_fraction
    instances: list[AssociatedInstance] = []
    next_id = 1
    for sequence_index, raw_detections in enumerate(detections_by_frame):
        detections = [_as_observation(item, sequence_index) for item in raw_detections]
        if not instances:
            for detection in sorted(
                detections, key=lambda item: (item.centroid[0], item.centroid[1], item.source_id)
            ):
                instances.append(AssociatedInstance(id=f"{phrase}-{next_id}", observations=[detection]))
                next_id += 1
            continue
        active = [instance for instance in instances if instance.observations]
        if not detections:
            continue
        cost = np.full((len(active), len(detections)), np.inf, dtype=float)
        for row, instance in enumerate(active):
            latest = instance.observations[-1]
            predicted = np.asarray(latest.centroid, dtype=float)
            if len(instance.observations) >= 2:
                previous = instance.observations[-2]
                predicted += np.asarray(latest.centroid) - np.asarray(previous.centroid)
            for column, detection in enumerate(detections):
                distance = float(np.linalg.norm(np.asarray(detection.centroid) - predicted))
                if distance <= gate:
                    # IoU distinguishes overlapping candidates without making a
                    # translating object require overlap with its old silhouette.
                    cost[row, column] = distance / image_width - 0.15 * _mask_iou(latest.mask, detection.mask)
        valid_rows, valid_columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))
        assigned: set[int] = set()
        for row, column in zip(valid_rows, valid_columns, strict=True):
            if not np.isfinite(cost[row, column]):
                continue
            active[row].observations.append(detections[column])
            assigned.add(int(column))
        for column, detection in enumerate(detections):
            if column not in assigned:
                instances.append(
                    AssociatedInstance(
                        id=f"{phrase}-{next_id}",
                        observations=[detection],
                        diagnostics=["new_instance_after_motion_gate"],
                    )
                )
                next_id += 1
    return instances


def stability_gate(track: MaskTrack) -> tuple[bool, list[str]]:
    """Apply the visibility, jump, and area-ratio track quality gates."""
    diagnostics = list(track.diagnostics)
    frames = sorted(track.frames, key=lambda frame: frame.frame_index)
    if track.total_frames <= 0:
        diagnostics.append("nonpositive_total_frames")
    visibility = len(frames) / track.total_frames if track.total_frames > 0 else 0.0
    if visibility < MIN_VISIBILITY:
        diagnostics.append(f"visibility_below_{MIN_VISIBILITY:.2f}:{visibility:.3f}")
    for previous, current in itertools.pairwise(frames):
        distance = float(np.linalg.norm(np.subtract(current.centroid, previous.centroid)))
        if distance > MAX_JUMP_FRACTION * track.image_width:
            diagnostics.append(f"centroid_jump:{previous.frame_index}->{current.frame_index}")
        if previous.area <= 0 or current.area <= 0:
            diagnostics.append(f"nonpositive_area:{previous.frame_index}->{current.frame_index}")
            continue
        ratio = current.area / previous.area
        if ratio > 3.0 or ratio < 1.0 / 3.0:
            diagnostics.append(f"area_ratio:{previous.frame_index}->{current.frame_index}:{ratio:.3f}")
    return not diagnostics, diagnostics


def _frame_from_mask(*, frame_index: int, timestamp_s: float, mask: np.ndarray, path: Path) -> MaskFrame:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("SAM produced an empty mask")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return MaskFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        mask_path=str(path.resolve()),
        centroid=(float(xs.mean()), float(ys.mean())),
        area=len(xs),
        box=(float(left), float(top), float(right), float(bottom)),
    )


def _paste_parsed_mask(record: Any, image_height: int, image_width: int) -> np.ndarray:
    from meta_sam_parser import decode_mask_to_raster

    decoded = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(
        record.mask.height, record.mask.width
    )
    bounds = record.bounds
    left, top = round(bounds.left), round(bounds.top)
    right, bottom = round(bounds.right), round(bounds.bottom)
    if right <= left or bottom <= top:
        right, bottom = left + decoded.shape[1], top + decoded.shape[0]
    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    dst_left, dst_top = max(0, left), max(0, top)
    dst_right, dst_bottom = min(image_width, right), min(image_height, bottom)
    if dst_right <= dst_left or dst_bottom <= dst_top:
        raise ValueError("parsed SAM mask lies outside video dimensions")
    src_left, src_top = dst_left - left, dst_top - top
    copy_height, copy_width = dst_bottom - dst_top, dst_right - dst_left
    canvas[dst_top:dst_bottom, dst_left:dst_right] = decoded[
        src_top : src_top + copy_height, src_left : src_left + copy_width
    ]
    return canvas


def _records(result: Any) -> list[Any]:
    parsed = result.get("parsed") if isinstance(result, dict) else getattr(result, "parsed", None)
    records = getattr(parsed, "records", None)
    return list(records) if records is not None else []


def _refined_phrase(client: ModelClient, object_info: ObjectInfo) -> str:
    """Use typed Wonder output when available; retain the observed noun on fallback."""
    try:
        reply = client.generate(
            ObjectList,
            model="muse-wonder-1.0",
            instructions="Return exactly one concise visible-object name for the requested noun phrase.",
            input=f"Refine the visible object noun phrase: {object_info.name}",
            stage="sam_phrase_refinement",
        )
    except Exception:  # noqa: BLE001 - unavailable critic/proposer retains original phrase
        return object_info.name
    if reply.objects:
        return reply.objects[0].name
    return object_info.name


class MockSamClient:
    """Synthetic persistent tracks supplied from the generator, not scene truth."""

    def __init__(self, tracks_path: Path) -> None:
        self.tracks_path = Path(tracks_path)

    def track(self, video_path: Path, objects: ObjectList) -> list[MaskTrack]:
        del video_path, objects
        raw = load_json(self.tracks_path)
        if not isinstance(raw, list):
            raise ValueError("synthetic tracks artifact must be a JSON list")  # noqa: TRY004 - artifact validation, not a Python argument error
        return [MaskTrack.model_validate(item) for item in raw]


class SamClient:
    """Real SAM adapter consuming only official ``meta_sam_parser`` records."""

    def __init__(
        self, client: ModelClient, output_dir: Path, *, box_prompts: dict[str, dict] | None = None
    ) -> None:
        self.client = client
        self.output_dir = Path(output_dir)
        self.box_prompts = box_prompts or {}

    def _recover_with_box(
        self, video_path: Path, phrase: str, records: Iterable[Any], width: int, height: int
    ) -> list[Any]:
        boxes = [record for record in records if getattr(record, "kind", None) == "box"]
        if not boxes:
            return []
        box = boxes[0]
        frame = getattr(getattr(box, "frame", None), "frame_index", 0)
        payload = {
            "size": (width, height),
            "box": (int(box.left), int(box.top), int(box.right), int(box.bottom)),
            "frame_index": int(frame),
            "object_id": 1,
        }
        try:
            return _records(self.client.sam_request(video_path, phrase, box_prompt=payload))
        except Exception:  # noqa: BLE001 - failed final recovery drops only the affected target
            return []

    def track(self, video_path: Path, objects: ObjectList) -> list[MaskTrack]:
        from splatsam.authorization import require_live

        require_live("B1", video_path)
        video_path = Path(video_path)
        video_frames, fps = video_info(video_path)
        reader = imageio.get_reader(video_path)
        try:
            height, width = np.asarray(reader.get_data(0)).shape[:2]
        finally:
            reader.close()
        tracks: list[MaskTrack] = []
        phrases: dict[str, ObjectInfo] = {}
        for item in objects.objects:
            key = item.name.strip().lower()
            if key in phrases:
                previous = phrases[key]
                phrases[key] = previous.model_copy(update={"count": previous.count + item.count})
            else:
                phrases[key] = item

        def decode(info: ObjectInfo, records: list[Any]) -> list[MaskTrack]:
            latest: dict[tuple[str, int], Any] = {}
            for record in records:
                if getattr(record, "kind", None) != "mask":
                    continue
                key = (str(record.object_id), int(record.frame.frame_index))
                previous = latest.get(key)
                if previous is None or record.revision >= previous.revision:
                    latest[key] = record
            observations: dict[int, list[InstanceObservation]] = defaultdict(list)
            for (source_id, index), record in latest.items():
                try:
                    raster = _paste_parsed_mask(record, height, width)
                    ys, xs = np.nonzero(raster)
                    if not len(xs):
                        continue
                    observations[index].append(
                        InstanceObservation(
                            index, (float(xs.mean()), float(ys.mean())), len(xs), source_id, raster
                        )
                    )
                except (ValueError, TypeError, AttributeError):
                    continue
            associated = associate_instances(
                [observations[i] for i in sorted(observations)], width, phrase=info.id
            )
            decoded = []
            for instance in associated:
                if not instance.observations:
                    continue
                source_id = instance.observations[0].source_id or instance.id
                samples = []
                for observation in sorted(instance.observations, key=lambda o: o.frame_index):
                    record = latest.get((observation.source_id, observation.frame_index))
                    if record is None:
                        continue
                    raster = _paste_parsed_mask(record, height, width)
                    safe_object = sha256(info.id.encode("utf-8")).hexdigest()[:16]
                    path = (
                        self.output_dir
                        / "masks"
                        / safe_object
                        / source_id
                        / f"frame_{observation.frame_index:06d}.png"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(path, (raster * 255).astype(np.uint8))
                    samples.append(
                        _frame_from_mask(
                            frame_index=observation.frame_index,
                            timestamp_s=observation.frame_index / fps,
                            mask=raster,
                            path=path,
                        )
                    )
                diagnostics = list(instance.diagnostics)
                if len({o.source_id for o in instance.observations}) > 1:
                    diagnostics.append("parser_id_changed_motion_association")
                track = MaskTrack(
                    id=f"{info.id}:{source_id}",
                    name=info.name,
                    frames=samples,
                    image_width=width,
                    image_height=height,
                    total_frames=video_frames,
                    source="live",
                    diagnostics=diagnostics,
                )
                stable, reasons = stability_gate(track)
                decoded.append(track.model_copy(update={"stable": stable, "diagnostics": reasons}))
            decoded.sort(
                key=lambda t: (not t.stable, t.frames[0].centroid[0] if t.frames else float("inf"), t.id)
            )
            return decoded[: info.count]

        for info in phrases.values():
            records: list[Any] = []
            candidates: list[MaskTrack] = []
            errors = []
            # Only failure triggers a paid phrase refinement, then a separate box
            # prompt. Never combine phrase text with the box grammar.
            for attempt in range(3):
                try:
                    if attempt == 0:
                        records = _records(
                            self.client.sam_request(
                                video_path, info.name, box_prompt=self.box_prompts.get(info.id)
                            )
                        )
                    elif attempt == 1:
                        records = _records(
                            self.client.sam_request(video_path, _refined_phrase(self.client, info))
                        )
                    else:
                        records = self._recover_with_box(video_path, info.name, records, width, height)
                    candidates = decode(info, records)
                    if candidates and all(t.stable for t in candidates):
                        break
                except Exception as error:  # noqa: BLE001 - retain explicit per-target diagnostics instead of aborting the scene
                    errors.append(type(error).__name__)
            if not candidates:
                candidates = [
                    MaskTrack(
                        id=info.id,
                        name=info.name,
                        frames=[],
                        image_width=width,
                        image_height=height,
                        total_frames=video_frames,
                        source="fallback",
                        stable=False,
                        diagnostics=["dropped_no_usable_masks", *errors],
                    )
                ]
            for candidate in candidates:
                if not candidate.stable and candidate.frames:
                    candidate = candidate.model_copy(
                        update={"diagnostics": candidate.diagnostics + ["clutter_only_unreliable_identity"]}
                    )
                tracks.append(candidate)
            self.client._trajectory(
                provenance="tracking_decision",
                stage="sam_stability",
                model="sam-3.1",
                request={"object": info.id},
                response=[{"id": t.id, "stable": t.stable, "diagnostics": t.diagnostics} for t in candidates],
            )
        return tracks


def _colour(identifier: str) -> tuple[int, int, int]:
    digest = sha256(identifier.encode("utf-8")).digest()
    return tuple(64 + value % 160 for value in digest[:3])


def _read_mask(path: str, root: Path) -> np.ndarray:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    mask = np.asarray(imageio.imread(resolved))
    return mask[..., 0] > 0 if mask.ndim == 3 else mask > 0


def save_overlays(video_path: Path, tracks: list[MaskTrack], out_dir: Path) -> Path:
    """Write a persistent-ID overlay video using real decoded RGB frames."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_index: dict[int, list[tuple[MaskTrack, MaskFrame]]] = defaultdict(list)
    for track in tracks:
        for frame in track.frames:
            by_index[frame.frame_index].append((track, frame))
    if not by_index:
        raise ValueError("cannot render overlay without mask frames")
    source_indexes = sorted(by_index)
    chosen = np.unique(np.linspace(0, len(source_indexes) - 1, min(12, len(source_indexes)), dtype=int))
    wanted = {source_indexes[int(item)] for item in chosen}
    rgb_by_index: dict[int, np.ndarray] = {}
    for index, image in enumerate(imageio.get_reader(video_path)):
        if index in wanted:
            rgb_by_index[index] = np.asarray(image)[..., :3].copy()
        if len(rgb_by_index) == len(wanted):
            break
    root = video_path.parent
    rendered: list[np.ndarray] = []
    timestamps: list[float] = []
    for index in sorted(wanted):
        image = rgb_by_index.get(index)
        if image is None:
            continue
        overlay = image.copy()
        for track, frame in by_index[index]:
            mask = _read_mask(frame.mask_path, root)
            colour = np.asarray(_colour(track.id), dtype=np.uint8)
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * colour).astype(np.uint8)
            left, top, _, _ = frame.box
            cv2.putText(
                overlay,
                f"{track.name} [{track.id.rsplit(':', 1)[-1]}]",
                (int(left), max(14, int(top) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tuple(int(v) for v in colour),
                1,
                cv2.LINE_AA,
            )
            timestamps.append(frame.timestamp_s)
        rendered.append(overlay)
    if not rendered:
        raise ValueError("selected overlay frames could not be decoded")
    ordered_times = sorted(set(timestamps))
    duration = max(1.0, ordered_times[-1] - ordered_times[0]) if len(ordered_times) > 1 else 1.0
    fps = max(1.0, len(rendered) / duration)
    output = out_dir / "overlay.mp4"
    imageio.mimwrite(output, rendered, fps=fps, codec="libx264", macro_block_size=1)
    return output
