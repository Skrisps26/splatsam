"""Presentation-only views. All outcomes and intervals come from saved artifacts."""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

COLORS = {"accent": "#aaa2ff", "mint": "#68e6c4", "muted": "#a6b2c7", "error": "#ff8e9d"}
FAILURES = {
    "slip_or_drop": "Slip / drop",
    "knock_over": "Knock-over",
    "occluder_collision": "Occluder collision",
    "instability": "Instability",
    "no_grasp_found": "No grasp",
    "timeout": "Timeout",
}


def mapping(value):
    return value if isinstance(value, dict) else {}


def items(value):
    return value if isinstance(value, list) else []


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def pct(value):
    numeric = number(value)
    return f"{numeric * 100:.1f}%" if numeric is not None else "Not reported"


def file(run: Path, relative) -> Path | None:
    if not isinstance(relative, str):
        return None
    candidate = (run / relative).resolve()
    return candidate if candidate.is_relative_to(run.resolve()) and candidate.is_file() else None


def media_file(run: Path, media: dict, key: str, fallback: str | None = None) -> Path | None:
    entry = mapping(mapping(media.get("assets")).get(key))
    # A declared asset that is absent must stay absent. Falling through to an
    # unlisted similarly named file can silently change its provenance.
    if entry:
        return file(run, entry.get("path"))
    return file(run, fallback)


def media_files(run: Path, media: dict, key: str) -> list[Path]:
    """Resolve a declared media sequence without guessing at sibling assets."""
    entry = mapping(mapping(media.get("assets")).get(key))
    paths = entry.get("path")
    if isinstance(paths, str):
        paths = [paths]
    return [resolved for path in items(paths) if (resolved := file(run, path))]


@st.cache_data(show_spinner=False)
def image_uri(path: str, modified: int) -> str:
    del modified
    try:
        data = Path(path).read_bytes()
        mime = "image/png" if Path(path).suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    except OSError:
        return ""


@st.cache_data(show_spinner=False)
def saved_track_previews(
    run_path: str, tracks_modified: int, frame_fraction: float
) -> list[tuple[str, str]]:
    """Return one saved mask from each track at the requested replay position."""
    del tracks_modified
    run = Path(run_path)
    try:
        tracks = json.loads((run / "tracks.json").read_text(encoding="utf-8"))
        masks = (run / "masks").resolve()
    except (OSError, ValueError):
        return []
    previews = []
    for track in items(tracks):
        frames = items(mapping(track).get("frames"))
        if not frames:
            continue
        index = min(round(frame_fraction * (len(frames) - 1)), len(frames) - 1)
        path = mapping(frames[index]).get("mask_path")
        if not isinstance(path, str):
            continue
        try:
            candidate = Path(path).resolve()
        except OSError:
            continue
        if candidate.is_relative_to(masks) and candidate.is_file():
            name = mapping(track).get("name") or mapping(track).get("id") or "tracked object"
            previews.append((str(candidate), str(name)))
    return previews


def track_previews(run: Path, frame_fraction: float) -> list[tuple[Path, str]]:
    tracks = run / "tracks.json"
    try:
        paths = saved_track_previews(
            str(run.resolve()), tracks.stat().st_mtime_ns, min(max(frame_fraction, 0.0), 1.0)
        )
    except OSError:
        return []
    return [(Path(path), name) for path, name in paths]


def show_image(path: Path | None, label: str, height=390):
    uri = image_uri(str(path), path.stat().st_mtime_ns) if path else ""
    if uri:
        st.markdown(
            f'<div class="reveal-frame"><img src="{uri}" alt="{html.escape(label, quote=True)}" style="width:100%;height:{height}px;object-fit:contain;background:#080c14;border-radius:12px"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty-state">This image was not exported for the recorded run.</div>',
            unsafe_allow_html=True,
        )


def show_annotation(source: Path | None, masks: list[tuple[Path, str]]) -> None:
    source_uri = image_uri(str(source), source.stat().st_mtime_ns) if source else ""
    mask_uris = [
        (image_uri(str(path), path.stat().st_mtime_ns), name) for path, name in masks
    ]
    mask_uris = [(uri, name) for uri, name in mask_uris if uri]
    if not source_uri:
        st.markdown(
            '<div class="empty-state">The declared working frame is unavailable, so its saved annotation cannot be overlaid.</div>',
            unsafe_allow_html=True,
        )
        return
    if mask_uris:
        label = (
            f"Saved persistent mask · {mask_uris[0][1]}"
            if len(mask_uris) == 1
            else f"Saved persistent masks · {len(mask_uris)} tracked objects"
        )
        overlays = "".join(
            f'<img class="annotation-mask" src="{uri}" alt="">'
            for uri, _ in mask_uris
        )
        st.markdown(
            f'<div class="annotation-frame reveal-frame"><img class="annotation-source" src="{source_uri}" alt="Saved source working frame">{overlays}<span class="annotation-tag">{html.escape(label)}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Saved source working frame with persisted tracking masks; the overlay is not a new inference.")
    else:
        st.markdown(
            f'<div class="annotation-frame reveal-frame"><img class="annotation-source" src="{source_uri}" alt="Saved source working frame"><span class="annotation-tag">No stable saved mask</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("The saved working frame is available, but no persisted tracking mask was exported.")


def show_video(path: Path | None, label: str, *, placement: str):
    if path:
        # st.video has no explicit key. Its accessible description participates
        # in Streamlit's element identity, so keep one stable description per
        # placement when a recorded clip is reused in the same render.
        st.video(str(path), autoplay=True, loop=True, muted=True, alt=f"{label} · {placement}")
        st.caption(label)
    else:
        st.markdown(
            '<div class="empty-state">This declared recorded video is unavailable in the current bundle.</div>',
            unsafe_allow_html=True,
        )


def section(title, detail=""):
    st.markdown(
        f'<div class="section-head"><h2>{html.escape(title)}</h2><p>{html.escape(detail)}</p></div>',
        unsafe_allow_html=True,
    )


def plot_style(fig, *, height=250):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1625",
        font={"family": "Manrope, sans-serif", "color": "#c7d3e8", "size": 11},
        height=height,
        margin={"l": 12, "r": 12, "t": 16, "b": 56},
        legend={"orientation": "h", "y": -0.3, "x": 0, "font": {"size": 10}},
        hoverlabel={"bgcolor": "#152038"},
    )
    fig.update_xaxes(gridcolor="#202c40", zeroline=False)
    fig.update_yaxes(gridcolor="#202c40", zeroline=False)
    return fig


def render_tuning(run: Path, report: dict, media: dict, state: dict, position: float, context: dict):
    del position
    clip_source = (
        "recorded reconstruction" if context.get("source_mode") == "live" else "saved recorded example"
    )
    rounds = [mapping(r) for r in items(state.get("rounds"))]
    section("The improvement loop", "Automatic parameter tuning · not a learned policy")
    left, right = st.columns([1.05, 1], gap="large")
    paired = mapping(media.get("paired"))
    round_clips = mapping(media.get("round_clips"))
    accepted_rounds = [r for r in rounds if r.get("accepted")]
    visible_round = str(accepted_rounds[-1].get("round")) if accepted_rounds else None
    round_asset = mapping(round_clips.get(visible_round)) if visible_round else {}
    total_accepted = sum(bool(r.get("accepted")) for r in items(report.get("rounds")))
    all_accepted_visible = bool(accepted_rounds) and len(accepted_rounds) >= total_accepted
    selected_replay = mapping(mapping(media.get("assets")).get("selected_planner_replay"))
    with left:
        if context.get("source_mode") == "live" and selected_replay.get("status") == "ready":
            clip = file(run, selected_replay.get("path"))
            label = (
                "Selected-planner training illustration · saved simulation clip "
                "(not a sealed test or real robot)"
            )
        elif round_asset:
            clip = file(run, round_asset.get("path"))
            label = f"Accepted round {visible_round} · {clip_source} scenario {round_asset.get('seed', '—')}"
        elif all_accepted_visible:
            clip = media_file(run, media, "recorded_final_replay")
            label = f"{clip_source.capitalize()} · selected-planner demonstration"
        else:
            clip = file(run, mapping(paired.get("baseline")).get("path"))
            label = f"{clip_source.capitalize()} · baseline scenario {paired.get('seed', '—')}"
        if clip:
            show_video(clip, label, placement="tuning-primary")
        else:
            st.info("No saved demonstration clip is available for the recorded tuning evidence.")
        episodes = [mapping(e) for e in items(state.get("episodes")) if e.get("phase") == "tune"]
        section("Episode wall", "Saved evaluations, revealed as events arrive")
        latest = {}
        for episode in episodes:
            latest[episode.get("seed")] = episode
        display = list(latest.values())[:200]
        slot_count = len(display)
        squares = []
        for episode in display:
            status = "success" if episode.get("success") else "failure"
            label = f"Seed {episode.get('seed')}: {'success' if episode.get('success') else episode.get('failure_mode')}"
            squares.append(
                f'<span class="episode-cell {status}" title="{html.escape(label, quote=True)}"></span>'
            )
        if squares:
            st.markdown(
                '<div class="episode-wall" role="img" aria-label="Recorded episode outcome grid">'
                + "".join(squares)
                + f'</div><div class="wall-legend"><span><i style="background:#58cfa8"></i>Success</span><span><i style="background:#df778c"></i>Failure</span><span>{slot_count} recorded scenario slots</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Recorded episode outcomes have not arrived in this replay.")
        st.caption(
            "Cached evaluations; planner-variant/round association was not recorded. The wall is not a new success-rate estimate."
        )
    with right:
        current = (
            mapping(
                rounds[-1].get(
                    "candidate_validation" if rounds[-1].get("accepted") else "baseline_validation"
                )
            )
            if rounds
            else {}
        )
        initial = mapping(rounds[0].get("baseline_validation")) if rounds else {}
        rate, baseline = number(current.get("success_rate")), number(initial.get("success_rate"))
        delta = (
            f"{(rate - baseline) * 100:+.1f} pp vs baseline"
            if rate is not None and baseline is not None
            else "Awaiting round results"
        )
        runtime = {}
        try:
            runtime = json.loads((run / "runtime.json").read_text())
        except (OSError, ValueError):
            pass
        speed = number(runtime.get("episodes_per_second"))
        st.markdown(
            f'<div class="metrics"><div class="metric"><span>Selected validation</span><strong>{pct(rate)}</strong><em>{delta}</em></div><div class="metric"><span>Recorded evaluations</span><strong>{len(episodes):,}</strong><em class="neutral">Revealed in replay</em></div><div class="metric"><span>CPU benchmark</span><strong>{speed:.1f}/s</strong><em class="neutral">Saved 100-episode pilot</em></div></div>'
            if speed is not None
            else f'<div class="metrics"><div class="metric"><span>Selected validation</span><strong>{pct(rate)}</strong><em>{delta}</em></div><div class="metric"><span>Recorded evaluations</span><strong>{len(episodes):,}</strong><em class="neutral">Revealed in replay</em></div><div class="metric"><span>Episodes / second</span><strong>—</strong><em class="neutral">Runtime not recorded</em></div></div>',
            unsafe_allow_html=True,
        )
        if rounds:
            x = [r.get("round", i + 1) for i, r in enumerate(rounds)]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[mapping(r.get("training")).get("success_rate") for r in rounds],
                    name="Train · incumbent",
                    mode="lines+markers",
                    line={"color": COLORS["accent"], "width": 3},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[mapping(r.get("candidate_validation")).get("success_rate") for r in rounds],
                    name="Validation · candidate",
                    mode="lines+markers",
                    line={"color": COLORS["mint"], "width": 3},
                )
            )
            if baseline is not None:
                fig.add_hline(
                    y=baseline,
                    line_dash="dash",
                    line_color="#8794ad",
                    annotation_text="Baseline validation",
                    annotation_position="top left",
                )
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_xaxes(title="Proposal round", dtick=1)
            st.plotly_chart(
                plot_style(fig),
                width="stretch",
                config={"displayModeBar": False},
                key="round_success",
            )
            failures = go.Figure()
            palette = ["#ff8e9d", "#f3c380", "#b19df2", "#647fa6", "#76bdcb", "#c5b58e"]
            for (failure, label), color in zip(FAILURES.items(), palette, strict=True):
                failures.add_trace(
                    go.Bar(
                        x=x,
                        y=[
                            mapping(mapping(r.get("candidate_validation")).get("failure_modes")).get(
                                failure, 0
                            )
                            for r in rounds
                        ],
                        name=label,
                        marker_color=color,
                    )
                )
            failures.update_layout(barmode="stack")
            failures.update_yaxes(title="Validation failures")
            failures.update_xaxes(dtick=1)
            st.plotly_chart(
                plot_style(failures, height=220),
                width="stretch",
                config={"displayModeBar": False},
                key="round_failures",
            )
        else:
            st.markdown(
                '<div class="empty-state">Round charts appear when a recorded round finishes.</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            "Validation is reused to select edits and is optimistic. Rejected candidates are shown, not adopted."
        )
    if paired.get("status") == "available" and all_accepted_visible:
        with st.expander(
            f"Baseline vs selected planner · same scenario {paired.get('seed')}",
            expanded=bool(accepted_rounds),
        ):
            a, b = st.columns(2)
            with a:
                show_video(
                    file(run, mapping(paired.get("baseline")).get("path")),
                    f"Baseline · seed {paired.get('seed')} · failed",
                    placement="paired-baseline",
                )
            with b:
                show_video(
                    file(run, mapping(paired.get("final")).get("path")),
                    f"Selected planner · seed {paired.get('seed')} · succeeded",
                    placement="paired-final",
                )
            st.caption(
                paired.get(
                    "provenance",
                    "Saved training-scenario media recordings; not additional evaluation metrics.",
                )
            )
    elif paired.get("status") == "available":
        st.caption("Same-scenario comparison unlocks after the last recorded accepted edit.")
    else:
        st.caption(
            "Same-scenario comparison is unavailable: no recorded baseline trace was saved. No clip is invented."
        )
    section("Improvement log", "Every proposal. Every rejection.")
    params = {}
    try:
        params = json.loads((run / "baseline_params.json").read_text())
    except (OSError, ValueError):
        pass
    for row in reversed(rounds[-6:]):
        patch = mapping(row.get("patch"))
        before = dict(params)
        for previous in rounds:
            if previous.get("round", 0) >= row.get("round", 0):
                break
            if previous.get("accepted"):
                before.update(mapping(previous.get("patch")))
        changes = (
            "; ".join(f"{key}: {before.get(key, '—')} → {value}" for key, value in patch.items())
            or "No bounded edit returned"
        )
        accepted = bool(row.get("accepted"))
        st.markdown(
            f'<div class="edit-row"><div class="edit-top"><strong>Round {row.get("round", "—")}</strong><span class="badge {"success" if accepted else "rejected"}">{"ACCEPTED" if accepted else "REJECTED"}</span></div><p class="edit-params">{html.escape(changes)}</p><p>{html.escape(str(row.get("proposal_reason", "Reason not recorded")))}</p><p>Validation change {pct(row.get("gain"))} · {html.escape(str(row.get("reason", "Not reported")))}</p></div>',
            unsafe_allow_html=True,
        )
    if not rounds:
        st.caption("The feed will update from round_finished events.")
    section("Model activity", "Sanitized calls · provenance stays visible")
    logs = []
    for call in items(state.get("llm_calls"))[-10:]:
        usage = mapping(call.get("usage"))
        total = (number(usage.get("input_tokens")) or 0) + (number(usage.get("output_tokens")) or 0)
        token_text = f"{int(total):,} tokens" if usage else "tokens not reported"
        cost = number(call.get("cost_usd"))
        cost_text = f"USD {cost:.4f}" if cost is not None else "cost not reported"
        logs.append(
            f'<div class="trace-row"><b>{html.escape(str(call.get("model", "model")))}</b> · {html.escape(str(call.get("stage", "call")))} · {token_text} · {cost_text} · {html.escape(str(call.get("provenance", "unlabeled")))}</div>'
        )
    st.markdown(
        '<div class="trace-panel" role="log" aria-label="Recent sanitized model calls">'
        + ("".join(logs) or "Waiting for recorded model activity…")
        + "</div>",
        unsafe_allow_html=True,
    )


def comparison_slider(before: Path | None, after: Path | None):
    if not before or not after:
        st.info("Both comparison images have not been exported.")
        return
    a, b = image_uri(str(before), before.stat().st_mtime_ns), image_uri(str(after), after.stat().st_mtime_ns)
    st.iframe(
        f'''<style>*{{box-sizing:border-box}}body{{margin:0;background:#0d1625;color:#e7edfc;font:13px system-ui}}.compare{{position:relative;height:330px;border-radius:12px;overflow:hidden}}img{{position:absolute;width:100%;height:100%;object-fit:contain;background:#090e18}}#front{{clip-path:inset(0 50% 0 0)}}.labels{{position:absolute;inset:12px 12px auto;display:flex;justify-content:space-between}}.labels span{{background:#111b2de8;padding:6px 9px;border-radius:6px}}input{{width:calc(100% - 30px);margin:14px 15px;accent-color:#a99eff;min-height:26px}}input:focus-visible{{outline:3px solid #68e6c4;outline-offset:2px}}</style><div class="compare"><img src="{b}" alt="Estimated physics scene"><img id="front" src="{a}" alt="Recorded source frame"><div class="labels"><span>Recorded frame</span><span>Estimated simulation</span></div></div><label for="wipe" style="position:absolute;left:-10000px">Compare recorded frame and estimated simulation</label><input id="wipe" type="range" min="0" max="100" value="50" aria-label="Comparison slider"><script>document.getElementById('wipe').addEventListener('input',function(){{document.getElementById('front').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'}})</script>''',
        height=395,
        tab_index=0,
        alt="Recorded source and estimated simulation comparison",
    )
    st.caption(
        "Drag or use arrow keys. The views use different cameras; this is a visual comparison, not pixel registration."
    )


def render_scene(run: Path, report: dict, media: dict, state: dict, elapsed: float, context: dict):
    del elapsed
    section("A scene takes shape", "Recorded frames → masks → metric footprints → physics")
    stages = mapping(state.get("stages"))
    stage = next(
        (
            key
            for key in ("understand", "track", "scale", "build", "scenarios")
            if mapping(stages.get(key)).get("status") == "running"
        ),
        "build" if state.get("scene") else "understand",
    )
    source_video = media_file(run, media, "source_video")
    source_frame = media_file(run, media, "original_frame")
    mask_reveals = media_files(run, media, "mask_reveals")
    track_fraction = 1.0
    if stage == "track":
        elapsed_track = number(mapping(stages.get("track")).get("duration_s")) or 0
        total_track = number(mapping(state.get("stage_durations")).get("track")) or 1
        track_fraction = min(elapsed_track / max(total_track, 0.001), 1.0)
    persisted_masks = track_previews(run, track_fraction) if stage != "understand" else []
    track_labels = [
        frame.get("label", "Saved persistent mask reveal")
        for frame in (mapping(frame) for frame in items(media.get("reveal_frames")))
        if frame.get("stage") == "track"
    ]
    annotation_path = None
    annotation_label = ""
    if stage != "understand" and mask_reveals:
        index = min(int(track_fraction * len(mask_reveals)), len(mask_reveals) - 1) if stage == "track" else len(mask_reveals) - 1
        annotation_path = mask_reveals[index]
        annotation_label = (
            str(track_labels[index]) if index < len(track_labels) else "Saved persistent mask reveal"
        )
    if stage == "scale":
        annotation_path = media_file(run, media, "reference_scale")
        annotation_label = "Saved reference outline and metric scale"
    elif stage == "build" and mapping(stages.get("build")).get("status") == "running":
        build_elapsed = number(mapping(stages.get("build")).get("duration_s")) or 0
        build_total = number(mapping(state.get("stage_durations")).get("build")) or 1
        reveal_key = "observed_footprints" if build_elapsed < build_total / 2 else "scene_upright"
        annotation_path = media_file(run, media, reveal_key)
        annotation_label = "Metric footprints" if reveal_key == "observed_footprints" else "Reconstructed physics scene"
    if stage == "track" and persisted_masks:
        current = min(int(track_fraction * len(persisted_masks)), len(persisted_masks) - 1)
        persisted_masks = [persisted_masks[current]]
    source, annotation, replay = st.columns([1, 1, 1.05], gap="large")
    with source:
        st.markdown('<div class="media-title">Recorded source</div>', unsafe_allow_html=True)
        if source_video:
            show_video(source_video, "Declared source video", placement="scene-source")
        else:
            show_image(source_frame, "Saved source working frame", height=400)
            st.caption("Declared source video unavailable; showing its separately declared saved working frame.")
    with annotation:
        st.markdown('<div class="media-title">Saved annotations</div>', unsafe_allow_html=True)
        if annotation_path:
            show_image(annotation_path, annotation_label, height=400)
            st.caption(f"{annotation_label}. Recorded evidence; no inference runs in the viewer.")
        elif persisted_masks:
            show_annotation(source_frame, persisted_masks)
        elif mask_reveals:
            show_image(source_frame, "Saved source working frame", height=400)
            st.caption("Persistent mask reveals begin with the recorded tracking stage.")
        else:
            show_annotation(source_frame, [])
    with replay:
        st.markdown('<div class="media-title">Robot-hand playback</div>', unsafe_allow_html=True)
        if state.get("scene"):
            replay_asset = mapping(mapping(media.get("assets")).get("selected_planner_replay"))
            replay_label = str(
                replay_asset.get("caption", replay_asset.get("provenance", "Saved robot-hand simulation replay"))
            )
            replay_path = (
                media_file(run, media, "selected_planner_replay")
                if replay_asset.get("status", "ready") == "ready"
                else None
            )
            show_video(
                replay_path,
                f"{replay_label} · illustrative playback only; not a real robot or held-out evaluation",
                placement="scene-robot-replay",
            )
        else:
            st.markdown(
                '<div class="empty-state">Robot-hand playback appears with the saved physics scene.</div>',
                unsafe_allow_html=True,
            )

    details, scale_detail = st.columns([1.8, 1], gap="large")
    with details:
        detected = items(state.get("objects"))
        st.subheader(f"{len(detected)} objects understood")
        for obj in detected:
            st.markdown(f"**{obj.get('name', obj.get('id', 'Object'))}**")
            st.caption(
                f"{obj.get('material_guess', 'Material estimate unavailable')} · {'grasp target' if obj.get('graspable') else 'reference / clutter'}"
            )
        if not detected:
            st.markdown(
                '<div class="idle-visual" aria-label="Waiting for object events"><div class="orbit"></div></div>',
                unsafe_allow_html=True,
            )
    with scale_detail:
        scale = mapping(stages.get("scale"))
        reference = mapping(context.get("reference"))
        reference_name = reference.get("name")
        reference_source = reference.get("source")
        if scale.get("status") == "done":
            label = f"Scale locked · {reference_name}" if isinstance(reference_name, str) else "Scale locked"
            st.markdown(f'<span class="badge success">{html.escape(label)}</span>', unsafe_allow_html=True)
            detail = (
                f"Metric scale from {reference_source}. Heights and physics remain estimates."
                if isinstance(reference_source, str)
                else "Metric scale comes from the saved reference. Heights and physics remain estimates."
            )
            st.caption(detail)
        else:
            st.caption("A saved metric reference will be shown when the scale stage is available.")

    if state.get("scene"):
        with st.expander("Open saved physics-scene orbit"):
            show_video(
                media_file(run, media, "scene_orbit"),
                "Saved camera orbit of the reconstructed physics scene",
                placement="scene-orbit",
            )
        with st.expander("Compare input and simulation"):
            comparison_slider(
                media_file(run, media, "original_frame"),
                media_file(run, media, "scene_upright"),
            )
        with st.expander("Object estimates and visual consistency", expanded=stage == "scenarios"):
            rows = []
            for obj in items(report.get("objects")):
                size = " × ".join(
                    f"{v * 100:.1f}" for v in items(obj.get("size_m")) if isinstance(v, (float, int))
                )
                mass = number(obj.get("mass_kg"))
                rows.append(
                    {
                        "Object": obj.get("name", obj.get("id")),
                        "Shape": obj.get("shape"),
                        "Size · cm": size,
                        "Mass · g": round(mass * 1000, 1) if mass is not None else None,
                        "Mass source": obj.get("mass_source", "unknown"),
                        "Friction source": obj.get("friction_source", "unknown"),
                        "Visual consistency": pct(obj.get("consistency_iou")),
                    }
                )
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "Visual consistency measures footprint appearance only—not mass, height, friction or physical accuracy."
            )


def render_final(run: Path, report: dict, state: dict, context: dict):
    final = mapping(state.get("final"))
    comparison = mapping(final.get("comparison"))
    baseline, selected = mapping(comparison.get("baseline")), mapping(comparison.get("final"))
    if not baseline or not selected:
        st.markdown(
            '<div class="empty-state">The held-out result stays hidden until final_test_finished arrives.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="final-card"><span class="badge success">SEALED TEST · RUN ONCE</span><h2>The result, with the uncertainty left in.</h2><p>Held-out scenarios, not the validation set used to choose edits.</p></div>',
        unsafe_allow_html=True,
    )
    values = [number(baseline.get("success_rate")), number(selected.get("success_rate"))]
    if all(v is not None for v in values):
        data = json.dumps([v * 100 for v in values])
        st.iframe(
            f'''<style>body{{margin:0;font-family:system-ui;color:#edf3ff;background:transparent}}.row{{display:flex;gap:24px;padding:14px 0}}.card{{flex:1;padding:20px;border-radius:12px;background:#111c2e}}small{{color:#b6c4dc}}strong{{display:block;font-size:44px;letter-spacing:-2px;margin-top:8px}}.final{{color:#78e5c5}}</style><div class="row"><div class="card"><small>Baseline · sealed test</small><strong id="b" aria-label="{values[0] * 100:.1f} percent">{values[0] * 100:.1f}%</strong></div><div class="card"><small>Final · sealed test</small><strong class="final" id="f" aria-label="{values[1] * 100:.1f} percent">{values[1] * 100:.1f}%</strong></div></div><script>const ends={data};if(!matchMedia('(prefers-reduced-motion: reduce)').matches){{const start=performance.now();function draw(t){{const p=Math.min((t-start)/850,1);['b','f'].forEach((id,i)=>document.getElementById(id).textContent=(ends[i]*(1-Math.pow(1-p,3))).toFixed(1)+'%');if(p<1)requestAnimationFrame(draw)}}requestAnimationFrame(draw)}}</script>''',
            height=155,
            alt="Baseline and final sealed test success rates",
        )
    fig = go.Figure()
    for name, row, color in (("Baseline", baseline, COLORS["accent"]), ("Final", selected, COLORS["mint"])):
        rate = number(row.get("success_rate"))
        ci = items(row.get("ci95"))
        if rate is not None and len(ci) == 2:
            fig.add_trace(
                go.Scatter(
                    x=[rate],
                    y=[name],
                    mode="markers",
                    name=name,
                    marker={"size": 17, "color": color},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": [ci[1] - rate],
                        "arrayminus": [rate - ci[0]],
                        "thickness": 3,
                        "width": 9,
                    },
                    hovertemplate=f"{name}: {row.get('successes')}/{row.get('n')}<br>Wilson 95% CI: {pct(ci[0])}–{pct(ci[1])}<extra></extra>",
                )
            )
    fig.update_xaxes(title="Success rate · Wilson 95% confidence intervals", tickformat=".0%", range=[0, 1])
    fig.update_layout(showlegend=False)
    st.plotly_chart(
        plot_style(fig, height=200),
        width="stretch",
        config={"displayModeBar": False},
        key="sealed_ci",
    )
    source = "Recorded reconstruction" if context.get("source_mode") == "live" else "Saved recorded example"
    st.caption(
        f"{baseline.get('successes')}/{baseline.get('n')} baseline · {selected.get('successes')}/{selected.get('n')} final · paired gain {pct(comparison.get('gain'))}. {source}; counts are copied from the saved final-test artifact."
    )
    seal = str(final.get("seal_hash", "Not recorded"))
    st.code(f"SEALED / RUN ONCE   sha256:{seal}", language=None)
    st.caption(
        "Confidence intervals and counts are copied from the saved report. Validation was reused for selection and is optimistic."
    )
    cols = st.columns(3)
    for col, name, label, mime in zip(
        cols,
        ["report.json", "report.md", "trajectory.jsonl"],
        ["Download report · JSON", "Download report · Markdown", "Download run trajectory"],
        ["application/json", "text/markdown", "application/x-ndjson"],
        strict=True,
    ):
        with col:
            path = file(run, name)
            if path:
                try:
                    st.download_button(
                        label,
                        path.read_bytes(),
                        file_name=f"splatsam-{name}",
                        mime=mime,
                        width="stretch",
                    )
                except OSError:
                    st.caption(f"{name} is not available.")
            else:
                st.caption(f"{name} was not exported.")


def render_autopsy(run: Path, media: dict, state: dict):
    failures = [e for e in items(state.get("episodes")) if e.get("success") is False]
    if not failures:
        st.info("No failed episodes have arrived in the replay yet.")
        return
    # Outcomes are selected, never re-evaluated or reclassified here.
    by_seed = {f"{e.get('phase')} · seed {e.get('seed')} · {e.get('failure_mode')}": e for e in failures}
    label = st.selectbox("Inspect a recorded failure", list(by_seed)[:300], key="autopsy_seed")
    row = by_seed[label]
    a, b = st.columns(2)
    with a:
        st.metric("Simulator label", str(row.get("failure_mode", "unavailable")))
        st.json(
            {
                k: row.get(k)
                for k in (
                    "seed",
                    "target_id",
                    "detail",
                    "max_lift_m",
                    "max_penetration_m",
                    "end_finger_contact",
                )
            }
        )
    with b:
        pair = mapping(media.get("paired"))
        if row.get("seed") == pair.get("seed") and pair.get("status") == "available":
            show_video(
                file(run, mapping(pair.get("baseline")).get("path")),
                "Illustrative baseline clip for this seed; the selected cached planner variant was not recorded",
                placement="autopsy-baseline",
            )
        else:
            st.info(
                "Keyframes were not recorded for this episode. Its label and physical measurements are available; no replacement images are invented."
            )


def model_legend():
    st.markdown(
        '<div class="model-legend"><span><b>sam-3.1</b> object tracking</span><span><b>muse-wonder-1.0</b> scene semantics & tuning proposals</span><span><b>muse-spark-1.3</b> failure critic</span></div>',
        unsafe_allow_html=True,
    )
