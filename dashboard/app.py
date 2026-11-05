# THESIS: a recorded workspace becomes an inspectable physics sandbox, not a debug page.
# OWN-WORLD: midnight navy, lilac-to-mint display, translucent sixteen-pixel panels.
# STORY: choose a video, follow evidence, watch bounded tuning, inspect the sealed result.
# FIRST VIEWPORT: oversized SplatSAM title, clear replay provenance, real simulation media.
# FORM: user-pinned product-demo brief overrides direction roll cac93094; code-led Experience.
# FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from panels import model_legend, render_autopsy, render_final, render_scene, render_tuning
from replay_data import event_seconds, load_events, project_state
from run_catalog import (
    RecordedRun,
    catalog_signature,
    discover_recorded_runs,
    match_uploaded_video,
    sha256_bytes,
)

STAGES = {
    "understand": "Understand scene",
    "track": "Track objects · sam-3.1",
    "scale": "Measure scale reference",
    "build": "Build physics world",
    "scenarios": "Generate scenarios",
    "tune": "Automatic tuning",
    "sealed": "Sealed test",
}


@st.cache_data(show_spinner=False)
def skin(modified: int) -> str:
    del modified
    css = (HERE / "styles.css").read_text()
    fonts = []
    for name, weight in (("Manrope-Regular.ttf", 400), ("Manrope-Bold.ttf", 700)):
        path = HERE / "assets" / name
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode()
            fonts.append(
                f"@font-face{{font-family:Manrope;src:url(data:font/ttf;base64,{encoded}) format('truetype');font-weight:{weight};font-display:swap}}"
            )
    return "<style>" + "".join(fonts) + css + "</style>"


@st.cache_data(show_spinner=False)
def cached_json(path: str, modified: int) -> dict:
    del modified
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeDecodeError):
        return {}


def read_json(path: Path) -> dict:
    try:
        return cached_json(str(path), path.stat().st_mtime_ns)
    except OSError:
        return {}


@st.cache_data(show_spinner=False)
def cached_events(path: str, modified: int) -> list[dict]:
    del modified
    return load_events(Path(path))


def events_for(run: Path) -> list[dict]:
    try:
        return cached_events(str(run), (run / "events.jsonl").stat().st_mtime_ns)
    except OSError:
        return []


@st.cache_data(show_spinner=False)
def cached_recorded_runs(root: str, signature: tuple[tuple[str, int, int], ...]) -> list[RecordedRun]:
    del signature
    return discover_recorded_runs(Path(root))


def recorded_runs() -> list[RecordedRun]:
    return cached_recorded_runs(str(ROOT), catalog_signature(ROOT))


def presentation_context(presentation: dict, media: dict | None = None) -> dict:
    metadata = presentation.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    media = media if isinstance(media, dict) else {}
    media_metadata = media.get("metadata")
    media_metadata = media_metadata if isinstance(media_metadata, dict) else {}
    reference = presentation.get(
        "reference",
        metadata.get("reference", media.get("reference", media_metadata.get("reference", {}))),
    )
    return {
        "source_mode": presentation.get(
            "source_mode",
            metadata.get(
                "source_mode",
                media.get(
                    "source_mode",
                    media_metadata.get("source_mode", presentation.get("origin", "recorded")),
                ),
            ),
        ),
        "reference": reference if isinstance(reference, dict) else {},
        "timestamp_basis": presentation.get("timestamp_basis"),
    }


def source_label(context: dict) -> str:
    return "Recorded reconstruction" if context.get("source_mode") == "live" else "Recorded example"


def load_run(run: Path) -> tuple[dict, dict, dict, list[dict]]:
    presentation = read_json(run / "presentation.json")
    return (
        read_json(run / "report.json"),
        read_json(run / "presentation/media.json"),
        presentation,
        events_for(run),
    )


def asset(run: Path, relative: str | None) -> Path | None:
    if not isinstance(relative, str):
        return None
    candidate = (run / relative).resolve()
    return candidate if candidate.is_relative_to(run.resolve()) and candidate.is_file() else None


def replay_position() -> float:
    offset = float(st.session_state.get("replay_offset", 0))
    if st.session_state.get("replay_playing", False):
        offset += (
            time.monotonic() - st.session_state.get("replay_anchor", time.monotonic())
        ) * st.session_state.get("replay_speed", 1.0)
    return max(0.0, offset)


def start_replay(run_path: str) -> None:
    st.session_state.replay_run = run_path
    st.session_state.replay_offset = 0.0
    st.session_state.replay_anchor = time.monotonic()
    st.session_state.replay_playing = True
    st.session_state.replay_started = True


def pause_or_resume() -> None:
    position = replay_position()
    st.session_state.replay_offset = position
    st.session_state.replay_anchor = time.monotonic()
    st.session_state.replay_playing = not st.session_state.get("replay_playing", False)


def speed_changed() -> None:
    old_speed = st.session_state.get("replay_speed", 1.0)
    elapsed = time.monotonic() - st.session_state.get("replay_anchor", time.monotonic())
    if st.session_state.get("replay_playing", False):
        st.session_state.replay_offset = st.session_state.get("replay_offset", 0.0) + elapsed * old_speed
    st.session_state.replay_anchor = time.monotonic()
    st.session_state.replay_speed = st.session_state.speed_control


def seek_changed() -> None:
    st.session_state.replay_offset = float(st.session_state.seek_control)
    st.session_state.replay_anchor = time.monotonic()
    st.session_state.replay_playing = False


def chapter_changed() -> None:
    chapter = st.session_state.get("replay_chapter")
    offsets = st.session_state.get("chapter_offsets", {})
    if chapter in offsets:
        st.session_state.replay_offset = offsets[chapter]
        st.session_state.replay_anchor = time.monotonic()
        st.session_state.replay_playing = False


def render_stepper(state: dict) -> None:
    cells = []
    for index, (key, label) in enumerate(STAGES.items(), 1):
        item = state.get("stages", {}).get(key, {})
        status = item.get("status", "pending")
        duration = item.get("duration_s")
        text = f"{duration:.1f}s in replay" if isinstance(duration, (int, float)) else "Waiting for event"
        icon = "✓" if status == "done" else str(index)
        cells.append(
            f'<div class="step {status}"><span class="step-dot">{icon}</span><strong>{html.escape(label)}</strong><small>{text}</small></div>'
        )
    st.markdown(
        '<div class="stepper" aria-label="Pipeline stages">' + "".join(cells) + "</div>",
        unsafe_allow_html=True,
    )


def upload_changed() -> None:
    st.session_state.sample_selected = False


def render_upload(sample_run: Path, records: list[RecordedRun]) -> str | None:
    with st.container(border=True):
        st.subheader("Start with a workspace")
        st.caption("Match a previously processed video, or select the bundled example.")
        uploaded = st.file_uploader(
            "Workspace video",
            type=["mp4"],
            label_visibility="collapsed",
            key="workspace_upload",
            on_change=upload_changed,
            help="MP4/H.264 recommended. The viewer only compares local bytes and never starts a pipeline.",
        )
        left, right = st.columns([1, 1])
        with left:
            if st.button("Use bundled example", width="stretch"):
                st.session_state.sample_selected = True
        with right:
            st.caption("Select an example only when you want its recorded replay.")

        if st.session_state.get("sample_selected"):
            source = asset(sample_run, "presentation/source.mp4")
            if source:
                st.video(str(source), muted=True, width=360, alt="Bundled example input")
            ready = bool(events_for(sample_run))
            st.button(
                "Build my robot sandbox",
                type="primary",
                width="stretch",
                on_click=start_replay,
                args=(str(sample_run),),
                disabled=not ready,
            )
            if ready:
                st.caption("Bundled synthetic example · recorded replay only; no pipeline run starts.")
            else:
                st.info("The bundled example is being prepared; its recorded presentation is not ready.")
            return "synthetic"

        if uploaded is None:
            st.info("Choose a video to look up a recorded reconstruction, or select the bundled example.")
            st.button("Build my robot sandbox", type="primary", disabled=True, width="stretch")
            return None

        upload_bytes = uploaded.getvalue()
        match = match_uploaded_video(records, sha256_bytes(upload_bytes))
        st.video(upload_bytes, muted=True, width=360)
        if match is not None and match.has_presentation_events:
            st.success("Recorded reconstruction found for these exact bytes.")
            st.button(
                "Build my robot sandbox",
                type="primary",
                width="stretch",
                on_click=start_replay,
                args=(str(match.path),),
            )
            st.caption("Opens saved reconstruction evidence. No pipeline or API call starts.")
            return "live"
        if match is not None:
            st.info(
                "This upload matches a verified reconstruction. Its presentation is still being prepared, so replay is unavailable."
            )
        else:
            st.warning(
                "No recorded reconstruction matches these exact bytes. This upload will not be replaced by an example."
            )
            if st.button("Save upload for a later authorized run"):
                digest = sha256_bytes(upload_bytes)
                destination = ROOT / "data/videos" / f"upload_{digest[:12]}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(upload_bytes)
                st.success(f"Saved privately as {destination.name}. No inference started.")
        st.button("Build my robot sandbox", type="primary", disabled=True, width="stretch")
        st.caption("Preparation and authorization are required before a new reconstruction can be recorded.")
        return None


def render_replay(run: Path, report: dict, media: dict, presentation: dict, events: list[dict]) -> None:
    context = presentation_context(presentation, media)
    times = event_seconds(events)
    duration = max(times) if times else 75.0
    position = min(replay_position(), duration)
    state = project_state(events, position)
    offsets = {"Result": duration}
    chapter_stages = {"track": "Masks", "scale": "Scale", "build": "Physics", "tune": "Tuning"}
    round_times = []
    stage_starts = {}
    stage_durations = {}
    for event, timestamp in zip(events, times, strict=True):
        if event.get("type") == "stage_started":
            stage_starts[event.get("payload", {}).get("stage")] = timestamp
            chapter = chapter_stages.get(event.get("payload", {}).get("stage"))
            if chapter:
                offsets[chapter] = min(timestamp + 2, duration)
        elif event.get("type") == "stage_finished":
            stage = event.get("payload", {}).get("stage")
            if stage in stage_starts:
                stage_durations[stage] = max(0.0, timestamp - stage_starts[stage])
        elif event.get("type") == "scene_built":
            offsets["Physics"] = min(timestamp + 0.1, duration)
        elif event.get("type") == "round_finished":
            round_times.append(timestamp)
    if round_times:
        offsets["Tuning"] = round_times[max(0, len(round_times) - 2)] + 0.1
    st.session_state.chapter_offsets = offsets
    state["stage_durations"] = stage_durations
    cost = state.get("cost_usd") or {}
    recorded_cost = cost.get("meta_usd")
    if recorded_cost is None and state.get("llm_calls"):
        try:
            recorded_cost = sum(float(call.get("cost_usd", 0)) for call in state["llm_calls"])
        except (TypeError, ValueError):
            recorded_cost = None
    cost_text = (
        f"USD {recorded_cost:.4f}" if isinstance(recorded_cost, (int, float)) else "reported at reveal"
    )
    st.markdown(
        f'<div class="topbar"><span class="brand-small">SplatSAM</span><span class="replay-label">Replaying a recorded run</span><div class="topbar-right"><span>{html.escape(source_label(context))}</span><span>Recorded Meta: {cost_text}</span><span>{position:04.1f}s / {duration:.0f}s</span></div></div>',
        unsafe_allow_html=True,
    )
    controls = st.columns([1, 1, 3])
    with controls[0]:
        if position >= duration:
            st.button("Restart", on_click=start_replay, args=(str(run),), width="stretch")
        else:
            st.button(
                "Pause" if st.session_state.get("replay_playing") else "Resume",
                on_click=pause_or_resume,
                width="stretch",
            )
    with controls[1]:
        st.selectbox(
            "Replay speed",
            [0.5, 1.0, 2.0, 4.0],
            index=1,
            key="speed_control",
            on_change=speed_changed,
            format_func=lambda n: f"{n:g}×",
            label_visibility="collapsed",
        )
    with controls[2]:
        st.progress(min(position / max(duration, 1), 1.0))
    st.pills(
        "Review a recorded chapter",
        ["Masks", "Scale", "Physics", "Tuning", "Result"],
        key="replay_chapter",
        on_change=chapter_changed,
        label_visibility="collapsed",
    )
    render_stepper(state)
    if context.get("source_mode") == "synthetic":
        st.caption("Bundled synthetic example · saved presentation timeline, not measured pipeline runtime.")
    else:
        st.caption("Saved presentation timeline; no pipeline or API call is started by this replay.")
    st.markdown(
        '<div class="live-caption"><span class="live-dot"></span>'
        + html.escape(state.get("caption") or "Ready to reveal the workspace")
        + "</div>",
        unsafe_allow_html=True,
    )
    if context.get("source_mode") == "live" and not state.get("final"):
        st.info(
            "Completed recorded reconstruction and available tuning evidence are shown below. The sealed held-out test is pending and has not run."
        )
    if state.get("final"):
        render_final(run, report, state, context)
        with st.expander("Revisit the improvement loop"):
            render_tuning(run, report, media, state, position, context)
        with st.expander("Inspect the reconstructed scene"):
            render_scene(run, report, media, state, position, context)
    elif state.get("stages", {}).get("tune", {}).get("status") != "pending":
        tabs = st.tabs(["Improvement loop", "Scene evidence", "Failure autopsy"])
        with tabs[0]:
            render_tuning(run, report, media, state, position, context)
        with tabs[1]:
            render_scene(run, report, media, state, position, context)
        with tabs[2]:
            render_autopsy(run, media, state)
    else:
        render_scene(run, report, media, state, position, context)
    model_legend()
    with st.expander("Review a moment"):
        st.slider(
            "Seek in the recorded timeline (pauses playback)",
            0,
            int(duration),
            0,
            key="seek_control",
            on_change=seek_changed,
        )
    if position >= duration:
        st.session_state.replay_offset = duration
        st.session_state.replay_playing = False
        st.button("Watch it again", on_click=start_replay, args=(str(run),), type="primary")


def main() -> None:
    st.set_page_config(
        page_title="SplatSAM · Robot sandbox", layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(skin((HERE / "styles.css").stat().st_mtime_ns), unsafe_allow_html=True)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default="demo_data")
    args, _ = parser.parse_known_args()
    requested = (ROOT / args.run_dir).resolve()
    sample_run = (
        requested
        if requested.is_relative_to(ROOT)
        and read_json(requested / "presentation.json").get("origin") == "synthetic"
        else ROOT / "demo_data"
    )
    records = recorded_runs()
    if not st.session_state.get("replay_started", False):
        st.markdown(
            '<div class="topbar"><span class="brand-small">SplatSAM</span><div class="topbar-right"><span class="mode-pill">REPLAY</span><span>Local artifact viewer</span><span>No pipeline run starts here</span></div></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown(
                '<div class="hero"><h1>SplatSAM</h1><p>Phone video in.<br>Physics sandbox out.</p><p class="hero-detail">Inspect a recorded reconstruction with its saved masks, scene, and bounded tuning evidence.</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<div class="idle-visual" aria-label="Select a local video or recorded example"><div class="orbit"></div></div>',
                unsafe_allow_html=True,
            )
        render_upload(sample_run, records)
        st.markdown(
            '<div class="limits-strip">Simulation only. Rule-based grasp planner; floating gripper, no arm. Shapes, heights, mass, and friction remain estimates unless recorded otherwise. Sealed tests are shown only after saved final-test evidence exists.</div>',
            unsafe_allow_html=True,
        )
        return

    allowed_runs = {sample_run.resolve(), *(record.path.resolve() for record in records)}
    selected_value = st.session_state.get("replay_run")
    selected = Path(selected_value).resolve() if isinstance(selected_value, str) else None
    if selected not in allowed_runs:
        st.session_state.replay_started = False
        st.warning("The selected replay is no longer available. Choose a local video or recorded example.")
        return
    report, media, presentation, events = load_run(selected)
    if not events:
        st.warning(
            "This recorded reconstruction is still being prepared; its presentation event log is not ready."
        )
        return

    @st.fragment(run_every=0.5)
    def replay_fragment():
        render_replay(selected, report, media, presentation, events)

    replay_fragment()
    context = presentation_context(presentation, media)
    if context.get("source_mode") == "live":
        footer = (
            "Simulation only. Recorded reconstruction; no live compute runs here. "
            "Shapes and physical properties are estimates unless saved as measured."
        )
    else:
        footer = (
            "Simulation only. Rule-based grasp planner and floating gripper, with no arm. "
            "Shapes and physical properties are estimates unless saved as measured."
        )
    st.markdown(f'<div class="limits-strip">{html.escape(footer)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
