# THESIS: a recorded workspace becomes an inspectable physics sandbox, not a debug page.
# OWN-WORLD: midnight navy, lilac-to-mint display, translucent sixteen-pixel panels.
# STORY: choose a video, follow evidence, watch bounded tuning, inspect the sealed result.
# FIRST VIEWPORT: oversized SplatSAM title, clear replay provenance, real simulation media.
# FORM: user-pinned product-demo brief overrides direction roll cac93094; code-led Experience.
# FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
from __future__ import annotations

import argparse
import base64
import hashlib
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

STAGES = {
    "understand": "Understand scene",
    "track": "Track objects · sam-3.1",
    "scale": "Measure scale · A4 sheet",
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


def start_replay() -> None:
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


def render_upload(run: Path, media: dict) -> None:
    with st.container(border=True):
        st.subheader("Start with a workspace")
        st.caption("Drop in a phone video, or explore a complete recorded sandbox.")
        uploaded = st.file_uploader(
            "Workspace video",
            type=["mp4"],
            label_visibility="collapsed",
            key="workspace_upload",
            on_change=upload_changed,
            help="MP4/H.264 recommended. Uploads stay local and are never sent to a model by this replay-only viewer.",
        )
        left, right = st.columns([1, 1])
        with left:
            if st.button("Use sample video", use_container_width=True):
                st.session_state.sample_selected = True
        with right:
            st.caption("12 seconds · six objects · A4 scale reference")
        if uploaded is not None and not st.session_state.get("sample_selected"):
            st.video(uploaded.getvalue(), muted=True, width=360)
            st.info(
                "This upload is local only. Live execution is paused; its results are not replaced by the recorded sample."
            )
            if st.button("Save upload for a later live run"):
                digest = hashlib.sha256(uploaded.getbuffer()).hexdigest()
                destination = ROOT / "data/videos" / f"upload_{digest[:12]}.mp4"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(uploaded.getbuffer())
                st.success(f"Saved privately as {destination.name}. No inference started.")
            st.button("Build my robot sandbox", type="primary", disabled=True, use_container_width=True)
            st.caption(
                "Choose Use sample video to launch the replay. Live launch is hidden rather than bypassing privacy, calibration or sealed-test approvals."
            )
        else:
            source = asset(run, "presentation/source.mp4")
            if st.session_state.get("sample_selected") and source:
                st.video(str(source), muted=True, width=360, alt="Unsegmented synthetic workspace input")
            st.button(
                "Build my robot sandbox",
                type="primary",
                use_container_width=True,
                on_click=start_replay,
                disabled=not bool(events_for(run)),
            )
            st.caption("Starts a recorded synthetic replay—not a new API or GPU run.")


def render_replay(run: Path, report: dict, media: dict, events: list[dict]) -> None:
    times = event_seconds(events)
    duration = max(times) if times else 75.0
    position = min(replay_position(), duration)
    state = project_state(events, position)
    offsets = {"Result": duration}
    chapter_stages = {"track": "Masks", "scale": "Scale", "build": "Physics", "tune": "Tuning"}
    round_times = []
    for event, timestamp in zip(events, times, strict=True):
        if event.get("type") == "stage_started":
            chapter = chapter_stages.get(event.get("payload", {}).get("stage"))
            if chapter:
                offsets[chapter] = min(timestamp + 2, duration)
        elif event.get("type") == "round_finished":
            round_times.append(timestamp)
    if round_times:
        offsets["Tuning"] = round_times[max(0, len(round_times) - 2)] + 0.1
    st.session_state.chapter_offsets = offsets
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
        f'<div class="topbar"><span class="brand-small">SplatSAM</span><span class="replay-label">Replaying a recorded run</span><div class="topbar-right"><span>{html.escape(run.name)}</span><span>Recorded Meta: {cost_text}</span><span>{position:04.1f}s / {duration:.0f}s</span></div></div>',
        unsafe_allow_html=True,
    )
    controls = st.columns([1, 1, 3])
    with controls[0]:
        if position >= duration:
            st.button("Restart", on_click=start_replay, use_container_width=True)
        else:
            st.button(
                "Pause" if st.session_state.get("replay_playing") else "Resume",
                on_click=pause_or_resume,
                use_container_width=True,
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
    st.caption(
        "Synthetic sample · mocked model responses · presentation-paced event timeline, not measured pipeline runtime."
    )
    st.markdown(
        '<div class="live-caption"><span class="live-dot"></span>'
        + html.escape(state.get("caption") or "Ready to reveal the workspace")
        + "</div>",
        unsafe_allow_html=True,
    )
    if state.get("final"):
        render_final(run, report, state)
        with st.expander("Revisit the improvement loop"):
            render_tuning(run, report, media, state, position)
        with st.expander("Inspect the reconstructed scene"):
            render_scene(run, report, media, state, position)
    elif state.get("stages", {}).get("tune", {}).get("status") != "pending":
        tabs = st.tabs(["Improvement loop", "Scene evidence", "Failure autopsy"])
        with tabs[0]:
            render_tuning(run, report, media, state, position)
        with tabs[1]:
            render_scene(run, report, media, state, position)
        with tabs[2]:
            render_autopsy(run, media, state)
    else:
        render_scene(run, report, media, state, position)
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
        st.button("Watch it again", on_click=start_replay, type="primary")


def main() -> None:
    st.set_page_config(
        page_title="SplatSAM · Robot sandbox", layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(skin((HERE / "styles.css").stat().st_mtime_ns), unsafe_allow_html=True)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default="demo_data")
    args, _ = parser.parse_known_args()
    requested = (ROOT / args.run_dir).resolve()
    run = requested if (requested / "events.jsonl").exists() else ROOT / "demo_data"
    report = read_json(run / "report.json")
    media = read_json(run / "presentation/media.json")
    events = events_for(run)
    started = st.session_state.get("replay_started", False)
    if not started:
        st.markdown(
            '<div class="topbar"><span class="brand-small">SplatSAM</span><div class="topbar-right"><span class="mode-pill">REPLAY</span><span>Recorded synthetic workspace</span><span>No API key or GPU needed</span></div></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown(
                '<div class="hero"><h1>SplatSAM</h1><p>Phone video in.<br>Physics sandbox out.</p><p class="hero-detail">A robot that gets measurably better—in this recorded synthetic experiment. Rule-based planning, transparent tuning, and an honest final test.</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            preview = asset(run, "presentation/scene_upright.png") or asset(run, "sim.png")
            if preview:
                st.image(str(preview), use_container_width=True)
        render_upload(run, media)
    else:
        if not events:
            st.warning(
                "This run has no presentation event log. Choose the bundled sample or run scripts/export_ui_events.py on an existing artifact directory."
            )
            return

        @st.fragment(run_every=0.5)
        def replay_fragment():
            render_replay(run, report, media, events)

        replay_fragment()
    st.markdown(
        '<div class="limits-strip">Simulation only. Rule-based grasp planner, tuned by model proposals. Floating gripper, no arm. Shapes are estimated from video. Test set sealed until the end. The bundled experiment uses synthetic inputs and mocked model responses.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
