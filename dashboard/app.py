# THESIS: laboratory evidence — source/simulation first, honest sealed metrics.
# WORLD: native Streamlit with a restrained light palette and system font; no hero, cards, kicker, or icons.
# FORM: pinned Operate lab notebook (brief seed bbe0e52e), with paired evidence first.
# This viewer is read-only: it never invokes models, GPU, or tests.
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default="runs/synth-demo")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


ARGS = parse_args()
DEFAULT_RUN_DIR = (PROJECT_ROOT / ARGS.run_dir).resolve()


def safe_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


@st.cache_data(show_spinner=False)
def cached_report(path_text: str, modified_ns: int) -> dict[str, Any] | None:
    del modified_ns
    return safe_json(Path(path_text))


def read_report(run_dir: Path, refresh_live: bool) -> dict[str, Any] | None:
    report_path = run_dir / "report.json"
    try:
        modified_ns = report_path.stat().st_mtime_ns
    except OSError:
        return None
    if refresh_live:
        return safe_json(report_path)
    return cached_report(str(report_path), modified_ns)


def discover_runs(default_run_dir: Path) -> list[Path]:
    candidates: set[Path] = set()
    for base in (PROJECT_ROOT / "runs", PROJECT_ROOT / "demo_data"):
        if base.is_dir():
            candidates.update(path.parent.resolve() for path in base.rglob("report.json"))
    if (default_run_dir / "report.json").is_file():
        candidates.add(default_run_dir)
    return sorted(candidates, key=lambda path: str(path).casefold())


def run_label(run_dir: Path, report: Mapping[str, Any] | None) -> str:
    try:
        relative = run_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = run_dir
    mode = report.get("mode") if report else None
    suffix = f" · {mode}" if isinstance(mode, str) else ""
    return f"{relative}{suffix}"


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def percent(value: Any, decimals: int = 1) -> str:
    numeric = number(value)
    return "Not reported" if numeric is None else f"{numeric * 100:.{decimals}f}%"


def count(value: Any) -> str:
    numeric = number(value)
    return "Not reported" if numeric is None else f"{int(numeric):,}"


def interval(value: Any) -> str:
    values = as_list(value)
    if len(values) != 2 or any(number(item) is None for item in values):
        return "Not reported"
    return f"{percent(values[0])} to {percent(values[1])}"


def money(value: Any) -> str:
    numeric = number(value)
    return "Not reported" if numeric is None else f"USD {numeric:,.2f}"


def short_hash(value: Any) -> str:
    return value[:12] if isinstance(value, str) else "Not reported"


def first_existing(run_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def image_or_notice(path: Path | None, label: str) -> None:
    if path:
        st.image(str(path), caption=label, use_container_width=True)
    else:
        st.info(f"{label} is not available in this completed artifact.")


def video_or_notice(path: Path | None, label: str) -> None:
    if path:
        st.video(str(path))
        st.caption(label)
    else:
        st.info(f"{label} is not available in this completed artifact.")


def render_run_status(report: Mapping[str, Any], run_dir: Path, viewer_mode: str) -> None:
    mode = report.get("mode", "unlabeled")
    stages = as_list(report.get("stages"))
    stage_text = ", ".join(
        f"{item.get('stage', 'unnamed')}: {item.get('status', 'unknown')}"
        for item in stages
        if isinstance(item, Mapping)
    )
    mode_text = str(mode).capitalize()
    viewer_text = "cached local replay" if viewer_mode == "Cached replay" else "live artifact refresh"
    st.markdown(
        f"**Run:** `{run_dir.name}` &nbsp;&nbsp; **Evidence mode:** {mode_text} &nbsp;&nbsp; "
        f"**Viewer:** {viewer_text}"
    )
    if stage_text:
        st.caption(f"Completed stages: {stage_text}")
    if mode == "synthetic" or report.get("illustrative_only") is True:
        st.markdown('<span class="synthetic-badge">SYNTHETIC ILLUSTRATION — NOT REAL-VIDEO EVIDENCE</span>', unsafe_allow_html=True)
    elif mode == "live":
        st.caption("Live artifact viewer only. This page cannot dispatch inference, provision GPU, or run a final test.")


def render_media(run_dir: Path) -> None:
    st.subheader("Perception and scene evidence")
    media_choice = st.selectbox(
        "Perception view",
        ("Tracked overlay", "Source frame"),
        help="Switches only the local artifact shown at left.",
    )
    source_asset = (
        first_existing(run_dir, ("overlay.mp4",))
        if media_choice == "Tracked overlay"
        else first_existing(run_dir, ("source.png",))
    )
    source_label = "Persistent-ID tracking overlay" if media_choice == "Tracked overlay" else "Selected source frame"
    left, right = st.columns(2, gap="large")
    with left:
        if source_asset and source_asset.suffix.lower() == ".mp4":
            video_or_notice(source_asset, source_label)
        else:
            image_or_notice(source_asset, source_label)
    with right:
        image_or_notice(first_existing(run_dir, ("sim.png",)), "Reconstructed simulation scene")
    st.caption("The left artifact is perception evidence; the right is an estimated simulation, not a real-world replay. Primitive colors identify objects, not recovered textures.")


def render_held_out(report: Mapping[str, Any]) -> None:
    st.subheader("Sealed held-out measurement")
    held_out = as_mapping(report.get("held_out"))
    baseline = as_mapping(held_out.get("baseline"))
    final = as_mapping(held_out.get("final"))
    if not baseline or not final:
        st.warning("Final test not completed. Sealed baseline/final metrics are unavailable rather than treated as zero.")
        return

    rows = []
    for label, item in (("Baseline rule settings", baseline), ("Final rule settings", final)):
        rows.append(
            {
                "Planner": label,
                "Successes / n": f"{count(item.get('successes'))} / {count(item.get('n'))}",
                "Success rate": percent(item.get("success_rate")),
                "Wilson 95% CI": interval(item.get("ci95")),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    gain = number(held_out.get("gain"))
    gain_ci = held_out.get("gain_ci95")
    if gain is not None:
        st.markdown(f"**Paired gain:** {percent(gain)} &nbsp; **Paired bootstrap 95% CI:** {interval(gain_ci)}")
    else:
        st.caption("Paired gain was not reported.")
    measurement = report.get("measurement")
    if isinstance(measurement, str):
        st.caption(f"Scoring rule: {measurement}")


def render_validation(report: Mapping[str, Any]) -> None:
    st.subheader("Validation selection evidence")
    st.caption("Validation selected edits and is optimistic by design. It is separate from the sealed held-out measurement above.")
    rounds = [as_mapping(item) for item in as_list(report.get("rounds")) if isinstance(item, Mapping)]
    if not rounds:
        st.info("No completed validation rounds are available in this artifact.")
        return

    round_numbers = [item.get("round", index + 1) for index, item in enumerate(rounds)]
    incumbent = [number(as_mapping(item.get("baseline_validation")).get("success_rate")) for item in rounds]
    candidate = [number(as_mapping(item.get("candidate_validation")).get("success_rate")) for item in rounds]
    figure = go.Figure()
    if any(value is not None for value in incumbent):
        figure.add_trace(go.Scatter(x=round_numbers, y=incumbent, mode="lines+markers", name="Incumbent validation"))
    if any(value is not None for value in candidate):
        figure.add_trace(go.Scatter(x=round_numbers, y=candidate, mode="lines+markers", name="Candidate validation"))
    figure.update_layout(
        title="Validation rounds — optimistic selection evidence, not held-out",
        height=320,
        margin=dict(l=10, r=10, t=48, b=100),
        yaxis=dict(tickformat=".0%", rangemode="tozero", title="Success rate"),
        xaxis=dict(title="Round", dtick=1),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    accepted_count = sum(item.get("accepted") is True for item in rounds)
    st.caption(
        f"{len(rounds)} evaluated rounds; {accepted_count} accepted. "
        f"{count(report.get('validation_evaluations'))} validation planner-scenario evaluations, "
        f"{count(report.get('fresh_validation_simulations'))} fresh simulations."
    )
    rows = []
    for item in rounds:
        patch = as_mapping(item.get("patch"))
        patch_text = ", ".join(f"{key}={value}" for key, value in patch.items()) or "No patch reported"
        rows.append(
            {
                "Round": item.get("round", "Not reported"),
                "Decision": "Accepted" if item.get("accepted") is True else "Rejected",
                "Patch": patch_text,
                "Validation gain": percent(item.get("gain")),
                "Bootstrap lower 10%": percent(item.get("bootstrap_lower10")),
                "Reason": item.get("reason", "Not reported"),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_failure_breakdown(report: Mapping[str, Any]) -> None:
    held_out = as_mapping(report.get("held_out"))
    baseline = as_mapping(held_out.get("baseline"))
    final = as_mapping(held_out.get("final"))
    baseline_failures = as_mapping(baseline.get("failure_modes"))
    final_failures = as_mapping(final.get("failure_modes"))
    if not baseline_failures and not final_failures:
        return
    labels = list(dict.fromkeys([*baseline_failures.keys(), *final_failures.keys()]))
    if not labels:
        return
    figure = go.Figure()
    figure.add_trace(go.Bar(name="Baseline", x=labels, y=[number(baseline_failures.get(label)) for label in labels]))
    figure.add_trace(go.Bar(name="Final", x=labels, y=[number(final_failures.get(label)) for label in labels]))
    figure.update_layout(
        title="Held-out failure distribution",
        barmode="group",
        height=330,
        margin=dict(l=10, r=10, t=48, b=100),
        yaxis=dict(title="Episodes", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_geometry(report: Mapping[str, Any], run_dir: Path) -> None:
    st.subheader("Reconstruction and geometry evidence")
    scale = as_mapping(report.get("scale"))
    scale_source = scale.get("source")
    independent_error = number(scale.get("independent_error"))
    scale_bits = []
    if isinstance(scale_source, str):
        scale_bits.append(f"Scale source: {scale_source}")
    if independent_error is not None:
        scale_bits.append(f"Independent synthetic scale error: {percent(independent_error, 3)}")
    st.caption(" · ".join(scale_bits) if scale_bits else "Scale provenance was not reported.")

    objects = [as_mapping(item) for item in as_list(report.get("objects")) if isinstance(item, Mapping)]
    if objects:
        rows = []
        for item in objects:
            size = as_list(item.get("size_m"))
            size_text = " × ".join(f"{value:.3f}" for raw in size if (value := number(raw)) is not None)
            rows.append(
                {
                    "Object": item.get("id", item.get("name", "Not reported")),
                    "Primitive": item.get("shape", "Not reported"),
                    "Footprint IoU": percent(item.get("consistency_iou"), 1),
                    "Reconstruction": item.get("reconstruction_source", "Not reported"),
                    "Dimensions (m)": size_text or "Not reported",
                    "Mass (kg)": f"{number(item.get('mass_kg')):.4f}" if number(item.get("mass_kg")) is not None else "Not reported",
                    "Mass source": item.get("mass_source", "Not reported"),
                    "Friction": str(item.get("friction", "Not reported")),
                    "Friction source": item.get("friction_source", "Not reported"),
                    "Height source": item.get("height_source", "Not reported"),
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No per-object reconstruction evidence is available in this artifact.")

    comparison_paths = sorted((run_dir / "consistency").glob("*_comparison.png")) if (run_dir / "consistency").is_dir() else []
    if comparison_paths:
        choices = {path.stem.removesuffix("_comparison"): path for path in comparison_paths}
        selected = st.selectbox("Object silhouette comparison", list(choices), key="comparison_object")
        st.image(str(choices[selected]), caption=f"Observed and fitted footprint comparison: {selected}", use_container_width=True)


def render_grasp_and_trace(run_dir: Path) -> None:
    st.subheader("Physical replay and trace")
    video_or_notice(first_existing(run_dir, ("grasp.mp4",)), "Representative MuJoCo grasp replay")
    trajectory_path = run_dir / "trajectory.jsonl"
    if trajectory_path.is_file():
        try:
            lines = trajectory_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        with st.expander(f"Sanitized trajectory trace ({len(lines)} records)"):
            if lines:
                st.code("\n".join(lines), language="json")
            else:
                st.info("The trajectory artifact is empty.")
    else:
        st.info("No sanitized trajectory trace is available in this artifact.")


def render_limits(report: Mapping[str, Any]) -> None:
    limits = [item for item in as_list(report.get("limits")) if isinstance(item, str)]
    st.divider()
    st.subheader("Interpretation limits")
    if limits:
        for limit in limits:
            st.markdown(f"- {limit}")
    else:
        st.caption("No run-specific interpretation limits were reported.")


def main() -> None:
    st.set_page_config(page_title="Splatsam", page_icon=None, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background: #fbfaf7; color: #182126; }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #142d36; letter-spacing: -0.025em; }
        h1 { font-weight: 680; }
        [data-testid="stSidebar"] { background: #f2f1eb; }
        [data-testid="stSidebar"] .stButton > button, .stDownloadButton > button { border-radius: 6px; }
        [data-testid="stDataFrame"] { width: 100%; }
        .synthetic-badge { display: inline-block; margin: .35rem 0 .9rem; padding: .22rem .5rem; border: 1px solid #9b6700; color: #5e3d00; background: #fff4d3; font-size: .78rem; font-weight: 650; letter-spacing: .02em; }
        *:focus-visible { outline: 3px solid #2c6e7e !important; outline-offset: 2px; }
        ::selection { background: #b9dde0; color: #142d36; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Splatsam")
    st.caption("A read-only laboratory notebook for video-derived scene evidence, simulation, and sealed grasp measurements.")

    discovered = discover_runs(DEFAULT_RUN_DIR)
    report_lookup = {run_dir: read_report(run_dir, refresh_live=False) for run_dir in discovered}
    usable = [run_dir for run_dir in discovered if report_lookup.get(run_dir) is not None]
    if not usable:
        st.warning("No completed report artifact is available. Run `make synth-demo` to create the local synthetic evidence bundle.")
        st.caption("This viewer does not create runs, dispatch models, provision GPU, or run final tests.")
        return

    with st.sidebar:
        st.header("Run evidence")
        saved_scope = st.session_state.get("artifact_scope", "Synthetic")
        scope = st.radio("Evidence mode", ("Synthetic", "Live"), index=0 if saved_scope == "Synthetic" else 1)
        st.session_state.artifact_scope = scope
        matching = [
            run_dir
            for run_dir in usable
            if str(as_mapping(report_lookup[run_dir]).get("mode", "")).casefold() == scope.casefold()
        ]
        if not matching:
            matching = usable
            st.caption(f"No completed {scope.casefold()} report was found; showing available reports.")
        default_index = matching.index(DEFAULT_RUN_DIR) if DEFAULT_RUN_DIR in matching else 0
        selected_run = st.selectbox(
            "Completed artifact",
            matching,
            index=default_index,
            format_func=lambda item: run_label(item, report_lookup.get(item)),
        )
        saved_viewer = st.session_state.get("viewer_mode", "Cached replay")
        viewer_mode = st.radio(
            "Viewer behavior",
            ("Cached replay", "Live artifact refresh"),
            index=0 if saved_viewer == "Cached replay" else 1,
            help="Both modes read completed local artifacts only. Live refresh never starts inference, GPU work, or a final test.",
        )
        st.session_state.viewer_mode = viewer_mode
        if st.button("Refresh artifacts", use_container_width=True):
            cached_report.clear()
            st.rerun()

    @st.fragment(run_every=3.0 if viewer_mode == "Live artifact refresh" else None)
    def render_selected() -> None:
        report = read_report(selected_run, refresh_live=viewer_mode == "Live artifact refresh")
        if report is None:
            st.error("The selected artifact is not readable. Choose another run or refresh after the writer completes.")
            return
        render_run_status(report, selected_run, viewer_mode)
        render_media(selected_run)
        costs = as_mapping(report.get("costs"))
        metadata_left, metadata_right = st.columns(2)
        with metadata_left:
            st.caption(f"Scene hash: `{short_hash(report.get('scene_hash'))}`")
        with metadata_right:
            st.caption(f"Recorded spend — Meta: {money(costs.get('meta_usd'))}; GPU: {money(costs.get('gpu_usd'))}")
        render_held_out(report)
        render_failure_breakdown(report)
        render_validation(report)
        render_geometry(report, selected_run)
        render_grasp_and_trace(selected_run)
        st.download_button("Download report JSON", data=(selected_run / "report.json").read_bytes(), file_name=f"{selected_run.name}-report.json", mime="application/json")
        render_limits(report)
    render_selected()


if __name__ == "__main__":
    main()
