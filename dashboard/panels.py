"""Presentation-only views. All outcomes and intervals come from saved artifacts."""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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


@st.cache_data(show_spinner=False)
def image_uri(path: str, modified: int) -> str:
    del modified
    try:
        data = Path(path).read_bytes()
        mime = "image/png" if Path(path).suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    except OSError:
        return ""


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


def show_video(path: Path | None, label: str):
    if path:
        st.video(str(path), autoplay=True, loop=True, muted=True, alt=label)
        st.caption(label)
    else:
        st.markdown(
            '<div class="empty-state">No recorded clip is available for this planner yet.</div>',
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


def render_tuning(run: Path, report: dict, media: dict, state: dict, position: float):
    del position
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
    with left:
        if round_asset:
            clip = file(run, round_asset.get("path"))
            label = (
                f"Accepted round {visible_round} · recorded synthetic scenario {round_asset.get('seed', '—')}"
            )
        elif all_accepted_visible:
            clip = file(
                run, mapping(mapping(media.get("assets")).get("recorded_final_replay")).get("path")
            ) or file(run, "grasp.mp4")
            label = "Saved final-planner demonstration · synthetic scenario 0"
        else:
            clip = file(run, paired.get("baseline"))
            label = f"Saved baseline demonstration · synthetic scenario {paired.get('seed', '—')}"
        if not clip:
            clip = file(run, "grasp.mp4")
            label = "Recorded final-planner example—not a live/current-round camera"
        show_video(clip, label)
        episodes = [mapping(e) for e in items(state.get("episodes")) if e.get("phase") == "tune"]
        section("Episode wall", "Saved evaluations, revealed as events arrive")
        latest = {}
        for episode in episodes:
            latest[episode.get("seed")] = episode
        display = list(latest.values())[:200]
        squares = []
        for index in range(200):
            episode = display[index] if index < len(display) else None
            status = "success" if episode and episode.get("success") else "failure" if episode else "pending"
            label = (
                f"Seed {episode.get('seed')}: {'success' if episode.get('success') else episode.get('failure_mode')}"
                if episode
                else "Awaiting a recorded episode"
            )
            squares.append(
                f'<span class="episode-cell {status}" title="{html.escape(label, quote=True)}"></span>'
            )
        st.markdown(
            '<div class="episode-wall" role="img" aria-label="Recorded episode outcome grid">'
            + "".join(squares)
            + '</div><div class="wall-legend"><span><i style="background:#58cfa8"></i>Success</span><span><i style="background:#df778c"></i>Failure</span><span>200 scenario slots</span></div>',
            unsafe_allow_html=True,
        )
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
                use_container_width=True,
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
                use_container_width=True,
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
            f"Baseline vs improved · same synthetic scenario {paired.get('seed')}",
            expanded=bool(accepted_rounds),
        ):
            a, b = st.columns(2)
            with a:
                show_video(
                    file(run, paired.get("baseline")), f"Baseline · seed {paired.get('seed')} · failed"
                )
            with b:
                show_video(
                    file(run, paired.get("final")), f"Final planner · seed {paired.get('seed')} · succeeded"
                )
            st.caption(
                paired.get(
                    "provenance",
                    "Illustrative training-seed media recordings; not additional evaluation metrics.",
                )
            )
    elif paired.get("status") == "available":
        st.caption("Same-seed baseline/final comparison unlocks after the last recorded accepted edit.")
    else:
        st.caption(
            "Same-seed baseline/improved comparison is unavailable: no recorded baseline trace was saved. No clip is invented."
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
    components.html(
        f'''<style>*{{box-sizing:border-box}}body{{margin:0;background:#0d1625;color:#e7edfc;font:13px system-ui}}.compare{{position:relative;height:330px;border-radius:12px;overflow:hidden}}img{{position:absolute;width:100%;height:100%;object-fit:contain;background:#090e18}}#front{{clip-path:inset(0 50% 0 0)}}.labels{{position:absolute;inset:12px 12px auto;display:flex;justify-content:space-between}}.labels span{{background:#111b2de8;padding:6px 9px;border-radius:6px}}input{{width:calc(100% - 30px);margin:14px 15px;accent-color:#a99eff;min-height:26px}}input:focus-visible{{outline:3px solid #68e6c4;outline-offset:2px}}</style><div class="compare"><img src="{b}" alt="Estimated physics scene"><img id="front" src="{a}" alt="Recorded source frame"><div class="labels"><span>Recorded frame</span><span>Estimated simulation</span></div></div><label for="wipe" style="position:absolute;left:-10000px">Compare recorded frame and estimated simulation</label><input id="wipe" type="range" min="0" max="100" value="50" aria-label="Comparison slider"><script>document.getElementById('wipe').addEventListener('input',function(){{document.getElementById('front').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'}})</script>''',
        height=395,
        scrolling=False,
    )
    st.caption(
        "Drag or use arrow keys. The views use different cameras; this is a visual comparison, not pixel registration."
    )


def render_scene(run: Path, report: dict, media: dict, state: dict, elapsed: float):
    section("A scene takes shape", "Recorded frames → masks → metric footprints → physics")
    stages = mapping(state.get("stages"))
    frames = [mapping(x) for x in items(media.get("reveal_frames"))]
    stage = next(
        (
            key
            for key in ("understand", "track", "scale", "build", "scenarios")
            if mapping(stages.get(key)).get("status") == "running"
        ),
        "build" if state.get("scene") else "understand",
    )
    candidates = [f for f in frames if f.get("stage") == stage and f.get("kind") == "image"]
    if stage == "track":
        count = len(items(state.get("objects")))
        selected = candidates[min(max(count - 1, 0), len(candidates) - 1)] if candidates else {}
    elif stage == "build" and candidates:
        duration = number(mapping(stages.get("build")).get("duration_s")) or 0
        selected = candidates[-1] if duration > 2 or state.get("scene") else candidates[0]
    elif stage == "scenarios":
        selected = next((f for f in frames if f.get("path") == "presentation/scene_upright.png"), {})
    else:
        selected = candidates[-1] if candidates else next(iter(frames), {})
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        show_image(
            file(run, selected.get("path")) or file(run, "source.png"),
            selected.get("label", "Recorded workspace"),
            height=400,
        )
        st.caption(selected.get("label", "Recorded workspace frame"))
    with right:
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
        scale = mapping(stages.get("scale"))
        if scale.get("status") == "done":
            st.markdown(
                '<span class="badge success">Scale locked · A4 reference</span>', unsafe_allow_html=True
            )
            st.caption("Paper-derived metric scale. Heights and physics remain estimates.")
    if state.get("scene"):
        orbit = file(run, "presentation/scene_orbit.mp4")
        if orbit and 22 <= elapsed <= 32:
            show_video(orbit, "Rendered orbit of the saved physics scene; no new simulation")
        with st.expander("Compare input and simulation"):
            comparison_slider(
                file(run, "source.png"), file(run, "presentation/scene_upright.png") or file(run, "sim.png")
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
                st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption(
                "Visual consistency measures footprint appearance only—not mass, height, friction or physical accuracy."
            )


def render_final(run: Path, report: dict, state: dict):
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
        components.html(
            f'''<style>body{{margin:0;font-family:system-ui;color:#edf3ff;background:transparent}}.row{{display:flex;gap:24px;padding:14px 0}}.card{{flex:1;padding:20px;border-radius:12px;background:#111c2e}}small{{color:#b6c4dc}}strong{{display:block;font-size:44px;letter-spacing:-2px;margin-top:8px}}.final{{color:#78e5c5}}</style><div class="row"><div class="card"><small>Baseline · sealed test</small><strong id="b" aria-label="{values[0] * 100:.1f} percent">{values[0] * 100:.1f}%</strong></div><div class="card"><small>Final · sealed test</small><strong class="final" id="f" aria-label="{values[1] * 100:.1f} percent">{values[1] * 100:.1f}%</strong></div></div><script>const ends={data};if(!matchMedia('(prefers-reduced-motion: reduce)').matches){{const start=performance.now();function draw(t){{const p=Math.min((t-start)/850,1);['b','f'].forEach((id,i)=>document.getElementById(id).textContent=(ends[i]*(1-Math.pow(1-p,3))).toFixed(1)+'%');if(p<1)requestAnimationFrame(draw)}}requestAnimationFrame(draw)}}</script>''',
            height=155,
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
        use_container_width=True,
        config={"displayModeBar": False},
        key="sealed_ci",
    )
    st.caption(
        f"{baseline.get('successes')}/{baseline.get('n')} baseline · {selected.get('successes')}/{selected.get('n')} final · paired gain {pct(comparison.get('gain'))}. Synthetic simulation illustration, not real-video results."
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
                        use_container_width=True,
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
                file(run, pair.get("baseline")), "Saved baseline clip for this synthetic training seed"
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
