# THESIS: a recorded workspace becomes an inspectable physics sandbox, not a debug page.
# OWN-WORLD: midnight navy, lilac-to-mint display, translucent sixteen-pixel panels.
# STORY: choose a video, follow evidence, watch bounded tuning, inspect the sealed result.
# FIRST VIEWPORT: oversized SplatSAM title, clear replay provenance, real simulation media.
# FORM: user-pinned product-demo brief overrides direction roll cac93094; code-led Experience.
# FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from app_legacy import render_geometry, render_held_out, render_media


@st.cache_data(show_spinner=False)
def skin() -> str:
    css = (HERE / "styles.css").read_text()
    fonts = []
    for name, weight in (("Manrope-Regular.ttf", 400), ("Manrope-Bold.ttf", 700)):
        path = HERE / "assets" / name
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode()
            fonts.append(f"@font-face{{font-family:Manrope;src:url(data:font/ttf;base64,{encoded}) format('truetype');font-weight:{weight};font-display:swap}}")
    return "<style>" + "".join(fonts) + css + "</style>"


def safe_report(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def main() -> None:
    st.set_page_config(page_title="SplatSAM · Robot sandbox", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(skin(), unsafe_allow_html=True)
    run = ROOT / "demo_data"
    report = safe_report(run / "report.json")
    st.markdown('<div class="topbar"><span class="brand-small">SplatSAM</span><div class="topbar-right"><span class="mode-pill">REPLAY</span><span>Recorded synthetic workspace</span><span>USD 0 replay cost</span></div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="hero"><h1>SplatSAM</h1><p>Phone video in.<br>Physics sandbox out.</p><p class="hero-detail">A floating gripper. A rule-based planner. A measurable improvement loop—with the evidence to back it up.</p></div>', unsafe_allow_html=True)
        st.markdown('<span class="badge">Replaying a recorded run · synthetic illustration</span>', unsafe_allow_html=True)
        st.caption("No API key. No GPU. Every result below comes from saved artifacts.")
    with right:
        if (run / "sim.png").is_file():
            st.image(str(run / "sim.png"), caption="An estimated MuJoCo workspace—not real-robot evidence", use_container_width=True)
        else:
            st.info("Scene preview is not available. Choose another recorded run.")
    st.divider()
    render_media(run)
    render_held_out(report)
    with st.expander("Inspect object estimates"):
        render_geometry(report, run)


if __name__ == "__main__":
    main()
