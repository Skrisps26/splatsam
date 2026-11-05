# Product

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
Python Streamlit, Plotly, MuJoCo-rendered saved media, local files, and bundled offline CSS. The viewer uses native Streamlit controls plus two minimal native `st.iframe` embeds for the before/after comparison and sealed-result count-up; it adds no UI dependency or network requirement.

## Users
Robotics developers and hackathon judges reviewing saved simulation evidence, or a privately verified recorded reconstruction from an authorized actual video.

## Product Purpose
Turn a completed video-to-physics experiment into an inspectable, honest replay. The viewer reveals saved object tracking, estimated primitives, validation-gated rule-based planner tuning, and a sealed comparison only when one exists; it never reruns a stage.

## Operating Context
`make setup && make demo` launches the local, read-only, **75-second presentation-paced synthetic replay**. Its event log was reconstructed from completed artifact snapshots, so replay timestamps are not runtime measurements. A private live replay is available only for an uploaded byte sequence that matches a saved live `input.json` digest and whose ignored local source video still verifies to that digest. Viewing, uploading, or replaying never dispatches inference, provisions GPU, authorizes live work, or alters a seal. The real final test has **not** run.

## Capabilities and Constraints
The first screen offers a local MP4 drop zone and the bundled synthetic example. An exact-byte match opens its recorded private reconstruction; a matching run without completed presentation events remains unavailable. An unmatched video remains its own preview or private saved upload—never a cached sample result. Sample controls and “synthetic” copy appear only for the bundled example. Live replays retain recorded provenance and explicitly label scene/grasp imagery as simulation.

The replay exposes Pause/Resume, 0.5×–4× speed, chapter pills, seek, a seven-stage progress stepper, progressive masks, saved scene/replay media, validation charts, accepted and rejected patches, failure records, provenance, and saved report/trajectory controls. Native tabs, expanders, Plotly charts, and a keyboard-operable comparison slider make evidence inspectable without fabricating a missing asset or recomputing a result. Narrow layouts stack and reflow this evidence.

## Brand Commitments
Dark technical product demo, not a light evidence notebook: midnight navy field, restrained radial depth, lilac-to-mint headline gradient, translucent bordered panels, compact provenance labels, and saved simulation imagery. Use embedded Manrope where bundled, system sans fallback otherwise. The product stays evidence-first: source mode and recorded provenance are visible before results; rich presentation never converts synthetic illustration or a video-derived simulation into a real-world claim.

## Evidence on Hand
The bundled synthetic run reports **69/200 (34.5%)** baseline and **95/200 (47.5%)** final paired sealed scenarios, with their saved Wilson intervals; it contains six rounds, three accepted and three rejected. These figures are specific to the bundled synthetic sample and were copied from saved artifacts for display. No new evaluation metric was generated for the UI.

The recorded Meta format spike is live-confirmed for synthetic Wonder JSON, Spark image input, Files upload, and streamed SAM parsing. `docs/api_notes.md` records the observed fixture and USD 0.028378 total. This is transport-format evidence only: it is neither actual-video quality evidence nor a real final result. No real held-out comparison is published, and no planner improvement is promised or implied.

## Product Principles
- Read saved artifacts only; the UI never recomputes measurements, scores, intervals, or simulation outcomes.
- State recorded/synthetic or recorded/live provenance and presentation pacing explicitly; never imply stage latency.
- Match a private live replay by uploaded bytes plus local source re-verification; never substitute the bundled sample for an unmatched upload.
- Keep validation selection distinct from any once-sealed final result; a real final is separately gated.
- Label estimates, masks, and all reported success figures with their source; do not claim real-robot transfer.
