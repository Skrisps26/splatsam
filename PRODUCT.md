# Product

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
Python Streamlit, Plotly, MuJoCo-rendered saved media, local files, and bundled offline CSS. The viewer uses native Streamlit controls plus two minimal native `st.iframe` embeds for the before/after comparison and sealed-result count-up; it adds no UI dependency or network requirement.

## Users
Robotics developers reviewing a synthetic simulation workspace, and hackathon judges following a short product demonstration from a public repository.

## Product Purpose
Turn a completed video-to-physics experiment into an inspectable, honest replay. The demo reveals saved object tracking, estimated primitives, validation-gated rule-based planner tuning, and a sealed synthetic comparison without rerunning any stage.

## Operating Context
`make setup && make demo` launches a local, read-only, **75-second presentation-paced** replay. The event log was reconstructed from completed artifact snapshots because original stage timing was not retained; replay timestamps are not runtime measurements. Phase A synthetic work is complete. The real Phase B repair/tuning continuation is paused, and the real final test has **not** run. Viewing, uploading, or replaying never dispatches inference, provisions GPU, authorizes live work, or alters a seal.

## Capabilities and Constraints
The first screen is labeled **REPLAY** and offers a local MP4 drop zone or the bundled synthetic sample. Uploads remain local; the only enabled launch starts the recorded sample. A disabled live-launch path explains that privacy, calibration, and sealed-test approvals have not been bypassed.

The replay exposes Pause/Resume, 0.5×–4× speed, chapter pills, seek, a seven-stage progress stepper, progressive masks, saved scene/replay media, validation charts, accepted and rejected patches, failure records, provenance, and saved report/trajectory controls. Native tabs, expanders, Plotly charts, and a keyboard-operable comparison slider make evidence inspectable without fabricating a missing asset or recomputing a result. Narrow layouts stack and reflow this evidence.

## Brand Commitments
Dark technical product demo, not a light evidence notebook: midnight navy field, restrained radial depth, lilac-to-mint headline gradient, translucent bordered panels, compact provenance labels, and real saved simulation imagery. Use embedded Manrope where bundled, system sans fallback otherwise. The product stays evidence-first: status and synthetic provenance are visible before results; rich presentation never converts synthetic illustration into a real-world claim.

## Evidence on Hand
The bundled synthetic run reports **69/200 (34.5%)** baseline and **95/200 (47.5%)** final paired sealed scenarios, with their saved Wilson intervals; it contains six rounds, three accepted and three rejected. These figures are synthetic-only and were copied from saved artifacts for display. No new evaluation metric was generated for the UI.

The recorded Meta format spike is live-confirmed for synthetic Wonder JSON, Spark image input, Files upload, and streamed SAM parsing. `docs/api_notes.md` records the observed fixture and USD 0.028378 total. This is transport-format evidence only: it is neither real-video quality evidence nor a real final result. The real final remains absent.

## Product Principles
- Read saved artifacts only; the UI never recomputes measurements, scores, intervals, or simulation outcomes.
- State replay provenance and presentation pacing explicitly; never imply stage latency.
- Keep uploads local and Live launch intentionally unavailable until the required approvals exist.
- Keep validation selection distinct from the once-sealed synthetic result.
- Label estimates, masks, and all reported success figures as synthetic evidence; do not claim real-robot transfer.
