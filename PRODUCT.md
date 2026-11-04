# Product

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
User-specified Python Streamlit dashboard, Plotly, MuJoCo renderer and local files. No image-generation service or external font requirement. Approved implementation uses native controls and a task-first Operate layout.

## Users
Robotics developers assembling simulation workspaces, and hackathon judges inspecting a 2–3 minute demonstration and public repository.

## Product Purpose
Convert a short desk video into tracked objects, an estimated simulation, seeded scenarios and honest measurements of a rule-based grasp planner's parameter improvement.

## Operating Context
Meta Global AI Developer Hackathon, Indie track, Push the Frontier. Phase A is synthetic and CPU-only except one credential-gated small API format spike. Phase B is separately authorized by go and per-action confirmations. Model/GPU costs have hard caps; cached replay must incur no inference spend.

## Capabilities and Constraints
The dashboard reads run artifacts; it cannot provision GPU resources, run the final test or authorize live work. Show mask overlay, source frame against simulated scene, grasp replay, baseline/final n and Wilson intervals, validation-round chart, failure counts, accepted/rejected edits, trajectory, per-object consistency and physics provenance. Prominently distinguish synthetic illustration from real captured-scene evidence, and validation from sealed held-out results. Missing evidence is not zero success.

## Brand Commitments
Name Splatsam. Technical, evidence-first language without real-robot transfer or neural-training claims. Approved layout is native, restrained and readable, with side-by-side media on desktop and stacked media at narrow widths. No aesthetic interview is needed: the supplied brief and approved plan fix these choices.

## Evidence on Hand
`runs/synth-demo/`: actual generated video perception/reconstruction, MuJoCo episodes, report, overlay, grasp replay and sanitized mock trajectory. Current synthetic held-out n=200: baseline 34.5%, final 47.5%, paired gain 13pp. Live Meta spike unverified because MODEL_API_KEY absent; no real-video final results exist. These absences must remain explicit.

## Product Principles
- Deterministic physical scoring, never LLM success judgments.
- Validation may select edits; sealed results may only measure the final pair.
- Estimated geometry/physics and synthetic data remain visibly labeled.
- Local read-only replay never dispatches inference or provisions resources.
