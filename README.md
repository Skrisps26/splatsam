# Splatsam

**A desk video → an estimated physics scene → an honestly evaluated grasp planner.**

Built for the Meta Global AI Developer Hackathon, Indie track / Push the Frontier. This is **simulation only**: a floating two-finger hand, no arm or inverse kinematics, and a **rule-based planner with tunable parameters—not a trained neural policy**. No real-robot transfer is claimed.

## Run the offline demo

Prerequisites: Linux, [uv](https://docs.astral.sh/uv/), GNU make and a working Mesa OpenGL implementation. The project selects software OSMesa when installed, otherwise software EGL/llvmpipe. Python 3.12 and Python dependencies are managed by uv.

```sh
git clone https://github.com/Skrisps26/splatsam.git
cd splatsam
make setup && make demo
```

Open **http://127.0.0.1:8501**. The committed synthetic bundle is local, with hash-verified core artifacts, and requires **no API key, model weights or GPU**. Setup may download ordinary dependencies, not model weights. The default example is a **75-second presentation-paced replay** of a completed synthetic run, not a live pipeline execution. `STREAMLIT_SERVER_PORT=8502 make demo` changes the local port when necessary.

| Command | Behavior |
| --- | --- |
| `make setup` | Synchronize locked Python 3.12 dependencies; create ignored `data/videos/` |
| `make test` | Behavioral tests, including actual MuJoCo grasps and renderer checks |
| `make synth-demo` | Generate a synthetic video, reconstruct from mock segmentation, improve on train/validation, run a one-shot synthetic sealed test, write report, then export replay artifacts |
| `make demo` | Copy the complete cached synthetic bundle to `runs/demo` and launch the offline replay viewer |
| `make export-ui RUN_DIR=runs/synth-demo` | Export presentation media and append idempotent presentation events from an already complete synthetic run; it never runs pipeline, API, physics, or evaluation |
| `make final-test RUN_DIR=... DATA_ROOT=...` | Explicit one-shot final evaluator; real input requires Phase B authorization |

`make synth-demo` writes `runs/synth-demo/report.json` and `report.md`. Identical reruns verify and reuse outcomes instead of evaluating the sealed set again. Changed simulation/reconstruction inputs receive a separate synthetic revision; old evidence is retained. The real one-shot marker cannot be bypassed by changing a run ID.

Optional mesh decomposition: `uv sync --extra mesh` installs CoACD. Native primitives do not need it.

## Recorded replay and private live evidence

The product demo is a dark, desktop-to-mobile Streamlit viewer with a lilac-to-mint title, glass-like evidence panels, a seven-stage stepper, persistent-mask reveals, saved scene media, validation rounds, and recorded results where they exist. The bundled example is a **75-second presentation-paced synthetic replay** and is explicitly labeled **“Replaying a recorded run.”** Its timeline is derived from an artifact snapshot, not measured stage latency.

Choose **Use bundled example** only to view that synthetic replay. An upload is previewed locally and its bytes are SHA-256 hashed. The dashboard opens a private live replay only when those exact bytes match a discovered `runs/*/input.json` live-video hash **and** the catalog can re-hash and verify that run's private source video under ignored `data/videos/`. The lookup is content-based; it has no filename, object-name, or run-name convention. An unmatched upload is never replaced with the bundled sample: it can only be saved privately for a later authorized run.

Real-mode copy calls the evidence **recorded** and labels simulator renders as a video-derived simulation, not robot footage. It does not use sample labels. The same recorded/replay and simulation limits remain visible; matching a video does not authorize inference, alter a seal, or create a final result.

### Presentation exporters

The synthetic exporter remains output-only:

```sh
make export-ui RUN_DIR=runs/synth-demo
```

It invokes:

```sh
uv run python scripts/export_ui_media.py --run-dir runs/synth-demo --bundle runs/synth-demo
uv run python scripts/export_ui_events.py --run-dir runs/synth-demo --destination runs/synth-demo
```

For an existing private live B1/B2 run, use the separate real exporter:

```sh
uv run python scripts/export_real_ui.py --run-dir runs/<run>
uv run python scripts/export_real_ui.py --run-dir runs/<run> --record-grasp --seed 0
```

It verifies the live `input.json` source, digest, and private video before copying saved source/track/scene evidence into `runs/<run>/presentation/`, including `media.json`; it writes the private `runs/<run>/presentation.json` and `events.jsonl` alongside the run. Without `--record-grasp`, it exports static saved-artifact media only. With `--record-grasp --seed 0` (the seed defaults to `0` and accepts `0..9999`), it records an illustrative scenario from saved frozen `family.json`, live `scene.json`, and `final_params.json` as `presentation/selected_planner_replay.mp4`, marked excluded from evaluation. Missing saved inputs create missing metadata, never a synthetic fallback. Neither command publishes raw video or runs a final test. A live presentation snapshot is immutable: if its source artifacts change after events exist, export refuses to mix them into that event stream. Preserve a separate presentation revision instead.

The synthetic seed-7 baseline/final clip remains illustrative TRAIN media only. It is excluded from reports, the seal, and all published evaluation metrics. See [the UI artifact inventory](docs/ui_artifacts.md) for origins, private paths, and limits.

## Current evidence: synthetic illustration only

The generated 12-second source video contains an A4 sheet, two boxes, one sphere and two horizontal cylinders. MuJoCo supplies ground-truth segmentation masks through `MockSamClient`; reconstruction receives masks and semantic height/material priors, **not hidden scene positions or sizes**. Planar accuracy is checked independently against the generator's isolated truth afterward.

| Planner on 200 paired sealed synthetic scenarios | Success | Wilson 95% interval |
| --- | ---: | --- |
| Simple default settings | 69 / 200 = 34.5% | 28.3%–41.3% |
| Validation-selected final settings | 95 / 200 = 47.5% | 40.7%–54.4% |

Paired gain: **13.0 percentage points**, paired-bootstrap 95% interval **8.5–17.5 points**. Six parameter patches were evaluated, three accepted; the loop stopped after three consecutive rejections. The final grip safety factor is 2.4 versus the default 1.2. Proposals and critic calls in this run are explicitly **mocked**, not live Meta usage.

- Independent scale error: **0.251%**; largest planar size error: **8.13%**.
- Object footprint IoUs: **0.901–0.991**. The ball is a sphere; markers are rolling horizontal cylinders.
- Train/validation/test: **150 / 100 / 200** concrete scenarios from disjoint seed ranges.
- Validation: **1,200** planner-scenario evaluations, **700** fresh simulations, on the same 100 validation seeds. Validation selected edits and is optimistic.
- **65/200 instability failures remain for both planners.** Solver/estimated-geometry limitations are visible, not relabeled or excluded from the denominator.
- The synthetic pipeline itself spent **$0**. A subsequent live synthetic format spike verified Wonder JSON, Spark image input, Files upload and streamed SAM for **$0.028378 Meta total**; GPU spend remains **$0**. Observed fixtures are saved in `tests/fixtures/meta_api_spike.json`.

Inspect the [full report](demo_data/report.md), [machine-readable report](demo_data/report.json), [sanitized trajectory](runs/evidence/trajectory.jsonl), and [write-up](docs/writeup.md). These numbers are not real-video results or evidence of real-robot transfer.

## Pipeline

```mermaid
flowchart LR
  V[Phone or synthetic video] --> F[Sample frames]
  F --> O[Wonder semantic objects]
  V --> S[SAM persistent masks]
  O --> S
  S --> H[A4 metric homography]
  H --> P[Native primitive reconstruction]
  P --> C[MuJoCo silhouette consistency]
  C --> E[Estimated simulation scene]
  E --> T[Frozen family and seeded scenarios]
  T --> L[Train failures and bounded edits]
  L --> G[Paired validation gate]
  G --> L
  G --> X[One-shot sealed comparison]
  X --> R[Report and read-only dashboard]
```

Muse Wonder 1.0 handles scene semantics, family design and parameter proposals; Muse Spark 1.3 handles optional ambiguous-failure diagnoses; SAM 3.1 supplies masks through Files + Responses. The official OpenAI SDK uses Meta's explicit URL/key, and `meta-sam-parser` handles the token grammar. **The simulator—not an LLM—scores success and failure.** Synthetic mode follows the same data contracts with typed mocks and real local physics.

The free hand is dynamically welded to a gradually moved mocap target, so finger friction can physically lift objects. Success requires at least 5 cm lift throughout a 2-second stationary hold, plus finger contact at the end. Nominal estimated mass/friction determine grip force; randomized true simulator multipliers are hidden from the planner.

## Mandatory Phase A / Phase B boundary

Phase A builds and tests with synthetic data under `data_synth/`. It never processes real video, creates a real seal, runs the real final test, or provisions GPU resources. **Plan approval and synthetic success are not the live `go`.**

### Live prerequisites and confirmations

1. Supply `MODEL_API_KEY` through the environment or ignored `.env`, never chat or the repo. The small synthetic format spike has passed; rerunning it is opt-in (`uv run --env-file .env python scripts/spike_api.py --max-usd 10`). [API notes](docs/api_notes.md) contain observed request formats, fixture and usage evidence.
2. Say **go**, confirm the real video is ready in ignored `data/videos/`, and confirm it contains no personal documents, screens, mail or ID cards. B1 binds authorization to its SHA256 and produces overlays/scene only. **User mask approval is required before B2.**
3. Confirm B2 inputs: freeze the scene/family, seal fresh `data/splits/`, run baseline and train/validation improvement. No real held-out outcomes are read here. If an authorized B2 run must continue, use:

   ```sh
   uv run --env-file .env python scripts/resume_live_loop.py \
     --run-dir runs/<run> --max-total-rounds 8
   ```

   The continuation preserves the frozen seal inputs and immutable prior history (including the three preserved rejected fallback rounds), adds at most five rounds, stops after three new consecutive rejections, and does not read held-out outcomes. It refuses a started or completed real final test; B4 remains separately confirmed and one-shot.
4. B3 is optional and additionally spend-approved. Runpod account access was verified read-only; resource creation/cleanup and actual SAM 3D inference remain unexercised. The selected live path uses CPU primitives with no GPU spending.
5. Confirm B4 final inputs: run the real baseline/final sealed comparison **exactly once**. `results/FINAL_TEST.started` is exclusive and remains after an interruption; the real evaluator refuses a second attempt, including with a different run ID.

An operator records the user's actual confirmations in ignored `data/authorization.json`; flags are not inferred from file presence. Initial safe template:

```json
{
  "go": false,
  "privacy_ready": false,
  "video_sha256": "SHA256_OF_THE_CONFIRMED_VIDEO",
  "masks_approved": false,
  "gpu_spend_approved": false,
  "actions": {"B1": false, "B2": false, "B3": false, "B4": false}
}
```

Only after those real confirmations, the operator uses the generic live command with a private run path and a video below `data/videos/`; the dashboard cannot create authorization or run evaluations. Optional measured overrides go in ignored `data/measured.yaml`:

```yaml
objects:
  stable_object_id:
    height_m: 0.025
    mass_kg: 0.12
    friction: 0.6
```

Valid individual measurements override estimates; malformed values are diagnosed and leave valid siblings intact.

## Spending, privacy and failure handling

- Shared, file-locked `runs/meta_budget.json`: **$10 Phase A**, **$50 cumulative development**, **at least $70 reserve**. Configuration can lower but cannot raise hard caps. Every attempted paid request is reserved first; unknown/failed metering is charged conservatively. SDK automatic retries are disabled.
- Diskcache keys include request/model/schema/media content. Successful Meta and optional HTTP inference are replayable without spending. Trajectory rows record safe hashes, schema/usage, cost, latency and provenance; secrets and media payloads are excluded. The opt-in synthetic spike separately records replayable working request formats after observed success.
- Optional GPU caps: RunPod **$4**, AWS **$25** only after its separate account/quota/credit gates, Modal no more than confirmed free credit. At most one owned resource, bounded runtime/idle watchdog policy, ledger and independent cleanup verification. **No project GPU resource was created in Phase A.**
- The optional HTTP service's CPU `stub` returns a clearly labeled fake GLB; it is never selected as live simulation geometry. The real SAM 3D adapter is lazy, spend/phase-gated, and exports upstream `output["glb"]`; its GPU runtime has not been exercised. See [API/contract notes](docs/api_notes.md).
- Expected stage failures leave a diagnostic report and explicit degraded/blocked state. Budgets, authorization and seals fail closed; no real scene is silently replaced with synthetic geometry.
- Raw video, measurements, credentials, caches and transient runs are ignored. Only synthetic artifacts and sanitized submission evidence are automatically published. Raw real video requires separate explicit approval to commit.

## Interpretation limits

Simulation only; floating hand/no arm or IK; rule-based parameter tuning, not a neural policy; no claimed real-robot transfer. Object heights and hidden geometry are estimated, and mass/friction are estimates unless labeled measured. Footprint IoU does not validate height, mass, friction or collision geometry. Paper curl, object height and camera tilt cause scale uncertainty of a few percent; thin objects remain sensitive to raster resolution and parallax. Results are specific to the captured scene. The local hash seal guards mistakes, not a malicious repository owner.

### Calibration references

Synthetic runs retain the A4 `reference` configuration. Real B1 uses the separate `live_reference` settings: a closed laptop rectangle, 357 × 255 mm, nominal surface elevation 23.5 mm, plastic, never a grasp target. Dimensions and reference identity are explicit inputs to the same OpenCV homography/reconstruction path; pens are identified from the real frames rather than assumed to be markers.

The laptop elevation is a model-specification estimate, not a camera-height measurement. The homography is fitted on its lid plane; without metric camera height/pose, floor-plane scale/position bias remains uncorrected and is listed in the report. Near-overhead framing reduces ambiguity but does not prove the real scene's metric accuracy.

Optional operator-assisted recovery uses ignored `data/perception.yaml`: `box_prompts` maps a stable object ID to a source-frame `size`, `frame_index`, `box` and `object_id`; `object_overrides` may correct only a detected object's name, estimated material or graspable flag. These do not fabricate masks or measured physics. SAM still predicts every mask, and semantic corrections are recorded with operator provenance in the trajectory. Do not describe such a run as unassisted.



Library searches and implementation choices: [decisions](docs/decisions.md). Replay design and observed desktop/mobile surface evidence: [design](DESIGN.md). Bundle contents, exporter contract, and provenance: [UI artifact inventory](docs/ui_artifacts.md).

Measured offline checks, unresolved prerequisites, spend estimates and the exact gated live sequence: [Phase A readiness](docs/readiness.md).
