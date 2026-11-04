# Phase A readiness

## Boundary status

The offline synthetic acceptance gates pass. The required small live Meta-format spike now also **passes**: Wonder structured JSON, Spark image input, Files upload and streamed SAM completed on synthetic media. Recorded cost is **USD 0.028378**, with no pending reservations. Saved fixtures replay through Pydantic and the official SAM parser (two boxes and two masks). Credentials were loaded from ignored `.env` without printing them.

No real video has been processed, no real scenario family/test seal has been created, no real final test has run, and no GPU job/pod has been created. Phase B requires the user's literal **go**, then the separate input/mask/spend confirmations below.

Repository: https://github.com/Skrisps26/splatsam . Implementation checkpoint tested from GitHub: `196c0ed`. Working directory: `/tmp/splatsam`. Read-only local dashboard: http://127.0.0.1:8501 .

## Exercised checks

- Original working checkout: `make test`: **56 passed in 23.49 s**. Ruff check passed after formatting.
- Fresh GitHub clone at `/tmp/splatsam-clean-proof`, with MODEL_API_KEY, HF_TOKEN and GPU-provider credentials removed: `make setup test synth-demo` completed successfully in **252.66 s** overall. Its test run passed **56 tests in 47.55 s**.
- Fresh synthetic pilot: **37/100 successes**, **100 episodes in 9.792 s**, local CPU. Full source video, tracked-mask replay, metric reconstruction, physics, validation loop, sealed synthetic evaluation and report generation completed.
- Second fresh-clone `make synth-demo`: verified cached synthetic results reused; **no sealed evaluation repeated**; completed in **1.79 s**.
- Fresh-clone `make demo` on port 8502 loaded the committed bundle with API/GPU clients disabled. Browser inspection showed the correct statistics and reconstruction tables. Both videos decoded and advanced during playback (11.933 s mask overlay and 8.84 s physical grasp replay). The duplicate service was stopped; the working-checkout dashboard remains on port 8501.
- Earlier desktop/mobile browser checks exercised view/run selection, trajectory expansion, empty-result state and local live-artifact refresh. Screenshots are in `.impeccable/review/`.
- Actual authenticated CPU-stub HTTP reconstruction returned a valid 12-face GLB; missing authentication returned 401. A repeat request succeeded from disk cache after the server stopped. Fake geometry remains explicitly labeled and is not live simulation geometry.
- Behavioral tests cover deterministic serial/parallel simulation, scoring boundaries, split disjointness and loop access restrictions, sealed hash/one-shot protections, budget limits, validation acceptance, tracking stability/identical-instance association, schemas, independent geometry alignment and synthetic scale/size accuracy.

## Measured synthetic illustration — not real-video evidence

The same numbers were independently reproduced by the fresh clone. Masks and model proposals are mocks with explicit provenance; MuJoCo physics, rendered media, statistics and seal enforcement are real local computations.

| Planner | Successes / sealed scenarios | Success rate | Wilson 95% interval |
| --- | --- | --- | --- |
| Simple default settings | 69 / 200 | 34.5% | 28.26–41.32% |
| Validation-selected settings | 95 / 200 | 47.5% | 40.69–54.40% |

Paired gain: **13.0 percentage points**, paired bootstrap 95% interval **8.5–17.5 points**. Six edits evaluated, three accepted, then three consecutive rejections. Validation was reused for acceptance: 1,200 planner/scenario evaluations, 700 fresh simulations, 100 distinct validation seeds. Validation is optimistic.

Paper-homography scale error: **0.251%**. Maximum planar-size error: **8.13%**. Per-object footprint IoU: **0.901–0.991**. Both planners had **65/200 instability failures**; baseline/final slip-or-drop counts were 31/6. Full breakdowns and provenance are in `demo_data/report.json` and `demo_data/report.md`.

## Spend and risks

- **Meta spent: USD 0.028378. GPU spent: USD 0.** The actual project GPU ledger has no resources. No account-credit balance has been verified.
- Hard gates: USD 10 Phase A; USD 50 cumulative development; configured USD 70 minimum reserve. No reserve has been spent. RunPod optional cap is USD 4; no fallback provider is authorized merely because RunPod is unavailable.
- Live synthetic request formats and observed usage are now exercised; exact requests/results are in `tests/fixtures/meta_api_spike.json` and API notes. This verifies transport/schema/parser compatibility, not real-video quality, Files-ID lifetime, box prompting or long-video stability.
- Runpod's existing research-profile stdio entry was added to the default profile without changing AWS or publishing credentials. An independent MCP SDK connection exposed 68 tools; authenticated read-only `list-pods` succeeded with zero pods. The conversation still needs `/mcp reload` to mount these tools. No SAM 3D weights, upstream GPU runtime, provider launch or provider cleanup has been exercised. The optional HTTP adapter/server is proven only with the CPU stub. GPU mesh promotion is not active; the working demonstration uses native primitives. A frozen scene hash cannot silently change after sealing.
- The 32.5% instability rate is a material contact/physics-fidelity risk. Do not suppress these failures or claim physical accuracy from silhouette agreement.
- Reconstruction heights/hidden surfaces and nominal mass/friction are estimates unless overridden by measurements. Thin objects are resolution-sensitive; paper curl, parallax and camera tilt cause scale uncertainty. Live masks may fail or merge identical instances; overlays require human approval.
- Simulation only, floating hand/no arm or IK, bounded rule-planner parameter tuning rather than neural-policy training. No real-robot transfer or cross-scene generalization is claimed. The local seal protects against accidents, not a malicious repository owner.

## Exact Phase B sequence after go

1. **Completed prerequisite:** the <=USD 10 synthetic format spike passed for Wonder JSON, Spark image, Files upload and SAM. Keep the credential in the environment/ignored `.env`; no personal data was sent.
2. **B1 — confirm input/privacy ready:** user places the intended video in ignored `data/videos/`, confirms its hash and that only the intended workspace may be sent externally. Optional measurements go in ignored `data/measured.yaml`. Authorize B1, run real perception and primitive scene construction, show overlays/diagnostics, then **stop until the user approves the masks**. No real seal or evaluation yet.
3. **B2 — confirm masks and evaluation input ready:** freeze the real family and scene, seal a fresh real 200-scenario test set, run the default baseline and bounded train/validation improvement loop. Keep test specs inaccessible to the loop. Record all accept/reject decisions and costs.
4. **B3 — optional, separately approved:** only if requested and the missing provider integration/access is available, verify gated weights/license and live price, authorize <=USD 4 RunPod spend, batch one bounded reconstruction job, cache results, stop resources and independently verify cleanup. Otherwise skip with primitives. Do not modify a sealed scene silently; optional geometry cannot invalidate the frozen evaluation contract.
5. **B4 — confirm final inputs ready:** verify the real seal and frozen planner artifacts; run paired baseline/final real-scene simulation evaluation **exactly once**. Produce confidence intervals/report, show the live artifact dashboard, preserve cached playback as backup, and update the write-up with honest real-scene numbers. Raw video is not committed without separate approval.

Authorization lives only in ignored `data/authorization.json`; README documents the action flags. The assistant must not manufacture user confirmations.

## Phase B planning estimate

**[INFERENCE — not a quote or measured bill]** Budget approximately **USD 2–5 additional Meta** for a successful primitive-path live run, excluding extra retries. An illustrative workload is ten Wonder calls with 10,000 input / 2,048 output tokens each (USD 0.8096), twenty-four Spark calls at the same token counts (USD 0.508896), and four 15 s / 30 fps SAM noun-phrase requests (USD 0.36): about USD 1.68 before uncertainty. Actual image tokens, frame rate, phrase count, diagnostics and retries vary; request reservations and the global cap, not this estimate, authorize spending. Unknown pricing uses the conservative configured fallback.

GPU estimate is **USD 0 by default**. Optional RunPod spending cannot exceed **USD 4** and requires live price/access checks and explicit approval. All CPU simulation remains local.

**STOP: waiting for go.**
