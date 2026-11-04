# Splatsam — video to a testable workspace

## Problem and target user
Robotics developers repeatedly build physics scenes by hand for new workspaces, then often tune and measure on overlapping scenarios. Splatsam converts a short tabletop video into an estimated MuJoCo scene and evaluates a simple grasp planner with a frozen scenario family and held-out seeds.

## Meta model integration
Muse Wonder 1.0 reads scene semantics, proposes one bounded scenario family and bounded planner-parameter edits. Muse Spark 1.3 provides optional failure diagnoses from three keyframes. SAM 3.1 supplies persistent object masks from a reused Files upload. The model never scores success; simulator state and contact history do. Every model call is hashed, cached and recorded with provenance and cost. Synthetic mocks are labeled; no mock response is claimed as live evidence.

## Method
A4 paper defines an image-to-table-plane metric homography. Masks become native boxes, spheres and vertical/horizontal cylinders. A free two-finger hand is dynamically welded to a moving mocap target. A principal-axis, clearance-scored parallel-jaw rule planner uses estimated nominal mass/friction to select grip safety factor and motion settings. Randomized simulator physics is hidden from those nominal estimates.

Training seeds are 0–9999, validation 10000–19999, test 900000+. Concrete frozen sets use 150/100/200 episodes. Validation selects edits using paired scenarios, at least 2 percentage points mean gain, a positive 10th-percentile paired bootstrap and failure-mode regression guard. Validation is consequently optimistic. A hash-verified one-shot final evaluator measures baseline and final on the sealed test set with Wilson 95% success intervals and a paired gain interval.

## Evidence status
Phase A ran the complete synthetic video → tracked-mask replay → metric primitives → physics → tuning → sealed evaluation pipeline on local CPU. Model proposals and masks were explicitly mocked; these numbers do not demonstrate live Meta API performance or real-video reconstruction.

| Sealed synthetic outcome | Success | Wilson 95% interval |
| --- | --- | --- |
| Simple default settings | 69/200 (34.5%) | 28.26–41.32% |
| Validation-selected parameters | 95/200 (47.5%) | 40.69–54.40% |

The paired improvement was 13.0 percentage points, with a paired bootstrap 95% interval of 8.5–17.5 points. Six proposed edits were evaluated; three were accepted before three consecutive rejections stopped tuning. There were 1,200 validation planner/scenario evaluations, including 700 fresh simulations and repeated use of the same 100 validation seeds. Neither validation nor synthetic results establish generalization to another workspace.

Paper-derived scale error was 0.251%; maximum planar-size error was 8.13%. Per-object footprint IoU ranged from 0.901 to 0.991. Both planners had 65/200 instability failures: collision/contact fidelity remains a material limitation, not a result to hide.

The cached report, source/overlay video, actual simulation replay and sanitized trajectory are included in the public demo bundle. The offline synthetic pipeline spent USD 0. A subsequent live synthetic API-format spike verified Wonder JSON, Spark image input, Files upload and streamed SAM for USD 0.028378 total Meta spend; GPU spend remains USD 0. Exact observed fixtures are committed separately from mock evidence. No real video, real seal, real final evaluation or GPU job has run. Phase B remains explicitly gated.

## Limits
Simulation only. The hand floats; there is no arm, reachability model or inverse kinematics. This is parameter tuning of a rule-based planner, not neural-policy training. No real-robot transfer is claimed. Object heights and hidden-side geometry are estimated. Mass and friction are estimates unless measured. Bird-eye IoU validates footprint appearance only—not height, mass, friction or collision geometry. Paper curl, object height and camera tilt imply scale uncertainty of a few percent. Results are specific to the captured scene. A local hash seal prevents accidental leakage/tampering, not a malicious repository owner.

## Roadmap
Multiple independently captured workspaces and physically measured geometry/physics would broaden evidence. SAM 3D is an optional separately budgeted reconstruction comparison, never a prerequisite for the primitive demo.
