# Splatsam — video to a testable workspace

## Problem and target user
Robotics developers repeatedly build physics scenes by hand for new workspaces, then often tune and measure on overlapping scenarios. Splatsam converts a short tabletop video into an estimated MuJoCo scene and evaluates a simple grasp planner with a frozen scenario family and held-out seeds.

## Meta model integration
Muse Wonder 1.0 reads scene semantics, proposes one bounded scenario family and bounded planner-parameter edits. Muse Spark 1.3 provides optional failure diagnoses from three keyframes. SAM 3.1 supplies persistent object masks from a reused Files upload. The model never scores success; simulator state and contact history do. Every model call is hashed, cached and recorded with provenance and cost. Synthetic mocks are labeled; no mock response is claimed as live evidence.

## Method
A4 paper defines an image-to-table-plane metric homography. Masks become native boxes, spheres and vertical/horizontal cylinders. A free two-finger hand is dynamically welded to a moving mocap target. A principal-axis, clearance-scored parallel-jaw rule planner uses estimated nominal mass/friction to select grip safety factor and motion settings. Randomized simulator physics is hidden from those nominal estimates.

Training seeds are 0–9999, validation 10000–19999, test 900000+. Concrete frozen sets use 150/100/200 episodes. Validation selects edits using paired scenarios, at least 2 percentage points mean gain, a positive 10th-percentile paired bootstrap and failure-mode regression guard. Validation is consequently optimistic. A hash-verified one-shot final evaluator measures baseline and final on the sealed test set with Wilson 95% success intervals and a paired gain interval.

## Evidence status
Phase A is synthetic, CPU-only and offline except a separately gated small API format spike. Real-video results are not available before Phase B. No numerical improvement is claimed until measured artifacts exist; if no edit qualifies, report no improvement found.

## Limits
Simulation only. The hand floats; there is no arm, reachability model or inverse kinematics. This is parameter tuning of a rule-based planner, not neural-policy training. No real-robot transfer is claimed. Object heights and hidden-side geometry are estimated. Mass and friction are estimates unless measured. Bird-eye IoU validates footprint appearance only—not height, mass, friction or collision geometry. Paper curl, object height and camera tilt imply scale uncertainty of a few percent. Results are specific to the captured scene. A local hash seal prevents accidental leakage/tampering, not a malicious repository owner.

## Roadmap
Multiple independently captured workspaces and physically measured geometry/physics would broaden evidence. SAM 3D is an optional separately budgeted reconstruction comparison, never a prerequisite for the primitive demo.
