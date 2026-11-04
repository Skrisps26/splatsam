"""Submission reports with provenance and simulator-only interpretation."""

from pathlib import Path

from splatsam.llm.schemas import SceneSpec
from splatsam.utils import digest, save_json

LIMITS = [
    "Simulation only; floating two-finger hand, no arm or inverse kinematics.",
    "Rule-based planner parameter tuning, not training a neural policy.",
    "No real-robot transfer is claimed.",
    "Object heights and hidden-side geometry are estimated unless explicitly measured.",
    "Mass and friction are estimates except fields labeled measured.",
    "Consistency IoU checks footprint appearance only, not height, mass, friction or collision geometry.",
    "Scale may be uncertain by a few percent due to paper curl, object height and camera tilt.",
    "Results are specific to the captured scene and this simulation/scoring configuration.",
    "Validation was used for accept/reject decisions and is therefore optimistic.",
    "Local hash seals guard accidental leakage/tampering, not a malicious repository owner.",
]


def generate_report(
    run_dir: Path,
    scene: SceneSpec,
    final_test: dict | None,
    history: list[dict],
    costs: dict,
    *,
    validation_evaluations: int = 0,
    fresh_validation_simulations: int = 0,
    stages: list[dict] | None = None,
) -> dict:
    comparison = final_test.get("comparison", final_test) if final_test else None
    if comparison and "baseline" not in comparison:
        comparison = None
    limits = list(LIMITS)
    elevation = scene.reference_validation.get("reference_elevation_m", 0.0)
    if elevation > 0:
        limits.append(
            f"The calibration surface is nominally {elevation * 1000:g} mm above the support plane. "
            "This is an estimated reference height, not a camera measurement. Its plane-offset "
            "scale/position bias remains uncorrected without metric camera height or pose."
        )
    report = {
        "schema_version": 1,
        "mode": scene.source,
        "scene_hash": digest(scene),
        "what_improves": "bounded parameters of a rule-based top-down parallel-jaw planner",
        "measurement": "physical lift >=5cm throughout 2s stationary hold, finger contact at end",
        "held_out": comparison,
        "illustrative_only": scene.source != "live",
        "rounds": history,
        "accepted_edits": sum(bool(r.get("accepted")) for r in history),
        "validation_evaluations": validation_evaluations,
        "fresh_validation_simulations": fresh_validation_simulations,
        "scale": {
            "source": scene.scale_source,
            "independent_error": scene.scale_error,
            "validation": scene.reference_validation,
        },
        "objects": [
            {
                "id": o.id,
                "name": o.name,
                "shape": o.shape,
                "size_m": o.size,
                "consistency_iou": o.consistency_iou,
                "reconstruction_source": o.reconstruction_source,
                "mass_kg": o.mass,
                "mass_source": o.mass_source,
                "friction": o.friction,
                "friction_source": o.friction_source,
                "height_source": o.height_source,
            }
            for o in scene.objects
        ],
        "costs": costs,
        "limits": limits,
        "stages": stages or [],
        "conclusion": (
            "Planner improvement not evaluated"
            if not history
            else "No improvement found"
            if not any(r.get("accepted") for r in history)
            else "Validation-selected planner; held-out effect reported separately"
        ),
    }
    save_json(run_dir / "report.json", report)
    lines = [
        "# Splatsam evaluation report",
        "",
        f"**Mode: {scene.source}.** "
        + (
            "Synthetic illustration only, not real-video evidence."
            if scene.source != "live"
            else "Scene-specific simulation evidence, not real-robot transfer."
        ),
        "",
        report["conclusion"],
        "",
        "## Sealed test",
    ]
    if comparison:
        lines += [
            "",
            "| Planner | Successes / n | Success rate | Wilson 95% CI |",
            "| --- | ---: | ---: | --- |",
        ]
        for key, label in [("baseline", "Simple default settings"), ("final", "Final rule-based settings")]:
            s = comparison[key]
            ci = s["ci95"]
            lines.append(
                f"| {label} | {s['successes']} / {s['n']} | {s['success_rate']:.1%} | [{ci[0]:.1%}, {ci[1]:.1%}] |"
            )
        if "gain" in comparison:
            lines += [
                "",
                f"Paired gain: {comparison['gain']:.1%}; paired bootstrap 95% CI {comparison.get('gain_ci95')}.",
            ]
        lines += ["", "### Failure breakdown", "", "| Mode | Baseline | Final |", "| --- | ---: | ---: |"]
        for mode in sorted(
            set(comparison["baseline"]["failure_modes"]) | set(comparison["final"]["failure_modes"])
        ):
            lines.append(
                f"| {mode} | {comparison['baseline']['failure_modes'].get(mode, 0)} | {comparison['final']['failure_modes'].get(mode, 0)} |"
            )
    else:
        lines += ["", "No completed held-out test. Missing evidence is not a zero success rate."]
    lines += [
        "",
        "## Validation selection",
        "",
        f"{len(history)} rounds; {report['accepted_edits']} accepted. {validation_evaluations} planner-scenario validation evaluations, {fresh_validation_simulations} fresh simulations. Validation is optimistic because used for selection.",
        "",
        "| Round | Accepted | Patch | Reason |",
        "| --- | --- | --- | --- |",
    ]
    for i, r in enumerate(history):
        lines.append(f"| {i + 1} | {r.get('accepted')} | {r.get('patch')} | {r.get('reason', '')} |")
    lines += [
        "",
        "## Reconstruction and physics",
        "",
        f"Scale source: {scene.scale_source}; independently measured synthetic error: {scene.scale_error}. Sheet self-consistency: {scene.reference_validation}.",
        "",
        "| ID | Shape/source | IoU | Mass kg/source | Friction/source | Height source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for o in scene.objects:
        lines.append(
            f"| {o.id} | {o.shape}/{o.reconstruction_source} | {o.consistency_iou} | {o.mass:.4g}/{o.mass_source} | {o.friction:.3g}/{o.friction_source} | {o.height_source} |"
        )
    lines += ["", "## Spend", "", str(costs), "", "## Limits", ""] + ["- " + s for s in LIMITS]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report
