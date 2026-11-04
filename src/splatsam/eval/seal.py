"""Freeze and execute sealed held-out synthetic/authorized evaluations."""
from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any

from splatsam.authorization import require_live
from splatsam.curriculum.sampler import TEST_SEEDS, sample_scenario
from splatsam.llm.schemas import ScenarioFamily, ScenarioSpec, SceneSpec
from splatsam.planner.params import PlannerParams
from splatsam.utils import ROOT, config, digest, load_json, save_json

SCORING_SETTINGS: dict[str, Any] = {
    "timestep_s": 0.002,
    "settle_s": 0.5,
    "close_s": 0.5,
    "lift_m": 0.10,
    "lift_success_m": 0.05,
    "hold_s": 2.0,
    "penetration_limit_m": 0.005,
    "velocity_limit_m_s": 5.0,
    "angular_velocity_limit_rad_s": 100.0,
    "failure_precedence": [
        "instability",
        "timeout",
        "occluder_collision",
        "knock_over",
        "no_grasp_found",
        "slip_or_drop",
    ],
}
_SCHEMA_VERSION = 1
_DEPENDENCIES = ("mujoco", "numpy", "scipy", "statsmodels", "pydantic")


class SealError(RuntimeError):
    """A frozen split is unavailable, altered, or incompatible with its inputs."""


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_real_root(root: Path) -> bool:
    data_root = (ROOT / "data").resolve()
    return root == data_root or root.is_relative_to(data_root)


def _is_synthetic_root(root: Path) -> bool:
    synth_root = (ROOT / "data_synth").resolve()
    if root == synth_root or root.is_relative_to(synth_root):
        return True
    temp_root = Path(tempfile.gettempdir()).resolve()
    # This checkout happens to live under /tmp; its non-data_synth paths are
    # not disposable synthetic namespaces.
    return root.is_relative_to(temp_root) and not root.is_relative_to(ROOT)


def _guard_root(root: Path, action: str) -> bool:
    """Return whether this is real data after refusing unauthorized paths."""
    if _is_real_root(root):
        require_live(action)
        return True
    if not _is_synthetic_root(root):
        raise SealError("Phase A seals may only use data_synth or a temporary synthetic namespace")
    return False


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _params_hash(params: PlannerParams) -> str:
    return params.hash


def _input_hashes(scene: SceneSpec, family: ScenarioFamily, baseline: PlannerParams) -> dict[str, str]:
    dependencies = _dependency_versions()
    return {
        "scene": digest(scene),
        "family": digest(family),
        "baseline": _params_hash(baseline),
        "scoring": digest(SCORING_SETTINGS),
        "config": digest(config()),
        "dependencies": digest(dependencies),
    }


def _paths(root: Path) -> tuple[Path, Path, Path]:
    splits = root / "splits"
    return splits / "test.json", splits / "SEALED.manifest.json", splits / "SEALED.lock"


def _write_lock(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value + "\n")
    os.replace(temporary, path)


def _manifest_for(
    *, scene: SceneSpec, family: ScenarioFamily, baseline: PlannerParams, scenarios: list[ScenarioSpec], real: bool
) -> dict[str, Any]:
    scenario_data = [scenario.model_dump(mode="json") for scenario in scenarios]
    dependencies = _dependency_versions()
    return {
        "schema_version": _SCHEMA_VERSION,
        "mode": "real" if real else "synthetic",
        "test_seed_range": [TEST_SEEDS.start, TEST_SEEDS.stop - 1],
        "scenario_count": len(scenarios),
        "test_spec_hash": digest(scenario_data),
        "hashes": _input_hashes(scene, family, baseline),
        "dependency_versions": dependencies,
        "scoring_settings": SCORING_SETTINGS,
    }


def _read_manifest(root: Path) -> tuple[dict[str, Any], list[ScenarioSpec]]:
    test_path, manifest_path, lock_path = _paths(root)
    if not (test_path.is_file() and manifest_path.is_file() and lock_path.is_file()):
        raise SealError("sealed test artifacts are incomplete")
    manifest = load_json(manifest_path)
    if lock_path.read_text().strip() != digest(manifest):
        raise SealError("SEALED.lock does not match the frozen manifest")
    payload = load_json(test_path)
    if payload.get("schema_version") != _SCHEMA_VERSION or not isinstance(payload.get("scenarios"), list):
        raise SealError("frozen test specification has an invalid format")
    try:
        scenarios = [ScenarioSpec.model_validate(value) for value in payload["scenarios"]]
    except Exception as exc:
        raise SealError("frozen test specification is invalid") from exc
    return manifest, scenarios


def _validate_manifest(
    manifest: dict[str, Any], scenarios: list[ScenarioSpec], *, scene: SceneSpec, family: ScenarioFamily,
    baseline: PlannerParams, real: bool,
) -> None:
    expected_hashes = _input_hashes(scene, family, baseline)
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise SealError("unsupported seal schema version")
    if manifest.get("mode") != ("real" if real else "synthetic"):
        raise SealError("seal mode does not match the selected data root")
    if manifest.get("hashes") != expected_hashes:
        raise SealError("sealed scene, family, baseline, scoring, config, or dependencies changed")
    if manifest.get("dependency_versions") != _dependency_versions():
        raise SealError("sealed dependency versions changed")
    if manifest.get("scoring_settings") != SCORING_SETTINGS:
        raise SealError("sealed scoring settings changed")
    scenario_data = [scenario.model_dump(mode="json") for scenario in scenarios]
    if manifest.get("test_spec_hash") != digest(scenario_data):
        raise SealError("frozen test scenarios were altered")
    expected_seeds = list(TEST_SEEDS)
    if [scenario.seed for scenario in scenarios] != expected_seeds:
        raise SealError("frozen test seeds are not the canonical held-out range")
    if manifest.get("scenario_count") != len(expected_seeds):
        raise SealError("frozen test scenario count changed")


def seal_test(scene: SceneSpec, family: ScenarioFamily, baseline: PlannerParams, root: Path | str) -> dict[str, Any]:
    """Create immutable concrete held-out specs without running any episodes."""
    root = _resolved(root)
    real = _guard_root(root, "B2")
    test_path, manifest_path, lock_path = _paths(root)
    existing = [path.exists() for path in (test_path, manifest_path, lock_path)]
    if any(existing):
        if not all(existing):
            raise SealError("refusing to complete a partially written seal")
        manifest, scenarios = _read_manifest(root)
        _validate_manifest(manifest, scenarios, scene=scene, family=family, baseline=baseline, real=real)
        return manifest

    scenarios = [sample_scenario(family, seed, scene) for seed in TEST_SEEDS]
    manifest = _manifest_for(scene=scene, family=family, baseline=baseline, scenarios=scenarios, real=real)
    save_json(test_path, {"schema_version": _SCHEMA_VERSION, "scenarios": [item.model_dump(mode="json") for item in scenarios]})
    save_json(manifest_path, manifest)
    _write_lock(lock_path, digest(manifest))
    return manifest



def _validate_episode_coverage(results: list[Any], scenarios: list[ScenarioSpec], label: str) -> None:
    expected = [(scenario.seed, scenario.target_id) for scenario in scenarios]
    actual = [(result.seed, result.target_id) for result in results]
    if actual != expected:
        raise SealError(f"{label} evaluation did not return every sealed scenario in stable order")

def _result_paths(root: Path, real: bool) -> tuple[Path, Path]:
    results = ROOT / "results" if real else root / "results"
    return results / "final_test.json", results / "FINAL_TEST.started"


def _claim_final(started_path: Path) -> None:
    started_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(started_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SealError("final test was already started and cannot be retried") from exc
    with os.fdopen(descriptor, "w") as marker:
        marker.write("final test started; this marker intentionally remains after completion or failure\n")
        marker.flush()
        os.fsync(marker.fileno())


def run_final(
    scene: SceneSpec,
    family: ScenarioFamily,
    baseline: PlannerParams,
    final: PlannerParams,
    root: Path | str,
    run_dir: Path | str,
    workers: int = 4,
) -> dict[str, Any]:
    """Run both planners once against the concrete sealed test cases.

    The evaluator import occurs only after the one-shot marker is claimed so
    creating or validating a seal never initializes MuJoCo.
    """
    root = _resolved(root)
    real = _guard_root(root, "B4")
    result_path, started_path = _result_paths(root, real)
    if result_path.exists():
        raise SealError("final test result already exists and cannot be replaced")
    _claim_final(started_path)

    manifest, scenarios = _read_manifest(root)
    _validate_manifest(manifest, scenarios, scene=scene, family=family, baseline=baseline, real=real)

    from splatsam.eval.statistics import compare, summarize
    from splatsam.sim.episode import evaluate

    baseline_results = evaluate(scene, scenarios, baseline, workers=workers)
    _validate_episode_coverage(baseline_results, scenarios, "baseline")
    final_results = evaluate(scene, scenarios, final, workers=workers)
    _validate_episode_coverage(final_results, scenarios, "final")
    comparison = compare(baseline_results, final_results)
    result = {
        "schema_version": _SCHEMA_VERSION,
        "mode": "real" if real else "synthetic",
        "run_id": Path(run_dir).name,
        "seal_hash": digest(manifest),
        "hashes": {
            "scene": digest(scene),
            "family": digest(family),
            "baseline": _params_hash(baseline),
            "final": _params_hash(final),
            "scoring": digest(SCORING_SETTINGS),
            "config": digest(config()),
            "dependencies": digest(_dependency_versions()),
        },
        "baseline": {
            "params": baseline.to_dict(),
            "results": [item.model_dump(mode="json") for item in baseline_results],
            "summary": summarize(baseline_results),
        },
        "final": {
            "params": final.to_dict(),
            "results": [item.model_dump(mode="json") for item in final_results],
            "summary": summarize(final_results),
        },
        "comparison": comparison,
    }
    save_json(result_path, result)
    return result
