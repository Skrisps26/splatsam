"""Held-out evaluation summaries and paired confidence intervals."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.proportion import proportion_confint

from splatsam.llm.schemas import EpisodeResult

_FAILURE_MODES = (
    "no_grasp_found",
    "occluder_collision",
    "knock_over",
    "slip_or_drop",
    "instability",
    "timeout",
)


def summarize(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Summarize completed episodes with an exact Wilson 95% interval."""
    n = len(results)
    successes = sum(result.success for result in results)
    failures = Counter(result.failure_mode for result in results if result.failure_mode is not None)
    if n:
        lower, upper = proportion_confint(successes, n, alpha=0.05, method="wilson")
        ci95: list[float | None] = [float(lower), float(upper)]
    else:
        ci95 = [None, None]
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "ci95": ci95,
        "failure_modes": {mode: int(failures[mode]) for mode in _FAILURE_MODES},
    }


def _paired_successes(
    baseline: Sequence[EpisodeResult], final: Sequence[EpisodeResult]
) -> tuple[np.ndarray, np.ndarray]:
    baseline_keys = [(result.seed, result.target_id) for result in baseline]
    final_keys = [(result.seed, result.target_id) for result in final]
    if len(baseline_keys) != len(set(baseline_keys)) or len(final_keys) != len(set(final_keys)):
        raise ValueError("paired results must have unique seed/target identities")
    final_by_key = {(result.seed, result.target_id): result for result in final}
    if set(baseline_keys) != set(final_by_key):
        raise ValueError("baseline and final results must cover identical scenarios")
    old = np.asarray([result.success for result in baseline], dtype=float)
    new = np.asarray([final_by_key[(result.seed, result.target_id)].success for result in baseline], dtype=float)
    return old, new


def compare(baseline: Sequence[EpisodeResult], final: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Compute the seeded percentile paired-bootstrap gain interval.

    Identical paired differences are handled directly because resampling cannot
    add uncertainty in that case and SciPy's general interval is unnecessary.
    """
    old, new = _paired_successes(baseline, final)
    if not len(old):
        raise ValueError("cannot compare empty held-out evaluations")
    differences = new - old
    gain = float(np.mean(differences))
    if np.all(differences == differences[0]):
        gain_ci95 = [float(differences[0]), float(differences[0])]
    else:
        interval = bootstrap(
            (old, new),
            lambda before, after: float(np.mean(after - before)),
            paired=True,
            method="percentile",
            confidence_level=0.95,
            n_resamples=10_000,
            rng=np.random.default_rng(42),
            vectorized=False,
        ).confidence_interval
        gain_ci95 = [float(interval.low), float(interval.high)]
    return {
        "baseline": summarize(baseline),
        "final": summarize(final),
        "gain": gain,
        "gain_ci95": gain_ci95,
    }
