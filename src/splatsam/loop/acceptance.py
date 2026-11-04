"""Paired validation gate. Never imports or opens sealed-test data."""

import numpy as np

from splatsam.llm.schemas import EpisodeResult


def acceptance(old: list[EpisodeResult], new: list[EpisodeResult], seed: int = 42) -> dict:
    if not old or [(r.seed, r.target_id) for r in old] != [(r.seed, r.target_id) for r in new]:
        raise ValueError("validation must contain identical paired scenarios in stable order")
    differences = np.array([int(b.success) - int(a.success) for a, b in zip(old, new)], dtype=float)
    gain = float(differences.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(old), size=(10000, len(old)))
    lower = float(np.quantile(differences[draws].mean(axis=1), 0.10))
    regressions = {}
    for mode in sorted({r.failure_mode for r in old if r.failure_mode}):
        n = sum(r.failure_mode == mode for r in old)
        if n >= 15:
            change = (sum(r.failure_mode == mode for r in new) - n) / len(old)
            if change > 0.05 + 1e-12:
                regressions[mode] = change
    accepted = gain >= 0.02 - 1e-12 and lower > 0 and not regressions
    reason = (
        "accepted"
        if accepted
        else (
            "mean gain below 2pp"
            if gain < 0.02 - 1e-12
            else "paired lower bound not positive"
            if lower <= 0
            else "protected failure regression"
        )
    )
    return {
        "accepted": accepted,
        "gain": gain,
        "bootstrap_lower10": lower,
        "protected_regressions": regressions,
        "reason": reason,
    }
