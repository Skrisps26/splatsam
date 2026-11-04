import pytest

from splatsam.llm.schemas import EpisodeResult
from splatsam.loop.acceptance import acceptance


def rows(successes):
    return [
        EpisodeResult(
            seed=i,
            target_id="box",
            success=i < successes,
            failure_mode=None if i < successes else "slip_or_drop",
        )
        for i in range(100)
    ]


def test_paired_gain_and_bootstrap_gates():
    assert not acceptance(rows(50), rows(51))["accepted"]
    assert not acceptance(rows(50), rows(50))["accepted"]
    assert acceptance(rows(50), rows(70))["accepted"]
    with pytest.raises(ValueError):
        acceptance(rows(50), list(reversed(rows(70))))


def test_mode_regression_guard_even_when_total_gain_positive():
    old = rows(40)
    new = rows(70)
    # Two modes protected by >=15 incumbent episodes. Improve total but worsen
    # one by 10pp, which must reject rather than hide regression in aggregate.
    for i in range(40, 60):
        old[i] = old[i].model_copy(update={"failure_mode": "knock_over"})
    for i in range(70, 100):
        new[i] = new[i].model_copy(update={"failure_mode": "knock_over"})
    decision = acceptance(old, new)
    assert not decision["accepted"] and decision["protected_regressions"]["knock_over"] == 0.10
