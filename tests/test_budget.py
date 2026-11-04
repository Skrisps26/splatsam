from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import multiprocessing
from pathlib import Path

import pytest

from splatsam.budget import BudgetExceeded, BudgetTracker


def _reserve_half_dollar(runs: str) -> bool:
    tracker = BudgetTracker(Path(runs) / "worker")
    try:
        tracker.reserve("unpriced-model", input_tokens=125_000)
    except BudgetExceeded:
        return False
    return True


def test_phase_a_cap_includes_pending_reservations(tmp_path: Path) -> None:
    tracker = BudgetTracker(tmp_path / "runs" / "first")
    reservation = tracker.reserve("unpriced-model", input_tokens=2_500_000)

    assert tracker.total_usd == Decimal("10")
    with pytest.raises(BudgetExceeded):
        tracker.reserve("unpriced-model", input_tokens=1)

    tracker.settle(reservation)
    snapshot = tracker.snapshot()
    assert snapshot["committed_usd"] == Decimal("10")
    assert snapshot["pending_usd"] == Decimal("0")
    assert snapshot["phase_totals"]["A"]["total_usd"] == Decimal("10")


def test_development_cap_applies_outside_phase_a(tmp_path: Path) -> None:
    tracker = BudgetTracker(tmp_path / "runs" / "first", phase="B")
    tracker.reserve("unpriced-model", input_tokens=12_500_000)

    with pytest.raises(BudgetExceeded):
        tracker.reserve("unpriced-model", input_tokens=1)


def test_concurrent_reservations_share_one_locked_ledger(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=8, mp_context=context) as pool:
        accepted = list(pool.map(_reserve_half_dollar, [str(runs)] * 24))

    tracker = BudgetTracker(runs / "reader")
    snapshot = tracker.snapshot()
    assert sum(accepted) == 20
    assert snapshot["pending_reservations"] == 20
    assert snapshot["total_usd"] == Decimal("10")


def test_missing_or_excess_actual_cost_is_conservative_and_persistent(tmp_path: Path) -> None:
    tracker = BudgetTracker(tmp_path / "runs" / "first")
    failed = tracker.reserve("unpriced-model", input_tokens=100_000)
    tracker.settle(failed)

    excessive = tracker.reserve("unpriced-model", input_tokens=25_000)
    with pytest.raises(BudgetExceeded):
        tracker.settle(excessive, "0.2")

    restarted = BudgetTracker(tmp_path / "runs" / "second")
    snapshot = restarted.snapshot()
    assert snapshot["committed_usd"] == Decimal("0.6")
    assert snapshot["pending_usd"] == Decimal("0")
    assert snapshot["dispatch_blocked"] is True
    with pytest.raises(BudgetExceeded):
        restarted.reserve("unpriced-model", input_tokens=1)


def test_sam_frame_minimum_and_image_prices(tmp_path: Path) -> None:
    tracker = BudgetTracker(tmp_path / "runs" / "first")
    frames = tracker.reserve("sam-3.1", frames=3)
    images = tracker.reserve("sam-3.1", images=2)

    assert tracker.total_usd == Decimal("0.0150")
    tracker.settle(frames)
    tracker.settle(images)
    assert tracker.total_usd == Decimal("0.0150")
