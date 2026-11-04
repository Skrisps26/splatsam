"""File-locked, conservative accounting for Meta development spend."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from splatsam.utils import config

_ZERO = Decimal(0)
_MILLION = Decimal(1000000)


class BudgetExceeded(RuntimeError):
    """A reservation or reported settlement would breach a configured cap."""


class BudgetTracker:
    """Persist reservations in the shared ``runs/meta_budget.json`` ledger.

    A reservation is spend immediately for cap purposes and remains so after a
    process exits.  A caller must settle it with metered cost when available;
    settling without a cost charges the complete reservation conservatively.
    """

    def __init__(self, path: Path, phase: str = "A") -> None:
        self.phase = str(phase)
        self._settings = config()
        self.path = self._ledger_path(Path(path))
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")

    @staticmethod
    def _ledger_path(path: Path) -> Path:
        """Accept a run directory, a runs directory, or an explicit ledger."""
        if path.suffix == ".json":
            return path
        if path.name == "runs":
            return path / "meta_budget.json"
        if path.parent.name == "runs":
            return path.parent / "meta_budget.json"
        return path / "runs" / "meta_budget.json"

    @staticmethod
    def _decimal(value: Decimal | float | str, field: str) -> Decimal:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be numeric, not boolean")  # noqa: TRY004 - unified budget-domain validation
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"{field} must be a finite nonnegative amount") from error
        if not amount.is_finite() or amount < _ZERO:
            raise ValueError(f"{field} must be a finite nonnegative amount")
        return amount

    @staticmethod
    def _count(value: int, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
        return value

    @contextmanager
    def _locked_ledger(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                ledger = self._load()
                try:
                    yield ledger
                finally:
                    # A high actual settlement must reach disk even though its
                    # caller receives BudgetExceeded after accounting for it.
                    self._write(ledger)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "committed_by_phase": {},
                "dispatch_blocked": False,
                "reservations": {},
            }
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"budget ledger is unreadable: {self.path}") from error
        if not isinstance(raw, dict):
            raise RuntimeError(f"budget ledger is invalid: {self.path}")  # noqa: TRY004 - persisted ledger corruption
        committed = raw.get("committed_by_phase", {})
        reservations = raw.get("reservations", {})
        dispatch_blocked = raw.get("dispatch_blocked", False)
        if (
            not isinstance(committed, dict)
            or not isinstance(reservations, dict)
            or not isinstance(dispatch_blocked, bool)
        ):
            raise RuntimeError(f"budget ledger is invalid: {self.path}")  # noqa: TRY004 - persisted ledger corruption
        # Validate monetary fields before deciding a cap; corrupted values must
        # never accidentally reopen the spending gate.
        for phase, amount in committed.items():
            self._decimal(amount, f"committed_by_phase[{phase!r}]")
        for reservation_id, reservation in reservations.items():
            if not isinstance(reservation, dict):
                raise RuntimeError(f"budget reservation {reservation_id!r} is invalid")  # noqa: TRY004 - persisted ledger corruption
            self._decimal(reservation.get("reserved_usd"), f"reservation {reservation_id!r}")
            if not isinstance(reservation.get("phase"), str):
                raise RuntimeError(f"budget reservation {reservation_id!r} has no phase")  # noqa: TRY004 - persisted ledger corruption
        return {
            "version": 1,
            "committed_by_phase": {str(key): str(value) for key, value in committed.items()},
            "dispatch_blocked": dispatch_blocked,
            "reservations": reservations,
        }

    def _write(self, ledger: dict[str, Any]) -> None:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(ledger, output, sort_keys=True, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _cost(self, model: str, input_tokens: int, output_tokens: int, frames: int, images: int) -> Decimal:
        if not isinstance(model, str) or not model:
            raise ValueError("model is required")
        input_tokens = self._count(input_tokens, "input_tokens")
        output_tokens = self._count(output_tokens, "output_tokens")
        frames = self._count(frames, "frames")
        images = self._count(images, "images")

        prices = self._settings.get("prices", {})
        selected = prices.get(model, {})
        wonder = prices.get("muse-wonder-1.0", {"input": 4.0, "output": 20.0})
        if "input" in selected and "output" in selected:
            token_prices = selected
        elif "per_frame" in selected or "per_image" in selected:
            token_prices = None
        else:
            # An unrecognized text model is charged at Wonder's public rates,
            # rather than being allowed through the gate for free.
            token_prices = wonder

        cost = _ZERO
        if token_prices is not None:
            cost += Decimal(input_tokens) * self._decimal(token_prices["input"], "input price") / _MILLION
            cost += Decimal(output_tokens) * self._decimal(token_prices["output"], "output price") / _MILLION

        if frames or images:
            sam_prices = prices.get("sam-3.1", {})
            if frames:
                minimum = int(sam_prices.get("minimum_frames", 50))
                unit = self._decimal(sam_prices.get("per_frame", "0.0002"), "SAM frame price")
                cost += Decimal(max(minimum, frames)) * unit
            else:
                unit = self._decimal(sam_prices.get("per_image", "0.0025"), "SAM image price")
                cost += Decimal(images) * unit
        return cost

    @staticmethod
    def _totals(ledger: dict[str, Any]) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        committed = {
            str(phase): BudgetTracker._decimal(amount, f"committed_by_phase[{phase!r}]")
            for phase, amount in ledger["committed_by_phase"].items()
        }
        pending: dict[str, Decimal] = {}
        for reservation in ledger["reservations"].values():
            phase = reservation["phase"]
            pending[phase] = pending.get(phase, _ZERO) + BudgetTracker._decimal(
                reservation["reserved_usd"], "reservation cost"
            )
        return committed, pending

    def _assert_caps(self, ledger: dict[str, Any]) -> None:
        committed, pending = self._totals(ledger)
        total = sum(committed.values(), _ZERO) + sum(pending.values(), _ZERO)
        budget = self._settings["budget"]
        development_cap = min(self._decimal(budget["development_usd"], "development cap"), Decimal(50))
        reserve = max(self._decimal(budget["reserve_usd"], "reserve"), Decimal(70))
        initial = self._decimal(budget["initial_usd"], "initial budget")
        if total > development_cap:
            raise BudgetExceeded(f"development budget exceeded: ${total} > ${development_cap}")
        if total > initial - reserve:
            raise BudgetExceeded(f"required reserve breached: ${total} > ${initial - reserve}")
        configured_phase_cap = budget.get("phase_a_usd", 10)
        phase_cap = min(
            self._decimal(10 if configured_phase_cap is None else configured_phase_cap, "phase A cap"),
            Decimal(10),
        )
        phase_total = committed.get("A", _ZERO) + pending.get("A", _ZERO)
        if phase_total > phase_cap:
            raise BudgetExceeded(f"Phase A budget exceeded: ${phase_total} > ${phase_cap}")

    @staticmethod
    def _assert_dispatch_open(ledger: dict[str, Any]) -> None:
        if ledger["dispatch_blocked"]:
            raise BudgetExceeded("dispatch is blocked after a provider reported cost above its reservation")

    def reserve(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        frames: int = 0,
        images: int = 0,
    ) -> str:
        """Reserve worst-case cost before an API attempt and return its ID."""
        cost = self._cost(model, input_tokens, output_tokens, frames, images)
        with self._locked_ledger() as ledger:
            self._assert_dispatch_open(ledger)
            reservation_id = uuid4().hex
            ledger["reservations"][reservation_id] = {
                "phase": self.phase,
                "model": model,
                "reserved_usd": str(cost),
            }
            try:
                self._assert_caps(ledger)
            except BudgetExceeded:
                del ledger["reservations"][reservation_id]
                raise
            return reservation_id

    def settle(self, reservation_id: str, cost: Decimal | float | str | None = None) -> None:
        """Commit actual cost, charging the reservation when metering is absent.

        If a provider reports an amount above its reservation, the amount is
        committed first so the ledger never understates spend, then the caller
        receives :class:`BudgetExceeded` and all subsequent reserves fail.
        """
        with self._locked_ledger() as ledger:
            reservation = ledger["reservations"].pop(reservation_id, None)
            if reservation is None:
                raise ValueError(f"unknown or already settled reservation: {reservation_id}")
            reserved = self._decimal(reservation["reserved_usd"], "reserved cost")
            actual = reserved if cost is None else self._decimal(cost, "cost")
            phase = reservation["phase"]
            previous = self._decimal(ledger["committed_by_phase"].get(phase, "0"), "committed cost")
            ledger["committed_by_phase"][phase] = str(previous + actual)
            exceeded_reservation = actual > reserved
            if exceeded_reservation:
                ledger["dispatch_blocked"] = True
            self._assert_caps(ledger)
            if exceeded_reservation:
                raise BudgetExceeded(
                    f"reported cost ${actual} exceeded reservation ${reserved}; future dispatch is blocked"
                )

    @property
    def total_usd(self) -> Decimal:
        """Committed plus pending costs, which is the safe spend position."""
        return self.snapshot()["total_usd"]

    def snapshot(self) -> dict[str, Any]:
        """Return exact Decimal accounting totals, including persisted pending work."""
        with self._locked_ledger() as ledger:
            committed, pending = self._totals(ledger)
            phases = sorted(set(committed) | set(pending))
            phase_totals = {
                phase: {
                    "committed_usd": committed.get(phase, _ZERO),
                    "pending_usd": pending.get(phase, _ZERO),
                    "total_usd": committed.get(phase, _ZERO) + pending.get(phase, _ZERO),
                }
                for phase in phases
            }
            committed_total = sum(committed.values(), _ZERO)
            pending_total = sum(pending.values(), _ZERO)
            return {
                "committed_usd": committed_total,
                "pending_usd": pending_total,
                "total_usd": committed_total + pending_total,
                "phase_totals": phase_totals,
                "pending_reservations": len(ledger["reservations"]),
                "dispatch_blocked": ledger["dispatch_blocked"],
            }
