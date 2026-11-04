"""Fail-closed accounting and cleanup for optional GPU resources.

This module deliberately has no cloud SDK.  A provider adapter must be supplied
by an authorized Phase B caller; the default RunPod path is unavailable when no
lifecycle control route is mounted.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Protocol
from uuid import uuid4


ZERO = Decimal("0")
_PROVIDER_DEFAULT_CAPS = {"runpod": Decimal("4"), "aws": Decimal("25")}


class GpuLedgerError(RuntimeError):
    pass


class ProviderUnavailable(GpuLedgerError):
    pass


class GpuLifecycleAdapter(Protocol):
    """Minimal provider interface; no endpoint or persistent-volume capability."""

    def create(self, *, provider: str, reason: str, max_runtime_s: int) -> str: ...

    def terminate(self, resource_id: str) -> None: ...

    def list_active(self) -> list[dict[str, Any]]: ...


class UnavailableRunPodAdapter:
    """Explicitly reject creation when an authenticated RunPod control route is absent."""

    def create(self, *, provider: str, reason: str, max_runtime_s: int) -> str:
        raise ProviderUnavailable("RunPod lifecycle control is unavailable; no resource was created")

    def terminate(self, resource_id: str) -> None:
        raise ProviderUnavailable("RunPod lifecycle control is unavailable")

    def list_active(self) -> list[dict[str, Any]]:
        return []


def _money(value: Decimal | str | int | float, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite nonnegative amount")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be a finite nonnegative amount") from error
    if not amount.is_finite() or amount < ZERO:
        raise ValueError(f"{field} must be a finite nonnegative amount")
    return amount


class GpuLedger:
    """File-locked single-resource lifecycle ledger with conservative metering."""

    def __init__(
        self,
        path: Path,
        adapter: GpuLifecycleAdapter | None = None,
        *,
        phase: str = "A",
        provider_caps: dict[str, Decimal | str | int | float] | None = None,
        teardown_margin_s: int = 300,
    ) -> None:
        if teardown_margin_s < 0:
            raise ValueError("teardown_margin_s must be nonnegative")
        self.path = Path(path) if Path(path).suffix == ".json" else Path(path) / "gpu_ledger.json"
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.adapter = adapter or UnavailableRunPodAdapter()
        self.phase = phase
        self.teardown_margin_s = teardown_margin_s
        self.provider_caps = dict(_PROVIDER_DEFAULT_CAPS)
        for provider, cap in (provider_caps or {}).items():
            key = str(provider).lower()
            configured = _money(cap, f"cap for {provider}")
            self.provider_caps[key] = min(configured, _PROVIDER_DEFAULT_CAPS[key]) if key in _PROVIDER_DEFAULT_CAPS else configured

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                ledger = self._load()
                yield ledger
                self._write(ledger)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "resources": []}
        try:
            value = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise GpuLedgerError(f"GPU ledger is unreadable: {self.path}") from error
        if not isinstance(value, dict) or not isinstance(value.get("resources"), list):
            raise GpuLedgerError(f"GPU ledger is invalid: {self.path}")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        value["total_usd"] = str(sum((_money(record.get("spent_usd", "0"), "spent_usd") for record in value["resources"]), ZERO))
        fd, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(value, output, sort_keys=True, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _active(ledger: dict[str, Any]) -> list[dict[str, Any]]:
        return [record for record in ledger["resources"] if record.get("active") is True]

    @staticmethod
    def _now(value: float | None = None) -> float:
        return time.time() if value is None else float(value)

    @staticmethod
    def _spent(record: dict[str, Any], now: float) -> Decimal:
        price = _money(record["price_per_hour_usd"], "price_per_hour_usd")
        started = float(record["started_at"])
        ended = now if record.get("active") else float(record.get("stopped_at", now))
        return price * Decimal(str(max(0.0, ended - started))) / Decimal("3600")

    def _cap(self, provider: str) -> Decimal:
        try:
            return self.provider_caps[provider.lower()]
        except KeyError as error:
            raise ProviderUnavailable(f"no configured credit cap for provider {provider!r}") from error

    def start(
        self,
        *,
        provider: str,
        price_per_hour_usd: Decimal | str | int | float,
        reason: str,
        max_runtime_s: int,
        now: float | None = None,
    ) -> str:
        """Create exactly one resource after all phase, cap, and reservation gates."""
        if self.phase.upper() != "B":
            raise ProviderUnavailable("Phase A forbids every provider-create call; only confirmed Phase B is eligible")
        if not isinstance(max_runtime_s, int) or isinstance(max_runtime_s, bool) or max_runtime_s <= 0:
            raise ValueError("max_runtime_s must be a positive integer")
        provider = provider.lower()
        price = _money(price_per_hour_usd, "price_per_hour_usd")
        if price <= 0:
            raise GpuLedgerError("a verified positive hourly price is required")
        cap = self._cap(provider)
        if isinstance(self.adapter, UnavailableRunPodAdapter):
            raise ProviderUnavailable("GPU lifecycle control is unavailable; no resource was created")
        current_time = self._now(now)
        with self._locked() as ledger:
            if self._active(ledger):
                raise GpuLedgerError("at most one GPU resource may be active")
            prior = sum((self._spent(record, current_time) for record in ledger["resources"] if record["provider"] == provider), ZERO)
            reserved = price * Decimal(max_runtime_s + self.teardown_margin_s) / Decimal("3600")
            if prior + reserved > cap:
                raise GpuLedgerError("GPU cap cannot cover runtime plus teardown margin")
            # Adapter invocation occurs only after all local gates. If it raises,
            # no ledger resource is asserted and no fake ID is written.
            resource_id = self.adapter.create(provider=provider, reason=reason, max_runtime_s=max_runtime_s)
            if not isinstance(resource_id, str) or not resource_id:
                raise GpuLedgerError("provider returned no resource ID")
            ledger["resources"].append(
                {
                    "id": uuid4().hex,
                    "provider": provider,
                    "resource_id": resource_id,
                    "price_per_hour_usd": str(price),
                    "reason": reason,
                    "started_at": current_time,
                    "max_runtime_s": max_runtime_s,
                    "active": True,
                    "stop_reason": None,
                    "stopped_at": None,
                    "cleanup_verified": False,
                    "spent_usd": "0",
                    "last_activity_at": current_time,
                    "usable_mesh_at": None,
                }
            )
            return resource_id

    def _record(self, ledger: dict[str, Any], resource_id: str) -> dict[str, Any]:
        for record in ledger["resources"]:
            if record.get("resource_id") == resource_id:
                return record
        raise GpuLedgerError(f"unknown GPU resource {resource_id!r}")

    def mark_activity(self, resource_id: str, *, now: float | None = None, usable_mesh: bool = False) -> None:
        with self._locked() as ledger:
            record = self._record(ledger, resource_id)
            if not record.get("active"):
                return
            current = self._now(now)
            record["last_activity_at"] = current
            if usable_mesh:
                record["usable_mesh_at"] = current
            record["spent_usd"] = str(self._spent(record, current))

    def stop(self, resource_id: str, *, reason: str, now: float | None = None) -> bool:
        """Terminate and independently verify absence; preserve active state on failure."""
        current = self._now(now)
        with self._locked() as ledger:
            record = self._record(ledger, resource_id)
            if not record.get("active"):
                return bool(record.get("cleanup_verified"))
            record["stop_reason"] = reason
            record["stop_attempted_at"] = current
            record["spent_usd"] = str(self._spent(record, current))
            try:
                self.adapter.terminate(resource_id)
                active_ids = {
                    str(item.get("resource_id", item.get("id", ""))) for item in self.adapter.list_active()
                }
            except Exception as error:  # A failed cleanup must remain visible/actionable.
                record["cleanup_error"] = str(error)
                record["cleanup_verified"] = False
                return False
            if resource_id in active_ids:
                record["cleanup_error"] = "provider still lists resource active"
                record["cleanup_verified"] = False
                return False
            record["active"] = False
            record["stopped_at"] = current
            record["cleanup_verified"] = True
            record.pop("cleanup_error", None)
            return True

    def watchdog(self, resource_id: str, *, now: float | None = None, finished: bool = False) -> str | None:
        """Stop on finish, ten idle minutes, runtime/cap, or 45 minutes without a usable mesh."""
        current = self._now(now)
        with self._locked() as ledger:
            record = self._record(ledger, resource_id)
            if not record.get("active"):
                return None
            elapsed = current - float(record["started_at"])
            idle = current - float(record.get("last_activity_at", record["started_at"]))
            no_mesh = record.get("usable_mesh_at") is None and elapsed >= 45 * 60
            provider_spend = sum((self._spent(item, current) for item in ledger["resources"] if item["provider"] == record["provider"]), ZERO)
            at_cap = provider_spend >= self._cap(str(record["provider"]))
            if finished:
                reason = "finished"
            elif idle >= 600:
                reason = "idle_timeout"
            elif elapsed >= int(record["max_runtime_s"]):
                reason = "runtime_cap"
            elif at_cap:
                reason = "cost_cap"
            elif no_mesh:
                reason = "no_usable_mesh_timeout"
            else:
                record["spent_usd"] = str(self._spent(record, current))
                return None
        self.stop(resource_id, reason=reason, now=current)
        return reason

    def cleanup_all(self, *, reason: str = "finally", now: float | None = None) -> dict[str, bool]:
        """Best-effort finally cleanup; failed IDs intentionally stay active in the ledger."""
        with self._locked() as ledger:
            identifiers = [str(record["resource_id"]) for record in self._active(ledger)]
        return {identifier: self.stop(identifier, reason=reason, now=now) for identifier in identifiers}

    def records(self) -> list[dict[str, Any]]:
        with self._locked() as ledger:
            return json.loads(json.dumps(ledger["resources"]))


__all__ = [
    "GpuLedger", "GpuLedgerError", "GpuLifecycleAdapter", "ProviderUnavailable", "UnavailableRunPodAdapter",
]
