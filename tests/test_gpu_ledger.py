from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from splatsam.gpu_ledger import GpuLedger, GpuLedgerError, ProviderUnavailable


class FakeAdapter:
    def __init__(self, *, fail_terminate: bool = False) -> None:
        self.active: dict[str, dict[str, str]] = {}
        self.create_calls = 0
        self.fail_terminate = fail_terminate

    def create(self, *, provider: str, reason: str, max_runtime_s: int) -> str:
        self.create_calls += 1
        identifier = f"{provider}-{self.create_calls}"
        self.active[identifier] = {"resource_id": identifier}
        return identifier

    def terminate(self, resource_id: str) -> None:
        if self.fail_terminate:
            raise RuntimeError("provider cleanup failed")
        self.active.pop(resource_id, None)

    def list_active(self) -> list[dict[str, str]]:
        return list(self.active.values())


def test_phase_a_never_calls_provider_create(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    ledger = GpuLedger(tmp_path / "runs", adapter, phase="A")
    with pytest.raises(ProviderUnavailable, match="Phase A"):
        ledger.start(provider="runpod", price_per_hour_usd="1", reason="mesh", max_runtime_s=60)
    assert adapter.create_calls == 0


def test_concurrent_create_gate_allows_only_one_active_resource(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    ledger = GpuLedger(tmp_path / "runs", adapter, phase="B")

    def create() -> str:
        return ledger.start(provider="runpod", price_per_hour_usd="1", reason="mesh", max_runtime_s=60)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create), pool.submit(create)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except GpuLedgerError as error:
                outcomes.append(error)
    # One transaction is permitted, while the competing transaction is rejected.
    assert outcomes.count("runpod-1") == 1
    assert sum(isinstance(outcome, GpuLedgerError) for outcome in outcomes) == 1
    assert adapter.create_calls == 1
    assert len(ledger.records()) == 1 and ledger.records()[0]["active"]




def test_cleanup_failure_preserves_active_provider_id(tmp_path: Path) -> None:
    adapter = FakeAdapter(fail_terminate=True)
    ledger = GpuLedger(tmp_path / "runs", adapter, phase="B")
    identifier = ledger.start(provider="runpod", price_per_hour_usd="1", reason="mesh", max_runtime_s=60, now=100.0)
    assert not ledger.stop(identifier, reason="finally", now=110.0)
    record = ledger.records()[0]
    assert record["resource_id"] == identifier
    assert record["active"] is True
    assert record["cleanup_verified"] is False
    assert "provider cleanup failed" in record["cleanup_error"]


def test_watchdog_and_verified_cleanup_account_boot_and_idle_time(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    ledger = GpuLedger(tmp_path / "runs", adapter, phase="B")
    identifier = ledger.start(provider="runpod", price_per_hour_usd="2", reason="mesh", max_runtime_s=1_000, now=100.0)
    assert ledger.watchdog(identifier, now=701.0) == "idle_timeout"
    record = ledger.records()[0]
    assert record["active"] is False and record["cleanup_verified"] is True
    assert float(record["spent_usd"]) > 0


def test_unconfigured_provider_and_runtime_reservation_fail_closed(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    ledger = GpuLedger(tmp_path / "runs", adapter, phase="B")
    with pytest.raises(ProviderUnavailable, match="no configured credit"):
        ledger.start(provider="modal", price_per_hour_usd="1", reason="mesh", max_runtime_s=60)
    with pytest.raises(GpuLedgerError, match="teardown"):
        ledger.start(provider="runpod", price_per_hour_usd="100", reason="mesh", max_runtime_s=60)
    assert adapter.create_calls == 0

def test_hard_caps_cannot_be_raised_by_configuration(tmp_path):
    adapter=FakeAdapter()
    ledger=GpuLedger(tmp_path,adapter,phase="B",provider_caps={"runpod":100},teardown_margin_s=0)
    with pytest.raises(GpuLedgerError,match="cap"):
        ledger.start(provider="runpod",price_per_hour_usd=1,reason="too long",max_runtime_s=18000,now=0)
    assert adapter.create_calls==0

def test_total_cost_remains_visible_after_cleanup(tmp_path):
    import json
    ledger=GpuLedger(tmp_path,FakeAdapter(),phase="B",teardown_margin_s=0)
    resource=ledger.start(provider="runpod",price_per_hour_usd=1,reason="batch",max_runtime_s=3600,now=0)
    assert ledger.stop(resource,reason="finished",now=3600)
    assert float(json.loads((tmp_path/"gpu_ledger.json").read_text())["total_usd"])==1
