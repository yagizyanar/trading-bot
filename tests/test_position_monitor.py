"""Tests for the 5-minute position monitor.

Lockfile guard is the only piece testable without a DB; the SL/TP loop needs
SessionLocal + Binance, which we mock around with monkeypatching to avoid net.
"""
from __future__ import annotations

from datetime import datetime, timezone

from routines import position_monitor


def test_run_skips_when_lockfile_present(tmp_memory_dir, tmp_lockfile_path):
    tmp_lockfile_path.write_text("locked", encoding="utf-8")
    res = position_monitor.run()
    assert res.skipped is True
    assert "lockfile" in (res.skip_reason or "").lower()
    assert res.open_positions_checked == 0


def test_run_returns_monitor_result_shape(tmp_memory_dir, tmp_lockfile_path, monkeypatch):
    """Smoke test: run with no open trades and stubbed DB session — returns OK."""
    # Stub SessionLocal so the test doesn't need a real DB
    class _StubSession:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def query(self, *a, **kw): return self
        def filter(self, *a, **kw): return self
        def all(self): return []
        def add(self, *a, **kw): return None
        def commit(self): return None

    monkeypatch.setattr(position_monitor, "SessionLocal", _StubSession)

    res = position_monitor.run()
    assert res.skipped is False
    assert res.error is None
    assert res.open_positions_checked == 0
    assert res.stop_losses_triggered == 0
    assert res.take_profits_triggered == 0
    assert res.anomalies_logged == 0
    assert isinstance(res.started_at, datetime)
    assert res.started_at.tzinfo == timezone.utc
