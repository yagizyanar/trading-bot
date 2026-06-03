"""Tests for the 1-minute position monitor.

Freqtrade-backed path is tested by mocking dashboard.backend.freqtrade_client.fetch_status.
DB-backed fallback is tested by making the Freqtrade fetch return None.
"""
from __future__ import annotations

from datetime import datetime, timezone

from routines import position_monitor


def test_run_skips_when_lockfile_present(tmp_memory_dir, tmp_lockfile_path):
    tmp_lockfile_path.write_text("locked", encoding="utf-8")
    res = position_monitor.run()
    assert res.skipped is True
    assert "lockfile" in (res.skip_reason or "").lower()
    assert res.source == "none"
    assert res.open_positions_checked == 0


def test_run_freqtrade_source_no_trades(tmp_memory_dir, tmp_lockfile_path, monkeypatch):
    """Freqtrade reachable, zero open trades — monitor runs clean."""
    monkeypatch.setattr(
        "dashboard.backend.freqtrade_client.fetch_status",
        lambda: [],
    )
    res = position_monitor.run()
    assert res.source == "freqtrade"
    assert res.open_positions_checked == 0
    assert res.error is None


def test_run_freqtrade_source_observes_sl(tmp_memory_dir, tmp_lockfile_path, monkeypatch):
    """A Freqtrade trade past -5% PnL triggers a SL_NEAR log line."""
    monkeypatch.setattr(
        "dashboard.backend.freqtrade_client.fetch_status",
        lambda: [{
            "pair": "SOL/USDT:USDT",
            "is_short": True,
            "open_rate": 100.0,
            "current_rate": 106.0,
            "profit_pct": -6.0,        # -6%, past SL
        }],
    )
    # Make _five_min_change return None so the anomaly path doesn't fire
    monkeypatch.setattr(position_monitor, "_five_min_change", lambda coin: None)

    res = position_monitor.run()
    assert res.source == "freqtrade"
    assert res.open_positions_checked == 1
    assert res.stop_losses_triggered == 1
    assert res.take_profits_triggered == 0
    # SL_NEAR line should be in the trade log
    log_content = (tmp_memory_dir / "trade_log.md").read_text(encoding="utf-8")
    assert "SL_NEAR" in log_content
    assert "SOL" in log_content


def test_run_freqtrade_source_observes_tp(tmp_memory_dir, tmp_lockfile_path, monkeypatch):
    # TAKE_PROFIT_PCT is 0.15 (+15% ROI exit). profit_pct just past +15%
    # exercises the TP_NEAR alert path.
    monkeypatch.setattr(
        "dashboard.backend.freqtrade_client.fetch_status",
        lambda: [{
            "pair": "INJ/USDT:USDT",
            "is_short": False,
            "open_rate": 5.0,
            "current_rate": 5.8,
            "profit_pct": 16.0,        # +16%, just past TP (TAKE_PROFIT_PCT=0.15 = 15%)
        }],
    )
    monkeypatch.setattr(position_monitor, "_five_min_change", lambda coin: None)

    res = position_monitor.run()
    assert res.take_profits_triggered == 1
    log_content = (tmp_memory_dir / "trade_log.md").read_text(encoding="utf-8")
    assert "TP_NEAR" in log_content


def test_run_anomaly_logged_independently(tmp_memory_dir, tmp_lockfile_path, monkeypatch):
    """A 3%+ 5-min price move while a position is open → ANOMALY logged."""
    monkeypatch.setattr(
        "dashboard.backend.freqtrade_client.fetch_status",
        lambda: [{
            "pair": "SOL/USDT:USDT",
            "is_short": False,
            "open_rate": 100.0,
            "current_rate": 101.0,
            "profit_pct": 1.0,
        }],
    )
    # Force a +3.5% 5-min move
    monkeypatch.setattr(position_monitor, "_five_min_change", lambda coin: 0.035)

    res = position_monitor.run()
    assert res.anomalies_logged == 1
    log_content = (tmp_memory_dir / "trade_log.md").read_text(encoding="utf-8")
    assert "ANOMALY" in log_content


def test_run_falls_back_to_db_when_freqtrade_unreachable(tmp_memory_dir, tmp_lockfile_path, monkeypatch):
    """Freqtrade unreachable → DB fallback path executes (with zero trades here)."""
    monkeypatch.setattr(
        "dashboard.backend.freqtrade_client.fetch_status",
        lambda: None,
    )
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
    assert res.source == "db"
    assert res.open_positions_checked == 0
    assert res.error is None
    assert isinstance(res.started_at, datetime)
    assert res.started_at.tzinfo == timezone.utc
