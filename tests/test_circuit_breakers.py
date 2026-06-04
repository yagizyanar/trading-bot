"""Tests for circuit breaker logic and the lockfile."""
from __future__ import annotations

from risk.circuit_breakers import CircuitBreakerLevel, evaluate_circuit_breakers
from risk.lockfile import is_locked, write_lockfile


def test_nominal_when_no_losses(tmp_memory_dir, tmp_lockfile_path):
    s = evaluate_circuit_breakers(0.0, 0.0, 10000.0, 10000.0, log=False)
    assert s.level == CircuitBreakerLevel.NOMINAL
    assert s.size_multiplier == 1.0
    assert s.allow_new_positions is True


def test_daily_halve_at_2pct(tmp_memory_dir, tmp_lockfile_path):
    s = evaluate_circuit_breakers(-0.025, 0.0, 9750.0, 10000.0, log=False)
    assert s.level == CircuitBreakerLevel.HALVE_SIZES
    assert s.size_multiplier == 0.5


def test_daily_close_at_3pct(tmp_memory_dir, tmp_lockfile_path):
    s = evaluate_circuit_breakers(-0.035, 0.0, 9650.0, 10000.0, log=False)
    assert s.level == CircuitBreakerLevel.CLOSE_ALL
    assert s.must_close_all
    assert not s.allow_new_positions


def test_daily_pause_at_5pct(tmp_memory_dir, tmp_lockfile_path):
    s = evaluate_circuit_breakers(-0.055, 0.0, 9450.0, 10000.0, log=False)
    assert s.level == CircuitBreakerLevel.PAUSE
    assert s.must_close_all


def test_weekly_stop_at_8pct(tmp_memory_dir, tmp_lockfile_path):
    s = evaluate_circuit_breakers(0.0, -0.085, 9150.0, 10000.0, log=False)
    assert s.level == CircuitBreakerLevel.WEEKLY_STOP
    assert s.must_close_all


def test_drawdown_writes_lockfile(tmp_memory_dir, tmp_lockfile_path):
    # Drawdown just past DRAWDOWN_LOCK_PCT → LOCKED (threshold-relative, robust to tuning).
    from config.settings import DRAWDOWN_LOCK_PCT
    assert not tmp_lockfile_path.exists()
    peak = 10000.0
    eq = peak * (1.0 - DRAWDOWN_LOCK_PCT - 0.005)
    s = evaluate_circuit_breakers(0.0, 0.0, eq, peak, log=False)
    assert s.level == CircuitBreakerLevel.LOCKED
    assert tmp_lockfile_path.exists()
    assert "Drawdown" in tmp_lockfile_path.read_text()


def test_drawdown_below_lock_threshold_does_not_lock(tmp_memory_dir, tmp_lockfile_path):
    # A drawdown a couple points UNDER the lock (with no daily/weekly loss) must
    # NOT lock — guards that the relaxed 20% lock doesn't trip during normal ops.
    from config.settings import DRAWDOWN_LOCK_PCT
    peak = 10000.0
    eq = peak * (1.0 - DRAWDOWN_LOCK_PCT + 0.02)
    s = evaluate_circuit_breakers(0.0, 0.0, eq, peak, log=False)
    assert s.level != CircuitBreakerLevel.LOCKED


def test_lockfile_present_short_circuits(tmp_memory_dir, tmp_lockfile_path):
    write_lockfile("manual", 10000.0, 9500.0, 0.05)
    assert is_locked()
    s = evaluate_circuit_breakers(0.0, 0.0, 9500.0, 10000.0, log=False)
    assert s.level == CircuitBreakerLevel.LOCKED
    assert not s.allow_new_positions


def test_most_severe_wins_when_multiple_triggered(tmp_memory_dir, tmp_lockfile_path):
    s = evaluate_circuit_breakers(-0.025, -0.06, 9400.0, 10000.0, log=False)
    # daily HALVE (1) + weekly WEEKLY_REDUCE (2) — top is WEEKLY_REDUCE
    assert s.level == CircuitBreakerLevel.WEEKLY_REDUCE
