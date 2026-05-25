"""Tests for the routine base contract (lockfile guard, error handling)."""
from __future__ import annotations

from routines.base import BaseRoutine


class _DummyOK(BaseRoutine):
    name = "dummy_ok"
    def _run_inner(self, snapshot, portfolio, cb_state):
        return {"hello": "world"}


class _DummyError(BaseRoutine):
    name = "dummy_err"
    def _run_inner(self, snapshot, portfolio, cb_state):
        raise RuntimeError("boom")


def _stub_portfolio():
    return {"equity": 10000.0, "peak_equity": 10000.0, "daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0}


def test_routine_runs_to_completion(tmp_memory_dir, tmp_lockfile_path):
    res = _DummyOK(get_portfolio_snapshot=_stub_portfolio).run()
    assert res.success
    assert not res.skipped
    assert res.extra == {"hello": "world"}


def test_routine_aborts_when_lockfile_present(tmp_memory_dir, tmp_lockfile_path):
    tmp_lockfile_path.write_text("locked", encoding="utf-8")
    res = _DummyOK(get_portfolio_snapshot=_stub_portfolio).run()
    assert not res.success
    assert res.skipped
    assert "lockfile" in (res.skip_reason or "").lower()


def test_routine_logs_lesson_on_crash(tmp_memory_dir, tmp_lockfile_path):
    res = _DummyError(get_portfolio_snapshot=_stub_portfolio).run()
    assert not res.success
    assert res.error and "boom" in res.error
    # lessons file should have a FAILED entry
    content = (tmp_memory_dir / "lessons_learned.md").read_text(encoding="utf-8")
    assert "FAILED" in content
    assert "dummy_err" in content
