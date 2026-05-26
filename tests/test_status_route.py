"""Tests for the cron-file-driven /api/status/ route helpers.

Focus is on the cron parsing and next-fire logic. The route function itself
is a thin shell over those helpers, so we don't spin up FastAPI for it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dashboard.backend.routes import status as st


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    """Reset the module-level cron cache before each test."""
    monkeypatch.setattr(st, "_CRON_CACHE", {"ts": 0.0, "specs": None})


# ---------------------------------------------------------------------------
# _field_match
# ---------------------------------------------------------------------------
def test_field_match_star_matches_everything():
    for v in (0, 5, 23, 59):
        assert st._field_match("*", v) is True


def test_field_match_exact_value():
    assert st._field_match("0", 0)
    assert not st._field_match("0", 1)
    assert st._field_match("12", 12)


def test_field_match_range():
    for v in range(1, 6):
        assert st._field_match("1-5", v)
    assert not st._field_match("1-5", 0)
    assert not st._field_match("1-5", 6)


def test_field_match_step():
    # */2 matches 0, 2, 4, 6, ...
    for v in (0, 2, 4, 22):
        assert st._field_match("*/2", v)
    for v in (1, 3, 5, 7):
        assert not st._field_match("*/2", v)


def test_field_match_list():
    assert st._field_match("0,15,30,45", 30)
    assert not st._field_match("0,15,30,45", 31)


# ---------------------------------------------------------------------------
# _matches and _next_fire
# ---------------------------------------------------------------------------
def test_matches_hourly_weekday():
    """0 * * * 1-5 — hourly, weekdays only."""
    spec = ("0", "*", "*", "*", "1-5")
    # Mon 2026-05-25 12:00 UTC
    mon_noon = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    assert st._matches(mon_noon, *spec)
    # Sat (not in 1-5) — Python Mon=0..Sun=6, cron Sun=0..Sat=6
    sat_noon = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    assert not st._matches(sat_noon, *spec)
    # Mon 12:30 — wrong minute
    mon_thirty = datetime(2026, 5, 25, 12, 30, tzinfo=timezone.utc)
    assert not st._matches(mon_thirty, *spec)


def test_matches_every_two_hours():
    spec = ("0", "*/2", "*", "*", "*")
    even = datetime(2026, 5, 25, 14, 0, tzinfo=timezone.utc)
    odd  = datetime(2026, 5, 25, 15, 0, tzinfo=timezone.utc)
    assert st._matches(even, *spec)
    assert not st._matches(odd, *spec)


def test_matches_sunday_only():
    """0 20 * * 0 — Sunday 20:00 UTC (cron Sunday = 0)."""
    spec = ("0", "20", "*", "*", "0")
    sun = datetime(2026, 5, 31, 20, 0, tzinfo=timezone.utc)
    assert sun.weekday() == 6  # Python Sunday
    assert st._matches(sun, *spec)
    mon = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)
    assert not st._matches(mon, *spec)


def test_next_fire_hourly():
    after = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)
    nxt = st._next_fire("0 * * * *", after)
    assert nxt == datetime(2026, 5, 25, 15, 0, tzinfo=timezone.utc)


def test_next_fire_every_two_hours_from_odd():
    after = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
    nxt = st._next_fire("0 */2 * * *", after)
    assert nxt == datetime(2026, 5, 25, 16, 0, tzinfo=timezone.utc)


def test_next_fire_every_minute():
    after = datetime(2026, 5, 25, 14, 30, 45, tzinfo=timezone.utc)
    nxt = st._next_fire("* * * * *", after)
    assert nxt == datetime(2026, 5, 25, 14, 31, tzinfo=timezone.utc)


def test_next_fire_weekday_skips_weekend():
    """0 8 * * 1-5 — running on Saturday should skip to Monday."""
    sat = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)  # weekday()=5
    nxt = st._next_fire("0 8 * * 1-5", sat)
    assert nxt == datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    assert nxt.weekday() == 0  # Monday


# ---------------------------------------------------------------------------
# _read_cron_specs
# ---------------------------------------------------------------------------
def test_read_cron_specs_falls_back_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "CRON_FILE", tmp_path / "does-not-exist")
    specs = st._read_cron_specs()
    names = {n for n, _ in specs}
    assert "market_evaluation" in names
    assert "sentiment_update"  in names


def test_read_cron_specs_parses_real_file(tmp_path, monkeypatch):
    cron = tmp_path / "trading-bot"
    cron.write_text(
        "SHELL=/bin/bash\n"
        "PATH=/usr/local/bin\n"
        "TRADE_ROOT=/opt/trading-bot\n"
        "\n"
        "# trading-bot routines — all times UTC\n"
        "0 0  * * 1-5  root  cd /opt/trading-bot && .venv/bin/python -m routines.pre_market\n"
        "0 */2 * * *  root  cd /opt/trading-bot && .venv/bin/python -m routines.sentiment_update\n"
        "0 *   * * *  root  cd /opt/trading-bot && .venv/bin/python -m routines.market_evaluation\n"
        "0 12 * * 1-5  root  cd /opt/trading-bot && .venv/bin/python -m routines.midday_check\n"
        "0 16 * * 1-5  root  cd /opt/trading-bot && .venv/bin/python -m routines.day_close\n"
        "0 20 * * 0    root  cd /opt/trading-bot && .venv/bin/python -m routines.weekly_review\n"
        "* * * * *    root  cd /opt/trading-bot && .venv/bin/python -m routines.position_monitor\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(st, "CRON_FILE", cron)
    specs = dict(st._read_cron_specs())

    assert specs["pre_market"]        == "0 0 * * 1-5"
    assert specs["sentiment_update"]  == "0 */2 * * *"
    assert specs["market_evaluation"] == "0 * * * *"
    assert specs["midday_check"]      == "0 12 * * 1-5"
    assert specs["day_close"]         == "0 16 * * 1-5"
    assert specs["weekly_review"]     == "0 20 * * 0"
    assert specs["position_monitor"]  == "* * * * *"


def test_read_cron_specs_skips_comments_and_env_lines(tmp_path, monkeypatch):
    cron = tmp_path / "trading-bot"
    cron.write_text(
        "# a comment\n"
        "SHELL=/bin/bash\n"
        "PATH=/usr/bin\n"
        "0 0 * * * root cd /x && python -m routines.foo\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(st, "CRON_FILE", cron)
    specs = st._read_cron_specs()
    assert specs == [("foo", "0 0 * * *")]


# ---------------------------------------------------------------------------
# Integration: status() returns the right next_routine
# ---------------------------------------------------------------------------
def test_status_route_excludes_position_monitor_from_next_widget(tmp_path, monkeypatch):
    cron = tmp_path / "trading-bot"
    cron.write_text(
        "0 *   * * *  root cd /x && python -m routines.market_evaluation\n"
        "* * * * *    root cd /x && python -m routines.position_monitor\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(st, "CRON_FILE", cron)

    res = st.status()
    # next_routine should be market_evaluation (not position_monitor)
    assert res["next_routine"]["name"] == "market_evaluation"
    # upcoming list also omits position_monitor
    names_in_upcoming = {item["name"] for item in res["upcoming"]}
    assert "position_monitor" not in names_in_upcoming
    assert "market_evaluation" in names_in_upcoming


def test_status_route_returns_cron_spec_for_each_routine(tmp_path, monkeypatch):
    cron = tmp_path / "trading-bot"
    cron.write_text(
        "0 0 * * 1-5  root cd /x && python -m routines.pre_market\n"
        "0 */2 * * *  root cd /x && python -m routines.sentiment_update\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(st, "CRON_FILE", cron)

    res = st.status()
    by_name = {item["name"]: item for item in res["upcoming"]}
    assert by_name["sentiment_update"]["cron"] == "0 */2 * * *"
    assert by_name["pre_market"]["cron"]       == "0 0 * * 1-5"
