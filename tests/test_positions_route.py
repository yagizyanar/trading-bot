"""Regression test for the closed-positions ordering bug.

Freqtrade /api/v1/trades returns OLDEST-first. The route must fetch the full
history and return NEWEST-first, sliced to the display limit — otherwise the
newest trades vanish once total closed > display limit.
"""
from __future__ import annotations

from dashboard.backend.routes import positions as pos


def _fake_trade(i: int, close_day: int) -> dict:
    return {
        "trade_id": i, "pair": "SOL/USDT:USDT", "is_short": True, "is_open": False,
        "open_rate": 100.0, "close_rate": 99.0, "amount": 1.0, "leverage": 1.0,
        "open_date": f"2026-05-{close_day:02d} 00:00:00",
        "close_date": f"2026-05-{close_day:02d} 12:00:00",
        "close_profit_abs": 1.0, "close_profit": 0.01,
    }


def test_closed_positions_returns_newest_first(monkeypatch):
    # Oldest-first input, exactly how Freqtrade returns it (days 11..20).
    fake_oldest_first = [_fake_trade(i, 10 + i) for i in range(1, 11)]
    captured = {}

    def fake_fetch(limit=50):
        captured["limit"] = limit
        return fake_oldest_first

    monkeypatch.setattr(pos, "fetch_closed_trades", fake_fetch)

    out = pos.closed_positions(limit=3, session=None)

    # Must fetch MORE than the display limit (the fix), not just 3.
    assert captured["limit"] >= 100
    # Returns exactly the display limit, NEWEST first.
    assert len(out) == 3
    assert out[0]["exit_ts"].startswith("2026-05-20")   # newest
    assert out[1]["exit_ts"].startswith("2026-05-19")
    assert out[2]["exit_ts"].startswith("2026-05-18")
