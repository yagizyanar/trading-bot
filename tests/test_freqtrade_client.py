"""Tests for the Freqtrade REST client.

We don't actually hit Freqtrade — we monkeypatch requests.get / os.environ.
"""
from __future__ import annotations

import pytest

from dashboard.backend import freqtrade_client


class _MockResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clear_cache():
    freqtrade_client.invalidate_cache()
    yield
    freqtrade_client.invalidate_cache()


def test_returns_none_when_password_missing(monkeypatch):
    monkeypatch.delenv("FREQTRADE_API_PASSWORD", raising=False)
    assert freqtrade_client.fetch_balance() is None
    assert freqtrade_client.live_equity() is None


def test_fetch_balance_success(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    payload = {
        "currencies": [{"currency": "USDT", "balance": 10000.0}],
        "total": 10000.0,
        "value": 9990.88,
        "symbol": "USD",
        "note": "Simulated balances",
        "starting_capital_fiat": 5000.0,
    }
    calls = []
    def fake_get(url, auth, timeout):
        calls.append((url, auth, timeout))
        return _MockResponse(200, payload)
    monkeypatch.setattr(freqtrade_client.requests, "get", fake_get)
    b = freqtrade_client.fetch_balance()
    assert b is not None
    assert b["value"] == 9990.88
    assert len(calls) == 1
    assert calls[0][1] == ("freqtrader", "secret")


def test_live_equity_prefers_value_over_total(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    monkeypatch.setattr(freqtrade_client.requests, "get",
                        lambda *a, **kw: _MockResponse(200, {"total": 10000, "value": 9990.88}))
    assert freqtrade_client.live_equity() == pytest.approx(9990.88)


def test_live_equity_falls_back_to_total(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    monkeypatch.setattr(freqtrade_client.requests, "get",
                        lambda *a, **kw: _MockResponse(200, {"total": 10000}))
    assert freqtrade_client.live_equity() == 10000.0


def test_live_equity_falls_back_when_value_is_zero(monkeypatch):
    """When fiat_display_currency is disabled, Freqtrade returns value=0.0 — fall back to total."""
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    monkeypatch.setattr(freqtrade_client.requests, "get",
                        lambda *a, **kw: _MockResponse(200, {"total": 10000, "value": 0.0}))
    assert freqtrade_client.live_equity() == 10000.0


def test_returns_none_on_http_error(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    monkeypatch.setattr(freqtrade_client.requests, "get",
                        lambda *a, **kw: _MockResponse(500, None))
    assert freqtrade_client.fetch_balance() is None


def test_returns_none_on_connection_error(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    def boom(*a, **kw):
        raise ConnectionError("refused")
    monkeypatch.setattr(freqtrade_client.requests, "get", boom)
    assert freqtrade_client.fetch_balance() is None


def test_cache_avoids_repeated_calls(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    calls = []
    def fake_get(url, auth, timeout):
        calls.append(url)
        return _MockResponse(200, {"value": 100.0})
    monkeypatch.setattr(freqtrade_client.requests, "get", fake_get)
    freqtrade_client.fetch_balance()
    freqtrade_client.fetch_balance()
    freqtrade_client.fetch_balance()
    assert len(calls) == 1, f"expected 1 call (cached), got {len(calls)}"


def test_fetch_status_returns_list(monkeypatch):
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    monkeypatch.setattr(freqtrade_client.requests, "get",
                        lambda *a, **kw: _MockResponse(200, [{"pair": "SOL/USDT:USDT"}]))
    s = freqtrade_client.fetch_status()
    assert isinstance(s, list)
    assert s[0]["pair"] == "SOL/USDT:USDT"


def test_fetch_closed_trades_unwraps_payload(monkeypatch):
    """Freqtrade returns {"trades": [...]} — the client should return just the list."""
    monkeypatch.setenv("FREQTRADE_API_PASSWORD", "secret")
    payload = {"trades": [{"trade_id": 1, "pair": "SOL/USDT:USDT"}], "trades_count": 1}
    monkeypatch.setattr(freqtrade_client.requests, "get",
                        lambda *a, **kw: _MockResponse(200, payload))
    trades = freqtrade_client.fetch_closed_trades(limit=10)
    assert isinstance(trades, list)
    assert len(trades) == 1
    assert trades[0]["trade_id"] == 1


def test_map_freqtrade_trade_open_short():
    raw = {
        "trade_id": 1,
        "pair": "SOL/USDT:USDT",
        "is_short": True,
        "is_open": True,
        "amount": 5.84,
        "open_rate": 85.54,
        "current_rate": 85.52,
        "close_rate": None,
        "profit_pct": -0.03,           # Freqtrade reports as percentage
        "profit_abs": -0.14,
        "open_date": "2026-05-26 10:51:53",
        "close_date": None,
        "leverage": 1.0,
        "enter_tag": "",
        "exit_reason": None,
    }
    m = freqtrade_client.map_freqtrade_trade(raw)
    assert m["id"] == 1
    assert m["coin"] == "SOL"
    assert m["side"] == "SHORT"
    assert m["entry_price"] == 85.54
    assert m["exit_price"] is None
    assert m["leverage"] == 1
    assert m["pnl_usd"] == -0.14
    # frontend expects a fraction (0.05 == 5%), not Freqtrade's percent number
    assert m["pnl_pct"] == pytest.approx(-0.03 / 100.0)
    assert m["outcome"] == "OPEN"
    assert m["is_paper"] is True
    assert m["reason_in"] == "freqtrade"


def test_map_freqtrade_trade_closed_win_long():
    raw = {
        "trade_id": 7,
        "pair": "INJ/USDT:USDT",
        "is_short": False,
        "is_open": False,
        "amount": 88.2,
        "open_rate": 5.666,
        "current_rate": 6.516,
        "close_rate": 6.516,
        "profit_pct": 15.00,
        "profit_abs": 74.97,
        "open_date": "2026-05-26 10:51:00",
        "close_date": "2026-05-27 09:30:00",
        "leverage": 2.0,
        "enter_tag": "",
        "exit_reason": "roi",
    }
    m = freqtrade_client.map_freqtrade_trade(raw)
    assert m["side"] == "LONG"
    assert m["exit_price"] == 6.516
    assert m["leverage"] == 2
    assert m["outcome"] == "WIN"
    assert m["pnl_pct"] == pytest.approx(0.15)
    assert m["reason_out"] == "roi"


def test_map_freqtrade_trade_loss_outcome():
    raw = {
        "trade_id": 3,
        "pair": "ARB/USDT:USDT",
        "is_short": True,
        "is_open": False,
        "amount": 100.0,
        "open_rate": 1.0, "close_rate": 1.05,
        "profit_pct": -5.0, "profit_abs": -5.0,
        "open_date": "x", "close_date": "y",
        "leverage": 1.0,
        "exit_reason": "stop_loss",
    }
    m = freqtrade_client.map_freqtrade_trade(raw)
    assert m["outcome"] == "LOSS"
    assert m["reason_out"] == "stop_loss"
