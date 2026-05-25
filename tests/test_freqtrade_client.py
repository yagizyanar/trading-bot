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
