"""Tests for sentiment modules — pure logic, no network."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from config.settings import TARGET_COINS
from sentiment.analyzer import UnifiedScore, _blend, _label, _yfinance_to_signal
from sentiment.binance_data import volume_anomaly
from sentiment import crypto_news, coingecko_data
from sentiment.crypto_news import (
    HeadlineScore, NewsItem, score_headlines, _detect_coins, _parse_rss, fetch_crypto_news,
)
from sentiment.coingecko_data import COINGECKO_IDS, fetch_price_changes_7d
from sentiment.fear_greed import _classify_to_multiplier


class _FakeResp:
    """Minimal stand-in for requests.Response for mocked-network tests."""
    def __init__(self, text="", json_data=None, status=200):
        self.text = text
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item><title>Solana surges to a new high</title><link>http://x/1</link>
    <pubDate>Wed, 03 Jun 2026 12:00:00 +0000</pubDate>
    <description>SOL rallies hard</description></item>
  <item><title>Chainlink announces upgrade</title><link>http://x/2</link>
    <pubDate>Wed, 03 Jun 2026 11:00:00 +0000</pubDate>
    <description>LINK news</description></item>
</channel></rss>"""


def test_fear_greed_multiplier_tiers():
    assert _classify_to_multiplier(10) == 0.5    # extreme fear
    assert _classify_to_multiplier(30) == 0.75   # fear
    assert _classify_to_multiplier(50) == 1.0    # neutral
    assert _classify_to_multiplier(70) == 1.0    # greed
    assert _classify_to_multiplier(85) == 0.8    # extreme greed (contrarian caution)


def test_detect_coins_aliases():
    assert "SOL" in _detect_coins("Solana ETF approved")
    assert "POL" in _detect_coins("Polygon zk-EVM upgrade")
    assert "S" in _detect_coins("Sonic mainnet launches today")
    assert _detect_coins("Just random text") == ()


def test_sonic_bare_letter_does_not_false_match():
    """`S` alias is omitted on purpose — single-letter would match anywhere.
    Headlines like 'BTC and ETH rally' must NOT yield an 'S' detection."""
    assert "S" not in _detect_coins("Bitcoin and Ethereum rally as ETFs roll out")
    assert "S" not in _detect_coins("S&P 500 hits new highs")  # only "Sonic" / "Sonic Labs" should match


def test_score_headlines_aggregates_per_coin():
    items = [
        NewsItem("Solana rally to ATH", None, datetime.now(timezone.utc), ("SOL",)),
        NewsItem("Solana crashes 15%", None, datetime.now(timezone.utc), ("SOL",)),
        NewsItem("LINK breakout incoming", None, datetime.now(timezone.utc), ("LINK",)),
    ]
    scores = score_headlines(items)
    assert "SOL" in scores and "LINK" in scores
    assert isinstance(scores["SOL"], HeadlineScore)
    assert scores["SOL"].mention_count == 2


def test_blend_redistributes_for_missing():
    full = {"news": 0.5, "volume": 0.5, "yfinance": 0.5,
            "long_short_ratio": 0.5, "funding_rate": 0.5, "hyperliquid": 0.5}
    partial = {"news": 0.5, "volume": 0.5, "yfinance": None,
               "long_short_ratio": None, "funding_rate": None, "hyperliquid": None}
    assert _blend(full) == pytest.approx(0.5)
    assert _blend(partial) == pytest.approx(0.5)  # redistributed proportionally
    assert _blend({k: None for k in full}) == 0.0


def test_unified_label_thresholds():
    assert _label(0.3) == "BULLISH"
    assert _label(-0.3) == "BEARISH"
    assert _label(0.0) == "NEUTRAL"
    assert _label(0.2) == "NEUTRAL"  # boundary — strictly greater


def test_yfinance_to_signal_monotone():
    assert _yfinance_to_signal(-0.5) < _yfinance_to_signal(0) < _yfinance_to_signal(0.5)
    assert -1.0 <= _yfinance_to_signal(-10) <= 1.0
    assert -1.0 <= _yfinance_to_signal(10) <= 1.0


def test_volume_anomaly_spike():
    # 199 baseline bars @100, a closed "recent" bar @250 (>2x baseline), then a
    # still-forming in-progress bar that must be ignored (iloc[-1]).
    idx = pd.date_range("2024-01-01", periods=201, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [1.0] * 201, "high": [1.0] * 201, "low": [1.0] * 201, "close": [1.0] * 201,
        "volume": [100.0] * 199 + [250.0, 5.0],  # iloc[-2]=250 (scored), iloc[-1]=5 (in-progress)
    }, index=idx)
    s = volume_anomaly(df, hours_window=24 * 7)
    assert s == pytest.approx(1.0)


def test_volume_anomaly_drying_up():
    idx = pd.date_range("2024-01-01", periods=201, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": [1.0] * 201, "high": [1.0] * 201, "low": [1.0] * 201, "close": [1.0] * 201,
        "volume": [100.0] * 199 + [40.0, 5.0],  # iloc[-2]=40 (<0.5x baseline), iloc[-1] ignored
    }, index=idx)
    assert volume_anomaly(df, hours_window=24 * 7) == pytest.approx(-1.0)


def test_volume_anomaly_ignores_in_progress_bar():
    # Regression for the -1.0-for-every-coin bug: the final Binance candle is
    # still forming and has partial volume. Here the last *closed* bar equals
    # baseline (score ~0) while the in-progress bar is tiny. The old code scored
    # iloc[-1] and wrongly returned -1.0; the fix must ignore it and return ~0.0.
    idx = pd.date_range("2024-01-01", periods=201, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "volume": [100.0] * 200 + [3.0],  # iloc[-2]=100 (==baseline), iloc[-1]=3 (in-progress)
    }, index=idx)
    assert volume_anomaly(df, hours_window=24 * 7) == pytest.approx(0.0)


def test_volume_anomaly_returns_none_on_short_series():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"volume": [100.0] * 10}, index=idx)
    assert volume_anomaly(df) is None


# --- news: RSS parsing (replaces paywalled cryptocurrency.cv) ---------------

def test_parse_rss_extracts_items():
    rows = _parse_rss(_SAMPLE_RSS)
    assert len(rows) == 2
    title, link, ts, desc = rows[0]
    assert title == "Solana surges to a new high"
    assert link == "http://x/1"
    assert ts.tzinfo is not None              # tz-aware
    assert ts.year == 2026 and ts.month == 6
    assert "rallies" in desc


def test_parse_rss_malformed_returns_empty():
    assert _parse_rss("<not xml") == []


def test_fetch_crypto_news_from_rss(monkeypatch):
    monkeypatch.setattr(crypto_news.requests, "get",
                        lambda *a, **k: _FakeResp(text=_SAMPLE_RSS))
    items = fetch_crypto_news(feeds=["http://fake/rss"])
    assert len(items) == 2
    by_title = {it.title: it for it in items}
    assert "SOL" in by_title["Solana surges to a new high"].coins_mentioned
    assert "LINK" in by_title["Chainlink announces upgrade"].coins_mentioned
    # newest-first ordering
    assert items[0].ts >= items[1].ts


def test_fetch_crypto_news_dedups_across_feeds(monkeypatch):
    monkeypatch.setattr(crypto_news.requests, "get",
                        lambda *a, **k: _FakeResp(text=_SAMPLE_RSS))
    items = fetch_crypto_news(feeds=["http://a/rss", "http://b/rss"])  # same content twice
    assert len(items) == 2                    # de-duplicated by title


def test_fetch_crypto_news_one_bad_feed_doesnt_sink_others(monkeypatch):
    def fake_get(url, *a, **k):
        if "bad" in url:
            raise RuntimeError("boom")
        return _FakeResp(text=_SAMPLE_RSS)
    monkeypatch.setattr(crypto_news.requests, "get", fake_get)
    items = fetch_crypto_news(feeds=["http://bad/rss", "http://good/rss"])
    assert len(items) == 2                    # good feed still parsed


# --- price change: CoinGecko (replaces yfinance for the 8 failing coins) ----

def test_coingecko_ids_cover_all_target_coins():
    missing = [c for c in TARGET_COINS if c not in COINGECKO_IDS]
    assert missing == [], f"TARGET_COINS missing a CoinGecko id: {missing}"


def test_coingecko_price_changes_parsing(monkeypatch):
    payload = [
        {"id": "solana", "price_change_percentage_7d_in_currency": -10.92},
        {"id": "bonk", "price_change_percentage_7d_in_currency": -11.73},
        {"id": "aptos", "price_change_percentage_7d_in_currency": None},  # resolved but no 7d
    ]
    monkeypatch.setattr(coingecko_data.requests, "get",
                        lambda *a, **k: _FakeResp(json_data=payload))
    out = fetch_price_changes_7d(["SOL", "1000BONK", "APT"])
    # percent -> fraction, and 1000BONK maps back from the 'bonk' id
    assert out["SOL"] == pytest.approx(-0.1092)
    assert out["1000BONK"] == pytest.approx(-0.1173)
    assert "APT" not in out                   # None 7d value is dropped


def test_coingecko_empty_on_request_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(coingecko_data.requests, "get", boom)
    assert fetch_price_changes_7d(["SOL"]) == {}
