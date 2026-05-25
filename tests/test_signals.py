"""Tests for technical indicators and 3-layer signal gate."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from markov.regime_detector import RegimeResult
from sentiment.analyzer import UnifiedScore
from signals.technical import TechnicalSnapshot, compute_technical_indicators, technical_label
from signals.three_layer import evaluate_signal


def _ohlcv(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [1000.0] * len(prices),
    }, index=idx)


def test_compute_indicators_handles_short_series():
    df = _ohlcv([100.0] * 10)
    assert compute_technical_indicators(df) is None


def test_compute_indicators_uptrend():
    prices = list(np.linspace(100, 120, 60))
    snap = compute_technical_indicators(_ohlcv(prices))
    assert snap is not None
    assert snap.trend_up is True
    assert snap.macd > snap.macd_signal or snap.macd_hist > 0


def test_technical_label_bull_on_oversold_uptrend():
    label = technical_label(rsi=30.0, macd_hist=0.5, trend_up=True, volume_ratio=2.5, bb_pct=0.05)
    assert label == "BULL"


def test_technical_label_bear_on_overbought_downtrend():
    label = technical_label(rsi=72.0, macd_hist=-0.5, trend_up=False, volume_ratio=2.5, bb_pct=0.95)
    assert label == "BEAR"


def _make_regime(coin="SOL", signal=0.4, regime="Bull"):
    return RegimeResult(
        coin=coin, timestamp=datetime.now(timezone.utc),
        regime=regime, confidence=0.8,
        bull_prob=0.6, bear_prob=0.2, sideways_prob=0.2,
        markov_signal=signal, rows_used=500,
    )


def _make_sentiment(coin="SOL", unified=0.3, label="BULLISH"):
    return UnifiedScore(
        coin=coin, timestamp=datetime.now(timezone.utc),
        news_score=unified, senticrypt=unified, volume_anomaly=0.0,
        yfinance_change=0.01, fear_greed=60, fear_greed_multiplier=1.0,
        unified=unified, signal=label,
    )


def _make_tech(label="BULL"):
    return TechnicalSnapshot(
        rsi=30.0, macd=1.0, macd_signal=0.5, macd_hist=0.5,
        bb_pct=0.1, ema_fast=110.0, ema_slow=100.0,
        volume_ratio=2.0, trend_up=True, label=label,
    )


def test_three_layer_long_when_all_agree():
    d = evaluate_signal(
        coin="SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct > 0


def test_three_layer_skip_when_sentiment_neutral():
    d = evaluate_signal(
        coin="SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.05, label="NEUTRAL"),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "SKIP"


def test_three_layer_skip_on_crash():
    d = evaluate_signal(
        coin="SOL",
        regime_result=_make_regime(signal=-0.4, regime="Crash"),
        sentiment=_make_sentiment(unified=-0.5, label="BEARISH"),
        technical=_make_tech("BEAR"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "SKIP"
    assert "Crash" in (d.skip_reason or "")


def test_three_layer_skip_when_cb_disallows():
    d = evaluate_signal(
        coin="SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=0.0, cb_allows_new=False,
    )
    assert d.decision == "SKIP"


def test_three_layer_short_when_all_bearish():
    d = evaluate_signal(
        coin="SOL",
        regime_result=_make_regime(signal=-0.4, regime="Bear"),
        sentiment=_make_sentiment(unified=-0.3, label="BEARISH"),
        technical=TechnicalSnapshot(
            rsi=70.0, macd=-1.0, macd_signal=-0.5, macd_hist=-0.5,
            bb_pct=0.9, ema_fast=100.0, ema_slow=110.0,
            volume_ratio=2.0, trend_up=False, label="BEAR",
        ),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "SHORT"
    assert d.leverage == 1   # shorts default to 1x


def test_three_layer_long_uses_cb_size_multiplier():
    d = evaluate_signal(
        coin="SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull"),
        sentiment=_make_sentiment(unified=0.4),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=0.5, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert pytest.approx(d.position_size_pct, rel=1e-6) == 0.025  # 0.05 * 0.5
