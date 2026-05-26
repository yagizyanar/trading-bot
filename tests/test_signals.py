"""Tests for technical indicators and the Markov-driven signal gate.

Logic under test is signals.three_layer (refactored from the old 3-layer
all-or-nothing gate to Markov-primary + sentiment/technical multipliers).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from markov.regime_detector import RegimeResult
from sentiment.analyzer import UnifiedScore
from signals.technical import TechnicalSnapshot, compute_technical_indicators, technical_label
from signals.three_layer import (
    BASE_FULL_PCT, BASE_MEDIUM_PCT, BASE_SMALL_PCT,
    SIDEWAYS_SIZE_MULT, TECH_MULT_CONTRADICTS,
    evaluate_signal,
)


def _ohlcv(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": [1000.0] * len(prices),
    }, index=idx)


# ---------------------------------------------------------------------------
# Technical indicators — unchanged
# ---------------------------------------------------------------------------
def test_compute_indicators_handles_short_series():
    df = _ohlcv([100.0] * 10)
    assert compute_technical_indicators(df) is None


def test_compute_indicators_uptrend():
    prices = list(np.linspace(100, 120, 60))
    snap = compute_technical_indicators(_ohlcv(prices))
    assert snap is not None
    assert snap.trend_up is True


def test_technical_label_bull_on_oversold_uptrend():
    label = technical_label(rsi=30.0, macd_hist=0.5, trend_up=True, volume_ratio=2.5, bb_pct=0.05)
    assert label == "BULL"


def test_technical_label_bear_on_overbought_downtrend():
    label = technical_label(rsi=72.0, macd_hist=-0.5, trend_up=False, volume_ratio=2.5, bb_pct=0.95)
    assert label == "BEAR"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
        news_score=unified, volume_anomaly=0.0,
        yfinance_change=0.01,
        long_short_ratio=None, funding_rate=None, hyperliquid_score=None,
        fear_greed=60, fear_greed_multiplier=1.0,
        unified=unified, signal=label,
    )


def _make_tech(label="BULL", trend_up=True):
    return TechnicalSnapshot(
        rsi=30.0, macd=1.0, macd_signal=0.5, macd_hist=0.5,
        bb_pct=0.1, ema_fast=110.0, ema_slow=100.0,
        volume_ratio=2.0, trend_up=trend_up, label=label,
    )


# ---------------------------------------------------------------------------
# Markov-as-sole-gate
# ---------------------------------------------------------------------------
def test_long_base_size_tiers():
    """Markov signal alone decides direction + base size."""
    for sig, expect_pct in [
        (0.6, BASE_FULL_PCT),    # > 0.5
        (0.4, BASE_MEDIUM_PCT),  # > 0.3
        (0.15, BASE_SMALL_PCT),  # > 0.1
    ]:
        d = evaluate_signal(
            "SOL",
            regime_result=_make_regime(signal=sig, regime="Bull"),
            sentiment=_make_sentiment(unified=0.3),   # ×1.0
            technical=_make_tech("BULL"),             # ×1.0
            capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        )
        assert d.decision == "LONG", f"sig={sig}"
        assert d.position_size_pct == pytest.approx(expect_pct), f"sig={sig}"


def test_dead_zone_skips():
    """|markov| ≤ 0.1 → SKIP."""
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.05, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "SKIP"
    assert "dead zone" in (d.skip_reason or "")


def test_short_tiers():
    for sig, expect_pct in [
        (-0.6, BASE_FULL_PCT),
        (-0.4, BASE_MEDIUM_PCT),
        (-0.15, BASE_SMALL_PCT),
    ]:
        d = evaluate_signal(
            "SOL",
            regime_result=_make_regime(signal=sig, regime="Bear"),
            sentiment=_make_sentiment(unified=-0.3, label="BEARISH"),
            technical=_make_tech("BEAR", trend_up=False),
            capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        )
        assert d.decision == "SHORT", f"sig={sig}"
        assert d.position_size_pct == pytest.approx(expect_pct), f"sig={sig}"


# ---------------------------------------------------------------------------
# Sentiment as MULTIPLIER (not gate)
# ---------------------------------------------------------------------------
def test_sentiment_reduces_long_size_but_does_not_skip():
    """User's worked example: markov +0.4 (3% base) + sent -0.1 (×0.50) → 1.5%."""
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=-0.1, label="NEUTRAL"),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct == pytest.approx(0.03 * 0.50)
    assert d.dollars == pytest.approx(150.0)


def test_sentiment_negative_still_trades_long():
    """Even sentiment < -0.2 doesn't skip a LONG — just reduces by 75%."""
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=-0.5, label="BEARISH"),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct == pytest.approx(0.03 * 0.25)


def test_sentiment_aligned_for_short_keeps_size():
    """Negative sentiment is aligned with a SHORT — full size kept."""
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=-0.4, regime="Bear"),
        sentiment=_make_sentiment(unified=-0.3, label="BEARISH"),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "SHORT"
    assert d.position_size_pct == pytest.approx(BASE_MEDIUM_PCT * 1.0)


# ---------------------------------------------------------------------------
# Technical as soft confirmation (contradiction → -25%, never gates)
# ---------------------------------------------------------------------------
def test_tech_contradiction_reduces_size_25():
    """BEAR tech on a LONG: position halved by 25%."""
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct == pytest.approx(BASE_MEDIUM_PCT * TECH_MULT_CONTRADICTS)


def test_tech_neutral_does_not_reduce():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("NEUTRAL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct == pytest.approx(BASE_MEDIUM_PCT)


# ---------------------------------------------------------------------------
# Sideways regime: halve sizes + force 1x leverage
# ---------------------------------------------------------------------------
def test_sideways_halves_and_forces_1x():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Sideways"),
        sentiment=_make_sentiment(unified=0.5),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT * SIDEWAYS_SIZE_MULT)
    assert d.leverage == 1, "Sideways must force 1x even with strong sentiment"


# ---------------------------------------------------------------------------
# Crash regime: hard SKIP
# ---------------------------------------------------------------------------
def test_crash_hard_skip():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=-0.5, regime="Crash"),
        sentiment=_make_sentiment(unified=-0.6, label="BEARISH"),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "SKIP"
    assert "Crash" in (d.skip_reason or "")


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------
def test_cb_disallows_new_skips():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=0.0, cb_allows_new=False,
    )
    assert d.decision == "SKIP"


def test_cb_multiplier_scales_position():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull"),
        sentiment=_make_sentiment(unified=0.4),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=0.5, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    # Full base 5%, all multipliers 1.0 except cb 0.5 → 2.5%
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT * 0.5)


# ---------------------------------------------------------------------------
# Leverage selection
# ---------------------------------------------------------------------------
def test_2x_leverage_in_bull_with_strong_sentiment():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull"),
        sentiment=_make_sentiment(unified=0.5),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == 2


def test_1x_leverage_when_sentiment_weak():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull"),
        sentiment=_make_sentiment(unified=0.15),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == 1


def test_short_2x_in_bear_with_strong_negative_sentiment():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=-0.6, regime="Bear"),
        sentiment=_make_sentiment(unified=-0.5, label="BEARISH"),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == 2
