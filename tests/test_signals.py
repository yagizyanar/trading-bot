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
    TARGET_DAILY_VOL, VOL_NORM_MIN, VOL_NORM_MAX,
    MARKOV_FLIP_THRESHOLD,
    _vol_normalization_multiplier,
    _leverage_from_signal,
    evaluate_signal,
)
from config.settings import MAX_LEVERAGE


def _lev(s: float) -> int:
    """Expected leverage = signal tier capped by MAX_LEVERAGE — mirrors
    evaluate_signal so size assertions stay correct across leverage-config changes."""
    return min(_leverage_from_signal(s), MAX_LEVERAGE)


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
def _make_regime(coin="SOL", signal=0.4, regime="Bull", realized_vol=0.0):
    return RegimeResult(
        coin=coin, timestamp=datetime.now(timezone.utc),
        regime=regime, confidence=0.8,
        bull_prob=0.6, bear_prob=0.2, sideways_prob=0.2,
        markov_signal=signal, rows_used=500, realized_vol=realized_vol,
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
        (0.6, BASE_FULL_PCT / _lev(0.6)),     # >0.5 → 3x lev → size /3 (notional = base)
        (0.4, BASE_MEDIUM_PCT / _lev(0.4)),   # >0.3 → 2x lev → size /2
        (0.15, BASE_SMALL_PCT),       # >0.1 → 1x lev → size unchanged
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
        (-0.6, BASE_FULL_PCT / _lev(0.6)),    # 3x lev
        (-0.4, BASE_MEDIUM_PCT / _lev(0.4)),  # 2x lev
        (-0.15, BASE_SMALL_PCT),      # 1x lev
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
    assert d.position_size_pct == pytest.approx(0.03 * 0.50 / _lev(0.4))   # 0.4 signal → 2x → size /2
    assert d.dollars == pytest.approx(10000 * 0.03 * 0.50 / _lev(0.4))


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
    assert d.position_size_pct == pytest.approx(0.03 * 0.25 / _lev(0.4))   # 0.4 signal → 2x


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
    assert d.position_size_pct == pytest.approx(BASE_MEDIUM_PCT * 1.0 / _lev(0.4))   # -0.4 → 2x


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
    assert d.position_size_pct == pytest.approx(BASE_MEDIUM_PCT * TECH_MULT_CONTRADICTS / _lev(0.4))   # 0.4 → 2x


def test_tech_neutral_does_not_reduce():
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("NEUTRAL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.decision == "LONG"
    assert d.position_size_pct == pytest.approx(BASE_MEDIUM_PCT / _lev(0.4))   # 0.4 → 2x


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
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT * SIDEWAYS_SIZE_MULT / _lev(0.6))   # 0.6 → 3x
    assert d.leverage == _lev(0.6)   # leverage now purely |signal|-based (0.6 → 3x); no Sideways override


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
    # Full base 5% / _lev(0.6)x lev (0.6 signal) × cb 0.5 → 0.83% margin
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT * 0.5 / _lev(0.6))


# ---------------------------------------------------------------------------
# Leverage selection
# ---------------------------------------------------------------------------
def test_leverage_strong_signal_3x():
    # Dynamic leverage by |signal| (2026-06-05): |0.6| > 0.5 → 3x.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull"),
        sentiment=_make_sentiment(unified=0.5),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == _lev(0.6)


def test_leverage_medium_signal_2x():
    # 0.3 < |0.4| <= 0.5 → 2x (leverage is purely |signal|-based, not sentiment).
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.5),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == _lev(0.4)


def test_leverage_weak_signal_1x():
    # 0.1 < |0.15| <= 0.3 → 1x.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.15, regime="Bull"),
        sentiment=_make_sentiment(unified=0.5),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == 1


def test_leverage_short_strong_3x():
    # |-0.6| > 0.5 → 3x on the short side too.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=-0.6, regime="Bear"),
        sentiment=_make_sentiment(unified=-0.5, label="BEARISH"),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.leverage == _lev(0.6)


# ---------------------------------------------------------------------------
# Item 5 — volatility-normalized sizing
# ---------------------------------------------------------------------------
def test_vol_normalization_multiplier_math():
    # scalar = TARGET_DAILY_VOL / realized_vol, clipped to [MIN, MAX].
    # Inputs are expressed relative to TARGET_DAILY_VOL so the test is robust to
    # tuning the constant.
    assert _vol_normalization_multiplier(TARGET_DAILY_VOL) == pytest.approx(1.0)            # equal vol
    assert _vol_normalization_multiplier(2 * TARGET_DAILY_VOL) == pytest.approx(0.5)        # 2x vol → half
    assert _vol_normalization_multiplier(TARGET_DAILY_VOL / 1.5) == pytest.approx(1.5)      # lower vol → bigger
    assert _vol_normalization_multiplier(TARGET_DAILY_VOL / 10) == pytest.approx(VOL_NORM_MAX)  # clip high
    assert _vol_normalization_multiplier(TARGET_DAILY_VOL * 10) == pytest.approx(VOL_NORM_MIN)  # clip low
    assert _vol_normalization_multiplier(0.0) == 1.0                                       # unknown → neutral
    assert _vol_normalization_multiplier(-1.0) == 1.0


def test_vol_norm_disabled_high_vol_not_shrunk():
    # Item 5 DISABLED 2026-06-05 (flat sizing): vol_mult forced to 1.0, so a high-vol
    # full signal is NO LONGER shrunk — it stays at the 5% base.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull", realized_vol=2 * TARGET_DAILY_VOL),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT / _lev(0.6))   # 0.6 signal → 3x → margin /3


def test_vol_norm_disabled_low_vol_not_grown():
    # Item 5 DISABLED 2026-06-05 (flat sizing): vol_mult forced to 1.0, so a low-vol
    # full signal is NO LONGER grown/clipped — it stays at the 5% base.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull", realized_vol=TARGET_DAILY_VOL / 10),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT / _lev(0.6))   # 0.6 signal → 3x → margin /3


def test_vol_normalization_absent_is_neutral():
    # realized_vol unknown (0.0, e.g. degraded regime) → no scaling, base size unchanged.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.6, regime="Bull", realized_vol=0.0),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
    )
    assert d.position_size_pct == pytest.approx(BASE_FULL_PCT / _lev(0.6))   # 0.6 signal → 3x → margin /3


# ---------------------------------------------------------------------------
# Item 7 — hysteresis / re-entry band on flips
# ---------------------------------------------------------------------------
def test_hysteresis_disabled_flips_on_weak_opposite():
    # Item 7 DISABLED 2026-06-05 (MARKOV_FLIP_THRESHOLD=0): a weak opposite (-0.2)
    # now FLIPS the LONG straight to SHORT instead of holding.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=-0.2, regime="Bear"),
        sentiment=_make_sentiment(unified=-0.1, label="NEUTRAL"),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        current_position="LONG",
    )
    assert d.decision == "SHORT"


def test_hysteresis_allows_strong_flip():
    # Currently LONG; signal strongly bearish (-0.4 ≥ 0.3) → flip to SHORT.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=-0.4, regime="Bear"),
        sentiment=_make_sentiment(unified=-0.3, label="BEARISH"),
        technical=_make_tech("BEAR", trend_up=False),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        current_position="LONG",
    )
    assert d.decision == "SHORT"


def test_hysteresis_does_not_block_same_direction():
    # Currently LONG; signal still bullish → normal LONG (no hysteresis).
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.4, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        current_position="LONG",
    )
    assert d.decision == "LONG"


def test_hysteresis_does_not_block_fresh_entry():
    # No open position; a weak signal (0.15 > entry gate 0.1) opens normally —
    # hysteresis only governs FLIPS, not fresh entries.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.15, regime="Bull"),
        sentiment=_make_sentiment(unified=0.3),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        current_position=None,
    )
    assert d.decision == "LONG"


def test_hysteresis_disabled_short_flips_on_weak_bullish():
    # Item 7 DISABLED 2026-06-05: a weak bullish (0.2) now FLIPS the SHORT to LONG.
    d = evaluate_signal(
        "SOL",
        regime_result=_make_regime(signal=0.2, regime="Bull"),
        sentiment=_make_sentiment(unified=0.1, label="NEUTRAL"),
        technical=_make_tech("BULL"),
        capital=10000.0, cb_multiplier=1.0, cb_allows_new=True,
        current_position="SHORT",
    )
    assert d.decision == "LONG"
