"""Markov regime detector — wraps the markov-hedge-fund-method skill.

Skill output is 3 states (Bear/Sideways/Bull). PHASE 3 of the user spec also
asks for Crash and Euphoria. We layer these on top by examining the recent
rolling return magnitude + volatility:

  CRASH    : rolling_return < -3 * threshold AND vol > vol_p75
  EUPHORIA : rolling_return > +3 * threshold AND vol > vol_p75
  BULL/BEAR/NEUTRAL : as per skill's regime label of the *current* day

Markov signal (PHASE 5):
  signal = P(next=Bull | current_state) - P(next=Bear | current_state)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import MARKOV_MIN_TRAIN, MARKOV_THRESHOLD, MARKOV_WINDOW
from sentiment.binance_data import fetch_binance_ohlcv

log = logging.getLogger(__name__)

# Prefer the external Claude Code skill when present (it's the canonical source),
# fall back to our vendored copy otherwise. Both implementations are identical
# pure-NumPy/Pandas code; the fallback exists so the bot is portable to hosts
# that don't have the skill installed (e.g. the VPS).
_SKILL_DIR = Path.home() / ".claude" / "skills" / "markov-hedge-fund-method"
if _SKILL_DIR.exists():
    sys.path.insert(0, str(_SKILL_DIR))

try:
    from markov_hedge_fund_method.regime import (  # type: ignore
        STATES,
        build_transition_matrix,
        label_regimes,
        signal_from_matrix,
        stationary_distribution,
    )
    _MARKOV_SOURCE = "external_skill"
except ImportError:
    from ._skill_fallback import (
        STATES,
        build_transition_matrix,
        label_regimes,
        signal_from_matrix,
        stationary_distribution,
    )
    _MARKOV_SOURCE = "vendored_fallback"
    log.info("markov-hedge-fund-method skill not available; using vendored fallback")


@dataclass(frozen=True)
class RegimeResult:
    coin: str
    timestamp: datetime
    regime: str               # Bull / Bear / Sideways / Crash / Euphoria
    confidence: float         # 0..1
    bull_prob: float
    bear_prob: float
    sideways_prob: float
    markov_signal: float      # P(Bull) - P(Bear) | current state
    rows_used: int
    note: Optional[str] = None


_DEFAULT_DEGRADED = RegimeResult(
    coin="",
    timestamp=datetime.now(timezone.utc),
    regime="Sideways",
    confidence=0.0,
    bull_prob=0.33,
    bear_prob=0.33,
    sideways_prob=0.34,
    markov_signal=0.0,
    rows_used=0,
    note="degraded — skill unavailable or insufficient data",
)


def _extreme_regime(close: pd.Series, window: int, threshold: float) -> Optional[str]:
    """Layer Crash/Euphoria detection on top of the 3-state base."""
    if len(close) < window * 2:
        return None
    rolling_return = close.pct_change(window)
    daily_returns = close.pct_change()
    vol = daily_returns.rolling(window=window).std()
    if vol.dropna().empty:
        return None
    vol_p75 = float(vol.dropna().quantile(0.75))
    latest_vol = float(vol.iloc[-1]) if not np.isnan(vol.iloc[-1]) else 0.0
    latest_rr = float(rolling_return.iloc[-1]) if not np.isnan(rolling_return.iloc[-1]) else 0.0
    if latest_rr <= -3 * threshold and latest_vol >= vol_p75:
        return "Crash"
    if latest_rr >= 3 * threshold and latest_vol >= vol_p75:
        return "Euphoria"
    return None


def _base_label_name(state_idx: int) -> str:
    if state_idx == 2:
        return "Bull"
    if state_idx == 0:
        return "Bear"
    return "Sideways"


def detect_regime(close: pd.Series, coin: str) -> RegimeResult:
    """Compute regime + Markov signal from a Close-price series."""
    now = datetime.now(timezone.utc)
    # label_regimes / build_transition_matrix are always non-None now
    # (we fall back to the vendored copy at import time).

    if close is None or len(close) < MARKOV_MIN_TRAIN + 30:
        return RegimeResult(
            coin=coin,
            timestamp=now,
            regime="Sideways",
            confidence=0.0,
            bull_prob=0.33, bear_prob=0.33, sideways_prob=0.34,
            markov_signal=0.0,
            rows_used=len(close) if close is not None else 0,
            note="insufficient data",
        )

    labels = label_regimes(close, window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
    if labels.empty:
        return RegimeResult(
            coin=coin, timestamp=now, regime="Sideways", confidence=0.0,
            bull_prob=0.33, bear_prob=0.33, sideways_prob=0.34,
            markov_signal=0.0, rows_used=0, note="no labels",
        )

    P = build_transition_matrix(labels)
    current_state = int(labels.iloc[-1])
    next_row = P[current_state]   # P(next=Bear), P(next=Sideways), P(next=Bull)
    bear_p, side_p, bull_p = float(next_row[0]), float(next_row[1]), float(next_row[2])
    markov_signal = float(signal_from_matrix(P, current_state))

    extreme = _extreme_regime(close, MARKOV_WINDOW, MARKOV_THRESHOLD)
    regime = extreme or _base_label_name(current_state)
    confidence = float(max(bear_p, side_p, bull_p))

    return RegimeResult(
        coin=coin,
        timestamp=now,
        regime=regime,
        confidence=confidence,
        bull_prob=bull_p,
        bear_prob=bear_p,
        sideways_prob=side_p,
        markov_signal=markov_signal,
        rows_used=len(close),
    )


def compute_regime_for_coin(coin: str, days: int = 365) -> RegimeResult:
    """End-to-end: fetch Binance daily candles, run regime detection.

    Window: 365 days. A controlled walk-forward sweep (2026-06-02, evaluating
    252/365/540/730-day windows on IDENTICAL out-of-sample days) found window
    length is roughly immaterial for edge — Sharpe 2.26-2.32 across all windows
    on the survivorship-controlled majors universe, with 365 marginally best.
    Chose 365 over the previous 730 because a shorter window adapts a little
    faster to structural change while 365 daily bars is still ample for a
    stable 3x3 transition-matrix estimate. The difference is noise-level; this
    is a low-stakes, reversible choice. See backtest/daily_walk_forward.py.
    """
    pair = f"{coin}USDT"
    df = fetch_binance_ohlcv(pair, interval="1d", limit=min(days, 1500))
    if df is None or df.empty:
        return RegimeResult(
            coin=coin,
            timestamp=datetime.now(timezone.utc),
            regime="Sideways",
            confidence=0.0,
            bull_prob=0.33, bear_prob=0.33, sideways_prob=0.34,
            markov_signal=0.0,
            rows_used=0,
            note="binance fetch failed",
        )
    close = df["close"].dropna()
    return detect_regime(close, coin)
