"""Technical indicators: RSI, MACD, Bollinger Bands, EMA, volume spike.

Pure NumPy/Pandas — no TA-Lib (avoids the Windows C-compiler dance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    TECHNICAL_RSI_OVERBOUGHT,
    TECHNICAL_RSI_OVERSOLD,
    VOLUME_SPIKE_MULTIPLIER,
)


@dataclass(frozen=True)
class TechnicalSnapshot:
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_pct: float            # position within Bollinger band, 0=lower, 1=upper
    ema_fast: float
    ema_slow: float
    volume_ratio: float      # last_volume / 20-bar avg
    trend_up: bool           # EMA fast > slow
    label: str               # BULL / BEAR / NEUTRAL


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_pct(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    lower = sma - num_std * std
    upper = sma + num_std * std
    span = (upper - lower).replace(0, np.nan)
    return (close - lower) / span  # 0=lower, 1=upper


def compute_technical_indicators(df: pd.DataFrame) -> Optional[TechnicalSnapshot]:
    """Compute one snapshot from the latest bar of an OHLCV DataFrame.

    Expects columns: 'close', 'volume'. Returns None if insufficient bars.
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = df["close"].astype(float).dropna()
    volume = df["volume"].astype(float).dropna() if "volume" in df.columns else None
    if len(close) < 30:
        return None

    rsi = _rsi(close).iloc[-1]
    macd_line, sig_line, hist = _macd(close)
    bb_pct = _bollinger_pct(close).iloc[-1]
    ema_fast = close.ewm(span=12, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=26, adjust=False).mean().iloc[-1]

    if volume is not None and len(volume) >= 21:
        baseline = float(volume.iloc[-21:-1].mean())
        vol_ratio = float(volume.iloc[-1] / baseline) if baseline > 0 else 1.0
    else:
        vol_ratio = 1.0

    trend_up = ema_fast > ema_slow

    label = technical_label(
        rsi=float(rsi),
        macd_hist=float(hist.iloc[-1]),
        trend_up=trend_up,
        volume_ratio=vol_ratio,
        bb_pct=float(bb_pct) if not np.isnan(bb_pct) else 0.5,
    )

    return TechnicalSnapshot(
        rsi=float(rsi) if not np.isnan(rsi) else 50.0,
        macd=float(macd_line.iloc[-1]),
        macd_signal=float(sig_line.iloc[-1]),
        macd_hist=float(hist.iloc[-1]),
        bb_pct=float(bb_pct) if not np.isnan(bb_pct) else 0.5,
        ema_fast=float(ema_fast),
        ema_slow=float(ema_slow),
        volume_ratio=vol_ratio,
        trend_up=trend_up,
        label=label,
    )


def technical_label(
    rsi: float,
    macd_hist: float,
    trend_up: bool,
    volume_ratio: float,
    bb_pct: float,
) -> str:
    """Combine indicators into a BULL/BEAR/NEUTRAL label (Layer 3).

    Bullish: RSI<35 + uptrend, OR MACD positive cross + volume spike + trend_up.
    Bearish: RSI>65 + downtrend, OR MACD negative cross + volume spike + downtrend.
    """
    bull_votes = 0
    bear_votes = 0

    if rsi < TECHNICAL_RSI_OVERSOLD and trend_up:
        bull_votes += 2
    if rsi > TECHNICAL_RSI_OVERBOUGHT and not trend_up:
        bear_votes += 2

    if macd_hist > 0 and trend_up:
        bull_votes += 1
    if macd_hist < 0 and not trend_up:
        bear_votes += 1

    if volume_ratio >= VOLUME_SPIKE_MULTIPLIER:
        if trend_up:
            bull_votes += 1
        else:
            bear_votes += 1

    if bb_pct <= 0.1 and trend_up:
        bull_votes += 1
    if bb_pct >= 0.9 and not trend_up:
        bear_votes += 1

    if bull_votes >= bear_votes + 2:
        return "BULL"
    if bear_votes >= bull_votes + 2:
        return "BEAR"
    return "NEUTRAL"
