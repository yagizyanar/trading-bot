"""yfinance fetcher for supplementary price/volume data.

Crypto symbols on Yahoo are like 'SOL-USD', 'AVAX-USD'. We map our base
symbols to that format.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


def _yf_symbol(coin: str) -> str:
    return f"{coin}-USD"


def fetch_yf_price_change(coin: str, period: str = "7d") -> Optional[float]:
    """Return percentage price change over `period` (e.g. '7d'). None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; skipping yf data for %s", coin)
        return None
    try:
        ticker = yf.Ticker(_yf_symbol(coin))
        df = ticker.history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 2:
            return None
        return float((close.iloc[-1] / close.iloc[0]) - 1.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance fetch failed for %s: %s", coin, exc)
        return None


def fetch_yf_ohlcv(coin: str, period: str = "60d", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Return OHLCV DataFrame for a coin. None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.Ticker(_yf_symbol(coin)).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance OHLCV fetch failed for %s: %s", coin, exc)
        return None
