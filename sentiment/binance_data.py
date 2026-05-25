"""Binance Futures OHLCV + volume anomaly detector.

Volume spike rule (user PHASE 4):
  spike = current_volume / mean(volume over previous 7 days)
  spike > 2.0 → anomaly_score = +1.0
  spike < 0.5 → anomaly_score = -1.0 (volume drying up)
  else linear interp into [-1, +1]
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    VOLUME_SPIKE_MULTIPLIER,
)

log = logging.getLogger(__name__)


def _client():
    """Lazy import of python-binance to keep import cost low."""
    from binance.client import Client  # type: ignore
    return Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)


def fetch_binance_ohlcv(
    pair: str,
    interval: str = "1h",
    limit: int = 500,
) -> Optional[pd.DataFrame]:
    """Fetch futures OHLCV. `pair` like 'SOLUSDT' (no slash). None on failure."""
    try:
        cli = _client()
        klines = cli.futures_klines(symbol=pair, interval=interval, limit=limit)
        if not klines:
            return None
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("open_time")
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("Binance OHLCV fetch failed for %s: %s", pair, exc)
        return None


def volume_anomaly(df: pd.DataFrame, hours_window: int = 24 * 7) -> Optional[float]:
    """Compute volume anomaly score in [-1.0, +1.0].

    Compares the most recent 1h volume bar to the mean of the prior `hours_window` bars.
    """
    if df is None or df.empty or "volume" not in df.columns:
        return None
    if len(df) < hours_window + 2:
        return None
    recent = float(df["volume"].iloc[-1])
    baseline = float(df["volume"].iloc[-(hours_window + 1):-1].mean())
    if baseline <= 0:
        return None
    spike = recent / baseline
    if spike >= VOLUME_SPIKE_MULTIPLIER:
        return 1.0
    if spike <= 0.5:
        return -1.0
    # Linear interp between [0.5, 1.0] -> [-1, 0] and [1.0, VOLUME_SPIKE_MULTIPLIER] -> [0, +1]
    if spike < 1.0:
        return (spike - 1.0) / 0.5  # in (-1, 0)
    return (spike - 1.0) / (VOLUME_SPIKE_MULTIPLIER - 1.0)  # in (0, +1)
