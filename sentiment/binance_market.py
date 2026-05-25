"""Binance Futures public market-data endpoints (no API key required).

Three signals:
  - Long/Short Ratio  : retail-account ratio.  >1.5 → bearish (overleveraged longs),
                        <0.7 → bullish (capitulated longs).
  - Funding Rate      : >0.01% → bearish, <-0.01% → bullish (contrarian).
  - Open Interest     : compare last two 1h buckets; OI rising → trend confirmation
                        in whichever direction price moved.

All fetchers return None on failure (caller blends with redistribution).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from config.settings import API_RETRY_ATTEMPTS, API_TIMEOUT_SECONDS

log = logging.getLogger(__name__)

LONG_SHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OPEN_INTEREST_URL = "https://fapi.binance.com/futures/data/openInterestHist"

LONG_SHORT_BEARISH = 1.5
LONG_SHORT_BULLISH = 0.7
FUNDING_BEARISH = 0.0001    # +0.01%
FUNDING_BULLISH = -0.0001   # -0.01%


@dataclass(frozen=True)
class LongShortReading:
    coin: str
    ratio: float          # raw ratio
    long_account: float
    short_account: float
    timestamp: datetime
    signal: float         # in [-1, +1]: positive=bullish, negative=bearish


@dataclass(frozen=True)
class FundingReading:
    coin: str
    rate: float           # raw funding rate (e.g. 0.0001 = 0.01%)
    timestamp: datetime
    signal: float         # in [-1, +1]


@dataclass(frozen=True)
class OpenInterestReading:
    coin: str
    current_oi: float
    previous_oi: float
    change_pct: float     # (current - previous) / previous
    timestamp: datetime
    signal: float         # in [-1, +1]; needs price direction context for confirmation


def _get(url: str, params: dict) -> Optional[list]:
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=API_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("%s attempt %s failed: %s", url, attempt, exc)
    return None


def _long_short_signal(ratio: float) -> float:
    """Map ratio → [-1, +1]. Above 1.5 → bearish; below 0.7 → bullish."""
    if ratio >= LONG_SHORT_BEARISH:
        # Saturates at 2.5x → -1.0
        return max(-1.0, -(ratio - LONG_SHORT_BEARISH) / (LONG_SHORT_BEARISH - 0.5))
    if ratio <= LONG_SHORT_BULLISH:
        # Saturates at 0.3x → +1.0
        return min(1.0, (LONG_SHORT_BULLISH - ratio) / (LONG_SHORT_BULLISH - 0.3))
    return 0.0


def fetch_long_short_ratio(coin: str) -> Optional[LongShortReading]:
    pair = f"{coin}USDT"
    rows = _get(LONG_SHORT_URL, {"symbol": pair, "period": "1h", "limit": 1})
    if not rows:
        return None
    try:
        row = rows[0]
        ratio = float(row["longShortRatio"])
        return LongShortReading(
            coin=coin,
            ratio=ratio,
            long_account=float(row.get("longAccount", 0.0)),
            short_account=float(row.get("shortAccount", 0.0)),
            timestamp=datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=timezone.utc),
            signal=_long_short_signal(ratio),
        )
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("long_short parse failed for %s: %s", coin, exc)
        return None


def _funding_signal(rate: float) -> float:
    """Contrarian: positive funding (longs paying) → bearish; negative → bullish."""
    if rate >= FUNDING_BEARISH:
        # Saturate at 0.1% (10x our threshold)
        return max(-1.0, -(rate - FUNDING_BEARISH) / (0.001 - FUNDING_BEARISH))
    if rate <= FUNDING_BULLISH:
        return min(1.0, (FUNDING_BULLISH - rate) / (FUNDING_BULLISH - (-0.001)))
    return 0.0


def fetch_funding_rate(coin: str) -> Optional[FundingReading]:
    pair = f"{coin}USDT"
    rows = _get(FUNDING_URL, {"symbol": pair, "limit": 1})
    if not rows:
        return None
    try:
        row = rows[-1]
        rate = float(row["fundingRate"])
        return FundingReading(
            coin=coin,
            rate=rate,
            timestamp=datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=timezone.utc),
            signal=_funding_signal(rate),
        )
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("funding parse failed for %s: %s", coin, exc)
        return None


def fetch_open_interest(coin: str) -> Optional[OpenInterestReading]:
    """Compare the latest two 1h OI buckets.

    Returns a signal that represents the *magnitude* of OI change. The caller
    must combine with price direction to confirm trend (OI rising + price up
    = bullish; OI rising + price down = bearish; OI falling = position unwind).
    """
    pair = f"{coin}USDT"
    rows = _get(OPEN_INTEREST_URL, {"symbol": pair, "period": "1h", "limit": 2})
    if not rows or len(rows) < 2:
        return None
    try:
        # rows are oldest first
        prev = float(rows[-2]["sumOpenInterest"])
        curr = float(rows[-1]["sumOpenInterest"])
        if prev <= 0:
            return None
        change_pct = (curr - prev) / prev
        # Scale OI change into a magnitude in [-1, +1]; 5% change saturates
        signal = max(-1.0, min(1.0, change_pct / 0.05))
        return OpenInterestReading(
            coin=coin,
            current_oi=curr,
            previous_oi=prev,
            change_pct=change_pct,
            timestamp=datetime.fromtimestamp(int(rows[-1]["timestamp"]) / 1000, tz=timezone.utc),
            signal=signal,
        )
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        log.warning("open_interest parse failed for %s: %s", coin, exc)
        return None
