"""Hyperliquid leaderboard sentiment fetcher (free, no API key).

Two-step pull:
  1. POST /info {"type": "leaderboard"} → top trader addresses + PnL.
  2. For top 20 by PnL, POST /info {"type": "clearinghouseState", "user": addr}
     → that trader's open positions per coin.

Aggregate per coin: long_pct = (#traders long) / (#traders with any position).
Signal: long_pct > 0.60 → bullish, < 0.40 → bearish, else neutral.
Mapped to [-1.0, +1.0] linearly.

Hyperliquid coin tickers are bare ("BTC", "ETH", "SOL"…) — match against our
TARGET_COINS via exact symbol comparison.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import requests

from config.settings import API_RETRY_ATTEMPTS, API_TIMEOUT_SECONDS, TARGET_COINS

log = logging.getLogger(__name__)

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"
TOP_N_TRADERS = 20
PER_TRADER_TIMEOUT = 8.0  # tighter per-trader to bound total wall time


@dataclass(frozen=True)
class HyperliquidReading:
    coin: str
    longs: int
    shorts: int
    long_pct: float       # (longs / (longs+shorts)) — NaN if no positions
    signal: float         # in [-1, +1]
    timestamp: datetime
    sample_size: int      # how many traders contributed any position


def _post(payload: dict, timeout: float = API_TIMEOUT_SECONDS) -> Optional[dict]:
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(HYPERLIQUID_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("hyperliquid %s attempt %s failed: %s",
                        payload.get("type", "?"), attempt, exc)
    return None


def _fetch_leaderboard(top_n: int = TOP_N_TRADERS) -> list[str]:
    """Return top-N trader addresses sorted by recent PnL.

    Hyperliquid's payload typically contains a 'leaderboardRows' key. We pick
    the highest accountValue (or a windowPerformance score if present).
    """
    # Leaderboard is undocumented (not on hyperliquid gitbook); reverse-
    # engineered shape requires `timeWindow`. Valid: "day" | "week" | "month" | "allTime".
    payload = _post({"type": "leaderboard", "timeWindow": "day"})
    if not payload:
        return []

    rows = payload.get("leaderboardRows") or payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return []

    def _score(row: dict) -> float:
        wp = row.get("windowPerformances") or []
        if wp:
            # Each window has [window_name, {"pnl": "...", "roi": "..."}]; pick the
            # first window's PnL as a stable rank.
            try:
                return float(wp[0][1]["pnl"])
            except (IndexError, KeyError, ValueError, TypeError):
                pass
        try:
            return float(row.get("accountValue", "0"))
        except (ValueError, TypeError):
            return 0.0

    rows = [r for r in rows if isinstance(r, dict) and r.get("ethAddress")]
    rows.sort(key=_score, reverse=True)
    return [r["ethAddress"] for r in rows[:top_n]]


def _fetch_positions(address: str) -> dict[str, float]:
    """Return {coin: size} for one trader. Positive size = long, negative = short."""
    payload = _post({"type": "clearinghouseState", "user": address}, timeout=PER_TRADER_TIMEOUT)
    if not payload:
        return {}
    positions: dict[str, float] = {}
    for ap in payload.get("assetPositions", []) or []:
        pos = ap.get("position") if isinstance(ap, dict) else None
        if not isinstance(pos, dict):
            continue
        coin = pos.get("coin")
        szi = pos.get("szi")
        if not coin or szi is None:
            continue
        try:
            size = float(szi)
        except (ValueError, TypeError):
            continue
        if size == 0.0:
            continue
        positions[coin] = size
    return positions


def _long_pct_to_signal(long_pct: float) -> float:
    """Map long_pct in [0, 1] to signal in [-1, 1]. 0.60 → +0.5, 0.40 → -0.5."""
    if long_pct >= 0.60:
        # 0.60 -> 0.5; 1.00 -> 1.0
        return min(1.0, 0.5 + (long_pct - 0.60) / 0.80)
    if long_pct <= 0.40:
        return max(-1.0, -0.5 - (0.40 - long_pct) / 0.80)
    # Between 0.40 and 0.60: linear from -0.5 to +0.5
    return (long_pct - 0.50) * 5.0  # 0.60 -> 0.5; 0.40 -> -0.5; 0.50 -> 0.0


def fetch_top_trader_sentiment(
    coins: Iterable[str] = TARGET_COINS,
    top_n: int = TOP_N_TRADERS,
) -> dict[str, HyperliquidReading]:
    """Return {coin: HyperliquidReading} for each coin with at least one position.

    Coins with zero positions across the top traders are omitted (caller treats
    as None and redistributes weight).
    """
    addresses = _fetch_leaderboard(top_n=top_n)
    if not addresses:
        log.warning("hyperliquid leaderboard empty — skipping")
        return {}

    now = datetime.now(timezone.utc)
    long_count: dict[str, int] = defaultdict(int)
    short_count: dict[str, int] = defaultdict(int)
    trader_count: dict[str, set] = defaultdict(set)

    target_set = set(coins)
    for addr in addresses:
        positions = _fetch_positions(addr)
        if not positions:
            continue
        for coin, size in positions.items():
            if coin not in target_set:
                continue
            trader_count[coin].add(addr)
            if size > 0:
                long_count[coin] += 1
            elif size < 0:
                short_count[coin] += 1

    result: dict[str, HyperliquidReading] = {}
    for coin in coins:
        total = long_count[coin] + short_count[coin]
        if total == 0:
            continue
        long_pct = long_count[coin] / total
        result[coin] = HyperliquidReading(
            coin=coin,
            longs=long_count[coin],
            shorts=short_count[coin],
            long_pct=long_pct,
            signal=_long_pct_to_signal(long_pct),
            timestamp=now,
            sample_size=len(trader_count[coin]),
        )
    return result
