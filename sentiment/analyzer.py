"""Unified sentiment combiner.

Produces one unified score per coin in [-1.0, +1.0] by blending six per-coin
signals. SentiCrypt was retired in v3 (domain is dead); its 20% weight was
redistributed proportionally to the remaining sources, with extra emphasis
on news (better per-coin signal) and the futures market data (more direct
positioning information).

  news (FinBERT on headlines)            : 0.30  per-coin
  volume_anomaly (Binance)               : 0.20  per-coin
  long_short_ratio (Binance Futures)     : 0.20  per-coin
  funding_rate (Binance Futures)         : 0.15  per-coin
  yfinance momentum                      : 0.10  per-coin
  hyperliquid (top traders)              : 0.05  per-coin
  -------------------------------------- : 1.00
  fear_greed                             : MULTIPLIER on the weighted sum

The "hyperliquid" slot is named for legacy/DB-column reasons but is now
populated by Binance Futures `topLongShortPositionRatio` (smart-money
positioning). The original Hyperliquid CDN leaderboard broke 2026-05-27
when its ethAddresses stopped resolving to active clearinghouseState
accounts — see project_hyperliquid_broken in memory for context.

If a per-coin source is missing, its weight is redistributed proportionally
among the remaining sources for that coin.

Signal labels (PHASE 4 / signal-generation skill):
  > +0.2  -> BULLISH
  < -0.2  -> BEARISH
  else    -> NEUTRAL
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from config.settings import (
    SENTIMENT_BEAR_THRESHOLD,
    SENTIMENT_BULL_THRESHOLD,
    TARGET_COINS,
)

from .binance_data import fetch_binance_ohlcv, volume_anomaly
from .binance_market import (
    fetch_funding_rate,
    fetch_long_short_ratio,
    fetch_top_trader_position_ratio,
)
from .crypto_news import fetch_crypto_news, score_headlines
from .fear_greed import fetch_fear_greed
from .yfinance_data import fetch_yf_price_change

log = logging.getLogger(__name__)


_WEIGHTS = {
    "news": 0.30,
    "volume": 0.20,
    "long_short_ratio": 0.20,
    "funding_rate": 0.15,
    "yfinance": 0.10,
    "hyperliquid": 0.05,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "_WEIGHTS must sum to 1.0"


@dataclass(frozen=True)
class UnifiedScore:
    coin: str
    timestamp: datetime
    news_score: Optional[float]
    volume_anomaly: Optional[float]
    yfinance_change: Optional[float]
    long_short_ratio: Optional[float]
    funding_rate: Optional[float]
    hyperliquid_score: Optional[float]
    fear_greed: Optional[int]
    fear_greed_multiplier: float
    unified: float
    signal: str  # BULLISH / BEARISH / NEUTRAL


def _label(score: float) -> str:
    if score > SENTIMENT_BULL_THRESHOLD:
        return "BULLISH"
    if score < SENTIMENT_BEAR_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def _yfinance_to_signal(pct_change: float) -> float:
    """Map 7-day price change to [-1, +1] via tanh-like scaling."""
    import math
    return max(-1.0, min(1.0, math.tanh(pct_change * 5.0)))


def _blend(components: dict[str, Optional[float]]) -> float:
    """Weighted blend with proportional reweighting for missing sources."""
    active = {k: v for k, v in components.items() if v is not None}
    if not active:
        return 0.0
    total_weight = sum(_WEIGHTS[k] for k in active)
    if total_weight <= 0:
        return 0.0
    weighted = sum(_WEIGHTS[k] * active[k] for k in active)
    return max(-1.0, min(1.0, weighted / total_weight))


def compute_unified_scores(
    coins: Iterable[str] = TARGET_COINS,
    news_limit: int = 100,
) -> dict[str, UnifiedScore]:
    """Fetch all sources and produce one UnifiedScore per requested coin.

    Robust to any single source failing — uses redistribution + neutral fallback
    per the sentiment-pipeline skill error-handling rule.
    """
    coins = tuple(coins)
    now = datetime.now(timezone.utc)

    fg = fetch_fear_greed()
    fg_mult = fg.multiplier if fg else 1.0
    fg_value = fg.value if fg else None

    news_items = fetch_crypto_news(limit=news_limit)
    news_scores = score_headlines(news_items)

    result: dict[str, UnifiedScore] = {}
    for coin in coins:
        pair = f"{coin}USDT"
        df = fetch_binance_ohlcv(pair, interval="1h", limit=24 * 8 + 2)
        vol_signal = volume_anomaly(df) if df is not None else None

        yf_pct = fetch_yf_price_change(coin, period="7d")
        yf_signal = _yfinance_to_signal(yf_pct) if yf_pct is not None else None

        news_hs = news_scores.get(coin)
        news_signal = news_hs.score if news_hs else None

        ls = fetch_long_short_ratio(coin)
        ls_signal = ls.signal if ls else None
        ls_ratio = ls.ratio if ls else None

        fr = fetch_funding_rate(coin)
        fr_signal = fr.signal if fr else None
        fr_rate = fr.rate if fr else None

        # "hyperliquid" slot is now Binance top-trader position ratio
        # (per-coin, covers all 18, follows smart money not contrarian).
        # Hyperliquid CDN leaderboard broke 2026-05-27, see module docstring.
        tt = fetch_top_trader_position_ratio(coin)
        hl_signal = tt.signal if tt else None

        components = {
            "news": news_signal,
            "volume": vol_signal,
            "yfinance": yf_signal,
            "long_short_ratio": ls_signal,
            "funding_rate": fr_signal,
            "hyperliquid": hl_signal,
        }
        raw_blend = _blend(components)
        unified = max(-1.0, min(1.0, raw_blend * fg_mult))

        result[coin] = UnifiedScore(
            coin=coin,
            timestamp=now,
            news_score=news_signal,
            volume_anomaly=vol_signal,
            yfinance_change=yf_pct,
            long_short_ratio=ls_ratio,
            funding_rate=fr_rate,
            hyperliquid_score=hl_signal,
            fear_greed=fg_value,
            fear_greed_multiplier=fg_mult,
            unified=unified,
            signal=_label(unified),
        )
    return result


def persist_unified_scores(scores: dict[str, UnifiedScore]) -> None:
    """Insert/update sentiment_scores rows for each coin."""
    from database import SentimentScore, SessionLocal  # local import to avoid cycles

    with SessionLocal() as session:
        for coin, us in scores.items():
            row = SentimentScore(
                coin=coin,
                ts=us.timestamp,
                fear_greed=us.fear_greed,
                news_score=us.news_score,
                volume_anomaly=us.volume_anomaly,
                yfinance_change=us.yfinance_change,
                long_short_ratio=us.long_short_ratio,
                funding_rate=us.funding_rate,
                hyperliquid_score=us.hyperliquid_score,
                unified=us.unified,
                signal=us.signal,
            )
            session.merge(row)
        session.commit()
