"""Cryptocurrency news fetcher + FinBERT headline scoring.

Source: cryptocurrency.cv/api/news (free, no key).
Process:
  1. Fetch latest N headlines (with timestamps).
  2. For each headline, run FinBERT (ProsusAI/finbert) → +/-/neutral probs.
  3. Aggregate per coin: mention_count, weighted_score in [-1.0, +1.0].

The model loads lazily so importing this module is cheap.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Iterable, Optional

import requests

from config.settings import (
    API_RETRY_ATTEMPTS,
    API_TIMEOUT_SECONDS,
    CRYPTO_NEWS_URL,
    TARGET_COINS,
)

log = logging.getLogger(__name__)

_FINBERT_PIPE = None
_FINBERT_LOCK = Lock()
_FINBERT_MODEL_NAME = "ProsusAI/finbert"


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: Optional[str]
    ts: datetime
    coins_mentioned: tuple[str, ...]


@dataclass(frozen=True)
class HeadlineScore:
    coin: str
    score: float           # -1..+1 (positive - negative prob)
    mention_count: int


_COIN_ALIASES: dict[str, tuple[str, ...]] = {
    "SOL": ("SOL", "Solana"),
    "AVAX": ("AVAX", "Avalanche"),
    "LINK": ("LINK", "Chainlink"),
    "DOT": ("DOT", "Polkadot"),
    "MATIC": ("MATIC", "Polygon"),
    "INJ": ("INJ", "Injective"),
    "ARB": ("ARB", "Arbitrum"),
    "OP": ("OP", "Optimism"),
    "APT": ("APT", "Aptos"),
    "SUI": ("SUI", "Sui"),
    "NEAR": ("NEAR", "Near Protocol"),
    "FTM": ("FTM", "Fantom"),
    "ATOM": ("ATOM", "Cosmos"),
    "SAND": ("SAND", "Sandbox"),
    "MANA": ("MANA", "Decentraland"),
    "AXS": ("AXS", "Axie Infinity"),
    "DYDX": ("DYDX", "dYdX"),
    "GMX": ("GMX",),
}


def _detect_coins(title: str) -> tuple[str, ...]:
    matches: list[str] = []
    upper = title.upper()
    for coin in TARGET_COINS:
        aliases = _COIN_ALIASES.get(coin, (coin,))
        for alias in aliases:
            pattern = r"\b" + re.escape(alias.upper()) + r"\b"
            if re.search(pattern, upper):
                matches.append(coin)
                break
    return tuple(dict.fromkeys(matches))  # de-dup, preserve order


def _parse_ts(value) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def fetch_crypto_news(limit: int = 100) -> list[NewsItem]:
    """Fetch latest crypto news headlines. Returns [] on failure."""
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(
                CRYPTO_NEWS_URL,
                params={"limit": limit},
                timeout=API_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json()
            items_raw = (
                payload if isinstance(payload, list)
                else payload.get("articles") or payload.get("data") or payload.get("news") or []
            )
            result: list[NewsItem] = []
            for row in items_raw:
                title = str(row.get("title") or row.get("headline") or "").strip()
                if not title:
                    continue
                ts = _parse_ts(row.get("published_at") or row.get("timestamp") or row.get("date"))
                url = row.get("url") or row.get("link")
                coins = _detect_coins(title)
                result.append(NewsItem(title=title, url=url, ts=ts, coins_mentioned=coins))
            return result
        except Exception as exc:  # noqa: BLE001
            log.warning("crypto_news fetch attempt %s failed: %s", attempt, exc)
    return []


def _load_finbert():
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE
    with _FINBERT_LOCK:
        if _FINBERT_PIPE is not None:
            return _FINBERT_PIPE
        try:
            from transformers import pipeline  # lazy import (heavy)
            _FINBERT_PIPE = pipeline(
                "sentiment-analysis",
                model=_FINBERT_MODEL_NAME,
                tokenizer=_FINBERT_MODEL_NAME,
                truncation=True,
            )
            return _FINBERT_PIPE
        except Exception as exc:  # noqa: BLE001
            log.warning("FinBERT load failed (%s); falling back to keyword scoring", exc)
            _FINBERT_PIPE = False  # sentinel: tried and failed
            return False


_BULL_WORDS = {"surge", "rally", "soar", "bull", "gain", "rise", "breakout", "all-time", "ath", "pump"}
_BEAR_WORDS = {"crash", "plunge", "dump", "bear", "fall", "drop", "loss", "selloff", "decline", "fud"}


def _keyword_score(title: str) -> float:
    lower = title.lower()
    bulls = sum(1 for w in _BULL_WORDS if w in lower)
    bears = sum(1 for w in _BEAR_WORDS if w in lower)
    if bulls == bears == 0:
        return 0.0
    return (bulls - bears) / max(1, bulls + bears)


def _finbert_score(titles: list[str]) -> list[float]:
    """Return list of scores aligned with titles. +1 positive, -1 negative, 0 neutral."""
    pipe = _load_finbert()
    if pipe is False or pipe is None:
        return [_keyword_score(t) for t in titles]
    try:
        outputs = pipe(titles)
        scores: list[float] = []
        for out in outputs:
            label = str(out.get("label", "neutral")).lower()
            sc = float(out.get("score", 0.0))
            if label == "positive":
                scores.append(sc)
            elif label == "negative":
                scores.append(-sc)
            else:
                scores.append(0.0)
        return scores
    except Exception as exc:  # noqa: BLE001
        log.warning("FinBERT scoring failed (%s); falling back to keyword scoring", exc)
        return [_keyword_score(t) for t in titles]


def score_headlines(items: Iterable[NewsItem]) -> dict[str, HeadlineScore]:
    """Aggregate per-coin news sentiment. Returns coin -> HeadlineScore."""
    items = list(items)
    if not items:
        return {}
    raw_scores = _finbert_score([it.title for it in items])

    per_coin: dict[str, list[float]] = {c: [] for c in TARGET_COINS}
    for item, sc in zip(items, raw_scores):
        for coin in item.coins_mentioned:
            per_coin[coin].append(sc)

    result: dict[str, HeadlineScore] = {}
    for coin, scores in per_coin.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        result[coin] = HeadlineScore(
            coin=coin,
            score=max(-1.0, min(1.0, avg)),
            mention_count=len(scores),
        )
    return result
