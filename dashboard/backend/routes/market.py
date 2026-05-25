"""Market-data endpoints: Long/Short ratio, funding rate, Hyperliquid top traders.

All reads come from the latest sentiment_scores row per coin so the dashboard
is fast and we don't hammer Binance/Hyperliquid on every page load. Freshness
is bounded by the sentiment-refresh cadence (hourly).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from config.settings import TARGET_COINS
from database import SentimentScore
from ..deps import db

router = APIRouter()


def _latest_per_coin(session: Session) -> dict[str, SentimentScore]:
    subq = (
        session.query(SentimentScore.coin, func.max(SentimentScore.ts).label("max_ts"))
        .group_by(SentimentScore.coin)
        .subquery()
    )
    rows = (
        session.query(SentimentScore)
        .join(subq, (SentimentScore.coin == subq.c.coin) & (SentimentScore.ts == subq.c.max_ts))
        .all()
    )
    return {r.coin: r for r in rows}


def _ls_label(ratio):
    if ratio is None:
        return "unknown"
    if ratio >= 1.5:
        return "BEARISH"     # overleveraged longs
    if ratio <= 0.7:
        return "BULLISH"     # capitulated longs
    return "NEUTRAL"


def _fr_label(rate):
    if rate is None:
        return "unknown"
    if rate >= 0.0001:
        return "BEARISH"
    if rate <= -0.0001:
        return "BULLISH"
    return "NEUTRAL"


@router.get("/long-short")
def long_short(session: Session = Depends(db)) -> list[dict]:
    by_coin = _latest_per_coin(session)
    out: list[dict] = []
    for coin in TARGET_COINS:
        r = by_coin.get(coin)
        out.append({
            "coin": coin,
            "ratio": r.long_short_ratio if r else None,
            "label": _ls_label(r.long_short_ratio if r else None),
            "ts": r.ts.isoformat() if r else None,
        })
    return out


@router.get("/funding")
def funding(session: Session = Depends(db)) -> list[dict]:
    by_coin = _latest_per_coin(session)
    out: list[dict] = []
    for coin in TARGET_COINS:
        r = by_coin.get(coin)
        rate = r.funding_rate if r else None
        out.append({
            "coin": coin,
            "rate": rate,
            "rate_pct": (rate * 100.0) if rate is not None else None,
            "label": _fr_label(rate),
            "ts": r.ts.isoformat() if r else None,
        })
    return out


@router.get("/hyperliquid")
def hyperliquid(session: Session = Depends(db)) -> list[dict]:
    by_coin = _latest_per_coin(session)
    out: list[dict] = []
    for coin in TARGET_COINS:
        r = by_coin.get(coin)
        s = r.hyperliquid_score if r else None
        if s is None:
            label = "unknown"
        elif s >= 0.2:
            label = "BULLISH"
        elif s <= -0.2:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        out.append({
            "coin": coin,
            "score": s,
            "label": label,
            "ts": r.ts.isoformat() if r else None,
        })
    return out


@router.get("/summary")
def summary(session: Session = Depends(db)) -> dict:
    """One-shot snapshot — handy for the WebSocket pusher or quick polling."""
    return {
        "long_short": long_short(session),
        "funding": funding(session),
        "hyperliquid": hyperliquid(session),
    }
