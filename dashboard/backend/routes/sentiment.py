"""Sentiment grid endpoint."""
from __future__ import annotations

from sqlalchemy import func
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config.settings import TARGET_COINS
from database import SentimentScore
from ..deps import db

router = APIRouter()


@router.get("/latest")
def latest_per_coin(session: Session = Depends(db)) -> list[dict]:
    """One row per coin — the most recent sentiment score."""
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
    by_coin = {r.coin: r for r in rows}
    out: list[dict] = []
    for coin in TARGET_COINS:
        r = by_coin.get(coin)
        if r is None:
            out.append({
                "coin": coin, "unified": 0.0, "signal": "NEUTRAL",
                "fear_greed": None, "ts": None,
                "news_score": None,
                "volume_anomaly": None, "yfinance_change": None,
            })
        else:
            out.append({
                "coin": r.coin,
                "ts": r.ts.isoformat(),
                "unified": r.unified,
                "signal": r.signal,
                "fear_greed": r.fear_greed,
                "news_score": r.news_score,
                "volume_anomaly": r.volume_anomaly,
                "yfinance_change": r.yfinance_change,
            })
    return out
