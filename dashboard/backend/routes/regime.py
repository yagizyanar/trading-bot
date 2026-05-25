"""Markov regime endpoints."""
from __future__ import annotations

from sqlalchemy import func
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from config.settings import TARGET_COINS
from database import RegimeState
from ..deps import db

router = APIRouter()


@router.get("/latest")
def latest_per_coin(session: Session = Depends(db)) -> list[dict]:
    subq = (
        session.query(RegimeState.coin, func.max(RegimeState.ts).label("max_ts"))
        .group_by(RegimeState.coin)
        .subquery()
    )
    rows = (
        session.query(RegimeState)
        .join(subq, (RegimeState.coin == subq.c.coin) & (RegimeState.ts == subq.c.max_ts))
        .all()
    )
    by_coin = {r.coin: r for r in rows}
    out: list[dict] = []
    for coin in TARGET_COINS:
        r = by_coin.get(coin)
        if r is None:
            out.append({
                "coin": coin, "regime": "unknown", "confidence": 0.0,
                "bull_prob": 0.0, "bear_prob": 0.0, "sideways_prob": 0.0,
                "markov_signal": 0.0, "ts": None,
            })
        else:
            out.append({
                "coin": r.coin,
                "ts": r.ts.isoformat(),
                "regime": r.regime,
                "confidence": r.confidence,
                "bull_prob": r.bull_prob,
                "bear_prob": r.bear_prob,
                "sideways_prob": r.sideways_prob,
                "markov_signal": r.markov_signal,
            })
    return out
