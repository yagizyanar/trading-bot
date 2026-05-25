"""Fear & Greed widget endpoint (cached most-recent reading from DB)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import SentimentScore
from ..deps import db

router = APIRouter()


@router.get("/")
def latest(session: Session = Depends(db)) -> dict:
    row = (
        session.query(SentimentScore)
        .filter(SentimentScore.fear_greed.isnot(None))
        .order_by(SentimentScore.ts.desc())
        .first()
    )
    if row is None or row.fear_greed is None:
        return {"value": None, "label": "unknown", "ts": None}

    v = row.fear_greed
    if v <= 25:
        label = "Extreme Fear"
    elif v <= 45:
        label = "Fear"
    elif v <= 55:
        label = "Neutral"
    elif v <= 75:
        label = "Greed"
    else:
        label = "Extreme Greed"
    return {"value": v, "label": label, "ts": row.ts.isoformat()}
