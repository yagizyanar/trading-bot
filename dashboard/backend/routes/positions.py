"""Open & closed positions endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import Trade
from ..deps import db

router = APIRouter()


@router.get("/open")
def open_positions(session: Session = Depends(db)) -> list[dict]:
    rows = session.query(Trade).filter(Trade.outcome == "OPEN").order_by(Trade.entry_ts.desc()).all()
    return [_serialize(r) for r in rows]


@router.get("/closed")
def closed_positions(
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(db),
) -> list[dict]:
    rows = (
        session.query(Trade)
        .filter(Trade.outcome.in_(("WIN", "LOSS")))
        .order_by(Trade.exit_ts.desc())
        .limit(limit)
        .all()
    )
    return [_serialize(r) for r in rows]


def _serialize(t: Trade) -> dict:
    return {
        "id": t.id,
        "coin": t.coin,
        "side": t.side,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "leverage": t.leverage,
        "pnl_usd": t.pnl_usd,
        "pnl_pct": t.pnl_pct,
        "entry_ts": t.entry_ts.isoformat() if t.entry_ts else None,
        "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
        "reason_in": t.reason_in,
        "reason_out": t.reason_out,
        "outcome": t.outcome,
        "is_paper": t.is_paper,
    }
