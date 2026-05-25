"""Portfolio + performance endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import PerformanceSnapshot
from ..deps import db

router = APIRouter()


@router.get("/latest")
def latest_snapshot(session: Session = Depends(db)) -> dict:
    row = (
        session.query(PerformanceSnapshot)
        .order_by(PerformanceSnapshot.ts.desc())
        .first()
    )
    if row is None:
        return _empty()
    return _serialize(row)


@router.get("/history")
def history(days: int = 30, session: Session = Depends(db)) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        session.query(PerformanceSnapshot)
        .filter(PerformanceSnapshot.ts >= since)
        .order_by(PerformanceSnapshot.ts.asc())
        .all()
    )
    return [_serialize(r) for r in rows]


def _serialize(r: PerformanceSnapshot) -> dict:
    return {
        "ts": r.ts.isoformat(),
        "total_equity": r.total_equity,
        "daily_pnl_usd": r.daily_pnl_usd,
        "daily_pnl_pct": r.daily_pnl_pct,
        "weekly_pnl_usd": r.weekly_pnl_usd,
        "weekly_pnl_pct": r.weekly_pnl_pct,
        "drawdown_pct": r.drawdown_pct,
        "peak_equity": r.peak_equity,
        "open_positions": r.open_positions,
        "deployed_capital_pct": r.deployed_capital_pct,
    }


def _empty() -> dict:
    return {
        "ts": None, "total_equity": 0.0,
        "daily_pnl_usd": 0.0, "daily_pnl_pct": 0.0,
        "weekly_pnl_usd": 0.0, "weekly_pnl_pct": 0.0,
        "drawdown_pct": 0.0, "peak_equity": 0.0,
        "open_positions": 0, "deployed_capital_pct": 0.0,
    }
