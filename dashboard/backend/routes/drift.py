"""Live-vs-backtest drift metrics for the dashboard (roadmap item 8).

`/latest` returns the most recent DriftSnapshot (written daily by
routines.drift_monitor); `/history` returns the recent series for charting.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from analytics.drift import (
    BACKTEST_SHARPE_BASELINE, NEG_PROFIT_STREAK_ALERT, SHARPE_ALERT, WINRATE_ALERT,
)
from database import DriftSnapshot

from ..deps import db

router = APIRouter()


def _serialize(r: DriftSnapshot) -> dict:
    return {
        "ts": r.ts.isoformat(),
        "window_days": r.window_days,
        "trades": r.trades_30d,
        "rolling_sharpe": r.rolling_sharpe_30d,
        "win_rate": r.win_rate_30d,
        "avg_profit_per_trade": r.avg_profit_per_trade_30d,
        "actual_cost_bps": r.actual_cost_bps,
        "expected_cost_bps": r.expected_cost_bps,
        "consecutive_negative_days": r.consecutive_negative_days,
        "alerts": [a.strip() for a in r.alerts.split(";")] if r.alerts else [],
    }


@router.get("/latest")
def latest(session: Session = Depends(db)) -> dict:
    row = (
        session.query(DriftSnapshot)
        .order_by(DriftSnapshot.ts.desc())
        .first()
    )
    thresholds = {
        "sharpe_alert": SHARPE_ALERT,
        "winrate_alert": WINRATE_ALERT,
        "neg_streak_alert": NEG_PROFIT_STREAK_ALERT,
        "backtest_sharpe_baseline": BACKTEST_SHARPE_BASELINE,
    }
    if row is None:
        return {"available": False, "thresholds": thresholds}
    payload = _serialize(row)
    payload["available"] = True
    payload["thresholds"] = thresholds
    return payload


@router.get("/history")
def history(days: int = 30, session: Session = Depends(db)) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (
        session.query(DriftSnapshot)
        .filter(DriftSnapshot.ts >= since)
        .order_by(DriftSnapshot.ts.asc())
        .all()
    )
    return [_serialize(r) for r in rows]
