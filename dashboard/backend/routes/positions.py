"""Open & closed positions endpoints.

Source preference (same pattern as /api/performance/latest):
  1. **Freqtrade /api/v1/status** for open trades (the executor's ground truth)
  2. **Freqtrade /api/v1/trades** for closed trades
  3. Fall back to our `trades` table if Freqtrade is unreachable

Each response item includes a `source` field so the frontend can show a
"live" vs "snapshot" indicator if it wants.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import Trade
from ..deps import db
from ..freqtrade_client import fetch_closed_trades, fetch_status, map_freqtrade_trade

router = APIRouter()


@router.get("/open")
def open_positions(session: Session = Depends(db)) -> list[dict]:
    """Open trades — prefer Freqtrade live; fall back to the DB."""
    live = fetch_status()
    if live is not None:
        out = [map_freqtrade_trade(t) for t in live]
        # Newest first
        out.sort(key=lambda r: r.get("entry_ts") or "", reverse=True)
        for r in out:
            r["source"] = "freqtrade"
        return out

    rows = (
        session.query(Trade)
        .filter(Trade.outcome == "OPEN")
        .order_by(Trade.entry_ts.desc())
        .all()
    )
    return [{**_serialize(r), "source": "snapshot"} for r in rows]


@router.get("/closed")
def closed_positions(
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(db),
) -> list[dict]:
    """Closed trades — prefer Freqtrade live; fall back to the DB."""
    live = fetch_closed_trades(limit=limit)
    if live is not None:
        out = [map_freqtrade_trade(t) for t in live]
        # Closed trades come newest first from /api/v1/trades; preserve that
        # ordering, but enforce it explicitly in case of unexpected payloads.
        out.sort(key=lambda r: r.get("exit_ts") or r.get("entry_ts") or "", reverse=True)
        for r in out:
            r["source"] = "freqtrade"
        return out[:limit]

    rows = (
        session.query(Trade)
        .filter(Trade.outcome.in_(("WIN", "LOSS")))
        .order_by(Trade.exit_ts.desc())
        .limit(limit)
        .all()
    )
    return [{**_serialize(r), "source": "snapshot"} for r in rows]


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
