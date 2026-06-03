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


# Freqtrade's /api/v1/trades returns OLDEST-first. Fetching only `limit` rows
# therefore EXCLUDES the newest trades once total closed > limit (which is why
# the dashboard table silently stopped showing recent trades after ~50-100
# closes). Fetch the full history, sort newest-first, then slice for display.
_CLOSED_FETCH_ALL = 1000  # covers ~2 months at current rate; bump or paginate beyond


@router.get("/closed")
def closed_positions(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(db),
) -> list[dict]:
    """Closed trades — newest-first, paginated.

    Fetches the full closed-trade history (not just `limit`) because Freqtrade
    returns them oldest-first; sorts newest-first; then returns the page
    [offset : offset+limit]. The frontend pages through with increasing offset
    ("Load more"); a short page (< limit) signals the end.
    """
    live = fetch_closed_trades(limit=_CLOSED_FETCH_ALL)
    if live is not None:
        out = [map_freqtrade_trade(t) for t in live]
        out.sort(key=lambda r: r.get("exit_ts") or r.get("entry_ts") or "", reverse=True)
        for r in out:
            r["source"] = "freqtrade"
        return out[offset:offset + limit]

    rows = (
        session.query(Trade)
        .filter(Trade.outcome.in_(("WIN", "LOSS")))
        .order_by(Trade.exit_ts.desc())
        .offset(offset)
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
