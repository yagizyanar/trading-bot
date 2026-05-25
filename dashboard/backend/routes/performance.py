"""Portfolio + performance endpoints.

`/latest` prefers Freqtrade's live balance over the most recent DB snapshot.
Falls back to the DB snapshot if Freqtrade is unreachable, and to a zero
dict if there's no snapshot at all.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import PerformanceSnapshot
from ..deps import db
from ..freqtrade_client import fetch_balance, fetch_status

router = APIRouter()


@router.get("/latest")
def latest_snapshot(session: Session = Depends(db)) -> dict:
    last_db = (
        session.query(PerformanceSnapshot)
        .order_by(PerformanceSnapshot.ts.desc())
        .first()
    )

    bal = fetch_balance()
    if bal is not None:
        return _live_to_snapshot(bal, last_db)

    # Freqtrade unreachable — fall back to DB snapshot
    if last_db is None:
        return _empty(source="none")
    payload = _serialize(last_db)
    payload["source"] = "snapshot"
    return payload


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


def _live_to_snapshot(bal: dict, last_db: PerformanceSnapshot | None) -> dict:
    """Synthesize a snapshot dict from Freqtrade's /balance response."""
    eq_raw = bal.get("value") if bal.get("value") is not None else bal.get("total", 0.0)
    try:
        equity = float(eq_raw) if eq_raw is not None else 0.0
    except (TypeError, ValueError):
        equity = 0.0

    # Peak: max of historical peak and current equity (don't regress peak below historical)
    historical_peak = float(last_db.peak_equity) if last_db else equity
    peak = max(historical_peak, equity)
    drawdown_pct = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

    # Freqtrade reports cumulative bot PnL as a *ratio* on the bot's starting capital.
    # We convert back to USD by multiplying by starting_capital_fiat (not by the full
    # wallet equity, which would inflate the number when tradable_balance_ratio < 1).
    raw_ratio = bal.get("starting_capital_fiat_ratio") or bal.get("starting_capital_ratio") or 0.0
    try:
        cum_pnl_pct = float(raw_ratio)
    except (TypeError, ValueError):
        cum_pnl_pct = 0.0
    try:
        bot_start = float(bal.get("starting_capital_fiat", 0.0))
    except (TypeError, ValueError):
        bot_start = 0.0
    cum_pnl_usd = bot_start * cum_pnl_pct

    # Open positions from /status (separate cached call). Falls back to 0.
    status = fetch_status()
    open_count = len(status) if isinstance(status, list) else 0

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total_equity": equity,
        "daily_pnl_usd": cum_pnl_usd,
        "daily_pnl_pct": cum_pnl_pct,
        "weekly_pnl_usd": cum_pnl_usd,    # /balance doesn't split by window
        "weekly_pnl_pct": cum_pnl_pct,
        "drawdown_pct": drawdown_pct,
        "peak_equity": peak,
        "open_positions": open_count,
        "deployed_capital_pct": 0.0,
        "source": "freqtrade",
        "currency_symbol": bal.get("symbol", "USD"),
        "dry_run": bool(bal.get("note") and "Simulated" in str(bal.get("note", ""))),
    }


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


def _empty(source: str = "none") -> dict:
    return {
        "ts": None, "total_equity": 0.0,
        "daily_pnl_usd": 0.0, "daily_pnl_pct": 0.0,
        "weekly_pnl_usd": 0.0, "weekly_pnl_pct": 0.0,
        "drawdown_pct": 0.0, "peak_equity": 0.0,
        "open_positions": 0, "deployed_capital_pct": 0.0,
        "source": source,
    }
