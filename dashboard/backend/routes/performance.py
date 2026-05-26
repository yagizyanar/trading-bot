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
from ..freqtrade_client import fetch_balance, fetch_pnl_breakdown, fetch_status

router = APIRouter()


def _baseline_equity_before(session, when: datetime, default: float) -> float:
    """Latest performance_snapshots.total_equity strictly before `when`. Fallback if none."""
    row = (
        session.query(PerformanceSnapshot)
        .filter(PerformanceSnapshot.ts < when)
        .order_by(PerformanceSnapshot.ts.desc())
        .first()
    )
    return float(row.total_equity) if row is not None else float(default)


@router.get("/latest")
def latest_snapshot(session: Session = Depends(db)) -> dict:
    last_db = (
        session.query(PerformanceSnapshot)
        .order_by(PerformanceSnapshot.ts.desc())
        .first()
    )

    bal = fetch_balance()
    pnl = fetch_pnl_breakdown()
    if bal is not None or pnl is not None:
        return _live_to_snapshot(session, bal or {}, pnl, last_db)

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


def _live_to_snapshot(session, bal: dict, pnl: dict | None,
                      last_db: PerformanceSnapshot | None) -> dict:
    """Compute a live snapshot using the reconcilable formula:

        equity = starting_wallet + open_pnl + closed_pnl

    This matches exactly what the positions table shows when you sum its
    P&L column (open_pnl) plus the closed-trades view (closed_pnl). No
    phantom drift vs Freqtrade's internal `balance.total` accounting.

    daily_pnl/weekly_pnl come from diffing current equity against the
    latest performance_snapshots row before today's / this week's
    00:00 UTC boundary. Falls back to starting_wallet if no snapshot
    exists yet (the bot just started).
    """
    from config.settings import PROJECT_ROOT
    import json

    # Starting wallet from config.json (dry_run_wallet)
    try:
        with open(PROJECT_ROOT / "config" / "config.json") as f:
            cfg = json.load(f)
        starting_wallet = float(cfg.get("dry_run_wallet", 10000))
    except Exception:
        starting_wallet = 10000.0

    open_pnl = float((pnl or {}).get("open_pnl", 0.0) or 0.0)
    closed_pnl = float((pnl or {}).get("closed_pnl", 0.0) or 0.0)
    equity = starting_wallet + open_pnl + closed_pnl

    # Daily / weekly baselines from prior snapshots
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday_start = today_start - timedelta(days=now.weekday())
    daily_base = _baseline_equity_before(session, today_start, default=starting_wallet)
    weekly_base = _baseline_equity_before(session, monday_start, default=starting_wallet)
    daily_pnl_usd = equity - daily_base
    weekly_pnl_usd = equity - weekly_base
    daily_pnl_pct = (daily_pnl_usd / daily_base) if daily_base > 0 else 0.0
    weekly_pnl_pct = (weekly_pnl_usd / weekly_base) if weekly_base > 0 else 0.0

    historical_peak = float(last_db.peak_equity) if last_db else equity
    peak = max(historical_peak, equity)
    drawdown_pct = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

    status = fetch_status()
    open_count = len(status) if isinstance(status, list) else 0

    return {
        "ts": now.isoformat(),
        "total_equity": equity,
        "open_pnl_usd": open_pnl,
        "closed_pnl_usd": closed_pnl,
        "daily_pnl_usd": daily_pnl_usd,
        "daily_pnl_pct": daily_pnl_pct,
        "weekly_pnl_usd": weekly_pnl_usd,
        "weekly_pnl_pct": weekly_pnl_pct,
        "drawdown_pct": drawdown_pct,
        "peak_equity": peak,
        "open_positions": open_count,
        "deployed_capital_pct": 0.0,
        "source": "freqtrade",
        "currency_symbol": bal.get("symbol") or bal.get("stake") or "USDT",
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
