"""Every-5-min equity snapshot writer.

Persists a `performance_snapshots` row every 5 minutes so the dashboard's
portfolio chart has continuity across page refreshes. Unlike `day_close`
(once per weekday at 16:00 UTC), this runs 24/7 — chart continuity needs
frequent samples regardless of weekday or circuit-breaker state.

Source of truth is Freqtrade's REST API. Equity uses the same reconcilable
formula as the rest of the bot:

    equity = DRY_RUN_WALLET + open_pnl + closed_pnl

Idempotency: each invocation writes ONE row tagged with the current UTC
timestamp. Duplicate rows from clock skew are harmless — they're additive
data points, not authoritative state.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from database import PerformanceSnapshot, SessionLocal
from routines.base import _default_portfolio_snapshot, setup_routine_logging

log = logging.getLogger(__name__)


def write_snapshot() -> dict:
    """Write a single performance_snapshots row from current Freqtrade state.

    Returns the row payload (also useful for tests). Raises on database errors.
    """
    from dashboard.backend.freqtrade_client import fetch_status

    portfolio = _default_portfolio_snapshot()
    live_open = fetch_status() or []
    open_positions = len(live_open)

    equity = float(portfolio["equity"])
    peak = float(portfolio["peak_equity"])
    daily_pnl_pct = float(portfolio["daily_pnl_pct"])
    weekly_pnl_pct = float(portfolio["weekly_pnl_pct"])

    deployed_pct = 0.0
    try:
        total_stake = sum(float(t.get("stake_amount") or 0) for t in live_open)
        deployed_pct = total_stake / max(equity, 1.0)
    except (TypeError, ValueError):
        deployed_pct = 0.0

    drawdown_pct = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

    row = {
        "ts": datetime.now(timezone.utc),
        "total_equity": equity,
        "daily_pnl_usd": equity * daily_pnl_pct,
        "daily_pnl_pct": daily_pnl_pct,
        "weekly_pnl_usd": equity * weekly_pnl_pct,
        "weekly_pnl_pct": weekly_pnl_pct,
        "drawdown_pct": drawdown_pct,
        "peak_equity": peak,
        "open_positions": open_positions,
        "deployed_capital_pct": deployed_pct,
    }

    with SessionLocal() as session:
        session.add(PerformanceSnapshot(**row))
        session.commit()

    return row


def main() -> int:
    setup_routine_logging()
    try:
        row = write_snapshot()
        log.info(
            "snapshot_writer: equity=$%.2f peak=$%.2f open=%d deployed=%.1f%% dd=%.2f%%",
            row["total_equity"], row["peak_equity"], row["open_positions"],
            row["deployed_capital_pct"] * 100, row["drawdown_pct"] * 100,
        )
        return 0
    except Exception:
        log.exception("snapshot_writer failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
