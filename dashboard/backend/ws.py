"""WebSocket endpoint for live dashboard pushes.

Pushes a rich snapshot every 5 seconds:
  - ts, locked (kill-switch state)
  - positions: list of OPEN Freqtrade trades, fully shaped for the UI
  - performance: live equity / peak / drawdown / open_positions count
  - sentiment_rows: count from the sentiment_scores DB table
  - equity, drawdown_pct, open_positions: convenience top-level mirrors

Frontend consumes this for the PortfolioChart and the open PositionsTable.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from database import PerformanceSnapshot, SentimentScore, SessionLocal
from risk.lockfile import is_locked

from .freqtrade_client import (
    fetch_balance,
    fetch_status,
    map_freqtrade_trade,
)

log = logging.getLogger(__name__)

_PUSH_INTERVAL_SECONDS = 5


def _live_performance(bal, last_db):
    """Inline copy of routes.performance._live_to_snapshot to avoid circular imports.

    Kept short and read-only; if you tweak the live-balance fields, mirror
    those changes in routes/performance.py as well.
    """
    from config.settings import STOP_LOSS_PCT  # noqa: F401 (kept for parity)
    eq_raw = bal.get("value") or bal.get("total") or 0.0
    try:
        equity = float(eq_raw)
    except (TypeError, ValueError):
        equity = 0.0

    historical_peak = float(last_db.peak_equity) if last_db else equity
    peak = max(historical_peak, equity)
    drawdown_pct = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

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

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total_equity": equity,
        "daily_pnl_usd": cum_pnl_usd,
        "daily_pnl_pct": cum_pnl_pct,
        "weekly_pnl_usd": cum_pnl_usd,
        "weekly_pnl_pct": cum_pnl_pct,
        "drawdown_pct": drawdown_pct,
        "peak_equity": peak,
        "open_positions": 0,  # filled in by caller from `positions` list
        "deployed_capital_pct": 0.0,
        "source": "freqtrade",
        "currency_symbol": bal.get("symbol") or bal.get("stake") or "USDT",
        "dry_run": bool(bal.get("note") and "Simulated" in str(bal.get("note", ""))),
    }


async def _snapshot() -> dict:
    payload: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "locked": is_locked(),
    }

    # Live open positions
    try:
        status_raw = fetch_status()
        if status_raw is None:
            payload["positions"] = None
        else:
            mapped = [map_freqtrade_trade(t) for t in status_raw]
            for r in mapped:
                r["source"] = "freqtrade"
            mapped.sort(key=lambda r: r.get("entry_ts") or "", reverse=True)
            payload["positions"] = mapped
    except Exception as exc:  # noqa: BLE001
        log.warning("ws positions fetch failed: %s", exc)
        payload["positions"] = None

    # Live performance snapshot
    try:
        bal = fetch_balance()
        with SessionLocal() as session:
            last_db = (
                session.query(PerformanceSnapshot)
                .order_by(PerformanceSnapshot.ts.desc())
                .first()
            )
            sent_count = session.query(SentimentScore).count()
        if bal is not None:
            perf = _live_performance(bal, last_db)
        elif last_db is not None:
            perf = {
                "ts": last_db.ts.isoformat(),
                "total_equity": last_db.total_equity,
                "peak_equity": last_db.peak_equity,
                "drawdown_pct": last_db.drawdown_pct,
                "daily_pnl_usd": last_db.daily_pnl_usd,
                "daily_pnl_pct": last_db.daily_pnl_pct,
                "weekly_pnl_usd": last_db.weekly_pnl_usd,
                "weekly_pnl_pct": last_db.weekly_pnl_pct,
                "open_positions": last_db.open_positions,
                "deployed_capital_pct": last_db.deployed_capital_pct,
                "source": "snapshot",
                "currency_symbol": "USDT",
                "dry_run": True,
            }
        else:
            perf = None
        payload["performance"] = perf
        payload["sentiment_rows"] = sent_count
    except Exception as exc:  # noqa: BLE001
        log.warning("ws performance fetch failed: %s", exc)
        payload["performance"] = None
        payload["sentiment_rows"] = 0

    # Convenience top-level mirrors for legacy widgets / quick reads
    perf = payload.get("performance") or {}
    payload["equity"] = perf.get("total_equity")
    payload["drawdown_pct"] = perf.get("drawdown_pct")
    pos = payload.get("positions") or []
    payload["open_positions"] = len(pos) if isinstance(pos, list) else 0
    if isinstance(perf, dict):
        perf["open_positions"] = payload["open_positions"]

    return payload


def register_ws(app: FastAPI) -> None:
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                snap = await _snapshot()
                await ws.send_text(json.dumps(snap, default=str))
                await asyncio.sleep(_PUSH_INTERVAL_SECONDS)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("ws push failed: %s", exc)
