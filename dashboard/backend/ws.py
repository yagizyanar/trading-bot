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

from .alerts import get_alerts
from .freqtrade_client import (
    fetch_balance,
    fetch_pnl_breakdown,
    fetch_status,
    map_freqtrade_trade,
)

log = logging.getLogger(__name__)

_PUSH_INTERVAL_SECONDS = 5


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

    # Live performance snapshot — uses the route's helper directly so the
    # WS payload and /api/performance/latest produce identical numbers.
    try:
        bal = fetch_balance()
        pnl = fetch_pnl_breakdown()
        with SessionLocal() as session:
            last_db = (
                session.query(PerformanceSnapshot)
                .order_by(PerformanceSnapshot.ts.desc())
                .first()
            )
            sent_count = session.query(SentimentScore).count()
            if bal is not None or pnl is not None:
                from .routes.performance import _live_to_snapshot
                perf = _live_to_snapshot(session, bal or {}, pnl, last_db)
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

    # System alerts (server-side 5-min cache, so cheap to read every push)
    try:
        payload["alerts"] = get_alerts()
    except Exception as exc:  # noqa: BLE001
        log.warning("ws alerts fetch failed: %s", exc)
        payload["alerts"] = []

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
