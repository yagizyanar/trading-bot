"""WebSocket endpoint for live dashboard pushes.

Broadcasts a snapshot of {status, performance, sentiment counts} every
5 seconds. Clients subscribe; no upstream input is processed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from database import PerformanceSnapshot, SentimentScore, SessionLocal
from risk.lockfile import is_locked

log = logging.getLogger(__name__)

_PUSH_INTERVAL_SECONDS = 5


async def _snapshot() -> dict:
    locked = is_locked()
    payload: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "locked": locked,
    }
    try:
        with SessionLocal() as session:
            perf = (
                session.query(PerformanceSnapshot)
                .order_by(PerformanceSnapshot.ts.desc())
                .first()
            )
            sent_count = session.query(SentimentScore).count()
        payload["equity"] = perf.total_equity if perf else None
        payload["drawdown_pct"] = perf.drawdown_pct if perf else None
        payload["open_positions"] = perf.open_positions if perf else 0
        payload["sentiment_rows"] = sent_count
    except Exception as exc:  # noqa: BLE001
        payload["db_error"] = str(exc)
    return payload


def register_ws(app: FastAPI) -> None:
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_text(json.dumps(await _snapshot()))
                await asyncio.sleep(_PUSH_INTERVAL_SECONDS)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("ws push failed: %s", exc)
