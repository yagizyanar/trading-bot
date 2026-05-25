"""FastAPI app exposing read-only dashboard data.

Run with:
    uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import LOG_LEVEL

from .routes import positions, sentiment, regime, performance, status, memory, fear_greed
from .ws import register_ws

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)

app = FastAPI(title="trade-sentiment-markov dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"])
app.include_router(regime.router, prefix="/api/regime", tags=["regime"])
app.include_router(performance.router, prefix="/api/performance", tags=["performance"])
app.include_router(status.router, prefix="/api/status", tags=["status"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])
app.include_router(fear_greed.router, prefix="/api/fear-greed", tags=["fear-greed"])

register_ws(app)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
