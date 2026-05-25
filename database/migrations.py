"""Create-all migration + idempotent column-additions for in-place upgrades.

Initial setup:
    python -c "from database.migrations import init_db; init_db()"

After model changes:
    python -c "from database.migrations import upgrade; upgrade()"

`upgrade()` is safe to re-run — uses `ADD COLUMN IF NOT EXISTS` (Postgres 9.6+).
For richer schema evolution, switch to Alembic (alembic is already a dep).
"""
from __future__ import annotations

from sqlalchemy import text

from .connection import engine
from .models import Base


_ADDITIVE_MIGRATIONS = (
    # Added 2026-05: new sentiment sources (Binance Futures market data + Hyperliquid)
    "ALTER TABLE sentiment_scores ADD COLUMN IF NOT EXISTS long_short_ratio DOUBLE PRECISION",
    "ALTER TABLE sentiment_scores ADD COLUMN IF NOT EXISTS funding_rate     DOUBLE PRECISION",
    "ALTER TABLE sentiment_scores ADD COLUMN IF NOT EXISTS hyperliquid_score DOUBLE PRECISION",
)


def init_db() -> None:
    """Create every table if it doesn't already exist."""
    Base.metadata.create_all(engine)
    print(f"Initialised database schema on {engine.url}")


def upgrade() -> None:
    """Apply additive migrations idempotently.

    `init_db` creates new tables but doesn't add columns to existing ones; that's
    what this function is for. Safe to run multiple times.
    """
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in _ADDITIVE_MIGRATIONS:
            conn.execute(text(stmt))
    print(f"Applied {len(_ADDITIVE_MIGRATIONS)} additive migrations on {engine.url}")
