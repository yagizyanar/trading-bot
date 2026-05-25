"""Create-all migration. Run once at install time:

    python -c "from database.migrations import init_db; init_db()"

For schema evolution, switch to Alembic (alembic.ini already pulled by deps).
"""
from __future__ import annotations

from .connection import engine
from .models import Base


def init_db() -> None:
    """Create every table if it doesn't already exist."""
    Base.metadata.create_all(engine)
    print(f"Initialised database schema on {engine.url}")
