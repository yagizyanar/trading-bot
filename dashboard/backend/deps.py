"""Shared FastAPI dependencies."""
from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session

from database import SessionLocal


def db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
