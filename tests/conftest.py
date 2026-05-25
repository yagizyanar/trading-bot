"""pytest config — adds project root to sys.path and isolates memory/lockfile."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Provide harmless defaults for env-required settings so importing
# config.settings doesn't blow up in CI.
os.environ.setdefault("BINANCE_API_KEY", "test")
os.environ.setdefault("BINANCE_SECRET_KEY", "test")
os.environ.setdefault("DRY_RUN", "true")

import pytest


@pytest.fixture
def tmp_memory_dir(tmp_path, monkeypatch):
    """Redirect memory I/O to a per-test temp directory."""
    import memory.memory_io as mio
    mdir = tmp_path / "memory"
    mdir.mkdir()
    monkeypatch.setattr(mio, "TRADE_LOG", mdir / "trade_log.md")
    monkeypatch.setattr(mio, "LESSONS", mdir / "lessons_learned.md")
    monkeypatch.setattr(mio, "MARKET_CONTEXT", mdir / "market_context.md")
    monkeypatch.setattr(mio, "STRATEGY_NOTES", mdir / "strategy_notes.md")
    monkeypatch.setattr(mio, "ALL_FILES", (
        mdir / "trade_log.md",
        mdir / "lessons_learned.md",
        mdir / "market_context.md",
        mdir / "strategy_notes.md",
    ))
    return mdir


@pytest.fixture
def tmp_lockfile_path(tmp_path, monkeypatch):
    """Point the lockfile to a temp path so tests can write/read freely."""
    import risk.lockfile as lf
    p = tmp_path / "TRADING_LOCKED.txt"
    monkeypatch.setattr(lf, "LOCKFILE_PATH", p)
    return p
