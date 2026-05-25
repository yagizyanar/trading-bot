"""TRADING_LOCKED.txt management.

The lockfile is the hard kill-switch. When it exists, every routine must abort
before doing any trading work. Only the user manually removes it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import LOCKFILE_PATH


def is_locked() -> bool:
    """True if TRADING_LOCKED.txt exists at project root."""
    return LOCKFILE_PATH.exists()


def lockfile_reason() -> Optional[str]:
    """Return the lockfile contents (date, reason, equity stats) or None."""
    if not LOCKFILE_PATH.exists():
        return None
    return LOCKFILE_PATH.read_text(encoding="utf-8")


def write_lockfile(
    reason: str,
    peak_equity: float,
    current_equity: float,
    drawdown_pct: float,
    path: Optional[Path] = None,
) -> None:
    """Atomically write the lockfile. Bot halts on next routine tick."""
    # Resolve LOCKFILE_PATH at call time so monkeypatch in tests is honoured.
    if path is None:
        path = LOCKFILE_PATH
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    content = (
        f"TRADING LOCKED\n"
        f"Locked at: {ts}\n"
        f"Reason: {reason}\n"
        f"Peak equity: ${peak_equity:.2f}\n"
        f"Current equity: ${current_equity:.2f}\n"
        f"Drawdown: {drawdown_pct:.2%}\n"
        f"\n"
        f"To resume: review trade_log.md and lessons_learned.md, then delete this file.\n"
    )
    path.write_text(content, encoding="utf-8")
