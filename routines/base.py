"""BaseRoutine — the contract every routine must follow.

From the routines-scheduling skill + user PHASE 8:
  1. Check TRADING_LOCKED.txt; abort if present.
  2. Read all memory files.
  3. Check circuit breakers.
  4. Run routine-specific logic.
  5. Write memory updates.
  6. Log completion.

Errors inside _run_inner are caught, logged to lessons_learned.md, and the
routine exits cleanly — never crashes the scheduler.
"""
from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from memory.memory_io import MemorySnapshot, append_lesson, read_all
from risk.circuit_breakers import CircuitBreakerState, evaluate_circuit_breakers
from risk.lockfile import is_locked, lockfile_reason

log = logging.getLogger(__name__)


@dataclass
class RoutineResult:
    name: str
    started_at: datetime
    finished_at: datetime
    success: bool
    skipped: bool
    skip_reason: Optional[str]
    error: Optional[str]
    extra: dict


class BaseRoutine(ABC):
    name: str = "base"

    def __init__(self, get_portfolio_snapshot=None):
        """
        get_portfolio_snapshot: optional callable returning a dict with
          {equity, peak_equity, daily_pnl_pct, weekly_pnl_pct}.
        Defaults to a placeholder that assumes paper-trading dry_run_wallet.
        """
        self._get_portfolio = get_portfolio_snapshot or _default_portfolio_snapshot

    def run(self) -> RoutineResult:
        started = datetime.now(timezone.utc)
        if is_locked():
            log.warning("[%s] aborting: TRADING_LOCKED.txt present", self.name)
            return RoutineResult(
                name=self.name, started_at=started, finished_at=datetime.now(timezone.utc),
                success=False, skipped=True,
                skip_reason=f"lockfile present: {lockfile_reason() or 'reason unknown'}",
                error=None, extra={},
            )

        try:
            snapshot = read_all()
            portfolio = self._get_portfolio()
            cb_state = evaluate_circuit_breakers(
                daily_pnl_pct=portfolio["daily_pnl_pct"],
                weekly_pnl_pct=portfolio["weekly_pnl_pct"],
                equity=portfolio["equity"],
                peak_equity=portfolio["peak_equity"],
            )
            extra = self._run_inner(snapshot, portfolio, cb_state) or {}
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.exception("[%s] failed: %s", self.name, exc)
            try:
                append_lesson(
                    observation=f"Routine {self.name} crashed: {exc}",
                    signal_involved="N/A",
                    worked_or_failed="FAILED",
                    action_next_time="Investigate exception; check API connectivity and DB",
                )
            except Exception:  # noqa: BLE001
                pass
            return RoutineResult(
                name=self.name, started_at=started, finished_at=datetime.now(timezone.utc),
                success=False, skipped=False, skip_reason=None,
                error=f"{exc}\n{tb}", extra={},
            )

        log.info("[%s] completed in %s", self.name, datetime.now(timezone.utc) - started)
        return RoutineResult(
            name=self.name, started_at=started, finished_at=datetime.now(timezone.utc),
            success=True, skipped=False, skip_reason=None, error=None, extra=extra,
        )

    @abstractmethod
    def _run_inner(
        self,
        snapshot: MemorySnapshot,
        portfolio: dict,
        cb_state: CircuitBreakerState,
    ) -> Optional[dict]:
        """Implement routine-specific logic. Return an optional extras dict."""


def _default_portfolio_snapshot() -> dict:
    """Fallback portfolio snapshot when no callable is provided.

    Pulls the most recent PerformanceSnapshot row from the DB. If none exists,
    returns a flat zero-state assuming a paper wallet of dry_run_wallet from config.
    """
    try:
        from database import PerformanceSnapshot, SessionLocal
        with SessionLocal() as session:
            row = (
                session.query(PerformanceSnapshot)
                .order_by(PerformanceSnapshot.ts.desc())
                .first()
            )
        if row is None:
            return _zero_state()
        return {
            "equity": row.total_equity,
            "peak_equity": row.peak_equity,
            "daily_pnl_pct": row.daily_pnl_pct,
            "weekly_pnl_pct": row.weekly_pnl_pct,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("portfolio snapshot DB lookup failed (%s) — returning zero state", exc)
        return _zero_state()


def _zero_state() -> dict:
    return {"equity": 10000.0, "peak_equity": 10000.0, "daily_pnl_pct": 0.0, "weekly_pnl_pct": 0.0}
