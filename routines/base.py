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


def setup_routine_logging() -> None:
    """Configure the root logger so routine log.info calls reach stdout.

    Cron redirects stdout+stderr to /var/log/trading-bot/<routine>.log. Without
    this call, log.info messages disappear (Python's root logger defaults to
    WARNING with no handlers) and the per-routine log files stay 0 bytes.

    Safe to call multiple times — only adds a handler if none is present.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _default_portfolio_snapshot() -> dict:
    """Live portfolio snapshot.

    Preferred source is Freqtrade's REST API — equity computed by the
    handoff §5 reconcilable formula:
        equity = DRY_RUN_WALLET + closed_pnl + sum(open profit_abs)
    Falls back to the most recent PerformanceSnapshot row, then to zero state.

    daily_pnl_pct = closed_pnl_since_midnight_UTC / DRY_RUN_WALLET
    weekly_pnl_pct uses the same denominator and Monday 00:00 UTC start.
    """
    try:
        from datetime import timedelta
        from config.settings import DRY_RUN_WALLET
        from dashboard.backend.freqtrade_client import (
            fetch_closed_trades, fetch_profit, fetch_status,
        )

        status = fetch_status()
        profit = fetch_profit()
        closed = fetch_closed_trades(limit=200)

        if status is None and profit is None:
            raise RuntimeError("Freqtrade API unreachable")

        open_pnl = sum(float(t.get("profit_abs") or 0) for t in (status or []))
        closed_pnl_all = float((profit or {}).get("profit_closed_coin", 0) or 0)
        equity = DRY_RUN_WALLET + open_pnl + closed_pnl_all

        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = midnight - timedelta(days=midnight.weekday())  # Monday 00:00 UTC

        daily_closed_pnl = 0.0
        weekly_closed_pnl = 0.0
        for t in (closed or []):
            close_str = t.get("close_date")
            if not close_str:
                continue
            try:
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)
            pnl = float(t.get("close_profit_abs") or 0)
            if close_dt >= midnight:
                daily_closed_pnl += pnl
            if close_dt >= week_start:
                weekly_closed_pnl += pnl

        daily_pnl_pct = daily_closed_pnl / DRY_RUN_WALLET if DRY_RUN_WALLET else 0.0
        weekly_pnl_pct = weekly_closed_pnl / DRY_RUN_WALLET if DRY_RUN_WALLET else 0.0

        # Peak equity: max of historical peak and current equity.
        peak = equity
        try:
            from database import PerformanceSnapshot, SessionLocal
            with SessionLocal() as session:
                row = (
                    session.query(PerformanceSnapshot)
                    .order_by(PerformanceSnapshot.peak_equity.desc())
                    .first()
                )
            if row is not None and row.peak_equity > peak:
                peak = row.peak_equity
        except Exception:  # noqa: BLE001
            pass

        return {
            "equity": equity,
            "peak_equity": peak,
            "daily_pnl_pct": daily_pnl_pct,
            "weekly_pnl_pct": weekly_pnl_pct,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("freqtrade portfolio snapshot failed (%s) — falling back to DB", exc)

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
