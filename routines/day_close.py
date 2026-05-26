"""16:00 UTC — day close.

Steps:
  1. Tally closed trades from today via Freqtrade's REST API (NOT our legacy
     Postgres `trades` table, which Freqtrade does not populate — see
     PROJECT_HANDOFF.md §4).
  2. Compute daily P&L.
  3. Persist a `performance_snapshots` row.
  4. Overwrite `market_context.md` with the daily snapshot.
  5. Append a [DAY CLOSE] summary block to `trade_log.md`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from database import PerformanceSnapshot, SessionLocal
from memory.memory_io import (
    MemorySnapshot,
    append_daily_summary,
    append_lesson,
    overwrite_market_context,
)
from risk.circuit_breakers import CircuitBreakerState

from .base import BaseRoutine, setup_routine_logging

log = logging.getLogger(__name__)


class DayCloseRoutine(BaseRoutine):
    name = "day_close"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        from dashboard.backend.freqtrade_client import fetch_closed_trades, fetch_status

        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        closed_all = fetch_closed_trades(limit=200) or []
        closed_today = []
        for t in closed_all:
            close_str = t.get("close_date")
            if not close_str:
                continue
            try:
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)
            if close_dt >= midnight:
                closed_today.append(t)

        wins = sum(1 for t in closed_today if float(t.get("close_profit_abs") or 0) > 0)
        losses = sum(1 for t in closed_today if float(t.get("close_profit_abs") or 0) <= 0)
        daily_pnl_usd = sum(float(t.get("close_profit_abs") or 0) for t in closed_today)

        live_open = fetch_status() or []
        open_positions = len(live_open)
        deployed_pct = 0.0
        try:
            deployed_pct = sum(float(t.get("stake_amount") or 0) for t in live_open) / max(portfolio["equity"], 1.0)
        except (TypeError, ValueError):
            deployed_pct = 0.0

        with SessionLocal() as session:
            session.add(PerformanceSnapshot(
                ts=now,
                total_equity=portfolio["equity"],
                daily_pnl_usd=daily_pnl_usd,
                daily_pnl_pct=portfolio["daily_pnl_pct"],
                weekly_pnl_usd=portfolio["equity"] * portfolio["weekly_pnl_pct"],
                weekly_pnl_pct=portfolio["weekly_pnl_pct"],
                drawdown_pct=cb_state.drawdown_pct,
                peak_equity=portfolio["peak_equity"],
                open_positions=open_positions,
                deployed_capital_pct=deployed_pct,
            ))
            session.commit()

        overwrite_market_context(
            regime="see latest market_evaluation",
            regime_confidence=0.0,
            fear_greed=None,
            overall_sentiment="see latest sentiment_update",
            active_positions=open_positions,
            portfolio_value=portfolio["equity"],
            deployed_pct=deployed_pct,
            daily_pnl_usd=daily_pnl_usd,
            daily_pnl_pct=portfolio["daily_pnl_pct"],
            weekly_pnl_usd=portfolio["equity"] * portfolio["weekly_pnl_pct"],
            weekly_pnl_pct=portfolio["weekly_pnl_pct"],
            drawdown_pct=cb_state.drawdown_pct,
            circuit_breaker_state=cb_state.level.value,
        )

        append_daily_summary(
            date_str=now.strftime("%Y-%m-%d"),
            equity=portfolio["equity"],
            wins=wins,
            losses=losses,
            daily_pnl_usd=daily_pnl_usd,
            daily_pnl_pct=portfolio["daily_pnl_pct"],
            open_positions=open_positions,
            circuit_breaker_state=cb_state.level.value,
        )

        if cb_state.level.value not in ("NOMINAL", "HALVE_SIZES"):
            append_lesson(
                observation=(
                    f"Day close: {wins} wins, {losses} losses, ${daily_pnl_usd:+.2f}, "
                    f"circuit breaker={cb_state.level.value}"
                ),
                signal_involved="circuit_breaker",
                worked_or_failed="TRIGGERED",
                action_next_time="Review trigger and confirm no systemic issue before next session",
            )

        log.info(
            "day_close: wins=%s losses=%s daily_pnl=$%.2f open=%s equity=$%.2f cb=%s",
            wins, losses, daily_pnl_usd, open_positions, portfolio["equity"], cb_state.level.value,
        )
        return {
            "wins": wins,
            "losses": losses,
            "daily_pnl_usd": daily_pnl_usd,
            "open_positions": open_positions,
        }


def main() -> None:
    setup_routine_logging()
    DayCloseRoutine().run()


if __name__ == "__main__":
    main()
