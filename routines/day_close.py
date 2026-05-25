"""16:00 UTC — day close.

Steps:
  1. Tally closed trades from today.
  2. Compute daily P&L.
  3. Apply daily circuit-breaker thresholds (already done by BaseRoutine.cb_state).
  4. Write a daily summary to memory files.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from database import PerformanceSnapshot, SessionLocal, Trade
from memory.memory_io import (
    MemorySnapshot,
    append_lesson,
    overwrite_market_context,
)
from risk.circuit_breakers import CircuitBreakerState

from .base import BaseRoutine

log = logging.getLogger(__name__)


class DayCloseRoutine(BaseRoutine):
    name = "day_close"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with SessionLocal() as session:
            closed_today = session.query(Trade).filter(
                Trade.exit_ts.isnot(None),
                Trade.exit_ts >= midnight,
            ).all()
            wins = sum(1 for t in closed_today if t.outcome == "WIN")
            losses = sum(1 for t in closed_today if t.outcome == "LOSS")
            daily_pnl_usd = sum((t.pnl_usd or 0.0) for t in closed_today)
            open_positions = session.query(Trade).filter(Trade.outcome == "OPEN").count()

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
                deployed_capital_pct=0.0,
            ))
            session.commit()

        overwrite_market_context(
            regime="see latest market_evaluation",
            regime_confidence=0.0,
            fear_greed=None,
            overall_sentiment="see latest sentiment_update",
            active_positions=open_positions,
            portfolio_value=portfolio["equity"],
            deployed_pct=0.0,
            daily_pnl_usd=daily_pnl_usd,
            daily_pnl_pct=portfolio["daily_pnl_pct"],
            weekly_pnl_usd=portfolio["equity"] * portfolio["weekly_pnl_pct"],
            weekly_pnl_pct=portfolio["weekly_pnl_pct"],
            drawdown_pct=cb_state.drawdown_pct,
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

        return {
            "wins": wins,
            "losses": losses,
            "daily_pnl_usd": daily_pnl_usd,
            "open_positions": open_positions,
        }


def main() -> None:
    DayCloseRoutine().run()


if __name__ == "__main__":
    main()
