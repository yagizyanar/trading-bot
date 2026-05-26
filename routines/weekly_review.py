"""Sunday 20:00 UTC — weekly review.

Steps:
  1. Compute weekly metrics: Sharpe, win rate, profit factor.
  2. Identify which signal layer worked best.
  3. Update strategy_notes.md.
  4. Trigger FreqAI model retrain by touching a sentinel file.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database import SessionLocal, Trade
from memory.memory_io import (
    MemorySnapshot,
    append_lesson,
    append_strategy_note,
)
from risk.circuit_breakers import CircuitBreakerState

from .base import BaseRoutine

log = logging.getLogger(__name__)

FREQAI_RETRAIN_SENTINEL = Path(__file__).resolve().parent.parent / "user_data" / "models" / ".retrain"


class WeeklyReviewRoutine(BaseRoutine):
    name = "weekly_review"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        with SessionLocal() as session:
            week_trades = session.query(Trade).filter(
                Trade.exit_ts.isnot(None),
                Trade.exit_ts >= week_ago,
            ).all()

        total = len(week_trades)
        wins = [t for t in week_trades if t.outcome == "WIN"]
        losses = [t for t in week_trades if t.outcome == "LOSS"]
        gross_win = sum((t.pnl_usd or 0.0) for t in wins)
        gross_loss = -sum((t.pnl_usd or 0.0) for t in losses)
        net = gross_win - gross_loss
        win_rate = (len(wins) / total) if total else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf

        returns = [(t.pnl_pct or 0.0) for t in week_trades]
        sharpe = _sharpe(returns)

        append_strategy_note(
            change="Weekly review snapshot",
            reason=(
                f"trades={total}, win_rate={win_rate:.1%}, "
                f"profit_factor={'inf' if profit_factor == math.inf else f'{profit_factor:.2f}'}, "
                f"sharpe={sharpe:.2f}, net=${net:+.2f}"
            ),
            expected_impact="No change — informational snapshot",
            validation_plan="Next week's review compares against these baselines",
        )

        if total > 0 and (win_rate < 0.45 or (profit_factor != math.inf and profit_factor < 1.1)):
            append_lesson(
                observation=(
                    f"Weekly performance below target: win_rate={win_rate:.1%}, "
                    f"profit_factor={profit_factor:.2f}"
                ),
                signal_involved="aggregate",
                worked_or_failed="FAILED",
                action_next_time=(
                    "Investigate worst-performing coins; consider tightening signal thresholds"
                ),
            )

        try:
            FREQAI_RETRAIN_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            FREQAI_RETRAIN_SENTINEL.write_text(now.isoformat(), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not write FreqAI retrain sentinel: %s", exc)

        return {
            "trades": total,
            "win_rate": win_rate,
            "profit_factor": (None if profit_factor == math.inf else profit_factor),
            "sharpe": sharpe,
            "net_pnl": net,
        }


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(52)


def main() -> None:
    from .base import setup_routine_logging
    setup_routine_logging()
    WeeklyReviewRoutine().run()


if __name__ == "__main__":
    main()
