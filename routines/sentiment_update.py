"""04:00 UTC — sentiment update.

Refresh every source, compute unified scores for all 18 coins, persist to DB.
"""
from __future__ import annotations

import logging

from config.settings import TARGET_COINS
from memory.memory_io import MemorySnapshot, overwrite_market_context
from risk.circuit_breakers import CircuitBreakerState
from sentiment.analyzer import compute_unified_scores, persist_unified_scores

from .base import BaseRoutine

log = logging.getLogger(__name__)


class SentimentUpdateRoutine(BaseRoutine):
    name = "sentiment_update"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        scores = compute_unified_scores(TARGET_COINS)
        try:
            persist_unified_scores(scores)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not persist sentiment scores: %s", exc)

        if scores:
            avg_unified = sum(s.unified for s in scores.values()) / len(scores)
        else:
            avg_unified = 0.0

        sentiment_label = (
            "bullish" if avg_unified > 0.2
            else "bearish" if avg_unified < -0.2
            else "neutral"
        )
        sample_fg = next(iter(scores.values())).fear_greed if scores else None

        overwrite_market_context(
            regime="unknown",
            regime_confidence=0.0,
            fear_greed=sample_fg,
            overall_sentiment=sentiment_label,
            active_positions=0,
            portfolio_value=portfolio["equity"],
            deployed_pct=0.0,
            daily_pnl_usd=portfolio["equity"] * portfolio["daily_pnl_pct"],
            daily_pnl_pct=portfolio["daily_pnl_pct"],
            weekly_pnl_usd=portfolio["equity"] * portfolio["weekly_pnl_pct"],
            weekly_pnl_pct=portfolio["weekly_pnl_pct"],
            drawdown_pct=cb_state.drawdown_pct,
            circuit_breaker_state=cb_state.level.value,
        )

        return {
            "coins_scored": len(scores),
            "avg_unified": avg_unified,
            "label": sentiment_label,
        }


def main() -> None:
    SentimentUpdateRoutine().run()


if __name__ == "__main__":
    main()
