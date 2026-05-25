"""00:00 UTC — pre-market analysis.

Steps (per routines-scheduling skill):
  1. Read memory files (handled by BaseRoutine).
  2. Fetch Fear & Greed Index.
  3. Scan latest news headlines.
  4. Update market_context.md with macro conditions.
  5. Flag major events for the 08:00 market_evaluation routine.
"""
from __future__ import annotations

import logging

from memory.memory_io import (
    MemorySnapshot,
    append_lesson,
    overwrite_market_context,
)
from risk.circuit_breakers import CircuitBreakerState
from sentiment.crypto_news import fetch_crypto_news
from sentiment.fear_greed import fetch_fear_greed

from .base import BaseRoutine

log = logging.getLogger(__name__)


class PreMarketRoutine(BaseRoutine):
    name = "pre_market"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        fg = fetch_fear_greed()
        news_items = fetch_crypto_news(limit=100)

        big_events = [
            it for it in news_items
            if any(k in it.title.lower() for k in ("crash", "hack", "sec", "etf", "ban", "rally", "ath"))
        ]

        regime_label = "unknown"
        regime_conf = 0.0

        sentiment_label = "neutral"
        if fg:
            if fg.value <= 25:
                sentiment_label = "extreme_fear"
            elif fg.value <= 45:
                sentiment_label = "fear"
            elif fg.value <= 55:
                sentiment_label = "neutral"
            elif fg.value <= 75:
                sentiment_label = "greed"
            else:
                sentiment_label = "extreme_greed"

        overwrite_market_context(
            regime=regime_label,
            regime_confidence=regime_conf,
            fear_greed=fg.value if fg else None,
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

        if big_events:
            append_lesson(
                observation=f"Pre-market scan flagged {len(big_events)} notable headlines",
                signal_involved="news",
                worked_or_failed="FLAGGED",
                action_next_time="Inspect titles in market_evaluation before sizing positions",
            )

        return {
            "fear_greed": fg.value if fg else None,
            "news_count": len(news_items),
            "flagged_event_count": len(big_events),
            "sentiment_label": sentiment_label,
        }


def main() -> None:
    PreMarketRoutine().run()


if __name__ == "__main__":
    main()
