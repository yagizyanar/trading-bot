"""08:00 UTC — market evaluation.

The heart of the bot:
  1. Run Markov regime detection on all 18 coins.
  2. Pull latest sentiment scores.
  3. Compute technical indicators on 1h Binance OHLCV.
  4. Apply three-layer gate.
  5. Persist decisions to signal_log.
  6. Decisions feed the Freqtrade strategy via cached DB lookup.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import TARGET_COINS
from database import SessionLocal, SignalLog
from markov.multi_coin_runner import persist_regimes, run_all_coins
from memory.memory_io import (
    MemorySnapshot,
    append_lesson,
    overwrite_market_context,
)
from risk.circuit_breakers import CircuitBreakerState
from risk.position_manager import can_open_position
from sentiment.analyzer import compute_unified_scores
from sentiment.binance_data import fetch_binance_ohlcv
from signals.technical import compute_technical_indicators
from signals.three_layer import SignalDecision, evaluate_signal

from .base import BaseRoutine

log = logging.getLogger(__name__)


class MarketEvaluationRoutine(BaseRoutine):
    name = "market_evaluation"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        regimes = run_all_coins(TARGET_COINS)
        try:
            persist_regimes(regimes)
        except Exception as exc:  # noqa: BLE001
            log.warning("persist_regimes failed: %s", exc)

        sentiments = compute_unified_scores(TARGET_COINS)

        decisions: list[SignalDecision] = []
        open_coins: set[str] = set(_currently_open_coins())
        deployed_pct = 0.0

        if cb_state.must_close_all:
            append_lesson(
                observation=f"Circuit breaker {cb_state.level.value} active — would close all positions",
                signal_involved="circuit_breaker",
                worked_or_failed="TRIGGERED",
                action_next_time=f"Trigger: {cb_state.trigger}",
            )

        for coin in TARGET_COINS:
            regime = regimes.get(coin)
            sent = sentiments.get(coin)
            if regime is None or sent is None:
                continue

            df = fetch_binance_ohlcv(f"{coin}USDT", interval="1h", limit=200)
            tech = compute_technical_indicators(df) if df is not None else None
            if tech is None:
                continue

            decision = evaluate_signal(
                coin=coin,
                regime_result=regime,
                sentiment=sent,
                technical=tech,
                capital=portfolio["equity"],
                cb_multiplier=cb_state.size_multiplier,
                cb_allows_new=cb_state.allow_new_positions,
            )

            if decision.decision in ("LONG", "SHORT"):
                ok, reason = can_open_position(
                    open_position_count=len(open_coins),
                    deployed_capital_pct=deployed_pct,
                    candidate_coin=coin,
                    open_coins=open_coins,
                    allow_new_positions=cb_state.allow_new_positions,
                )
                if not ok:
                    decision = SignalDecision(
                        coin=coin, decision="SKIP",
                        position_size_pct=0.0, dollars=0.0, leverage=1,
                        markov_signal=decision.markov_signal,
                        sentiment_score=decision.sentiment_score,
                        technical_label=decision.technical_label,
                        regime=decision.regime,
                        reason=f"portfolio rule: {reason}",
                        skip_reason=reason,
                    )
                else:
                    open_coins.add(coin)
                    deployed_pct += decision.position_size_pct

            decisions.append(decision)

        _persist_signal_log(decisions)

        long_count = sum(1 for d in decisions if d.decision == "LONG")
        short_count = sum(1 for d in decisions if d.decision == "SHORT")
        avg_regime_signal = (
            sum(r.markov_signal for r in regimes.values()) / len(regimes) if regimes else 0.0
        )

        if regimes:
            from collections import Counter
            regime_mode = Counter(r.regime for r in regimes.values()).most_common(1)[0][0]
            confidence_avg = sum(r.confidence for r in regimes.values()) / len(regimes)
        else:
            regime_mode = "unknown"
            confidence_avg = 0.0

        overwrite_market_context(
            regime=regime_mode,
            regime_confidence=confidence_avg,
            fear_greed=next(iter(sentiments.values())).fear_greed if sentiments else None,
            overall_sentiment=(
                "bullish" if avg_regime_signal > 0.2
                else "bearish" if avg_regime_signal < -0.2 else "neutral"
            ),
            active_positions=len(open_coins),
            portfolio_value=portfolio["equity"],
            deployed_pct=deployed_pct,
            daily_pnl_usd=portfolio["equity"] * portfolio["daily_pnl_pct"],
            daily_pnl_pct=portfolio["daily_pnl_pct"],
            weekly_pnl_usd=portfolio["equity"] * portfolio["weekly_pnl_pct"],
            weekly_pnl_pct=portfolio["weekly_pnl_pct"],
            drawdown_pct=cb_state.drawdown_pct,
            circuit_breaker_state=cb_state.level.value,
        )

        return {
            "long_count": long_count,
            "short_count": short_count,
            "skip_count": len(decisions) - long_count - short_count,
            "average_markov_signal": avg_regime_signal,
            "regime_mode": regime_mode,
        }


def _currently_open_coins() -> list[str]:
    try:
        from database import SessionLocal, Trade
        with SessionLocal() as session:
            rows = session.query(Trade.coin).filter(Trade.outcome == "OPEN").all()
        return [r[0] for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.warning("open-coins lookup failed: %s", exc)
        return []


def _persist_signal_log(decisions: list[SignalDecision]) -> None:
    if not decisions:
        return
    try:
        ts = datetime.now(timezone.utc)
        with SessionLocal() as session:
            for d in decisions:
                session.add(SignalLog(
                    coin=d.coin,
                    ts=ts,
                    markov_signal=d.markov_signal,
                    sentiment_score=d.sentiment_score,
                    technical_signal=d.technical_label,
                    decision=d.decision,
                    position_size_pct=d.position_size_pct,
                    skip_reason=d.skip_reason,
                ))
            session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("signal_log persist failed: %s", exc)


def main() -> None:
    MarketEvaluationRoutine().run()


if __name__ == "__main__":
    main()
