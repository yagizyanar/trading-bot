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

from config.settings import NET_BETA_BUDGET, TARGET_COINS
from database import SessionLocal, SignalLog
from markov.multi_coin_runner import persist_regimes, run_all_coins
from memory.memory_io import (
    MemorySnapshot,
    append_lesson,
    overwrite_market_context,
)
from risk.beta import compute_book_betas
from risk.circuit_breakers import CircuitBreakerState
from risk.position_manager import can_open_position, net_beta_allows
from sentiment.analyzer import compute_unified_scores
from sentiment.binance_data import fetch_binance_ohlcv
from signals.technical import compute_technical_indicators
from signals.three_layer import BASE_FULL_PCT, SignalDecision, evaluate_signal

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

        open_positions = _currently_open_positions()
        open_coins: set[str] = {p["coin"] for p in open_positions}
        position_dir: dict[str, str] = {p["coin"]: p["direction"] for p in open_positions}
        deployed_pct = 0.0

        # Item 6: aggregate net-BTC-beta cap. Compute each coin's beta and seed
        # the running net beta from the live book (signed full-position-equivalents
        # weighted by beta). `betas` defaults coins to 1.0 if data is unavailable.
        betas = compute_book_betas(TARGET_COINS)
        equity = portfolio.get("equity") or 1.0

        def _beta_units(coin: str, size_pct: float, direction: str) -> float:
            sign = 1.0 if direction == "LONG" else -1.0
            return sign * (size_pct / BASE_FULL_PCT) * betas.get(coin, 1.0)

        net_beta = sum(
            _beta_units(p["coin"], (p["stake"] / equity if equity > 0 else 0.0), p["direction"])
            for p in open_positions
        )

        if cb_state.must_close_all:
            append_lesson(
                observation=f"Circuit breaker {cb_state.level.value} active — would close all positions",
                signal_involved="circuit_breaker",
                worked_or_failed="TRIGGERED",
                action_next_time=f"Trigger: {cb_state.trigger}",
            )

        # Pass 1: compute raw signal decisions for every coin (no portfolio
        # gate yet). Decisions can be LONG / SHORT / SKIP based on regime
        # alignment, markov dead-zone, sentiment, etc.
        raw_decisions: list[SignalDecision] = []
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
                current_position=position_dir.get(coin),  # Item 7: hysteresis on flips
            )
            raw_decisions.append(decision)

        # Pass 2: apply the portfolio gate (max_open, deployed_pct, sector cap)
        # in CONVICTION ORDER so the strongest setups win contested sector
        # slots. Without this, TARGET_COINS list order alone determined who
        # got the slot — e.g. with WIF / 1000BONK / 1000PEPE all generating
        # SHORT signals in the MEME sector, WIF and 1000BONK (3.75% size from
        # weaker sentiment) used to claim both MEME slots before 1000PEPE
        # (5.00% size, also qualifying for 2x leverage) could be evaluated.
        # Stable sort: ties preserve TARGET_COINS order, so behaviour is
        # deterministic when convictions tie.
        skips_from_signal_layer = [d for d in raw_decisions if d.decision == "SKIP"]
        trade_candidates = [d for d in raw_decisions if d.decision in ("LONG", "SHORT")]
        trade_candidates.sort(key=lambda d: d.position_size_pct, reverse=True)

        decisions: list[SignalDecision] = list(skips_from_signal_layer)
        for decision in trade_candidates:
            contrib = _beta_units(decision.coin, decision.position_size_pct, decision.decision)
            ok, reason = can_open_position(
                open_position_count=len(open_coins),
                deployed_capital_pct=deployed_pct,
                candidate_coin=decision.coin,
                open_coins=open_coins,
                allow_new_positions=cb_state.allow_new_positions,
            )
            # Item 6: net-beta cap — block a trade that would over-concentrate
            # the book's directional exposure (a same-direction add when already
            # at the budget). Opposite-direction trades that diversify still pass.
            if ok and not net_beta_allows(net_beta, contrib, NET_BETA_BUDGET):
                ok = False
                reason = (
                    f"net-beta cap: net {net_beta:+.2f} {contrib:+.2f} would breach "
                    f"±{NET_BETA_BUDGET:.1f}"
                )
            if not ok:
                decision = SignalDecision(
                    coin=decision.coin, decision="SKIP",
                    position_size_pct=0.0, dollars=0.0, leverage=1,
                    markov_signal=decision.markov_signal,
                    sentiment_score=decision.sentiment_score,
                    technical_label=decision.technical_label,
                    regime=decision.regime,
                    reason=f"portfolio rule: {reason}",
                    skip_reason=reason,
                )
            else:
                open_coins.add(decision.coin)
                deployed_pct += decision.position_size_pct
                net_beta += contrib
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

        # Real deployed capital from live Freqtrade state — NOT the sum of
        # new-signal sizes from this cycle (which is misleading: most signals
        # SKIP, so it would always read 0-15% on the dashboard even when 40%+
        # of the wallet is actually deployed across pre-existing positions).
        actual_deployed_pct = _actual_deployed_pct(portfolio["equity"])

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
            deployed_pct=actual_deployed_pct,
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


def _actual_deployed_pct(equity: float) -> float:
    """Real deployed capital from Freqtrade's open positions, as a fraction of equity.

    Falls back to 0.0 if Freqtrade is unreachable. Used by market_context.md so
    the "deployed %" dashboard line matches what the positions table actually
    sums to, not the per-cycle new-signal sum (which would always be small
    because most signals SKIP).
    """
    if equity is None or equity <= 0:
        return 0.0
    try:
        from dashboard.backend.freqtrade_client import fetch_status
        live = fetch_status()
        if not live:
            return 0.0
        total_stake = sum(float(t.get("stake_amount") or 0) for t in live)
        return total_stake / float(equity)
    except Exception as exc:  # noqa: BLE001
        log.warning("actual_deployed_pct: freqtrade lookup failed: %s", exc)
        return 0.0


def _currently_open_coins() -> list[str]:
    """Return base symbols of currently open positions.

    Prefers Freqtrade's live `/api/v1/status` (the executor's ground truth),
    falls back to our `trades` DB table if Freqtrade is unreachable.

    The DB table is empty in normal operation because Freqtrade stores its
    trades in its own SQLite — reading from there gave us spurious correlation
    blocks (e.g., OP getting marked as "Already 2 open in sector L2" when only
    ARB was actually open and MATIC was a wished-for-but-undeliverable signal).
    """
    try:
        from dashboard.backend.freqtrade_client import fetch_status
        live = fetch_status()
        if live is not None:
            return [t.get("pair", "").split("/")[0] for t in live if t.get("pair")]
    except Exception as exc:  # noqa: BLE001
        log.warning("freqtrade fetch_status failed in market_evaluation: %s", exc)

    try:
        from database import SessionLocal, Trade
        with SessionLocal() as session:
            rows = session.query(Trade.coin).filter(Trade.outcome == "OPEN").all()
        return [r[0] for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.warning("open-coins lookup failed: %s", exc)
        return []


def _currently_open_positions() -> list[dict]:
    """[{coin, direction, stake}] for live open positions — for the net-beta seed.

    Same source as _currently_open_coins (Freqtrade /api/v1/status). `direction`
    is LONG/SHORT from `is_short`; `stake` is the position's stake in quote
    currency. Returns [] if Freqtrade is unreachable.
    """
    try:
        from dashboard.backend.freqtrade_client import fetch_status
        live = fetch_status()
        if not live:
            return []
        out: list[dict] = []
        for t in live:
            pair = t.get("pair", "")
            if not pair:
                continue
            out.append({
                "coin": pair.split("/")[0],
                "direction": "SHORT" if t.get("is_short") else "LONG",
                "stake": float(t.get("stake_amount") or 0.0),
            })
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("open-positions lookup failed: %s", exc)
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
    from .base import setup_routine_logging
    setup_routine_logging()
    MarketEvaluationRoutine().run()


if __name__ == "__main__":
    main()
