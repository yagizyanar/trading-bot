"""FreqAI strategy that consumes Markov regime + sentiment + technical signals.

Design:
- Freqtrade calls populate_indicators() on every candle close.
- We pull the most recent signal_log row for the pair from PostgreSQL.
- Entry/exit are gated on the cached SignalDecision from the latest routine run.
- FreqAI provides ML predictions as an additional confirmation feature.

If the database is unavailable we fall back to neutral signals — Freqtrade
keeps running, no trades opened.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from freqtrade.strategy import IStrategy, IntParameter, informative  # type: ignore
except ImportError:  # pragma: no cover - allow strategy file to import in tests
    IStrategy = object  # type: ignore[misc,assignment]
    IntParameter = lambda *a, **kw: None  # type: ignore[assignment]

    def informative(*_args, **_kwargs):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn
        return _decorator

log = logging.getLogger(__name__)


class FreqAISentimentStrategy(IStrategy):  # type: ignore[misc,valid-type]
    """Strategy wired to our sentiment + Markov + technical signal layer."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    stoploss = -0.05
    # Trailing TP widened 2026-06-04 (2%/+3% -> 8%/+10%) after live evidence: the
    # old 2% trail caused 63% of exits and capped winners at +2.4% avg (only 4
    # trades ever >+8%), amputating the 20-day-momentum fat tail. minimal_roi
    # disabled (was +15%, fired 2x ever) so the trail manages the upside.
    # 2026-06-05 trailing bake-off (oos_trail_compare.py, 15m, 2022-24): 8%/+10%
    # beat both 4%/+5% and 6%/+8% on EVERY 3yr metric — compounded -35.0% vs
    # -42.1% / -36.8%, best Sharpe, fewest trades (3876 vs 6707) and lowest fee
    # drag (11.9% vs 20.5%). The tighter trails ~doubled churn with no return
    # payoff, so the earlier-activation experiment is reverted to wider 8%/+10%.
    trailing_stop = True
    trailing_stop_positive = 0.08
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    minimal_roi = {"0": 10}
    # 1-day cooldown after a LOSING stop (StoplossGuard) — kills the 15m hard-stop
    # re-entry churn that the intraday harness exposed (item 9). Locks only the pair
    # that stopped, only after a losing stop (required_profit default 0), so signal
    # flips and winning trailing exits are unaffected. Strategy-level: config-level
    # protections are DEPRECATED in freqtrade 2026.4.
    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 96,    # 96 x 15m = 1 day
            "trade_limit": 1,
            "stop_duration_candles": 96,
            "only_per_pair": True,
        }
    ]
    startup_candle_count: int = 200

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "blue"},
            "ema_slow": {"color": "orange"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "purple"}},
            "MACD": {"macd": {"color": "blue"}, "macd_signal": {"color": "orange"}},
            "Sentiment": {"unified_sentiment": {"color": "green"}},
            "Markov": {"markov_signal": {"color": "red"}},
        },
    }

    # ------------------------------------------------------------------
    # FreqAI feature engineering
    # ------------------------------------------------------------------
    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, **_) -> pd.DataFrame:
        dataframe[f"%-rsi-period_{period}"] = self._rsi(dataframe["close"], period)
        dataframe[f"%-mfi-period_{period}"] = self._mfi(dataframe, period)
        dataframe[f"%-roc-period_{period}"] = dataframe["close"].pct_change(period)
        dataframe[f"%-volume-zscore-{period}"] = (
            (dataframe["volume"] - dataframe["volume"].rolling(period).mean())
            / dataframe["volume"].rolling(period).std()
        )
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **_) -> pd.DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-volume"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, **_) -> pd.DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **_) -> pd.DataFrame:
        # Classify next 24-period return as up (+1) / flat (0) / down (-1)
        future_return = dataframe["close"].shift(-24) / dataframe["close"] - 1
        dataframe["&-target"] = np.where(
            future_return > 0.02, 1, np.where(future_return < -0.02, -1, 0)
        )
        return dataframe

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        from signals.technical import compute_technical_indicators

        dataframe["rsi"] = self._rsi(dataframe["close"], 14)
        ema_fast = dataframe["close"].ewm(span=12, adjust=False).mean()
        ema_slow = dataframe["close"].ewm(span=26, adjust=False).mean()
        dataframe["ema_fast"] = ema_fast
        dataframe["ema_slow"] = ema_slow
        macd = ema_fast - ema_slow
        dataframe["macd"] = macd
        dataframe["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
        dataframe["macd_hist"] = dataframe["macd"] - dataframe["macd_signal"]

        # Pull latest sentiment + regime from our DB.
        coin = metadata["pair"].split("/")[0]
        sentiment_score, regime_signal, decision = _latest_decision(coin)
        dataframe["unified_sentiment"] = sentiment_score
        dataframe["markov_signal"] = regime_signal
        dataframe["routine_decision"] = decision  # LONG / SHORT / SKIP

        # Only invoke FreqAI when config.freqai.enabled is True. `self.freqai`
        # is a non-None stub object even when freqai is disabled — its `start()`
        # raises OperationalException, which used to silently kill every
        # candle's analysis. Check the config flag instead.
        if self.config.get("freqai", {}).get("enabled", False):  # type: ignore[attr-defined]
            dataframe = self.freqai.start(dataframe, metadata, self)  # type: ignore[attr-defined]

        # Compose technical label from latest row
        snapshot = compute_technical_indicators(
            dataframe.rename(columns={"date": "open_time"}).set_index(
                pd.DatetimeIndex(dataframe["date"])
            )
        )
        dataframe["tech_label"] = snapshot.label if snapshot else "NEUTRAL"
        return dataframe

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Trust `routine_decision` directly — the market_evaluation routine has
        # already applied the Markov-primary gate, sentiment/technical multipliers,
        # regime checks, correlation caps, and circuit breakers. Re-gating here
        # would duplicate (and historically conflicted with) that logic.
        #
        # Stop-loss (-5%) and take-profit (+15%) are still enforced by Freqtrade.
        dataframe.loc[dataframe["routine_decision"] == "LONG",  "enter_long"]  = 1
        dataframe.loc[dataframe["routine_decision"] == "SHORT", "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Exit only on a *strong opposite* routine decision — a SKIP isn't a
        # reason to close (it often just means correlation rules blocked a new
        # entry, not that the existing position is wrong).
        dataframe.loc[dataframe["routine_decision"] == "SHORT", "exit_long"]  = 1
        dataframe.loc[dataframe["routine_decision"] == "LONG",  "exit_short"] = 1
        return dataframe

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float | None, max_stake: float,
        leverage: float, entry_tag: str | None, side: str, **_: Any,
    ) -> float:
        """Override Freqtrade's default sizing with the routine's `position_size_pct`.

        The market_evaluation routine writes a per-coin size to signal_log that
        already factors in Markov strength, sentiment alignment, technical
        confirmation, regime (Sideways halving), and the circuit-breaker multiplier.
        We apply that fraction to the current total wallet:

            stake_USDT = position_size_pct × wallets.get_total(stake_currency)

        Fall back to Freqtrade's proposed_stake on any error (DB unreachable,
        no signal_log row, wallet info unavailable, etc.) — Freqtrade's default
        is also safe, just doesn't honour the routine's size decision.
        """
        coin = pair.split("/")[0]
        pct = _latest_position_size_pct(coin)
        if pct is None or pct <= 0:
            return float(proposed_stake)
        try:
            stake_currency = self.config.get("stake_currency", "USDT")  # type: ignore[attr-defined]
            total = float(self.wallets.get_total(stake_currency))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            log.warning("custom_stake_amount: wallet lookup failed (%s) — using proposed", exc)
            return float(proposed_stake)

        stake = total * float(pct)
        if min_stake is not None:
            stake = max(float(min_stake), stake)
        stake = min(stake, float(max_stake))
        return float(stake)

    def leverage(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_leverage: float, max_leverage: float, side: str, **_: Any,
    ) -> float:
        # Dynamic leverage by |markov signal| (2026-06-05): |s|>0.5 → 3x, >0.3 → 2x,
        # else 1x — matching signals.three_layer._leverage_from_signal (the size layer
        # divides position size by the same tier so NOTIONAL stays constant). Capped by
        # settings.MAX_LEVERAGE (=3) and the exchange max_leverage.
        from signals.three_layer import _leverage_from_signal
        from config.settings import MAX_LEVERAGE

        coin = pair.split("/")[0]
        _, markov, _ = _latest_decision(coin)
        lev = _leverage_from_signal(markov)
        return float(min(lev, max_leverage, MAX_LEVERAGE))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _mfi(df: pd.DataFrame, period: int) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        raw_flow = typical * df["volume"]
        delta = typical.diff()
        pos_flow = raw_flow.where(delta > 0, 0.0).rolling(period).sum()
        neg_flow = raw_flow.where(delta < 0, 0.0).rolling(period).sum()
        mfr = pos_flow / neg_flow.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + mfr))


# ----------------------------------------------------------------------
# DB-backed signal lookups (cached to keep per-candle latency low)
# ----------------------------------------------------------------------
@lru_cache(maxsize=128)
def _latest_decision_cached(coin: str, bucket: int) -> tuple[float, float, str]:
    """`bucket` is a coarsened timestamp so the cache invalidates every minute."""
    try:
        from database import SessionLocal, SignalLog
        with SessionLocal() as session:
            row = (
                session.query(SignalLog)
                .filter(SignalLog.coin == coin)
                .order_by(SignalLog.ts.desc())
                .first()
            )
        if row is None:
            return 0.0, 0.0, "SKIP"
        return float(row.sentiment_score), float(row.markov_signal), row.decision
    except Exception as exc:  # noqa: BLE001
        log.warning("signal_log lookup failed for %s: %s", coin, exc)
        return 0.0, 0.0, "SKIP"


def _latest_decision(coin: str) -> tuple[float, float, str]:
    bucket = int(datetime.now(timezone.utc).timestamp() // 60)
    return _latest_decision_cached(coin, bucket)


@lru_cache(maxsize=128)
def _latest_regime_cached(coin: str, bucket: int) -> str:
    try:
        from database import RegimeState, SessionLocal
        with SessionLocal() as session:
            row = (
                session.query(RegimeState)
                .filter(RegimeState.coin == coin)
                .order_by(RegimeState.ts.desc())
                .first()
            )
        return row.regime if row else "Sideways"
    except Exception as exc:  # noqa: BLE001
        log.warning("regime_states lookup failed for %s: %s", coin, exc)
        return "Sideways"


def _latest_regime(coin: str) -> str:
    bucket = int(datetime.now(timezone.utc).timestamp() // 60)
    return _latest_regime_cached(coin, bucket)


@lru_cache(maxsize=128)
def _latest_position_size_pct_cached(coin: str, bucket: int) -> Optional[float]:
    """Latest position_size_pct from signal_log for this coin. None if SKIP/missing."""
    try:
        from database import SessionLocal, SignalLog
        with SessionLocal() as session:
            row = (
                session.query(SignalLog)
                .filter(SignalLog.coin == coin)
                .order_by(SignalLog.ts.desc())
                .first()
            )
        if row is None or row.decision == "SKIP":
            return None
        return float(row.position_size_pct)
    except Exception as exc:  # noqa: BLE001
        log.warning("signal_log position_size_pct lookup failed for %s: %s", coin, exc)
        return None


def _latest_position_size_pct(coin: str) -> Optional[float]:
    bucket = int(datetime.now(timezone.utc).timestamp() // 60)
    return _latest_position_size_pct_cached(coin, bucket)
