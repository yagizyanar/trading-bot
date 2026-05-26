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
from typing import Any

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
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    minimal_roi = {"0": 0.15}
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
    def leverage(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_leverage: float, max_leverage: float, side: str, **_: Any,
    ) -> float:
        # Use the Markov-driven leverage rule from signals.three_layer:
        #   - Sideways or Crash regime → 1x
        #   - LONG  in Bull/Euphoria with sentiment > +0.3 → 2x
        #   - SHORT in Bear           with sentiment < -0.3 → 2x
        #   - else → 1x
        from signals.three_layer import _choose_leverage

        coin = pair.split("/")[0]
        sent, _, _ = _latest_decision(coin)
        regime = _latest_regime(coin)
        decision = "SHORT" if str(side).lower() == "short" else "LONG"
        lev = _choose_leverage(decision, sent, regime)
        return float(min(lev, max_leverage))

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
