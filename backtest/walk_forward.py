"""Walk-forward backtest (regime-PERSISTENCE study — NOT the live strategy).

⚠️  For an edge test that reflects a frequently-re-evaluating strategy WITH
    transaction costs, use `backtest.daily_walk_forward` instead. This module
    holds a single position static for the whole out-of-sample window (~180
    days) and models zero costs — it answers "does the regime label persist?",
    not "does the trading strategy have a net edge?". Kept for that narrower
    research question and backwards-compatibility.

Methodology (per backtesting-protocol skill):
  - In-sample window: 252 trading days
  - Out-of-sample window: 6 months (~180 days)
  - At every step, re-estimate Markov matrix on in-sample data only
  - Trade out-of-sample, then roll the window forward
  - Report aggregate metrics across all windows
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_IN_SAMPLE_DAYS,
    BACKTEST_OUT_SAMPLE_DAYS,
    MARKOV_THRESHOLD,
    MARKOV_WINDOW,
)
from markov.regime_detector import (
    build_transition_matrix,
    label_regimes,
    signal_from_matrix,
)

from .metrics import PerformanceMetrics, compute_metrics

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardResult:
    coin: str
    metrics: PerformanceMetrics
    n_windows: int
    in_sample_days: int
    out_sample_days: int
    daily_returns: pd.Series
    equity_curve: pd.Series


def run_walk_forward(
    close: pd.Series,
    coin: str,
    in_sample: int = BACKTEST_IN_SAMPLE_DAYS,
    out_sample: int = BACKTEST_OUT_SAMPLE_DAYS,
    sentiment_signal: Optional[pd.Series] = None,
    sentiment_threshold: float = 0.2,
) -> WalkForwardResult:
    """Run the walk-forward backtest for one coin.

    `sentiment_signal` (optional) is an aligned series of unified sentiment
    scores. When provided, trades only fire if the sentiment confirms
    (long needs > +threshold, short needs < -threshold).
    """
    if label_regimes is None or build_transition_matrix is None:
        log.error("Markov skill missing — walk-forward cannot run")
        return _empty_result(coin, in_sample, out_sample)

    close = close.dropna()
    if len(close) < in_sample + out_sample + 30:
        log.warning("Not enough data for walk-forward on %s (have %d)", coin, len(close))
        return _empty_result(coin, in_sample, out_sample)

    daily_returns = close.pct_change().dropna()
    all_strategy_returns: list[float] = []
    trade_returns: list[float] = []
    window_count = 0

    start = in_sample
    while start + out_sample <= len(close):
        train_close = close.iloc[:start]
        labels = label_regimes(train_close, window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
        if labels.empty:
            start += out_sample
            continue
        P = build_transition_matrix(labels)
        current_state = int(labels.iloc[-1])
        sig = float(signal_from_matrix(P, current_state))

        position = 0.0
        if sig > 0.2:
            if sentiment_signal is None or _avg_window(sentiment_signal, train_close.index) > sentiment_threshold:
                position = 1.0
        elif sig < -0.2:
            if sentiment_signal is None or _avg_window(sentiment_signal, train_close.index) < -sentiment_threshold:
                position = -1.0

        test_returns = daily_returns.iloc[start:start + out_sample]
        strat = (test_returns * position)
        all_strategy_returns.extend(strat.tolist())

        if position != 0.0:
            trade_pl = float((1.0 + strat).prod() - 1.0)
            trade_returns.append(trade_pl)

        window_count += 1
        start += out_sample

    if not all_strategy_returns:
        return _empty_result(coin, in_sample, out_sample)

    series_idx = daily_returns.index[in_sample:in_sample + len(all_strategy_returns)]
    daily_series = pd.Series(all_strategy_returns, index=series_idx, dtype=float)

    metrics = compute_metrics(
        trade_returns=trade_returns or all_strategy_returns,
        daily_returns=all_strategy_returns,
    )
    equity = (1.0 + daily_series.fillna(0.0)).cumprod()

    return WalkForwardResult(
        coin=coin,
        metrics=metrics,
        n_windows=window_count,
        in_sample_days=in_sample,
        out_sample_days=out_sample,
        daily_returns=daily_series,
        equity_curve=equity,
    )


def _avg_window(signal: pd.Series, index: pd.Index) -> float:
    overlap = signal.reindex(index, method="ffill").dropna()
    return float(overlap.tail(30).mean()) if not overlap.empty else 0.0


def _empty_result(coin: str, in_sample: int, out_sample: int) -> WalkForwardResult:
    empty_series = pd.Series(dtype=float)
    return WalkForwardResult(
        coin=coin,
        metrics=compute_metrics([]),
        n_windows=0,
        in_sample_days=in_sample,
        out_sample_days=out_sample,
        daily_returns=empty_series,
        equity_curve=empty_series,
    )
