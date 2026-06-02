"""Faithful daily walk-forward backtest with transaction costs.

WHY THIS EXISTS (vs `walk_forward.py`):
The original `run_walk_forward` trains once per window, takes a single position,
and holds it static for the entire out-of-sample window (~180 days). That yields
~2-3 "trades" per coin and models zero transaction costs — useful for studying
regime *persistence*, but it does NOT represent a strategy that re-evaluates
frequently. This module:

  - re-evaluates the Markov signal EVERY day on a rolling in-sample window
  - takes long / short / flat via the same ±gate the live bot uses
  - deducts transaction costs on every position change (entry / exit / flip)
  - approximates the live hard stop (-5%) on daily bars
  - exposes a multi-coin equal-weight portfolio with an optional position cap

NO LOOK-AHEAD: the position applied to day t is decided from data through t-1
only. `markov._skill_fallback.label_regimes` uses a *trailing* rolling return,
so labels never peek forward. `test_daily_walk_forward.py` asserts causality
(truncating future data doesn't change earlier positions).

IMPORTANT INTERPRETATION NOTE (2026-06-02): a controlled window-length sweep
(evaluating 252/365/540/730 on identical out-of-sample days) found the in-sample
window length is roughly immaterial — Sharpe ~2.3 on majors regardless. An
earlier uncontrolled comparison wrongly attributed a Sharpe drop to the window;
it was actually a different evaluation period. Always compare windows on the
SAME days (use `eval_start`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from config.settings import BACKTEST_IN_SAMPLE_DAYS, MARKOV_THRESHOLD, MARKOV_WINDOW
from markov.regime_detector import (
    build_transition_matrix,
    label_regimes,
    signal_from_matrix,
)

from .metrics import PerformanceMetrics, compute_metrics

DEFAULT_COST_PER_SIDE = 0.0006   # 0.04% Binance taker + ~0.02% slippage
DEFAULT_SIGNAL_GATE = 0.2        # matches live: signal > +gate long, < -gate short
DEFAULT_STOP = 0.05              # live hard stop
PERIODS_PER_YEAR = 365           # crypto trades 365 d/yr


# ---------------------------------------------------------------------------
# Pluggable signal functions. Each takes the in-sample close series (data
# through t-1 only) and returns a target position in [-1, +1].
# ---------------------------------------------------------------------------
def markov_position(train_close: pd.Series, gate: float = DEFAULT_SIGNAL_GATE) -> float:
    """Current live signal: quantized ±1/0 from the Markov transition matrix.

    NOTE: this collapses to sign-of-trailing-20d-return. The matrix is
    diagonal-dominant (consecutive rolling-return labels share 19/20 days),
    so P(Bear|Bear)≈0.9 for every coin → the signal magnitude is a near-
    constant ±0.87 carrying no per-coin information. See module + edge memo.
    """
    labels = label_regimes(train_close, window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
    if labels.empty:
        return 0.0
    P = build_transition_matrix(labels)
    sig = float(signal_from_matrix(P, int(labels.iloc[-1])))
    return 1.0 if sig > gate else (-1.0 if sig < -gate else 0.0)


def momentum_zscore(close: pd.Series, window: int = MARKOV_WINDOW) -> pd.Series:
    """Trailing W-day return expressed in units of W-day volatility.

    Continuous and vol-normalized — a 2% move means very different things for
    a 1%-vol coin vs a 10%-vol coin. Unlike the raw-return threshold the Markov
    label uses, this is comparable across coins, so it both filters noise on
    high-vol names and differentiates conviction per coin.
    """
    ret_w = close.pct_change(window)
    vol_w = close.pct_change().rolling(window).std() * math.sqrt(window)
    return ret_w / vol_w.replace(0.0, np.nan)


def momentum_position(
    train_close: pd.Series,
    window: int = MARKOV_WINDOW,
    gate_z: float = 0.5,
    sized: bool = False,
    z_scale: float = 2.0,
) -> float:
    """Continuous vol-normalized momentum signal.

    sized=False → direction-only ±1/0 (apples-to-apples vs markov_position).
    sized=True  → conviction-weighted position clip(z / z_scale, -1, +1),
                  zeroed inside the ±gate_z dead-zone.
    """
    z = momentum_zscore(train_close, window)
    if z.empty or pd.isna(z.iloc[-1]):
        return 0.0
    zv = float(z.iloc[-1])
    if sized:
        return float(np.clip(zv / z_scale, -1.0, 1.0)) if abs(zv) >= gate_z else 0.0
    return 1.0 if zv > gate_z else (-1.0 if zv < -gate_z else 0.0)


@dataclass(frozen=True)
class DailyWalkForwardResult:
    coin: str
    in_sample: int
    cost_per_side: float
    positions: pd.Series      # position held INTO each out-of-sample day (-1/0/+1)
    net_daily: pd.Series      # daily returns after cost
    gross_daily: pd.Series    # daily returns before cost
    n_flips: int
    metrics: PerformanceMetrics


def simulate_coin(
    close: pd.Series,
    coin: str = "?",
    in_sample: int = BACKTEST_IN_SAMPLE_DAYS,
    cost_per_side: float = DEFAULT_COST_PER_SIDE,
    stop: float = DEFAULT_STOP,
    gate: float = DEFAULT_SIGNAL_GATE,
    signal_fn: Optional[Callable[[pd.Series], float]] = None,
) -> DailyWalkForwardResult:
    """Daily-rebalanced long/short walk-forward for one coin.

    `signal_fn(train_close) -> target position in [-1, +1]`. Defaults to the
    quantized Markov signal (`markov_position`) so behaviour is unchanged.
    Pass `momentum_position`-style callables for the A/B.
    """
    if signal_fn is None:
        signal_fn = lambda tc: markov_position(tc, gate=gate)

    close = close.dropna().reset_index(drop=True)
    rets = close.pct_change().fillna(0.0).to_numpy()
    n = len(close)
    if n < in_sample + 30:
        empty = pd.Series(dtype=float)
        return DailyWalkForwardResult(
            coin, in_sample, cost_per_side, empty, empty, empty, 0, compute_metrics([])
        )

    net = np.full(n, np.nan)
    gross = np.full(n, np.nan)
    posarr = np.zeros(n)
    pos = 0.0
    flips = 0

    for t in range(in_sample, n):
        train = close.iloc[t - in_sample:t]           # data through t-1 only
        target = float(signal_fn(train))

        day_ret = rets[t] * pos
        # Hard-stop on the UNDERLYING adverse move (works for fractional pos):
        # if price moved more than `stop` against the position, exit at the stop,
        # loss scaled by position size. For pos=±1 this equals the old behaviour.
        if pos != 0.0 and (-rets[t] * np.sign(pos)) > stop:
            day_ret = -stop * abs(pos)
            target = 0.0

        gross[t] = day_ret
        posarr[t] = pos
        turnover = abs(target - pos)
        if turnover > 0:
            flips += 1
        net[t] = day_ret - cost_per_side * turnover
        pos = target

    net_s = pd.Series(net[in_sample:], dtype=float).reset_index(drop=True)
    gross_s = pd.Series(gross[in_sample:], dtype=float).reset_index(drop=True)
    pos_s = pd.Series(posarr[in_sample:], dtype=float).reset_index(drop=True)
    trade_returns = _segment_trade_returns(posarr[in_sample:], rets[in_sample:], cost_per_side)
    metrics = compute_metrics(
        trade_returns=trade_returns or net_s.tolist(),
        daily_returns=net_s.tolist(),
        periods_per_year=PERIODS_PER_YEAR,
    )
    return DailyWalkForwardResult(
        coin, in_sample, cost_per_side, pos_s, net_s, gross_s, flips, metrics
    )


def _segment_trade_returns(positions: np.ndarray, rets: np.ndarray, cost_per_side: float) -> list[float]:
    """Compress a daily position path into per-trade compounded returns (net of cost)."""
    trades: list[float] = []
    cur_pos = 0.0
    cur_factor = 1.0
    for i in range(len(positions)):
        p = positions[i]
        if p != cur_pos:
            if cur_pos != 0.0:
                cur_factor *= (1.0 - cost_per_side)   # exit cost
                trades.append(cur_factor - 1.0)
            cur_factor = 1.0
            if p != 0.0:
                cur_factor *= (1.0 - cost_per_side)   # entry cost
            cur_pos = p
        if cur_pos != 0.0 and i < len(rets):
            cur_factor *= (1.0 + rets[i] * cur_pos)
    if cur_pos != 0.0:
        trades.append(cur_factor - 1.0)
    return trades


def always_short_returns(close: pd.Series, in_sample: int = BACKTEST_IN_SAMPLE_DAYS,
                         cost_per_side: float = DEFAULT_COST_PER_SIDE) -> pd.Series:
    """Benchmark: permanent short from in_sample onward. The honest comparator
    for a strategy that is mostly short in a downtrend."""
    close = close.dropna().reset_index(drop=True)
    r = -close.pct_change().fillna(0.0).to_numpy()[in_sample:]
    if len(r):
        r = r.copy()
        r[0] -= cost_per_side
    return pd.Series(r, dtype=float).reset_index(drop=True)


def equal_weight_portfolio(
    closes: dict[str, pd.Series],
    universe: Iterable[str],
    in_sample: int = BACKTEST_IN_SAMPLE_DAYS,
    cost_per_side: float = DEFAULT_COST_PER_SIDE,
    stop: float = DEFAULT_STOP,
    gate: float = DEFAULT_SIGNAL_GATE,
    cap: Optional[int] = None,
    eval_start: Optional[int] = None,
    signal_fn: Optional[Callable[[pd.Series], float]] = None,
) -> Optional[pd.Series]:
    """Equal-weight daily-rebalanced portfolio net daily returns.

    `cap` limits simultaneous positions (selected by |20-day momentum| among
    coins with a non-flat signal), mirroring the live max_open_trades cap.
    `eval_start` truncates every coin to a common global day index so different
    in_sample windows are compared on IDENTICAL out-of-sample days.
    `signal_fn` selects the signal (default markov); pass momentum variants for A/B.
    """
    sims: dict[str, tuple] = {}
    for c in universe:
        cl = closes.get(c)
        if cl is None:
            continue
        cl = cl.dropna().reset_index(drop=True)
        need = (eval_start if eval_start is not None else in_sample) + 40
        if len(cl) < need:
            continue
        res = simulate_coin(cl, c, in_sample=in_sample, cost_per_side=0.0,
                            stop=stop, gate=gate, signal_fn=signal_fn)
        mom = cl.pct_change(20).abs().fillna(0.0).to_numpy()
        # align positions/gross to a full-length array indexed from 0
        full_pos = np.zeros(len(cl)); full_gross = np.zeros(len(cl))
        full_pos[in_sample:] = res.positions.to_numpy()
        full_gross[in_sample:] = np.nan_to_num(res.gross_daily.to_numpy())
        sims[c] = (full_pos, full_gross, mom)
    if not sims:
        return None

    coins = list(sims.keys())
    start = eval_start if eval_start is not None else in_sample
    common = min(len(sims[c][0]) - start for c in coins)
    if common <= 0:
        return None
    POS = np.vstack([sims[c][0][start:start + common] for c in coins])
    GROSS = np.vstack([sims[c][1][start:start + common] for c in coins])
    MOM = np.vstack([sims[c][2][start:start + common] for c in coins])

    net = np.zeros(common)
    prev_w = np.zeros(len(coins))
    for d in range(common):
        active = np.where(POS[:, d] != 0.0)[0]
        if cap is not None and len(active) > cap:
            active = active[np.argsort(-MOM[active, d])][:cap]
        w = np.zeros(len(coins))
        g = 0.0
        if len(active) > 0:
            wt = 1.0 / len(active)
            for i in active:
                w[i] = wt * POS[i, d]
                g += wt * GROSS[i, d]
        net[d] = g - cost_per_side * np.abs(w - prev_w).sum()
        prev_w = w
    return pd.Series(net, dtype=float)


def portfolio_sharpe(daily: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    d = daily.dropna().to_numpy()
    if len(d) < 2 or d.std(ddof=1) == 0:
        return 0.0
    return float(d.mean() / d.std(ddof=1) * math.sqrt(periods_per_year))
