"""Tests for backtest metrics, benchmarks, walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.benchmarks import buy_and_hold, random_entry, sma_200
from backtest.daily_walk_forward import (
    always_short_returns,
    equal_weight_portfolio,
    momentum_position,
    momentum_zscore,
    simulate_coin,
)
from backtest.metrics import compute_metrics, meets_minimum_requirements
from backtest.stress_tests import run_stress_tests
from backtest.walk_forward import run_walk_forward


@pytest.fixture
def synthetic_uptrend():
    n = 800
    rng = np.random.default_rng(42)
    returns = 0.0006 + 0.01 * rng.standard_normal(n)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(100.0 * (1.0 + returns).cumprod(), index=idx)


def test_compute_metrics_empty():
    m = compute_metrics([])
    assert m.n_trades == 0


def test_compute_metrics_basic():
    trade_returns = [0.10, -0.05, 0.08, -0.03, 0.12, -0.04]
    daily_returns = [0.001, -0.0005, 0.0008, -0.0003, 0.0012, -0.0004] * 20
    m = compute_metrics(trade_returns, daily_returns)
    assert m.n_trades == len(trade_returns)
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor > 0
    assert m.avg_win > 0
    assert m.avg_loss < 0


def test_minimum_requirements_failure_reasons():
    weak = compute_metrics([0.01] * 10 + [-0.04] * 10)
    ok, fails = meets_minimum_requirements(weak)
    assert not ok
    assert any("win_rate" in f or "profit_factor" in f or "sharpe" in f or "n_trades" in f for f in fails)


def test_buy_and_hold_runs(synthetic_uptrend):
    res = buy_and_hold(synthetic_uptrend)
    assert res.name == "buy_and_hold"
    assert len(res.daily_returns) > 0


def test_sma_200_runs(synthetic_uptrend):
    res = sma_200(synthetic_uptrend)
    assert res.name == "sma_200"


def test_random_entry_deterministic_with_seed(synthetic_uptrend):
    a = random_entry(synthetic_uptrend, seed=7)
    b = random_entry(synthetic_uptrend, seed=7)
    assert list(a.daily_returns.values) == list(b.daily_returns.values)


def test_walk_forward_runs(synthetic_uptrend):
    res = run_walk_forward(synthetic_uptrend, coin="TEST")
    assert res.coin == "TEST"
    assert res.in_sample_days == 252
    assert res.out_sample_days == 180


def test_walk_forward_short_series_degrades(synthetic_uptrend):
    short = synthetic_uptrend.iloc[:100]
    res = run_walk_forward(short, coin="TEST")
    assert res.n_windows == 0


def test_daily_walk_forward_runs(synthetic_uptrend):
    res = simulate_coin(synthetic_uptrend, coin="TEST", in_sample=252)
    assert res.coin == "TEST"
    assert len(res.net_daily) > 0
    assert len(res.positions) == len(res.net_daily)


def test_daily_walk_forward_short_series_degrades():
    short = pd.Series(100.0 + np.arange(50), dtype=float)
    res = simulate_coin(short, coin="TEST", in_sample=252)
    assert len(res.net_daily) == 0
    assert res.metrics.n_trades == 0


def test_daily_walk_forward_positions_are_causal(synthetic_uptrend):
    """No look-ahead: truncating FUTURE data must not change EARLIER positions."""
    full = simulate_coin(synthetic_uptrend, in_sample=252)
    truncated = simulate_coin(synthetic_uptrend.iloc[:600], in_sample=252)
    k = len(truncated.positions)
    assert k > 0
    assert full.positions.iloc[:k].tolist() == truncated.positions.iloc[:k].tolist()


def test_daily_walk_forward_costs_reduce_returns(synthetic_uptrend):
    cheap = simulate_coin(synthetic_uptrend, in_sample=252, cost_per_side=0.0)
    pricey = simulate_coin(synthetic_uptrend, in_sample=252, cost_per_side=0.005)
    assert cheap.net_daily.sum() >= pricey.net_daily.sum()


def test_equal_weight_portfolio_smoke(synthetic_uptrend):
    rng = np.random.default_rng(1)
    closes = {}
    for name in ("A", "B", "C"):
        r = 0.0004 + 0.012 * rng.standard_normal(800)
        closes[name] = pd.Series(100.0 * (1 + r).cumprod(), dtype=float)
    port = equal_weight_portfolio(closes, ["A", "B", "C"], in_sample=252, cap=2)
    assert port is not None and len(port) > 0


def test_always_short_returns_sign(synthetic_uptrend):
    short = always_short_returns(synthetic_uptrend, in_sample=252)
    long_daily = synthetic_uptrend.pct_change().fillna(0.0).reset_index(drop=True)[252:]
    # short return ≈ negated long return (minus the one entry cost on day 0)
    assert short.iloc[5] == pytest.approx(-long_daily.reset_index(drop=True).iloc[5], abs=1e-9)


def test_momentum_zscore_sign_tracks_trend():
    up = pd.Series(100.0 * (1.01 ** np.arange(60)), dtype=float)     # steady uptrend
    down = pd.Series(100.0 * (0.99 ** np.arange(60)), dtype=float)   # steady downtrend
    assert momentum_zscore(up, window=20).iloc[-1] > 0
    assert momentum_zscore(down, window=20).iloc[-1] < 0


def test_momentum_position_directionality():
    up = pd.Series(100.0 * (1.01 ** np.arange(60)), dtype=float)
    down = pd.Series(100.0 * (0.99 ** np.arange(60)), dtype=float)
    assert momentum_position(up, window=20, gate_z=0.5) == 1.0
    assert momentum_position(down, window=20, gate_z=0.5) == -1.0


def test_momentum_position_sized_is_continuous_and_bounded():
    up = pd.Series(100.0 * (1.02 ** np.arange(60)), dtype=float)
    p = momentum_position(up, window=20, gate_z=0.3, sized=True, z_scale=2.0)
    assert 0.0 < p <= 1.0          # conviction-weighted, capped at 1


def test_simulate_coin_accepts_momentum_signal(synthetic_uptrend):
    res = simulate_coin(synthetic_uptrend, coin="MOM", in_sample=252,
                        signal_fn=lambda tc: momentum_position(tc, sized=True))
    assert res.coin == "MOM"
    assert len(res.net_daily) > 0


def test_stress_tests_run_all_scenarios(synthetic_uptrend):
    report = run_stress_tests(synthetic_uptrend, "TEST")
    names = [s.name for s in report.scenarios]
    assert "black_swan_-15%" in names
    assert "sustained_bear_30d" in names
    assert "sentiment_outage_48h" in names
    assert len(report.scenarios) == 5
