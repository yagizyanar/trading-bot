"""Tests for backtest metrics, benchmarks, walk-forward."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.benchmarks import buy_and_hold, random_entry, sma_200
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


def test_stress_tests_run_all_scenarios(synthetic_uptrend):
    report = run_stress_tests(synthetic_uptrend, "TEST")
    names = [s.name for s in report.scenarios]
    assert "black_swan_-15%" in names
    assert "sustained_bear_30d" in names
    assert "sentiment_outage_48h" in names
    assert len(report.scenarios) == 5
