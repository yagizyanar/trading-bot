"""Tests for the live-vs-backtest drift monitor metrics + alert logic (item 8)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analytics.drift import (
    EXPECTED_COST_BPS_RT, NEG_PROFIT_STREAK_ALERT,
    compute_metrics, evaluate_alerts, format_alert,
)

NOW = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)


def _t(day_offset, profit, pnl_abs=None, stake=100.0, fee_pct=0.02):
    """Synthetic closed trade `day_offset` days before NOW."""
    d = NOW - timedelta(days=day_offset)
    return {
        "close_date": d.strftime("%Y-%m-%d %H:%M:%S"),
        "close_profit": profit,
        "close_profit_abs": pnl_abs if pnl_abs is not None else profit * stake,
        "stake_amount": stake,
        "fee_open_cost": stake * fee_pct / 100.0,
        "fee_close_cost": stake * fee_pct / 100.0,
        "funding_fees": 0.0,
    }


def test_empty_trades_yield_none_and_no_alerts():
    m = compute_metrics([], 30, NOW)
    assert m.trades == 0
    assert m.rolling_sharpe is None and m.win_rate is None and m.avg_profit_per_trade is None
    assert evaluate_alerts(m, 0) == []


def test_win_rate_and_avg_profit():
    trades = [_t(1, 0.05), _t(2, 0.03), _t(3, -0.02), _t(4, -0.01)]
    m = compute_metrics(trades, 30, NOW)
    assert m.trades == 4
    assert m.win_rate == 0.5
    assert abs(m.avg_profit_per_trade - (0.05 + 0.03 - 0.02 - 0.01) / 4) < 1e-9
    assert m.avg_profit_negative is False


def test_sharpe_positive_for_consistent_wins():
    trades = [_t(i, 0.02, pnl_abs=2.0) for i in range(1, 16)]
    m = compute_metrics(trades, 30, NOW)
    assert m.rolling_sharpe is not None and m.rolling_sharpe > 0


def test_cost_bps_round_trip():
    # 0.02% per side on a $100 stake → 4 bps round-trip
    m = compute_metrics([_t(1, 0.01, stake=100.0, fee_pct=0.02)], 30, NOW)
    assert abs(m.actual_cost_bps - 4.0) < 0.5
    assert m.expected_cost_bps == EXPECTED_COST_BPS_RT


def test_alert_low_sharpe():
    # 20 trades alternating +/-5 on consecutive days → Sharpe ~0 < 0.5
    trades = [_t(i, 0.05 if i % 2 == 0 else -0.05, pnl_abs=(5.0 if i % 2 == 0 else -5.0))
              for i in range(1, 21)]
    m = compute_metrics(trades, 30, NOW)
    assert any("Sharpe" in a for a in evaluate_alerts(m, 0))


def test_alert_low_win_rate():
    trades = [_t(i, -0.02, pnl_abs=-2.0) for i in range(1, 16)]   # 15 losers, 0% win
    m = compute_metrics(trades, 30, NOW)
    assert any("win rate" in a for a in evaluate_alerts(m, 0))


def test_alert_negative_streak_threshold():
    trades = [_t(i, 0.02, pnl_abs=2.0) for i in range(1, 16)]
    m = compute_metrics(trades, 30, NOW)
    assert any("consecutive" in a for a in evaluate_alerts(m, NEG_PROFIT_STREAK_ALERT))
    assert not any("consecutive" in a for a in evaluate_alerts(m, NEG_PROFIT_STREAK_ALERT - 1))


def test_no_sharpe_or_winrate_alert_on_thin_data():
    trades = [_t(i, -0.05, pnl_abs=-5.0) for i in range(1, 5)]    # 4 losers < MIN_TRADES
    m = compute_metrics(trades, 30, NOW)
    alerts = evaluate_alerts(m, 0)
    assert not any(("Sharpe" in a or "win rate" in a) for a in alerts)


def test_avg_profit_negative_flag_set():
    m = compute_metrics([_t(1, -0.02), _t(2, -0.01)], 30, NOW)
    assert m.avg_profit_negative is True


def test_format_alert_smoke():
    m = compute_metrics([_t(i, -0.02, pnl_abs=-2.0) for i in range(1, 16)], 30, NOW)
    msg = format_alert(m, 7, evaluate_alerts(m, 7))
    assert "DRIFT ALERT" in msg and "trades" in msg
