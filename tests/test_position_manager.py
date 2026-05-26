"""Tests for position-sizing, leverage selection, correlation gate."""
from __future__ import annotations

from risk.position_manager import (
    can_open_position,
    correlation_check,
    decide_leverage,
    pct_capital_for_signal,
)


def test_pct_capital_tiers():
    # Tiers (Video 4 "Option A" — small threshold relaxed from 0.2 to 0.1):
    #   |s| > 0.5 → 5%,  > 0.3 → 3%,  > 0.1 → 1%,  ≤ 0.1 → 0
    assert pct_capital_for_signal(0.6) == 0.05
    assert pct_capital_for_signal(0.4) == 0.03
    assert pct_capital_for_signal(0.15) == 0.01   # was 0.25 in old tier
    assert pct_capital_for_signal(0.10) == 0.0    # boundary — strictly greater
    assert pct_capital_for_signal(0.05) == 0.0
    # symmetric on shorts
    assert pct_capital_for_signal(-0.6) == 0.05
    assert pct_capital_for_signal(-0.15) == 0.01


def test_decide_leverage_requires_both_conditions():
    assert decide_leverage(0.5, "Bull") == 2
    assert decide_leverage(0.5, "Neutral") == 2
    assert decide_leverage(0.25, "Bull") == 1   # sentiment below threshold
    assert decide_leverage(0.5, "Bear") == 1    # wrong regime
    assert decide_leverage(0.5, "Crash") == 1


def test_correlation_blocks_third_l2():
    # POL replaced MATIC in TARGET_COINS / SECTOR_MAP (Polygon rebrand, 2024).
    ok, reason = correlation_check("OP", ["POL", "ARB"])
    assert not ok
    assert "L2" in reason


def test_correlation_allows_different_sectors():
    ok, _ = correlation_check("LINK", ["POL", "ARB"])
    assert ok


def test_can_open_position_respects_caps():
    ok, reason = can_open_position(10, 0.20, "SOL", [], True)
    assert not ok and "Max open positions" in reason

    ok, reason = can_open_position(5, 0.50, "SOL", [], True)
    assert not ok and "Deployed capital" in reason

    ok, reason = can_open_position(5, 0.20, "SOL", [], False)
    assert not ok and "Circuit breaker" in reason

    ok, _ = can_open_position(2, 0.10, "SOL", ["LINK"], True)
    assert ok
