"""Tests for position-sizing, leverage selection, correlation gate."""
from __future__ import annotations

from config.settings import MAX_LEVERAGE
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
    # When both conditions hold (sentiment > threshold AND Bull/Neutral regime),
    # leverage is the configured ceiling MAX_LEVERAGE; otherwise always 1x.
    # Asserting against MAX_LEVERAGE (rather than a literal 2) keeps this green
    # whether the ceiling is 1 (the 2026-06-03 go-live default) or 2.
    assert decide_leverage(0.5, "Bull") == MAX_LEVERAGE
    assert decide_leverage(0.5, "Neutral") == MAX_LEVERAGE
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
    ok, reason = can_open_position(15, 0.20, "SOL", [], True)
    assert not ok and "Max open positions" in reason

    ok, reason = can_open_position(5, 0.80, "SOL", [], True)
    assert not ok and "Deployed capital" in reason

    ok, reason = can_open_position(5, 0.20, "SOL", [], False)
    assert not ok and "Circuit breaker" in reason

    ok, _ = can_open_position(2, 0.10, "SOL", ["LINK"], True)
    assert ok
