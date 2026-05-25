"""Tests for hyperliquid helpers — pure-logic mappings only."""
from __future__ import annotations

import pytest

from sentiment.hyperliquid import _long_pct_to_signal


def test_long_pct_to_signal_at_neutral_50pct():
    assert _long_pct_to_signal(0.50) == 0.0


def test_long_pct_to_signal_bullish_above_60pct():
    assert _long_pct_to_signal(0.60) >= 0.5
    assert _long_pct_to_signal(0.80) > _long_pct_to_signal(0.60)
    assert _long_pct_to_signal(1.00) == pytest.approx(1.0)


def test_long_pct_to_signal_bearish_below_40pct():
    assert _long_pct_to_signal(0.40) <= -0.5
    assert _long_pct_to_signal(0.20) < _long_pct_to_signal(0.40)
    assert _long_pct_to_signal(0.0) == pytest.approx(-1.0)


def test_long_pct_to_signal_intermediate_zone():
    # 40-60% spans -0.5 to +0.5 linearly
    assert _long_pct_to_signal(0.45) < 0.0
    assert _long_pct_to_signal(0.55) > 0.0
    assert -_long_pct_to_signal(0.45) == pytest.approx(_long_pct_to_signal(0.55), abs=1e-9)


def test_long_pct_to_signal_monotone():
    pcts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    signals = [_long_pct_to_signal(p) for p in pcts]
    for a, b in zip(signals, signals[1:]):
        assert a <= b, f"non-monotone: {a} > {b}"
