"""Tests for binance_market helpers — pure-logic only (signal mappings)."""
from __future__ import annotations

import pytest

from sentiment.binance_market import (
    _funding_signal,
    _long_short_signal,
)


def test_long_short_signal_bearish_when_ratio_high():
    assert _long_short_signal(1.5) <= 0.0
    assert _long_short_signal(2.0) < -0.4
    # saturates at -1.0
    assert _long_short_signal(5.0) == pytest.approx(-1.0)


def test_long_short_signal_bullish_when_ratio_low():
    assert _long_short_signal(0.7) >= 0.0
    assert _long_short_signal(0.5) > 0.4
    # saturates at +1.0
    assert _long_short_signal(0.1) == pytest.approx(1.0)


def test_long_short_signal_neutral_zone():
    assert _long_short_signal(1.0) == 0.0
    assert _long_short_signal(1.2) == 0.0
    assert _long_short_signal(0.8) == 0.0


def test_funding_signal_bearish_when_positive():
    assert _funding_signal(0.0001) <= 0.0   # at threshold
    assert _funding_signal(0.0005) < -0.4   # half saturated
    assert _funding_signal(0.01) == pytest.approx(-1.0)


def test_funding_signal_bullish_when_negative():
    assert _funding_signal(-0.0001) >= 0.0
    assert _funding_signal(-0.0005) > 0.4
    assert _funding_signal(-0.01) == pytest.approx(1.0)


def test_funding_signal_neutral():
    assert _funding_signal(0.0) == 0.0
    assert _funding_signal(0.00005) == 0.0
    assert _funding_signal(-0.00005) == 0.0
