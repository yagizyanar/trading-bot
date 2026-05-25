"""Smoke tests on the updated analyzer weight scheme."""
from __future__ import annotations

import pytest

from sentiment.analyzer import _WEIGHTS, _blend


def test_weights_sum_to_one():
    assert sum(_WEIGHTS.values()) == pytest.approx(1.0)


def test_all_six_sources_present():
    expected = {"news", "volume", "yfinance",
                "long_short_ratio", "funding_rate", "hyperliquid"}
    assert set(_WEIGHTS.keys()) == expected


def test_news_is_top_weight():
    # After SentiCrypt's removal, news takes the largest single share.
    assert _WEIGHTS["news"] == max(_WEIGHTS.values())


def test_blend_handles_only_new_sources():
    # If only the three Binance/Hyperliquid sources are available, blend
    # redistributes weight across just them.
    partial = {
        "news": None, "volume": None, "yfinance": None,
        "long_short_ratio": 0.5, "funding_rate": -0.5, "hyperliquid": 0.0,
    }
    out = _blend(partial)
    # Weighted: 0.20*0.5 + 0.15*-0.5 + 0.05*0 = 0.025; total weight = 0.40
    # → 0.025 / 0.40 = 0.0625
    assert out == pytest.approx(0.025 / 0.40)


def test_blend_full_six_sources():
    full = {k: 0.4 for k in _WEIGHTS}
    assert _blend(full) == pytest.approx(0.4)


def test_blend_clips_to_range():
    full = {k: 5.0 for k in _WEIGHTS}      # absurd values
    assert _blend(full) == 1.0
    full_neg = {k: -5.0 for k in _WEIGHTS}
    assert _blend(full_neg) == -1.0
