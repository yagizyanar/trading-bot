"""Tests for BTC-beta estimation + the net-beta gate (Item 6)."""
from __future__ import annotations

import numpy as np
import pytest

from risk.beta import compute_beta_to_btc
from risk.position_manager import net_beta_allows

# 35 deterministic daily returns (no RNG — tests must be reproducible)
_R = np.array([
    0.01, -0.02, 0.015, -0.01, 0.02, -0.03, 0.025, -0.005, 0.01, -0.015,
    0.02, -0.01, 0.03, -0.02, 0.005, 0.01, -0.025, 0.02, -0.01, 0.015,
    0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.02, -0.01, 0.015, -0.02,
    0.01, 0.02, -0.015, 0.01, -0.02,
])


def _prices(returns):
    return list(100.0 * np.cumprod(1.0 + np.asarray(returns)))


def test_beta_doubled_returns_is_two():
    # coin moves exactly 2x BTC each day → corr 1, vol ratio 2 → beta 2.
    assert compute_beta_to_btc(_prices(2.0 * _R), _prices(_R)) == pytest.approx(2.0, abs=0.05)


def test_beta_inverse_returns_is_negative_one():
    assert compute_beta_to_btc(_prices(-1.0 * _R), _prices(_R)) == pytest.approx(-1.0, abs=0.05)


def test_beta_insufficient_data_defaults_one():
    assert compute_beta_to_btc([100, 101, 102], [100, 101, 102]) == 1.0


def test_beta_flat_btc_defaults_one():
    assert compute_beta_to_btc(_prices(_R), [100.0] * 35) == 1.0   # zero BTC vol


def test_net_beta_allows_within_budget():
    assert net_beta_allows(0.0, 1.0, 3.0) is True
    assert net_beta_allows(2.0, 0.9, 3.0) is True      # 2.9 <= 3.0


def test_net_beta_blocks_worsening_over_budget():
    assert net_beta_allows(2.5, 1.0, 3.0) is False     # -> 3.5, worsens past budget
    assert net_beta_allows(-2.8, -0.5, 3.0) is False   # -> -3.3, worsens


def test_net_beta_allows_reducing_over_budget():
    # already over budget, but the new trade diversifies (reduces |net|)
    assert net_beta_allows(3.5, -1.0, 3.0) is True      # -> 2.5
    assert net_beta_allows(-3.5, 1.0, 3.0) is True      # -> -2.5
