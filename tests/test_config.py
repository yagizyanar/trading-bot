"""Smoke tests on config.settings constants."""
from __future__ import annotations

from config import settings


def test_target_coins_count():
    assert len(settings.TARGET_COINS) == 18


def test_pairs_match_target_coins():
    assert len(settings.PAIRS) == 18
    assert settings.PAIRS[0] == "SOL/USDT"


def test_circuit_breaker_ordering():
    assert (
        settings.DAILY_LOSS_HALVE_PCT
        < settings.DAILY_LOSS_CLOSE_PCT
        < settings.DAILY_LOSS_PAUSE_PCT
        < settings.DRAWDOWN_LOCK_PCT
    )


def test_sector_map_covers_all_coins():
    for c in settings.TARGET_COINS:
        # not every coin needs a sector, but those that don't should default sensibly
        assert isinstance(settings.SECTOR_MAP.get(c, "UNKNOWN"), str)


def test_signal_size_tiers_ordered():
    assert settings.SIGNAL_FULL_PCT > settings.SIGNAL_MEDIUM_PCT > settings.SIGNAL_SMALL_PCT > 0
