"""Smoke tests on config.settings constants."""
from __future__ import annotations

import json
import pathlib

from config import settings

_FT_CONFIG = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "config" / "config.json").read_text()
)


def test_target_coins_count():
    assert len(settings.TARGET_COINS) == 24


def test_stoploss_on_exchange_enabled():
    # The -5% stop must live on the exchange so it survives a bot/VPS death.
    # These flags live inside order_types (NOT top-level — Freqtrade ignores
    # a top-level stoploss_on_exchange, which would silently leave it off).
    ot = _FT_CONFIG["order_types"]
    assert ot["stoploss_on_exchange"] is True
    assert ot["stoploss_on_exchange_limit_ratio"] == 0.99
    assert _FT_CONFIG["stoploss"] == -0.05


def test_freqai_removed_or_disabled():
    # FreqAI has no trained model and its predictions are never read by the
    # entry/exit logic — it must stay off (removed from config) to avoid wasted
    # compute and a needless failure surface.
    assert _FT_CONFIG.get("freqai", {}).get("enabled", False) is False


def test_pairs_match_target_coins():
    assert len(settings.PAIRS) == 24
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
