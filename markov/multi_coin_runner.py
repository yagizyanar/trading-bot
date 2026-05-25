"""Run Markov regime detection on every target coin and persist results.

Called hourly by the routines layer.
"""
from __future__ import annotations

import logging
from typing import Iterable

from config.settings import TARGET_COINS
from database import RegimeState, SessionLocal

from .regime_detector import RegimeResult, compute_regime_for_coin

log = logging.getLogger(__name__)


def run_all_coins(coins: Iterable[str] = TARGET_COINS) -> dict[str, RegimeResult]:
    """Run regime detection on every coin. Returns coin -> RegimeResult."""
    results: dict[str, RegimeResult] = {}
    for coin in coins:
        try:
            results[coin] = compute_regime_for_coin(coin)
        except Exception as exc:  # noqa: BLE001
            log.warning("regime detection failed for %s: %s", coin, exc)
    return results


def persist_regimes(results: dict[str, RegimeResult]) -> None:
    """Insert results into regime_states table."""
    with SessionLocal() as session:
        for coin, r in results.items():
            row = RegimeState(
                coin=coin,
                ts=r.timestamp,
                regime=r.regime,
                confidence=r.confidence,
                bull_prob=r.bull_prob,
                bear_prob=r.bear_prob,
                sideways_prob=r.sideways_prob,
                markov_signal=r.markov_signal,
            )
            session.merge(row)
        session.commit()
