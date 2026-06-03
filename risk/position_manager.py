"""Position rules: correlation, leverage choice, signal-to-size mapping."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from config.settings import (
    LEVERAGE_SENTIMENT_THRESHOLD,
    MAX_CAPITAL_DEPLOYED_PCT,
    MAX_LEVERAGE,
    MAX_OPEN_POSITIONS,
    SECTOR_MAP,
    SIGNAL_FULL_PCT,
    SIGNAL_MEDIUM_PCT,
    SIGNAL_SMALL_PCT,
)


CORRELATED_SECTOR_LIMIT = 3


def pct_capital_for_signal(signal_strength: float) -> float:
    """Map |signal| to % of capital. Below 0.1 returns 0.0 (skip).

    Tiers (Video 4 "Option A" — pure Markov sizing):
      |s| > 0.5  → 5% (full)
      |s| > 0.3  → 3% (medium)
      |s| > 0.1  → 1% (small)  ← was 0.2; relaxed so the bot actually trades
    """
    s = abs(signal_strength)
    if s > 0.5:
        return SIGNAL_FULL_PCT
    if s > 0.3:
        return SIGNAL_MEDIUM_PCT
    if s > 0.1:
        return SIGNAL_SMALL_PCT
    return 0.0


def decide_leverage(sentiment_score: float, regime: str) -> int:
    """2x only when sentiment > +0.3 AND regime is Bull or Neutral. Else 1x."""
    if sentiment_score > LEVERAGE_SENTIMENT_THRESHOLD and regime in ("Bull", "Neutral"):
        return MAX_LEVERAGE
    return 1


def net_beta_allows(current_net_beta: float, contribution: float, budget: float) -> bool:
    """Item 6: allow a new position unless it pushes |net beta| past `budget`
    in the WORSENING direction.

    `current_net_beta` and `contribution` are signed full-position-equivalents
    weighted by beta-to-BTC (LONG +, SHORT −). A position that REDUCES an
    already-over-budget imbalance is always allowed (it diversifies the book).
    """
    prospective = current_net_beta + contribution
    if abs(prospective) <= budget:
        return True
    return abs(prospective) < abs(current_net_beta)


def correlation_check(
    candidate_coin: str,
    open_coins: Iterable[str],
    sector_limit: int = CORRELATED_SECTOR_LIMIT,
) -> tuple[bool, Optional[str]]:
    """Return (allowed, reason). Block if opening would push the candidate's
    sector count to >= sector_limit (default 3)."""
    sector = SECTOR_MAP.get(candidate_coin)
    if sector is None:
        return True, None
    counts = Counter(SECTOR_MAP.get(c) for c in open_coins)
    if counts.get(sector, 0) >= sector_limit - 1:
        return False, f"Already {counts.get(sector, 0)} open positions in sector {sector}"
    return True, None


def can_open_position(
    open_position_count: int,
    deployed_capital_pct: float,
    candidate_coin: str,
    open_coins: Iterable[str],
    allow_new_positions: bool,
) -> tuple[bool, Optional[str]]:
    """Aggregate gate: portfolio caps + correlation + circuit-breaker-allowed."""
    if not allow_new_positions:
        return False, "Circuit breaker forbids new positions"
    if open_position_count >= MAX_OPEN_POSITIONS:
        return False, f"Max open positions reached ({MAX_OPEN_POSITIONS})"
    if deployed_capital_pct >= MAX_CAPITAL_DEPLOYED_PCT:
        return False, f"Deployed capital {deployed_capital_pct:.0%} >= cap {MAX_CAPITAL_DEPLOYED_PCT:.0%}"
    ok, reason = correlation_check(candidate_coin, open_coins)
    if not ok:
        return False, reason
    return True, None
