"""Position sizing from signal strength + circuit-breaker multiplier."""
from __future__ import annotations

from typing import Optional

from config.settings import (
    SIGNAL_FULL_PCT,
    SIGNAL_MEDIUM_PCT,
    SIGNAL_SMALL_PCT,
)
from risk.position_manager import pct_capital_for_signal


def size_for_signal(
    signal_strength: float,
    capital: float,
    cb_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Return (dollar_amount, pct_of_capital) for the given signal strength.

    `signal_strength` is the absolute Markov signal magnitude.
    `cb_multiplier` is the size_multiplier from the circuit breaker state (1.0/0.5/0.0).
    """
    base_pct = pct_capital_for_signal(signal_strength)
    final_pct = base_pct * cb_multiplier
    dollars = capital * final_pct
    return dollars, final_pct
