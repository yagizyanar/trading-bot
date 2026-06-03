"""Granular circuit breakers.

Levels (from circuit-breakers skill + user PHASE 6):
  NOMINAL        — nothing triggered
  HALVE_SIZES    — daily loss > 2% → cut new position sizes in half
  CLOSE_ALL      — daily loss > 3% → close all positions, no new entries
  PAUSE          — daily loss > 5% → bot enters PAUSED state for the day
  WEEKLY_REDUCE  — weekly loss > 5% → reduce sizes 50% for rest of week
  WEEKLY_STOP    — weekly loss > 8% → close all, no new trades this week
  LOCKED         — 10% drawdown from peak → writes TRADING_LOCKED.txt

Severity ordering: NOMINAL < HALVE_SIZES < WEEKLY_REDUCE < CLOSE_ALL <
                   PAUSE < WEEKLY_STOP < LOCKED
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.settings import (
    DAILY_LOSS_CLOSE_PCT,
    DAILY_LOSS_HALVE_PCT,
    DAILY_LOSS_PAUSE_PCT,
    DRAWDOWN_LOCK_PCT,
    WEEKLY_LOSS_REDUCE_PCT,
    WEEKLY_LOSS_STOP_PCT,
)
from memory.memory_io import log_circuit_breaker
from notifications import send_telegram
from .lockfile import is_locked, write_lockfile


class CircuitBreakerLevel(str, Enum):
    NOMINAL = "NOMINAL"
    HALVE_SIZES = "HALVE_SIZES"
    WEEKLY_REDUCE = "WEEKLY_REDUCE"
    CLOSE_ALL = "CLOSE_ALL"
    PAUSE = "PAUSE"
    WEEKLY_STOP = "WEEKLY_STOP"
    LOCKED = "LOCKED"


_SEVERITY: dict[CircuitBreakerLevel, int] = {
    CircuitBreakerLevel.NOMINAL: 0,
    CircuitBreakerLevel.HALVE_SIZES: 1,
    CircuitBreakerLevel.WEEKLY_REDUCE: 2,
    CircuitBreakerLevel.CLOSE_ALL: 3,
    CircuitBreakerLevel.PAUSE: 4,
    CircuitBreakerLevel.WEEKLY_STOP: 5,
    CircuitBreakerLevel.LOCKED: 6,
}


@dataclass(frozen=True)
class CircuitBreakerState:
    level: CircuitBreakerLevel
    trigger: str
    daily_pnl_pct: float
    weekly_pnl_pct: float
    drawdown_pct: float
    equity: float
    peak_equity: float
    size_multiplier: float
    allow_new_positions: bool
    must_close_all: bool


def _highest(*states: CircuitBreakerLevel) -> CircuitBreakerLevel:
    return max(states, key=lambda s: _SEVERITY[s])


def evaluate_circuit_breakers(
    daily_pnl_pct: float,
    weekly_pnl_pct: float,
    equity: float,
    peak_equity: float,
    log: bool = True,
) -> CircuitBreakerState:
    """Evaluate all tiers and return the most severe active level.

    Arguments are *signed* PnL percentages: -0.025 means -2.5%.
    Drawdown is computed from `equity` and `peak_equity`.
    """
    drawdown_pct = 0.0 if peak_equity <= 0 else max(0.0, (peak_equity - equity) / peak_equity)

    if is_locked():
        state = CircuitBreakerState(
            level=CircuitBreakerLevel.LOCKED,
            trigger="TRADING_LOCKED.txt present",
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            drawdown_pct=drawdown_pct,
            equity=equity,
            peak_equity=peak_equity,
            size_multiplier=0.0,
            allow_new_positions=False,
            must_close_all=False,  # already locked; manual review required
        )
        return state

    triggered: list[tuple[CircuitBreakerLevel, str]] = []

    daily_loss = -daily_pnl_pct
    weekly_loss = -weekly_pnl_pct

    if drawdown_pct >= DRAWDOWN_LOCK_PCT:
        reason = f"Drawdown {drawdown_pct:.2%} >= {DRAWDOWN_LOCK_PCT:.0%}"
        write_lockfile(reason, peak_equity, equity, drawdown_pct)
        if log:
            log_circuit_breaker("LOCKED", reason, equity, "Lockfile written. Manual restart required.")
            send_telegram(
                f"🔴 <b>CIRCUIT BREAKER: LOCKED</b>\n{reason}\n"
                f"Equity ${equity:,.0f} (peak ${peak_equity:,.0f}).\n"
                f"Trading halted — manual restart required."
            )
        return CircuitBreakerState(
            level=CircuitBreakerLevel.LOCKED,
            trigger=reason,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            drawdown_pct=drawdown_pct,
            equity=equity,
            peak_equity=peak_equity,
            size_multiplier=0.0,
            allow_new_positions=False,
            must_close_all=True,
        )

    if weekly_loss >= WEEKLY_LOSS_STOP_PCT:
        triggered.append((CircuitBreakerLevel.WEEKLY_STOP, f"Weekly loss {weekly_loss:.2%}"))
    elif weekly_loss >= WEEKLY_LOSS_REDUCE_PCT:
        triggered.append((CircuitBreakerLevel.WEEKLY_REDUCE, f"Weekly loss {weekly_loss:.2%}"))

    if daily_loss >= DAILY_LOSS_PAUSE_PCT:
        triggered.append((CircuitBreakerLevel.PAUSE, f"Daily loss {daily_loss:.2%}"))
    elif daily_loss >= DAILY_LOSS_CLOSE_PCT:
        triggered.append((CircuitBreakerLevel.CLOSE_ALL, f"Daily loss {daily_loss:.2%}"))
    elif daily_loss >= DAILY_LOSS_HALVE_PCT:
        triggered.append((CircuitBreakerLevel.HALVE_SIZES, f"Daily loss {daily_loss:.2%}"))

    if not triggered:
        return CircuitBreakerState(
            level=CircuitBreakerLevel.NOMINAL,
            trigger="",
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct,
            drawdown_pct=drawdown_pct,
            equity=equity,
            peak_equity=peak_equity,
            size_multiplier=1.0,
            allow_new_positions=True,
            must_close_all=False,
        )

    top_level = _highest(*(lv for lv, _ in triggered))
    top_reason = next(r for lv, r in triggered if lv == top_level)

    size_multiplier = 1.0
    allow_new = True
    must_close = False

    if top_level == CircuitBreakerLevel.HALVE_SIZES:
        size_multiplier = 0.5
    elif top_level == CircuitBreakerLevel.WEEKLY_REDUCE:
        size_multiplier = 0.5
    elif top_level == CircuitBreakerLevel.CLOSE_ALL:
        size_multiplier = 0.0
        allow_new = False
        must_close = True
    elif top_level == CircuitBreakerLevel.PAUSE:
        size_multiplier = 0.0
        allow_new = False
        must_close = True
    elif top_level == CircuitBreakerLevel.WEEKLY_STOP:
        size_multiplier = 0.0
        allow_new = False
        must_close = True

    if log:
        log_circuit_breaker(top_level.value, top_reason, equity)
        send_telegram(
            f"⚠️ <b>Circuit breaker: {top_level.value}</b>\n{top_reason}\n"
            f"Equity ${equity:,.0f} | size×{size_multiplier:.2f} | "
            f"new positions: {'no' if not allow_new else 'yes'}"
        )

    return CircuitBreakerState(
        level=top_level,
        trigger=top_reason,
        daily_pnl_pct=daily_pnl_pct,
        weekly_pnl_pct=weekly_pnl_pct,
        drawdown_pct=drawdown_pct,
        equity=equity,
        peak_equity=peak_equity,
        size_multiplier=size_multiplier,
        allow_new_positions=allow_new,
        must_close_all=must_close,
    )
