"""Memory file I/O — the bot's only persistent context.

Contract (from memory-architecture skill):
- Every routine reads all four files at start.
- Every routine writes the relevant files at end.
- trade_log.md, lessons_learned.md, strategy_notes.md are APPEND-ONLY.
- market_context.md is OVERWRITE (always reflects "now").
- Files must exist; if missing, create from template.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import MEMORY_DIR

TRADE_LOG = MEMORY_DIR / "trade_log.md"
LESSONS = MEMORY_DIR / "lessons_learned.md"
MARKET_CONTEXT = MEMORY_DIR / "market_context.md"
STRATEGY_NOTES = MEMORY_DIR / "strategy_notes.md"

ALL_FILES = (TRADE_LOG, LESSONS, MARKET_CONTEXT, STRATEGY_NOTES)


@dataclass(frozen=True)
class MemorySnapshot:
    trade_log: str
    lessons_learned: str
    market_context: str
    strategy_notes: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.stem.replace('_', ' ').title()}\n\n---\n", encoding="utf-8")


def read_one(path: Path) -> str:
    _ensure_exists(path)
    return path.read_text(encoding="utf-8")


def read_all() -> MemorySnapshot:
    """Read all four memory files. Call at the START of every routine."""
    return MemorySnapshot(
        trade_log=read_one(TRADE_LOG),
        lessons_learned=read_one(LESSONS),
        market_context=read_one(MARKET_CONTEXT),
        strategy_notes=read_one(STRATEGY_NOTES),
    )


def _append(path: Path, block: str) -> None:
    _ensure_exists(path)
    with path.open("a", encoding="utf-8") as f:
        if not block.startswith("\n"):
            f.write("\n")
        f.write(block.rstrip() + "\n")


def append_trade(
    coin: str,
    direction: str,
    entry: float,
    exit_price: Optional[float],
    quantity: float,
    leverage: int,
    pnl_usd: Optional[float],
    pnl_pct: Optional[float],
    reason_in: str,
    reason_out: str,
    outcome: str,
    ts: Optional[str] = None,
) -> None:
    """Append one trade entry to trade_log.md."""
    ts = ts or _utc_now()
    exit_str = f"${exit_price:.4f}" if exit_price is not None else "OPEN"
    pnl_str = (
        f"{pnl_usd:+.2f} ({pnl_pct:+.2%})"
        if pnl_usd is not None and pnl_pct is not None
        else "pending"
    )
    block = (
        f"\n## {ts}\n"
        f"Coin: {coin}\n"
        f"Direction: {direction}\n"
        f"Entry: ${entry:.4f}\n"
        f"Exit: {exit_str}\n"
        f"Quantity: {quantity:.6f}\n"
        f"Leverage: {leverage}x\n"
        f"P&L: {pnl_str}\n"
        f"Reason In: {reason_in}\n"
        f"Reason Out: {reason_out}\n"
        f"Outcome: {outcome}\n"
    )
    _append(TRADE_LOG, block)


def append_lesson(
    observation: str,
    signal_involved: str,
    worked_or_failed: str,
    action_next_time: str,
    ts: Optional[str] = None,
) -> None:
    """Append one observation to lessons_learned.md."""
    ts = ts or _utc_now()
    block = (
        f"\n## {ts}\n"
        f"Observation: {observation}\n"
        f"Signal involved: {signal_involved}\n"
        f"Worked or failed: {worked_or_failed}\n"
        f"Action next time: {action_next_time}\n"
    )
    _append(LESSONS, block)


def overwrite_market_context(
    regime: str,
    regime_confidence: float,
    fear_greed: Optional[int],
    overall_sentiment: str,
    active_positions: int,
    portfolio_value: float,
    deployed_pct: float,
    daily_pnl_usd: float,
    daily_pnl_pct: float,
    weekly_pnl_usd: float,
    weekly_pnl_pct: float,
    drawdown_pct: float,
    circuit_breaker_state: str,
) -> None:
    """Overwrite market_context.md with the latest snapshot."""
    content = (
        f"# Market Context\n\n"
        f"Last Updated: {_utc_now()}\n"
        f"Current Regime: {regime}\n"
        f"Regime Confidence: {regime_confidence:.1%}\n"
        f"Fear & Greed Index: {fear_greed if fear_greed is not None else 'unknown'}\n"
        f"Overall Sentiment: {overall_sentiment}\n"
        f"Active Positions: {active_positions}\n"
        f"Portfolio Status: ${portfolio_value:.2f} total, {deployed_pct:.1%} deployed\n"
        f"Daily P&L: ${daily_pnl_usd:+.2f} ({daily_pnl_pct:+.2%})\n"
        f"Weekly P&L: ${weekly_pnl_usd:+.2f} ({weekly_pnl_pct:+.2%})\n"
        f"Drawdown from Peak: {drawdown_pct:.2%}\n"
        f"Circuit Breaker State: {circuit_breaker_state}\n"
    )
    MARKET_CONTEXT.parent.mkdir(parents=True, exist_ok=True)
    MARKET_CONTEXT.write_text(content, encoding="utf-8")


def append_strategy_note(
    change: str,
    reason: str,
    expected_impact: str,
    validation_plan: str,
    ts: Optional[str] = None,
) -> None:
    """Append one strategy decision to strategy_notes.md."""
    ts = ts or _utc_now()
    block = (
        f"\n## {ts}\n"
        f"Change: {change}\n"
        f"Reason: {reason}\n"
        f"Expected impact: {expected_impact}\n"
        f"Validation plan: {validation_plan}\n"
    )
    _append(STRATEGY_NOTES, block)


def log_circuit_breaker(
    level: str,
    trigger: str,
    equity: float,
    extra: Optional[str] = None,
) -> None:
    """Log a circuit-breaker event into trade_log.md with the [CIRCUIT BREAKER] prefix."""
    ts = _utc_now()
    block = (
        f"\n## {ts} [CIRCUIT BREAKER]\n"
        f"Level: {level}\n"
        f"Trigger: {trigger}\n"
        f"Equity at trigger: ${equity:.2f}\n"
    )
    if extra:
        block += f"Notes: {extra}\n"
    _append(TRADE_LOG, block)
