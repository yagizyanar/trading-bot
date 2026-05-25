"""Tests for memory I/O helpers — append-only and overwrite contracts."""
from __future__ import annotations

from memory.memory_io import (
    append_lesson,
    append_strategy_note,
    append_trade,
    log_circuit_breaker,
    overwrite_market_context,
    read_all,
)


def test_append_trade_appends_block(tmp_memory_dir):
    append_trade(
        coin="SOL/USDT", direction="LONG", entry=100.0, exit_price=115.0,
        quantity=1.0, leverage=2, pnl_usd=15.0, pnl_pct=0.15,
        reason_in="3-layer match", reason_out="TP hit", outcome="WIN",
    )
    content = (tmp_memory_dir / "trade_log.md").read_text(encoding="utf-8")
    assert "SOL/USDT" in content
    assert "WIN" in content
    assert "+15.00" in content


def test_append_trade_open_position(tmp_memory_dir):
    append_trade(
        coin="AVAX/USDT", direction="SHORT", entry=50.0, exit_price=None,
        quantity=2.0, leverage=1, pnl_usd=None, pnl_pct=None,
        reason_in="x", reason_out="open", outcome="OPEN",
    )
    content = (tmp_memory_dir / "trade_log.md").read_text(encoding="utf-8")
    assert "OPEN" in content
    assert "pending" in content


def test_overwrite_market_context_replaces_file(tmp_memory_dir):
    overwrite_market_context(
        regime="Bull", regime_confidence=0.72, fear_greed=62,
        overall_sentiment="bullish", active_positions=3,
        portfolio_value=10000.0, deployed_pct=0.20,
        daily_pnl_usd=120.0, daily_pnl_pct=0.012,
        weekly_pnl_usd=400.0, weekly_pnl_pct=0.04,
        drawdown_pct=0.03, circuit_breaker_state="NOMINAL",
    )
    first = (tmp_memory_dir / "market_context.md").read_text(encoding="utf-8")
    assert "Bull" in first

    overwrite_market_context(
        regime="Bear", regime_confidence=0.50, fear_greed=24,
        overall_sentiment="bearish", active_positions=0,
        portfolio_value=9500.0, deployed_pct=0.0,
        daily_pnl_usd=-50.0, daily_pnl_pct=-0.005,
        weekly_pnl_usd=-100.0, weekly_pnl_pct=-0.01,
        drawdown_pct=0.05, circuit_breaker_state="HALVE_SIZES",
    )
    second = (tmp_memory_dir / "market_context.md").read_text(encoding="utf-8")
    assert "Bear" in second
    assert "Bull" not in second  # overwrite, not append


def test_read_all_creates_missing_files(tmp_memory_dir):
    snap = read_all()
    assert isinstance(snap.trade_log, str)
    assert isinstance(snap.market_context, str)
    for fname in ("trade_log.md", "lessons_learned.md", "market_context.md", "strategy_notes.md"):
        assert (tmp_memory_dir / fname).exists()


def test_log_circuit_breaker(tmp_memory_dir):
    log_circuit_breaker(level="HALVE_SIZES", trigger="daily -2.5%", equity=9750.0)
    content = (tmp_memory_dir / "trade_log.md").read_text(encoding="utf-8")
    assert "[CIRCUIT BREAKER]" in content
    assert "HALVE_SIZES" in content


def test_append_lesson_and_strategy_note(tmp_memory_dir):
    append_lesson(observation="x", signal_involved="markov", worked_or_failed="WORKED", action_next_time="y")
    append_strategy_note(change="tweak", reason="r", expected_impact="i", validation_plan="p")
    assert "WORKED" in (tmp_memory_dir / "lessons_learned.md").read_text(encoding="utf-8")
    assert "tweak" in (tmp_memory_dir / "strategy_notes.md").read_text(encoding="utf-8")
