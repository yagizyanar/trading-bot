"""SQLAlchemy schema for the trading bot.

Tables:
- sentiment_scores : per coin, per hour
- regime_states    : per coin, per hour
- trades           : every entry/exit
- signal_log       : every signal evaluation (whether traded or not)
- circuit_breaker_events
- performance_snapshots
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    coin: Mapped[str] = mapped_column(String(16), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fear_greed: Mapped[Optional[int]] = mapped_column(Integer)
    senticrypt: Mapped[Optional[float]] = mapped_column(Float)
    news_score: Mapped[Optional[float]] = mapped_column(Float)
    volume_anomaly: Mapped[Optional[float]] = mapped_column(Float)
    yfinance_change: Mapped[Optional[float]] = mapped_column(Float)
    unified: Mapped[float] = mapped_column(Float, nullable=False)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)  # BULLISH/BEARISH/NEUTRAL

    __table_args__ = (
        UniqueConstraint("coin", "ts", name="uq_sentiment_coin_ts"),
        Index("ix_sentiment_coin_ts", "coin", "ts"),
    )


class RegimeState(Base):
    __tablename__ = "regime_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    coin: Mapped[str] = mapped_column(String(16), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    regime: Mapped[str] = mapped_column(String(16), nullable=False)  # Bull/Bear/Sideways/Crash/Euphoria
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bull_prob: Mapped[float] = mapped_column(Float, nullable=False)
    bear_prob: Mapped[float] = mapped_column(Float, nullable=False)
    sideways_prob: Mapped[float] = mapped_column(Float, nullable=False)
    markov_signal: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("coin", "ts", name="uq_regime_coin_ts"),
        Index("ix_regime_coin_ts", "coin", "ts"),
    )


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    coin: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # LONG/SHORT
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[Optional[float]] = mapped_column(Float)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    pnl_usd: Mapped[Optional[float]] = mapped_column(Float)
    pnl_pct: Mapped[Optional[float]] = mapped_column(Float)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    reason_in: Mapped[str] = mapped_column(Text, nullable=False)
    reason_out: Mapped[Optional[str]] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(8), nullable=False, default="OPEN")  # WIN/LOSS/OPEN
    is_paper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_trade_coin_entry_ts", "coin", "entry_ts"),
        Index("ix_trade_outcome", "outcome"),
    )


class SignalLog(Base):
    __tablename__ = "signal_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    coin: Mapped[str] = mapped_column(String(16), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    markov_signal: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    technical_signal: Mapped[str] = mapped_column(String(16), nullable=False)  # BULL/BEAR/NEUTRAL
    decision: Mapped[str] = mapped_column(String(16), nullable=False)  # LONG/SHORT/SKIP
    position_size_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    skip_reason: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_signallog_coin_ts", "coin", "ts"),
        Index("ix_signallog_decision", "decision"),
    )


class CircuitBreakerEvent(Base):
    __tablename__ = "circuit_breaker_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    level: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger: Mapped[str] = mapped_column(Text, nullable=False)
    equity_at_trigger: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (Index("ix_cb_ts", "ts"),)


class PerformanceSnapshot(Base):
    __tablename__ = "performance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    total_equity: Mapped[float] = mapped_column(Float, nullable=False)
    daily_pnl_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    daily_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weekly_pnl_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weekly_pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_equity: Mapped[float] = mapped_column(Float, nullable=False)
    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    deployed_capital_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (Index("ix_perf_ts", "ts"),)
