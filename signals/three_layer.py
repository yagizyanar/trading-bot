"""Three-layer signal evaluation.

Layer 1 → Markov regime (Bull/Bear/Sideways/Crash/Euphoria) + Markov signal
Layer 2 → Unified sentiment score (-1.0 .. +1.0)
Layer 3 → Technical label (BULL/BEAR/NEUTRAL)

A LONG decision requires:
  - markov_signal > +0.2
  - regime in {Bull, Sideways/Neutral} AND regime != Crash
  - sentiment > +0.2
  - technical == BULL

A SHORT decision requires the mirrored conditions.

Anything else → SKIP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import (
    SENTIMENT_BEAR_THRESHOLD,
    SENTIMENT_BULL_THRESHOLD,
)
from markov.regime_detector import RegimeResult
from sentiment.analyzer import UnifiedScore

from .position_sizing import size_for_signal
from .technical import TechnicalSnapshot


@dataclass(frozen=True)
class SignalDecision:
    coin: str
    decision: str             # LONG / SHORT / SKIP
    position_size_pct: float  # fraction of capital
    dollars: float
    leverage: int             # 1 or 2
    markov_signal: float
    sentiment_score: float
    technical_label: str
    regime: str
    reason: str
    skip_reason: Optional[str] = None


def _allowed_long_regime(regime: str) -> bool:
    return regime in ("Bull", "Sideways", "Euphoria")


def _allowed_short_regime(regime: str) -> bool:
    return regime in ("Bear", "Sideways")


def evaluate_signal(
    coin: str,
    regime_result: RegimeResult,
    sentiment: UnifiedScore,
    technical: TechnicalSnapshot,
    capital: float,
    cb_multiplier: float,
    cb_allows_new: bool,
    leverage_chooser=None,
) -> SignalDecision:
    """Return a SignalDecision. Defaults to SKIP unless all 3 layers agree."""
    from risk.position_manager import decide_leverage

    leverage_chooser = leverage_chooser or decide_leverage

    if not cb_allows_new:
        return SignalDecision(
            coin=coin,
            decision="SKIP",
            position_size_pct=0.0,
            dollars=0.0,
            leverage=1,
            markov_signal=regime_result.markov_signal,
            sentiment_score=sentiment.unified,
            technical_label=technical.label,
            regime=regime_result.regime,
            reason="circuit breaker disallows new positions",
            skip_reason="circuit breaker disallows new positions",
        )

    if regime_result.regime == "Crash":
        return SignalDecision(
            coin=coin, decision="SKIP", position_size_pct=0.0, dollars=0.0,
            leverage=1, markov_signal=regime_result.markov_signal,
            sentiment_score=sentiment.unified, technical_label=technical.label,
            regime=regime_result.regime, reason="Crash regime — no new positions",
            skip_reason="Crash regime",
        )

    ms = regime_result.markov_signal
    sent = sentiment.unified
    tech = technical.label

    long_ok = (
        ms > 0.2
        and _allowed_long_regime(regime_result.regime)
        and sent > SENTIMENT_BULL_THRESHOLD
        and tech == "BULL"
    )
    short_ok = (
        ms < -0.2
        and _allowed_short_regime(regime_result.regime)
        and sent < SENTIMENT_BEAR_THRESHOLD
        and tech == "BEAR"
    )

    if long_ok:
        dollars, pct = size_for_signal(ms, capital, cb_multiplier)
        lev = leverage_chooser(sent, regime_result.regime)
        return SignalDecision(
            coin=coin, decision="LONG", position_size_pct=pct, dollars=dollars,
            leverage=lev, markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason=f"markov={ms:+.2f}, sent={sent:+.2f}, tech={tech}, regime={regime_result.regime}",
        )

    if short_ok:
        dollars, pct = size_for_signal(ms, capital, cb_multiplier)
        # Shorts default to 1x leverage regardless of sentiment magnitude
        return SignalDecision(
            coin=coin, decision="SHORT", position_size_pct=pct, dollars=dollars,
            leverage=1, markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason=f"markov={ms:+.2f}, sent={sent:+.2f}, tech={tech}, regime={regime_result.regime}",
        )

    fails: list[str] = []
    if abs(ms) <= 0.2:
        fails.append(f"|markov|={abs(ms):.2f}<=0.2")
    if abs(sent) <= 0.2:
        fails.append(f"|sent|={abs(sent):.2f}<=0.2")
    if tech == "NEUTRAL":
        fails.append("technical=NEUTRAL")
    if not (long_ok or short_ok) and not fails:
        fails.append("layers disagree on direction")

    return SignalDecision(
        coin=coin, decision="SKIP", position_size_pct=0.0, dollars=0.0,
        leverage=1, markov_signal=ms, sentiment_score=sent,
        technical_label=tech, regime=regime_result.regime,
        reason=f"3-layer gate not satisfied: {', '.join(fails)}",
        skip_reason=", ".join(fails),
    )
