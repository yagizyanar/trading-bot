"""Markov-driven signal evaluation (Video 4 "Option A" hedge fund method).

REPLACES the original 3-layer all-or-nothing gate. The old rule SKIPped a
trade if any of {Markov, sentiment, technical} didn't strictly agree —
which in practice meant the bot never opened a trade. New logic:

1. **Markov is the sole gate and the base sizer.** Direction + base position
   size come from `markov_signal = P(Bull|s) − P(Bear|s)`:

       |signal| > 0.5  →  5% of capital (full)
       |signal| > 0.3  →  3% of capital (medium)
       |signal| > 0.1  →  1% of capital (small)
       |signal| ≤ 0.1  →  no trade

2. **Sentiment is a multiplier**, not a gate. Aligned with trade direction
   (positive for LONG, negative for SHORT):

       aligned > +0.2   →  ×1.00  (keep)
       aligned 0..+0.2  →  ×0.75  (reduce 25%)
       aligned -0.2..0  →  ×0.50  (reduce 50%)
       aligned ≤ -0.2   →  ×0.25  (reduce 75%)

3. **Technical is a soft confirmation**, not a gate. Contradiction
   (BULL tech on a SHORT trade or vice versa) shaves another 25% off.

4. **Sideways regime**: trades still fire, but position size is halved
   and leverage is forced to 1x.

5. **Crash regime**: hard SKIP — no new positions, regardless of signal.

6. **Confidence is informational only.** The Markov matrix is allowed to
   trade even with low confidence; the dashboard shows the number for
   the user but the gate does not consult it.

Circuit breakers are still enforced upstream — `cb_allows_new=False` SKIPs
everything, and the `cb_multiplier` scales the final position size last.

Stop-loss (-5%) and take-profit (+15%) are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from markov.regime_detector import RegimeResult
from sentiment.analyzer import UnifiedScore

from .technical import TechnicalSnapshot


# Markov-signal → base position size (fraction of capital). Strictly greater-than.
MARKOV_FULL    = 0.5
MARKOV_MEDIUM  = 0.3
MARKOV_SMALL   = 0.1   # was 0.2 — relaxed per Video 4

BASE_FULL_PCT   = 0.05
BASE_MEDIUM_PCT = 0.03
BASE_SMALL_PCT  = 0.01

# Sentiment alignment thresholds for position-size adjustment.
SENT_ALIGNED_STRONG = 0.2    # > +0.2 → keep
# 0 to +0.2 → reduce 25%
# -0.2 to 0 → reduce 50%
# < -0.2 → reduce 75%

SENT_MULT_FULL    = 1.00
SENT_MULT_25_OFF  = 0.75
SENT_MULT_50_OFF  = 0.50
SENT_MULT_75_OFF  = 0.25

TECH_MULT_CONTRADICTS = 0.75
SIDEWAYS_SIZE_MULT    = 0.5

# Volatility-normalized sizing (Item 5, 2026-06-03): scale size so every
# position targets the same daily risk regardless of coin volatility.
# scalar = TARGET_DAILY_VOL / realized_daily_vol(ATR%), clipped to bounds so
# no single position becomes negligible or oversized. A 5% slug of a low-vol
# major can grow up to 2×; a high-vol memecoin shrinks toward 0.25×.
TARGET_DAILY_VOL = 0.05   # 5%/day target risk per position. Final go-live setting
                          # (2026-06-03, "Config 3"): the multi-year OOS bake-off
                          # showed items-ON @ 5% vol / budget 3.0 gives the best
                          # risk-adjusted profile (Sharpe ~3, DD ~13%, CB never trips
                          # at the 20% lock) and ~doubles the conservative config.
                          # DD is anchored by NET_BETA_BUDGET=3.0, not by this knob.
VOL_NORM_MIN     = 0.25
VOL_NORM_MAX     = 2.0

# Sentiment threshold for 2x leverage. Lowered 0.3 → 0.2 (2026-05-27) to
# increase 2x frequency — the previous 0.3 threshold was rarely hit because
# blended sentiment hovers in [-0.32, -0.08] under contrarian retail L/S
# weighting. With 0.2, LONG 2x triggers at sent > +0.2 in Bull/Euphoria,
# SHORT 2x at sent < -0.2 in Bear.
SENTIMENT_2X_THRESHOLD = 0.2

# Hysteresis / re-entry band (Item 7, 2026-06-03): once a position is open, a
# mere sign change isn't enough to FLIP it — the opposing Markov signal must
# clear this (stronger) threshold. Weak opposite signals (between the normal
# entry gate MARKOV_SMALL=0.1 and this) → SKIP, which HOLDS the current position
# (populate_exit_trend only exits on a *strong opposite* decision, not a SKIP).
# Cuts round-trip churn / fee drag from whipsawing on noise. 0.3 = the medium tier.
MARKOV_FLIP_THRESHOLD = 0.3

ALLOWED_LONG_REGIMES  = ("Bull", "Sideways", "Euphoria")
ALLOWED_SHORT_REGIMES = ("Bear", "Sideways")


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


def _base_size_from_markov(markov_signal: float) -> tuple[str, float]:
    """Return (decision, base_pct) from the Markov signal alone.

    decision is "LONG" / "SHORT" / "SKIP". For SKIP the base_pct is 0.
    """
    m = markov_signal
    if m > MARKOV_FULL:    return "LONG",  BASE_FULL_PCT
    if m > MARKOV_MEDIUM:  return "LONG",  BASE_MEDIUM_PCT
    if m > MARKOV_SMALL:   return "LONG",  BASE_SMALL_PCT
    if m < -MARKOV_FULL:   return "SHORT", BASE_FULL_PCT
    if m < -MARKOV_MEDIUM: return "SHORT", BASE_MEDIUM_PCT
    if m < -MARKOV_SMALL:  return "SHORT", BASE_SMALL_PCT
    return "SKIP", 0.0


def _sentiment_multiplier(sentiment: float, decision: str) -> float:
    """Aligned sentiment → size multiplier. For SHORT we invert."""
    aligned = sentiment if decision == "LONG" else -sentiment
    if aligned >  SENT_ALIGNED_STRONG: return SENT_MULT_FULL
    if aligned >  0.0:                 return SENT_MULT_25_OFF
    if aligned > -SENT_ALIGNED_STRONG: return SENT_MULT_50_OFF
    return SENT_MULT_75_OFF


def _technical_multiplier(tech_label: str, decision: str) -> float:
    """BULL tech on SHORT (or BEAR on LONG) shaves 25% off; otherwise no-op."""
    if decision == "LONG"  and tech_label == "BEAR": return TECH_MULT_CONTRADICTS
    if decision == "SHORT" and tech_label == "BULL": return TECH_MULT_CONTRADICTS
    return 1.0


def _regime_size_multiplier(regime: str) -> float:
    """Sideways → half. Crash is handled separately (hard SKIP)."""
    return SIDEWAYS_SIZE_MULT if regime == "Sideways" else 1.0


def _vol_normalization_multiplier(realized_vol: float) -> float:
    """Equal-risk scalar = TARGET_DAILY_VOL / realized_vol(ATR%), clipped.

    `realized_vol` is the coin's daily ATR% (e.g. 0.04 = 4%/day). Returns 1.0
    when vol is unknown (0/negative) so sizing is unchanged if the regime path
    couldn't compute ATR.
    """
    if not realized_vol or realized_vol <= 0:
        return 1.0
    return float(min(VOL_NORM_MAX, max(VOL_NORM_MIN, TARGET_DAILY_VOL / realized_vol)))


def _choose_leverage(decision: str, sentiment: float, regime: str) -> int:
    """1x in Sideways; otherwise 2x only when sentiment strongly favours the trade."""
    if regime == "Sideways":
        return 1
    if regime == "Crash":
        return 1
    if decision == "LONG"  and sentiment >  SENTIMENT_2X_THRESHOLD and regime in ("Bull", "Euphoria"):
        return 2
    if decision == "SHORT" and sentiment < -SENTIMENT_2X_THRESHOLD and regime == "Bear":
        return 2
    return 1


def evaluate_signal(
    coin: str,
    regime_result: RegimeResult,
    sentiment: UnifiedScore,
    technical: TechnicalSnapshot,
    capital: float,
    cb_multiplier: float,
    cb_allows_new: bool,
    current_position: Optional[str] = None,  # "LONG"/"SHORT"/None — for hysteresis
    leverage_chooser=None,  # kept for backwards-compat with existing callers
) -> SignalDecision:
    """Compute the trading decision for one coin.

    Markov decides direction + base size. Sentiment and technical adjust size.
    Sideways halves. Crash hard-skips. Circuit breakers scale everything last.
    """
    ms = regime_result.markov_signal
    sent = sentiment.unified
    tech = technical.label

    if not cb_allows_new:
        return SignalDecision(
            coin=coin, decision="SKIP",
            position_size_pct=0.0, dollars=0.0, leverage=1,
            markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason="circuit breaker disallows new positions",
            skip_reason="circuit breaker disallows new positions",
        )

    if regime_result.regime == "Crash":
        return SignalDecision(
            coin=coin, decision="SKIP",
            position_size_pct=0.0, dollars=0.0, leverage=1,
            markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason="Crash regime — no new positions",
            skip_reason="Crash regime",
        )

    decision, base_pct = _base_size_from_markov(ms)
    if decision == "SKIP":
        return SignalDecision(
            coin=coin, decision="SKIP",
            position_size_pct=0.0, dollars=0.0, leverage=1,
            markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason=f"|markov|={abs(ms):.2f} <= {MARKOV_SMALL} (dead zone)",
            skip_reason=f"markov in dead zone",
        )

    if decision == "LONG"  and regime_result.regime not in ALLOWED_LONG_REGIMES:
        return SignalDecision(
            coin=coin, decision="SKIP",
            position_size_pct=0.0, dollars=0.0, leverage=1,
            markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason=f"LONG not allowed in {regime_result.regime} regime",
            skip_reason=f"LONG vs {regime_result.regime}",
        )
    if decision == "SHORT" and regime_result.regime not in ALLOWED_SHORT_REGIMES:
        return SignalDecision(
            coin=coin, decision="SKIP",
            position_size_pct=0.0, dollars=0.0, leverage=1,
            markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason=f"SHORT not allowed in {regime_result.regime} regime",
            skip_reason=f"SHORT vs {regime_result.regime}",
        )

    # Hysteresis (Item 7): don't FLIP an open position on a weak opposite signal.
    # A flip needs |markov| ≥ MARKOV_FLIP_THRESHOLD; a weak opposite (0.1..0.3)
    # → SKIP, which HOLDS the current position (SKIP doesn't trigger an exit).
    # Same-direction or no open position → proceed normally.
    if (current_position in ("LONG", "SHORT")
            and decision != current_position
            and abs(ms) < MARKOV_FLIP_THRESHOLD):
        return SignalDecision(
            coin=coin, decision="SKIP",
            position_size_pct=0.0, dollars=0.0, leverage=1,
            markov_signal=ms, sentiment_score=sent,
            technical_label=tech, regime=regime_result.regime,
            reason=(f"hysteresis: |markov|={abs(ms):.2f} < flip {MARKOV_FLIP_THRESHOLD} "
                    f"— holding current {current_position}"),
            skip_reason="hysteresis: weak opposite signal, holding position",
        )

    sent_mult   = _sentiment_multiplier(sent, decision)
    tech_mult   = _technical_multiplier(tech, decision)
    regime_mult = _regime_size_multiplier(regime_result.regime)
    vol_mult    = _vol_normalization_multiplier(regime_result.realized_vol)

    final_pct = base_pct * sent_mult * tech_mult * regime_mult * vol_mult * cb_multiplier
    dollars   = capital * final_pct
    lev       = _choose_leverage(decision, sent, regime_result.regime)

    reason = (
        f"markov={ms:+.2f} ({decision} base {base_pct:.0%}) "
        f"× sent={sent_mult:.2f} × tech={tech_mult:.2f} "
        f"× regime={regime_mult:.2f} × vol={vol_mult:.2f} × cb={cb_multiplier:.2f} "
        f"→ {final_pct:.2%} lev={lev}x"
    )
    return SignalDecision(
        coin=coin, decision=decision,
        position_size_pct=final_pct, dollars=dollars,
        leverage=lev,
        markov_signal=ms, sentiment_score=sent,
        technical_label=tech, regime=regime_result.regime,
        reason=reason,
    )
