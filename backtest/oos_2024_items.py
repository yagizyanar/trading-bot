"""2024 OOS harness EXTENDED to measure roadmap items 5-7 (2026-06-03).

Faithfully models the live sizing/risk in a daily walk-forward so the three
mechanisms can be measured rather than estimated:

  Item 5 (vol-normalized sizing): each position's slot size = clip(
      TARGET_DAILY_VOL / ATR%_daily, 0.25, 2.0) instead of an equal slot.
  Item 6 (net-beta cap): each day, net_beta = Σ sign·slot·beta_to_btc (rolling
      30d). If |net_beta| > budget, the whole book is deleveraged by
      budget/|net_beta| (live SKIPs the breaching adds; same net effect on
      directional exposure).
  Item 7 (hysteresis): a position FLIPS only if |raw markov signal| ≥ 0.3, and
      HOLDS through the dead-zone (matches live, where a SKIP doesn't exit).

UNITS NOTE: positions are measured in "slots" where 1 slot = 1/CAP of capital
(so 10 equal slots = 100% deployed). net_beta is in slot·beta units — the same
space as the live NET_BETA_BUDGET (a budget of 3.0 ≈ 3 full one-way positions).

BASELINE RECONCILIATION: scenario 1 (items off, equal 1-slot, gate-0.2 path,
exit-on-deadzone) uses fixed 1/CAP slots rather than the original harness's
renormalized 1/N, so its absolute return is ~0.9× the published +369% on
partial-active days; Sharpe is identical (scale-invariant). All three scenarios
share this one engine, so the deltas between them are clean.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import MARKOV_THRESHOLD, MARKOV_WINDOW
from markov.regime_detector import (
    build_transition_matrix,
    label_regimes,
    signal_from_matrix,
)
from signals.three_layer import (
    MARKOV_FLIP_THRESHOLD,
    TARGET_DAILY_VOL,
    VOL_NORM_MAX,
    VOL_NORM_MIN,
)

UNIVERSE = ["SOL", "AVAX", "LINK", "DOT", "INJ", "OP", "APT",
            "NEAR", "ATOM", "SAND", "MANA", "AXS", "DYDX"]
IN_SAMPLE = 365
ENTRY_GATE = 0.2          # original harness signal gate (DEFAULT_SIGNAL_GATE)
FLIP = MARKOV_FLIP_THRESHOLD   # 0.3
STOP = 0.05
COST = 0.0007             # 0.05% fee + 0.02% slippage per side
CAP = 10
PER_SLOT = 1.0 / CAP      # 1 slot = 10% of capital → 10 slots = 100% deployed
START_CAP = 300.0
BETA_WINDOW = 30

EVAL_START = pd.Timestamp("2024-01-01", tz="UTC")
EVAL_END = pd.Timestamp("2024-12-31", tz="UTC")
LOOKBACK_START = pd.Timestamp("2023-01-01", tz="UTC")


def _fetch(sym):
    from sentiment.binance_data import fetch_binance_ohlcv
    df = fetch_binance_ohlcv(sym, interval="1d", limit=1500)
    if df is None or df.empty:
        return None
    return df


def _atr_pct_series(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return (tr.rolling(window).mean() / close).bfill().fillna(0.0)


def _raw_signals(close: pd.Series, in_sample: int) -> np.ndarray:
    """Continuous Markov signal P(Bull)-P(Bear) for each day t≥in_sample (causal)."""
    close = close.reset_index(drop=True)
    n = len(close)
    raw = np.full(n, np.nan)
    for t in range(in_sample, n):
        train = close.iloc[t - in_sample:t]
        labels = label_regimes(train, window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
        if labels.empty:
            raw[t] = 0.0
        else:
            P = build_transition_matrix(labels)
            raw[t] = float(signal_from_matrix(P, int(labels.iloc[-1])))
    return raw


def _derive_path(raw, rets, in_sample, stop, mode):
    """From raw signals → (positions, gross) applying the -5% stop.

    mode='baseline': enter/flip/exit on |s|>ENTRY_GATE, flat in the dead-zone.
    mode='hysteresis': flip only if |s|≥FLIP; hold through dead-zone / weak opposite.
    """
    n = len(raw)
    positions = np.zeros(n)
    gross = np.full(n, np.nan)
    pos = 0.0
    for t in range(in_sample, n):
        if pos != 0.0 and (-rets[t] * np.sign(pos)) > stop:   # stop hit
            gross[t] = -stop * abs(pos)
            positions[t] = pos
            pos = 0.0
            continue
        gross[t] = rets[t] * pos
        positions[t] = pos
        s = raw[t]
        desired = 1.0 if s > ENTRY_GATE else (-1.0 if s < -ENTRY_GATE else 0.0)
        if mode == "baseline":
            target = desired
        else:  # hysteresis
            if pos == 0.0:
                target = desired
            elif desired != 0.0 and desired != pos:
                target = desired if abs(s) >= FLIP else pos   # flip only if strong
            else:
                target = pos                                   # same dir / dead-zone → hold
        pos = target
    return positions, gross


def _rolling_beta(coin_close: pd.Series, btc_close: pd.Series, window=BETA_WINDOW) -> pd.Series:
    rc = coin_close.pct_change()
    rb = btc_close.reindex(coin_close.index).pct_change()
    corr = rc.rolling(window).corr(rb)
    beta = corr * (rc.rolling(window).std() / rb.rolling(window).std())
    return beta.shift(1).fillna(1.0)   # causal, neutral default


def run_portfolio(coins, common, POS, GROSS, VOLM, BETA, MOM, *, item5, item6, budget,
                  dd_lock=None):
    """dd_lock: if set (e.g. 0.10), trading HALTS the first day drawdown-from-peak
    reaches it (the live drawdown circuit breaker locks the bot). Equity is then
    frozen → total_return is the REALIZED return after the lock. None = no CB."""
    nc, nd = len(coins), len(common)
    eq = START_CAP
    peak = START_CAP
    locked = False
    locked_day = None
    net = np.zeros(nd)
    eq_curve = np.zeros(nd)
    fee_dollars = 0.0
    prev_frac = np.zeros(nc)
    active_counts = []
    capped_days = 0
    for d in range(nd):
        if locked:                      # drawdown lock tripped — trading halted, equity frozen
            eq_curve[d] = eq
            active_counts.append(0)
            continue
        active = [i for i in range(nc) if POS[i, d] != 0.0]
        if len(active) > CAP:
            active = sorted(active, key=lambda i: -MOM[i, d])[:CAP]
        active_counts.append(len(active))
        slots = {i: (VOLM[i, d] if item5 else 1.0) for i in active}
        net_beta = sum(np.sign(POS[i, d]) * slots[i] * BETA[i, d] for i in active)
        scale = 1.0
        if item6 and abs(net_beta) > budget and abs(net_beta) > 0:
            scale = budget / abs(net_beta)
            capped_days += 1
        frac = np.zeros(nc)
        g = 0.0
        for i in active:
            frac[i] = slots[i] * scale * PER_SLOT
            g += frac[i] * GROSS[i, d]
        turnover = float(np.abs(frac - prev_frac).sum())
        cost = COST * turnover
        net[d] = g - cost
        fee_dollars += eq * cost
        eq *= (1.0 + net[d])
        eq_curve[d] = eq
        prev_frac = frac
        if eq > peak:
            peak = eq
        if dd_lock is not None and peak > 0 and (peak - eq) / peak >= dd_lock:
            locked, locked_day = True, d
    net_s = pd.Series(net)
    sd = net_s.std(ddof=1)
    sharpe = float(net_s.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0
    runmax = np.maximum.accumulate(eq_curve)
    max_dd = float(((eq_curve - runmax) / runmax).min())
    return dict(total_return=eq / START_CAP - 1.0, final=eq, sharpe=sharpe,
                max_dd=max_dd, fee=fee_dollars, avg_active=float(np.mean(active_counts)),
                capped_days=capped_days, locked=locked, locked_day=locked_day,
                peak_return=peak / START_CAP - 1.0)


def main() -> None:
    print("=" * 82)
    print("2024 OOS — MEASURED items 5-7  |  13-coin universe  |  365d lookback  |  cost 0.07%/side")
    print("=" * 82)

    btc_df = _fetch("BTCUSDT")
    btc_close = btc_df["close"].astype(float)

    per = {}
    for c in UNIVERSE:
        df = _fetch(f"{c}USDT")
        if df is None:
            print(f"  WARN no data {c}")
            continue
        df = df[(df.index >= LOOKBACK_START) & (df.index <= EVAL_END)]
        close = df["close"].astype(float)
        rets = close.pct_change().fillna(0.0).to_numpy()
        raw = _raw_signals(close, IN_SAMPLE)
        base_pos, base_gross = _derive_path(raw, rets, IN_SAMPLE, STOP, "baseline")
        hyst_pos, hyst_gross = _derive_path(raw, rets, IN_SAMPLE, STOP, "hysteresis")
        atr = _atr_pct_series(df).to_numpy()
        beta = _rolling_beta(close, btc_close).to_numpy()
        mom = close.pct_change(20).abs().fillna(0.0).to_numpy()
        idx = close.index
        per[c] = pd.DataFrame({
            "base_pos": base_pos, "base_gross": np.nan_to_num(base_gross),
            "hyst_pos": hyst_pos, "hyst_gross": np.nan_to_num(hyst_gross),
            "atr": atr, "beta": beta, "mom": mom,
        }, index=idx)
        per[c] = per[c][(per[c].index >= EVAL_START) & (per[c].index <= EVAL_END)]

    coins = list(per.keys())
    common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()
    print(f"Universe {len(coins)} coins | 2024 days {len(common)}")

    def mat(col, path_mode):
        return np.vstack([per[c].reindex(common)[col].fillna(0.0).to_numpy() for c in coins])

    BASE_POS, BASE_GROSS = mat("base_pos", None), mat("base_gross", None)
    HYST_POS, HYST_GROSS = mat("hyst_pos", None), mat("hyst_gross", None)
    ATR = mat("atr", None)
    BETA = mat("beta", None)
    MOM = mat("mom", None)

    def volm_for(target_vol: float) -> np.ndarray:
        """vol_mult matrix for a given TARGET_DAILY_VOL: clip(target/ATR%, MIN, MAX)."""
        safe = np.where(ATR > 0, ATR, 1.0)
        v = np.where(ATR > 0, target_vol / safe, 1.0)
        return np.clip(v, VOL_NORM_MIN, VOL_NORM_MAX)

    ONES = np.ones_like(ATR)
    baseline = run_portfolio(coins, common, BASE_POS, BASE_GROSS, ONES, BETA, MOM,
                             item5=False, item6=False, budget=3.0)

    # TARGET_DAILY_VOL sweep (items 5+6+7 active, hysteresis path).
    sweep = [
        ("1) tgt=2.0%  budget=3.0  (current/conservative)", 0.020, 3.0),
        ("2) tgt=3.5%  budget=3.0  (medium)",               0.035, 3.0),
        ("3) tgt=5.0%  budget=3.0  (aggressive)",           0.050, 3.0),
        ("4) tgt=5.0%  budget=8.0  (full aggressive)",      0.050, 8.0),
    ]
    rows = [("0) BASELINE (items off, reference)", baseline)]
    detail = []
    for label, tv, bud in sweep:
        r = run_portfolio(coins, common, HYST_POS, HYST_GROSS, volm_for(tv), BETA, MOM,
                          item5=True, item6=True, budget=bud)
        rows.append((label, r))
        detail.append((label, r))

    print("\n" + "-" * 92)
    print(f"  {'scenario':<48}{'total ret':>11}{'final $':>10}{'Sharpe':>8}{'maxDD':>8}{'fee $':>8}")
    print("-" * 92)
    for label, r in rows:
        print(f"  {label:<48}{r['total_return']:>+10.1%}{r['final']:>10,.0f}"
              f"{r['sharpe']:>8.2f}{r['max_dd']:>8.1%}{r['fee']:>8,.0f}")
    print("-" * 92)
    for label, r in detail:
        print(f"    {label[:12]:<13} avg_active={r['avg_active']:.1f}  "
              f"beta-capped_days={r['capped_days']}/{len(common)}")
    print("=" * 92)


if __name__ == "__main__":
    main()
