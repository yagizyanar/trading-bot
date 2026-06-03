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


def run_portfolio(coins, common, POS, GROSS, VOLM, BETA, MOM, *, item5, item6, budget):
    nc, nd = len(coins), len(common)
    eq = START_CAP
    net = np.zeros(nd)
    eq_curve = np.zeros(nd)
    fee_dollars = 0.0
    prev_frac = np.zeros(nc)
    active_counts = []
    capped_days = 0
    for d in range(nd):
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
    net_s = pd.Series(net)
    sd = net_s.std(ddof=1)
    sharpe = float(net_s.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0
    runmax = np.maximum.accumulate(eq_curve)
    max_dd = float(((eq_curve - runmax) / runmax).min())
    return dict(total_return=eq / START_CAP - 1.0, final=eq, sharpe=sharpe,
                max_dd=max_dd, fee=fee_dollars, avg_active=float(np.mean(active_counts)),
                capped_days=capped_days)


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
        volm = np.clip(np.where(atr > 0, TARGET_DAILY_VOL / np.where(atr > 0, atr, 1), 1.0),
                       VOL_NORM_MIN, VOL_NORM_MAX)
        beta = _rolling_beta(close, btc_close).to_numpy()
        mom = close.pct_change(20).abs().fillna(0.0).to_numpy()
        idx = close.index
        per[c] = pd.DataFrame({
            "base_pos": base_pos, "base_gross": np.nan_to_num(base_gross),
            "hyst_pos": hyst_pos, "hyst_gross": np.nan_to_num(hyst_gross),
            "volm": volm, "beta": beta, "mom": mom,
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
    VOLM = mat("volm", None)
    BETA = mat("beta", None)
    MOM = mat("mom", None)

    # Scenario 1: baseline (items OFF) — baseline path, equal 1-slot, no beta cap.
    s1 = run_portfolio(coins, common, BASE_POS, BASE_GROSS, VOLM, BETA, MOM,
                       item5=False, item6=False, budget=3.0)
    # Scenario 2: items 5+6+7, NET_BETA_BUDGET=3.0 (deployed).
    s2 = run_portfolio(coins, common, HYST_POS, HYST_GROSS, VOLM, BETA, MOM,
                       item5=True, item6=True, budget=3.0)
    # Scenario 3: items 5+6+7, NET_BETA_BUDGET=5.0.
    s3 = run_portfolio(coins, common, HYST_POS, HYST_GROSS, VOLM, BETA, MOM,
                       item5=True, item6=True, budget=5.0)

    print("\n" + "-" * 82)
    print(f"  {'scenario':<34}{'total ret':>12}{'final $':>11}{'Sharpe':>9}{'maxDD':>9}{'fee $':>9}")
    print("-" * 82)
    for name, s in [("1) Baseline (items off)", s1),
                    ("2) Items 5+6+7, budget=3.0", s2),
                    ("3) Items 5+6+7, budget=5.0", s3)]:
        print(f"  {name:<34}{s['total_return']:>+11.1%}{s['final']:>11,.0f}"
              f"{s['sharpe']:>9.2f}{s['max_dd']:>9.1%}{s['fee']:>9,.0f}")
    print("-" * 82)
    print(f"  avg active positions: base {s1['avg_active']:.1f} | items {s2['avg_active']:.1f}")
    print(f"  beta-capped days: budget3.0 {s2['capped_days']} | budget5.0 {s3['capped_days']} (of {len(common)})")
    print("=" * 82)


if __name__ == "__main__":
    main()
