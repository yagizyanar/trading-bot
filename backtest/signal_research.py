"""Signal research (MEASURE ONLY — no deploy).

Task 1: 4-hour Markov vs daily Markov (same exit stack).
Task 2: entry/exit timing tweaks on the daily signal:
        (a) entry-window  — only enter within N bars of a NEW signal (no stale entries)
        (b) weaken-exit   — exit when |markov| drops below a threshold (not just on a flip)
        (c) funding-avoid — skip entries within N bars before an 8h funding mark

Exit stack held = current live: -5% stop, 3% trail / +5% activation, 1d cooldown, hysteresis.
Real Markov on BOTH daily and 4H (so the daily baseline matches the deployed config).
Equal-weight clean set, 15m intrabar replay, no lookahead (signal known at period start,
applied forward; entries/exits at the bar OPEN).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from binance.enums import HistoricalKlinesType

from config.settings import MARKOV_THRESHOLD, MARKOV_WINDOW, PROJECT_ROOT
from markov.regime_detector import label_regimes, build_transition_matrix, signal_from_matrix
from backtest.oos_multiyear import _client
from backtest.oos_intraday import (
    _fetch_15m, _funding_events, _bars_arrays, _daily_targets, _daily_returns,
    _portfolio, _sharpe, _maxdd, _exitmix, UNIVERSE, YEARS,
    ENTRY_GATE, FLIP, STOP_PCT, TRAIL_PCT, TRAIL_ACTIVATE, FEE,
)

# Research baseline = 1-day post-stop cooldown (15m bars), kept INDEPENDENT of the
# live StoplossGuard (synced to 2d/192 in oos_intraday on 2026-06-09) so Task 1/2 and
# the Task 3 baseline stay reproducible. Task 3 passes its own cooldown_bars (96*d).
COOLDOWN_BARS = 96

LB4H = 240   # 4H bars (~40d) for the transition matrix
CACHE4 = PROJECT_ROOT / ".cache_4h"
RUN_4H = os.environ.get("RUN_4H", "1") == "1"   # set RUN_4H=0 to skip the slow 4H builds + Task 1


def _fetch_4h(coin):
    CACHE4.mkdir(exist_ok=True)
    fp = CACHE4 / f"{coin}.pkl"
    if fp.exists():
        return pickle.load(open(fp, "rb"))
    try:
        kl = _client().get_historical_klines(f"{coin}USDT", "4h", "2021-08-01", "2025-01-01",
                                             klines_type=HistoricalKlinesType.FUTURES)
    except Exception:
        kl = []
    if not kl:
        pickle.dump(None, open(fp, "wb")); return None
    df = pd.DataFrame(kl, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"])
    s = pd.Series(pd.to_numeric(df["c"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df["t"], unit="ms", utc=True)))
    s = s[~s.index.duplicated(keep="last")]
    pickle.dump(s, open(fp, "wb")); return s


def _4h_targets(coin, year, mw):
    """Markov targets on 4H bars (same logic as _daily_targets). mw=20 -> 80h momentum
    (literal '4H bars'); mw=120 -> 20-day momentum sampled at 4H (fair horizon)."""
    es = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ee = pd.Timestamp(f"{year}-12-31 23:59", tz="UTC")
    ws = es - pd.Timedelta(hours=4 * (LB4H + mw + 5))
    c4all = _fetch_4h(coin)
    if c4all is None:
        return {}, {}
    c4 = c4all[(c4all.index >= ws) & (c4all.index <= ee)].dropna()
    if len(c4) < LB4H + mw + 10:
        return {}, {}
    labels = label_regimes(c4, window=mw, threshold=MARKOV_THRESHOLD)
    if labels is None or labels.empty:
        return {}, {}
    idx = list(c4.index)
    targets, strength, pos = {}, {}, 0.0
    for t in range(LB4H, len(c4)):
        lab = labels.iloc[t - LB4H:t]
        if lab.empty:
            continue
        s = float(signal_from_matrix(build_transition_matrix(lab), int(lab.iloc[-1])))
        desired = 1.0 if s > ENTRY_GATE else (-1.0 if s < -ENTRY_GATE else 0.0)
        if pos == 0.0:
            pos = desired
        elif desired != 0.0 and desired != pos:
            pos = desired if abs(s) >= FLIP else pos
        if idx[t] >= es:
            targets[idx[t]] = int(pos)
            strength[idx[t]] = abs(s)
    return targets, strength


def _per_bar(arr, targets, strength, mode):
    ts_ms, o, h, l, c, days = arr
    if mode == "daily":
        keys = days
    else:  # 4h
        keys = pd.DatetimeIndex(pd.to_datetime(ts_ms, unit="ms", utc=True)).floor("4h")
    d = np.array([float(targets.get(k, 0)) for k in keys])
    s = np.array([float(strength.get(k, 0.0)) for k in keys])
    return d, s


def replay2(arr, desired, strength, fund_ts, fund_rate, cooldown_bars,
            entry_window=0, weaken_exit=0.0, fund_avoid=0):
    ts_ms, o, h, l, c, days = arr
    n = len(c)
    sig_start = np.zeros(n, dtype=np.int64)
    last = 0
    for i in range(n):
        if i == 0 or desired[i] != desired[i - 1]:
            last = i
        sig_start[i] = last
    near = np.zeros(n, dtype=bool)
    if fund_avoid > 0 and len(fund_ts):
        ft = 0
        for i in range(n):
            while ft < len(fund_ts) and fund_ts[ft] < ts_ms[i]:
                ft += 1
            if ft < len(fund_ts) and 0 <= (fund_ts[ft] - ts_ms[i]) / 9e5 <= fund_avoid:
                near[i] = True
    pos = 0; entry = mfe = prev_c = 0.0; entry_ms = 0; cooldown_until = 0
    equity = 1.0; fee_frac = 0.0; fund_frac = 0.0
    trades = []; reasons = Counter(); daily_eq = {}
    fi = 0
    for i in range(n):
        day = days[i]
        while fi < len(fund_ts) and fund_ts[fi] <= ts_ms[i]:
            if pos != 0:
                equity *= (1 - pos * fund_rate[fi]); fund_frac += pos * fund_rate[fi]
            fi += 1
        if pos != 0:
            exit_px = None; reason = None
            if weaken_exit > 0 and strength[i] < weaken_exit:
                exit_px = o[i]; reason = "weaken_exit"
            if exit_px is None and pos > 0:
                if mfe >= entry * (1 + TRAIL_ACTIVATE):
                    lvl = mfe * (1 - TRAIL_PCT)
                    if l[i] <= lvl:
                        exit_px = o[i] if o[i] < lvl else lvl; reason = "trailing_stop"
                elif l[i] <= entry * (1 - STOP_PCT):
                    lvl = entry * (1 - STOP_PCT); exit_px = o[i] if o[i] < lvl else lvl; reason = "hard_stop"
            elif exit_px is None and pos < 0:
                if mfe <= entry * (1 - TRAIL_ACTIVATE):
                    lvl = mfe * (1 + TRAIL_PCT)
                    if h[i] >= lvl:
                        exit_px = o[i] if o[i] > lvl else lvl; reason = "trailing_stop"
                elif h[i] >= entry * (1 + STOP_PCT):
                    lvl = entry * (1 + STOP_PCT); exit_px = o[i] if o[i] > lvl else lvl; reason = "hard_stop"
            if exit_px is not None:
                equity *= (1 + pos * (exit_px / prev_c - 1)); equity *= (1 - FEE); fee_frac += FEE
                trades.append(dict(ret=pos * (exit_px / entry - 1), dur=(ts_ms[i] - entry_ms) / 3.6e6, reason=reason))
                reasons[reason] += 1
                if reason == "hard_stop":
                    cooldown_until = i + cooldown_bars
                pos = 0
            else:
                equity *= (1 + pos * (c[i] / prev_c - 1)); prev_c = c[i]
                mfe = max(mfe, h[i]) if pos > 0 else min(mfe, l[i])
        d = desired[i]
        if pos == 0:
            if d != 0 and i >= cooldown_until:
                stale = entry_window > 0 and (i - sig_start[i]) > entry_window
                if not stale and not (fund_avoid > 0 and near[i]):
                    pos = d; entry = mfe = prev_c = c[i]; entry_ms = ts_ms[i]
                    equity *= (1 - FEE); fee_frac += FEE
        elif d != 0 and d != pos:
            equity *= (1 - FEE); fee_frac += FEE
            trades.append(dict(ret=pos * (c[i] / entry - 1), dur=(ts_ms[i] - entry_ms) / 3.6e6, reason="signal_flip"))
            reasons["signal_flip"] += 1
            pos = d; entry = mfe = prev_c = c[i]; entry_ms = ts_ms[i]
            equity *= (1 - FEE); fee_frac += FEE
        daily_eq[day] = equity
    return _daily_returns(daily_eq), trades, fee_frac, fund_frac, reasons


# -------- precompute per (coin, year): 15m arr, funding, and the 3 signal variants --------
print("precomputing signals (real Markov: daily" + (" + 4H-80h + 4H-20d" if RUN_4H else " only") + ")...")
PRE = {}
for year in YEARS:
    for coin in UNIVERSE:
        dt, ds = _daily_targets(coin, year, return_strength=True)
        if not dt:
            continue
        bars = _fetch_15m(coin)
        bars = bars[(bars.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                    (bars.index <= pd.Timestamp(f"{year}-12-31 23:59", tz="UTC"))]
        if len(bars) < 20000:
            continue
        arr = _bars_arrays(bars)
        fts, frate = _funding_events(coin)
        f4_fast = _4h_targets(coin, year, mw=20) if RUN_4H else ({}, {})
        f4_fair = _4h_targets(coin, year, mw=120) if RUN_4H else ({}, {})
        PRE[(coin, year)] = dict(arr=arr, fts=fts, frate=frate,
                                 daily=(dt, ds), fast=f4_fast, fair=f4_fair)
    print(f"  {year}: {sum(1 for k in PRE if k[1] == year)} coins")


def run_config(signal_key, mode, rules, cooldown_bars=COOLDOWN_BARS):
    """signal_key in {daily,fast,fair}; mode {daily,4h}; rules dict for replay2."""
    out = {}
    for year in YEARS:
        rets, tr, fe, rs = [], [], [], Counter()
        for coin in UNIVERSE:
            p = PRE.get((coin, year))
            if not p:
                continue
            targets, strength = p[signal_key]
            if not targets:
                continue
            d, s = _per_bar(p["arr"], targets, strength, mode)
            r, t, f, _, reasons = replay2(p["arr"], d, s, p["fts"], p["frate"], cooldown_bars, **rules)
            rets.append(r.rename(coin)); tr += t; fe.append(f); rs += reasons
        if not rets:
            continue
        port = _portfolio(rets)
        durs = [t["dur"] for t in tr] or [0]
        out[year] = dict(ret=float(np.prod(1 + port.to_numpy()) - 1), sharpe=_sharpe(port),
                         dd=_maxdd(port), n=len(tr), dur=np.mean(durs) / 24, fee=np.mean(fe) * 100,
                         mix=_exitmix(rs))
    return out


def show(name, res):
    for y in YEARS:
        m = res.get(y)
        if m:
            print(f"  {name:<20}{y}  ret {m['ret']:>+7.1%}  Sharpe {m['sharpe']:>+5.2f}  maxDD {m['dd']:>6.1%}"
                  f"  {m['n']:>5}tr {m['dur']:>4.1f}d  fee {m['fee']:>4.1f}%  [{m['mix']}]")


NO = dict(entry_window=0, weaken_exit=0.0, fund_avoid=0)
if RUN_4H:
    print("\n" + "=" * 110)
    print("TASK 1 — 4-HOUR MARKOV vs DAILY MARKOV (same exit stack)")
    print("=" * 110)
    daily = run_config("daily", "daily", NO)
    fast = run_config("fast", "4h", NO)      # 4H, 20-bar = 80h momentum (literal '4H bars')
    fair = run_config("fair", "4h", NO)      # 4H, 120-bar = 20-day momentum (fair horizon, faster updates)
    show("DAILY (baseline)", daily)
    print()
    show("4H-80h (literal)", fast)
    print()
    show("4H-20d (fair)", fair)
    print("\nreaction speed (avg trade duration & trades/yr — shorter/more = faster-reacting signal):")
    for nm, res in [("daily", daily), ("4H-80h", fast), ("4H-20d", fair)]:
        avg_dur = np.mean([res[y]["dur"] for y in res]) if res else 0
        avg_tr = np.mean([res[y]["n"] for y in res]) if res else 0
        print(f"  {nm:<8} avg hold {avg_dur:.2f}d  ~{avg_tr:.0f} trades/yr")

print("\n" + "=" * 110)
print("TASK 2 — ENTRY/EXIT TIMING TWEAKS (daily signal)")
print("=" * 110)
base = run_config("daily", "daily", NO)
a = run_config("daily", "daily", dict(entry_window=16, weaken_exit=0.0, fund_avoid=0))   # first 4h (16 bars)
b = run_config("daily", "daily", dict(entry_window=0, weaken_exit=0.30, fund_avoid=0))   # |markov|<0.3 -> exit
c = run_config("daily", "daily", dict(entry_window=0, weaken_exit=0.0, fund_avoid=4))    # skip last 1h before funding
combo = run_config("daily", "daily", dict(entry_window=16, weaken_exit=0.30, fund_avoid=4))
show("BASELINE", base)
print()
show("(a) entry<=4h", a)
print()
show("(b) weaken-exit .30", b)
print()
show("(c) funding-avoid 1h", c)
print()
show("(a+b+c) combined", combo)


def compounded(res):
    return float(np.prod([1 + res[y]["ret"] for y in YEARS if y in res]) - 1)


print("\n3-year compounded return (vs baseline):")
for nm, res in [("baseline", base), ("(a) entry<=4h", a), ("(b) weaken-exit", b),
                ("(c) funding-avoid", c), ("(a+b+c)", combo)]:
    print(f"  {nm:<18} {compounded(res):>+7.1%}")
print("=" * 110)


# ============================ TASK 3 — COOLDOWN SWEEP ============================
print("\n" + "=" * 110)
print("TASK 3 — COOLDOWN AFTER A STOP-LOSS (daily signal; lock re-entry N days after a -5% hard stop)")
print("=" * 110)
CDAYS = [1, 2, 3, 5]
cds = {d: run_config("daily", "daily", NO, cooldown_bars=96 * d) for d in CDAYS}   # 96 bars = 1 day (15m)
a_ref = run_config("daily", "daily", dict(entry_window=16, weaken_exit=0.0, fund_avoid=0))  # rule (a)
for d in CDAYS:
    show("cooldown %dd%s" % (d, " (base)" if d == 1 else ""), cds[d])
    print()
show("rule (a) entry<=4h", a_ref)


def _avgm(res, k):
    return float(np.mean([res[y][k] for y in YEARS if y in res])) if res else 0.0


def _chop(res):   # avg return across the two whipsaw years (2022 & 2024)
    ys = [y for y in (2022, 2024) if y in res]
    return float(np.mean([res[y]["ret"] for y in ys])) if ys else 0.0


print("\nwhipsaw-vs-trend balance:")
print(f"  {'config':<20}{'chop 22+24':>12}{'trend 23':>10}{'3yr':>8}{'avgSh':>7}{'tr/yr':>8}{'fee%':>7}")
for nm, res in [("cooldown 1d (base)", cds[1]), ("cooldown 2d", cds[2]),
                ("cooldown 3d", cds[3]), ("cooldown 5d", cds[5]),
                ("rule (a) entry<=4h", a_ref)]:
    t23 = res.get(2023, {}).get("ret", 0.0)
    print(f"  {nm:<20}{_chop(res):>+11.1%} {t23:>+9.1%} {compounded(res):>+7.1%}"
          f" {_avgm(res, 'sharpe'):>+6.2f} {_avgm(res, 'n'):>7.0f} {_avgm(res, 'fee'):>6.1f}")
print("  chop 22+24 = avg return across the two whipsaw years; trend 23 = the trend year.")
print("  best 'balance' = lifts chop 22+24 toward 0 while giving back the LEAST of trend 23.")
print("=" * 110)
