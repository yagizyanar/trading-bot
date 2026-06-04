"""15-minute intraday execution harness (roadmap item 9).

The daily harness checks exits only at the daily close, so it CANNOT see the live
exit mechanics — the -5% stop, the 8%/+10% trailing stop, intraday wicks. This
replays the SAME daily Markov signal on real 15m FUTURES bars with the deployed
exit stack, resolving the intraday sequence (which level was hit first), then
compares to the daily-close model on the identical data/signal/coins.

Live config replicated (config.json):
  stop -5% from entry · trailing 8% activating at +10% profit · no flat TP
  fee 0.05%/side · funding applied at the 8h marks (00:00/08:00/16:00 UTC)

Signal: daily Markov (20d) + hysteresis, computed on daily closes (causal),
executed on 15m bars. After a stop/trailing exit, no same-day re-entry (≈ a 1-day
cooldown; live Freqtrade has no CooldownPeriod protection, so it could re-enter
sooner — flagged as a finding). Universe = 13-coin clean set. Primary year 2024.

Reports per book (15m vs daily-close): return, Sharpe, maxDD, avg trade duration,
fee+funding drag, and the exit-reason mix (trailing vs hard stop vs signal flip).
"""
from __future__ import annotations

import math
import os
import pickle
from collections import Counter
from datetime import date as _date, timezone

import numpy as np
import pandas as pd

from config.settings import MARKOV_THRESHOLD, MARKOV_WINDOW, PROJECT_ROOT
from markov.regime_detector import build_transition_matrix, label_regimes, signal_from_matrix
from backtest.oos_multiyear import UNIVERSE, _client, _spot

ENTRY_GATE = 0.2
FLIP = 0.3
STOP_PCT = 0.05
TRAIL_PCT = 0.08
TRAIL_ACTIVATE = 0.10
FEE = 0.0005                      # 0.05% per side
LOOKBACK = 365
CACHE = PROJECT_ROOT / ".cache_15m"


# ----------------------------------------------------------------------------- data
def _fetch_15m(coin: str, start="2023-12-20", end="2025-01-01") -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{coin}_15m.pkl"
    if fp.exists():
        return pickle.load(open(fp, "rb"))
    from binance.enums import HistoricalKlinesType
    cli = _client()
    kl = cli.get_historical_klines(f"{coin}USDT", "15m", start, end,
                                   klines_type=HistoricalKlinesType.FUTURES)
    df = pd.DataFrame(kl, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"])
    df = df[["t", "o", "h", "l", "c"]].astype({"o": float, "h": float, "l": float, "c": float})
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    pickle.dump(df, open(fp, "wb"))
    return df


def _funding_events(coin: str):
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{coin}_fund.pkl"
    if fp.exists():
        return pickle.load(open(fp, "rb"))
    cli = _client()
    rows, start, end = [], int(pd.Timestamp("2023-12-01", tz="UTC").timestamp() * 1000), \
        int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    while start < end:
        try:
            batch = cli.futures_funding_rate(symbol=f"{coin}USDT", startTime=start, endTime=end, limit=1000)
        except Exception:
            batch = []
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        start = int(batch[-1]["fundingTime"]) + 1
    ts = np.array([int(r["fundingTime"]) for r in rows], dtype=np.int64)
    rate = np.array([float(r["fundingRate"]) for r in rows], dtype=float)
    pickle.dump((ts, rate), open(fp, "wb"))
    return ts, rate


# ----------------------------------------------------------------------------- signal
def _daily_targets(coin: str, year: int) -> dict:
    """date -> desired position (+1/-1/0) from daily Markov+hysteresis (no stop)."""
    es = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=LOOKBACK + 10)
    df = _spot(f"{coin}USDT")
    if df is None:
        return {}
    sl = df[(df.index >= ss) & (df.index <= ee)]
    close = sl["close"].reset_index(drop=True)
    n = len(close)
    if int((sl.index < es).sum()) < LOOKBACK:
        return {}
    dates = list(sl.index)
    pos = 0.0
    targets = {}
    for t in range(LOOKBACK, n):
        train = close.iloc[t - LOOKBACK:t]
        labels = label_regimes(train, window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
        s = 0.0 if labels.empty else float(signal_from_matrix(build_transition_matrix(labels), int(labels.iloc[-1])))
        desired = 1.0 if s > ENTRY_GATE else (-1.0 if s < -ENTRY_GATE else 0.0)
        if pos == 0.0:
            pos = desired
        elif desired != 0.0 and desired != pos:
            pos = desired if abs(s) >= FLIP else pos
        d = dates[t]
        if d >= es:
            targets[d.date()] = int(pos)
    return targets


# ----------------------------------------------------------------------------- replays
def replay_intraday(bars: pd.DataFrame, targets: dict, fund_ts, fund_rate):
    """Walk 15m bars with real intrabar stop/trailing. Returns (daily_returns, trades, fee_frac, fund_frac, reasons)."""
    ts_ms = np.array([int(t.timestamp() * 1000) for t in bars.index], dtype=np.int64)  # robust across index resolution
    o, h, l, c = bars["o"].to_numpy(), bars["h"].to_numpy(), bars["l"].to_numpy(), bars["c"].to_numpy()
    days = np.array([d.date() for d in bars.index])
    n = len(c)

    pos = 0; entry = mfe = prev_c = 0.0; entry_ms = 0; stopped_day = None
    equity = 1.0
    fee_frac = 0.0; fund_frac = 0.0
    trades = []; reasons = Counter(); daily_eq = {}
    fi = 0
    for i in range(n):
        day = days[i]
        if stopped_day is not None and stopped_day != day:
            stopped_day = None
        # funding at 8h marks
        while fi < len(fund_ts) and fund_ts[fi] <= ts_ms[i]:
            if pos != 0:
                equity *= (1 - pos * fund_rate[fi]); fund_frac += pos * fund_rate[fi]
            fi += 1
        # manage open position
        if pos != 0:
            exit_px = None; reason = None
            if pos > 0:
                if mfe >= entry * (1 + TRAIL_ACTIVATE):
                    lvl = mfe * (1 - TRAIL_PCT)
                    if l[i] <= lvl:
                        exit_px = o[i] if o[i] < lvl else lvl; reason = "trailing_stop"
                elif l[i] <= entry * (1 - STOP_PCT):
                    lvl = entry * (1 - STOP_PCT); exit_px = o[i] if o[i] < lvl else lvl; reason = "hard_stop"
            else:
                if mfe <= entry * (1 - TRAIL_ACTIVATE):
                    lvl = mfe * (1 + TRAIL_PCT)
                    if h[i] >= lvl:
                        exit_px = o[i] if o[i] > lvl else lvl; reason = "trailing_stop"
                elif h[i] >= entry * (1 + STOP_PCT):
                    lvl = entry * (1 + STOP_PCT); exit_px = o[i] if o[i] > lvl else lvl; reason = "hard_stop"
            if exit_px is not None:
                equity *= (1 + pos * (exit_px / prev_c - 1))
                equity *= (1 - FEE); fee_frac += FEE
                trades.append(dict(ret=pos * (exit_px / entry - 1), dur=(ts_ms[i] - entry_ms) / 3.6e6, reason=reason))
                reasons[reason] += 1
                pos = 0; stopped_day = day
            else:
                equity *= (1 + pos * (c[i] / prev_c - 1)); prev_c = c[i]
                mfe = max(mfe, h[i]) if pos > 0 else min(mfe, l[i])
        # signal entry / flip at this bar
        desired = targets.get(day, 0)
        if pos == 0:
            if desired != 0 and stopped_day != day:
                pos = desired; entry = mfe = prev_c = c[i]; entry_ms = ts_ms[i]
                equity *= (1 - FEE); fee_frac += FEE
        elif desired != 0 and desired != pos:                    # flip (price already MTM'd to close)
            equity *= (1 - FEE); fee_frac += FEE
            trades.append(dict(ret=pos * (c[i] / entry - 1), dur=(ts_ms[i] - entry_ms) / 3.6e6, reason="signal_flip"))
            reasons["signal_flip"] += 1
            pos = desired; entry = mfe = prev_c = c[i]; entry_ms = ts_ms[i]
            equity *= (1 - FEE); fee_frac += FEE
        daily_eq[day] = equity
    return _daily_returns(daily_eq), trades, fee_frac, fund_frac, reasons


def replay_daily_close(daily_close: pd.Series, targets: dict, fund_ts, fund_rate):
    """Daily-harness model: exits checked only at the daily CLOSE (cumulative from entry)."""
    idx = list(daily_close.index)
    px = daily_close.to_numpy()
    ts_ms = np.array([int(pd.Timestamp(d).timestamp() * 1000) for d in idx], dtype=np.int64)
    n = len(px)
    pos = 0; entry = mfe = prev_c = 0.0; entry_ms = 0; stopped_day = None
    equity = 1.0; fee_frac = 0.0; fund_frac = 0.0
    trades = []; reasons = Counter(); daily_eq = {}
    fi = 0
    for i in range(n):
        day = idx[i].date() if hasattr(idx[i], "date") else idx[i]
        if stopped_day is not None and stopped_day != day:
            stopped_day = None
        while fi < len(fund_ts) and fund_ts[fi] <= ts_ms[i]:
            if pos != 0:
                equity *= (1 - pos * fund_rate[fi]); fund_frac += pos * fund_rate[fi]
            fi += 1
        if pos != 0:
            equity *= (1 + pos * (px[i] / prev_c - 1)); prev_c = px[i]
            mfe = max(mfe, px[i]) if pos > 0 else min(mfe, px[i])
            cum = pos * (px[i] / entry - 1); peak_cum = pos * (mfe / entry - 1)
            exit_now = reason = None
            if peak_cum >= TRAIL_ACTIVATE and (peak_cum - cum) >= TRAIL_PCT:
                exit_now, reason = True, "trailing_stop"
            elif cum <= -STOP_PCT:
                exit_now, reason = True, "hard_stop"
            if exit_now:
                equity *= (1 - FEE); fee_frac += FEE
                trades.append(dict(ret=cum, dur=(ts_ms[i] - entry_ms) / 3.6e6, reason=reason))
                reasons[reason] += 1
                pos = 0; stopped_day = day
        desired = targets.get(day, 0)
        if pos == 0:
            if desired != 0 and stopped_day != day:
                pos = desired; entry = mfe = prev_c = px[i]; entry_ms = ts_ms[i]
                equity *= (1 - FEE); fee_frac += FEE
        elif desired != 0 and desired != pos:
            equity *= (1 - FEE); fee_frac += FEE
            trades.append(dict(ret=pos * (px[i] / entry - 1), dur=(ts_ms[i] - entry_ms) / 3.6e6, reason="signal_flip"))
            reasons["signal_flip"] += 1
            pos = desired; entry = mfe = prev_c = px[i]; entry_ms = ts_ms[i]
            equity *= (1 - FEE); fee_frac += FEE
        daily_eq[day] = equity
    return _daily_returns(daily_eq), trades, fee_frac, fund_frac, reasons


def signal_only_returns(daily_close: pd.Series, targets: dict, cost=FEE) -> pd.Series:
    """Baseline: pure equal-weight daily signal, exit only on flip, NO stop/trail.
    Isolates whether the signal itself has edge in this construction."""
    ret = daily_close.pct_change().fillna(0.0).to_numpy()
    days = [(d.date() if hasattr(d, "date") else d) for d in daily_close.index]
    pos = np.array([targets.get(d, 0) for d in days], dtype=float)   # target[D] is causal (from D-1)
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pd.Series(pos * ret - cost * turn, index=daily_close.index)


# ----------------------------------------------------------------------------- metrics
def _daily_returns(daily_eq: dict) -> pd.Series:
    s = pd.Series(daily_eq).sort_index()
    return s.pct_change().fillna(0.0)


def _portfolio(per_coin_returns: list) -> pd.Series:
    df = pd.concat(per_coin_returns, axis=1).fillna(0.0)
    return df.mean(axis=1)


def _sharpe(r):
    r = np.asarray(r, float); sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def _maxdd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); rm = np.maximum.accumulate(eq)
    return float(((eq - rm) / rm).min())


def _summarize(label, port, all_trades, fee_fracs, fund_fracs, reasons):
    tot = float(np.prod(1 + port.to_numpy()) - 1)
    durs = [t["dur"] for t in all_trades]
    tot_ex = sum(reasons.values()) or 1
    print(f"\n  {label}")
    print(f"    return {tot:+.1%}   Sharpe {_sharpe(port):.2f}   maxDD {_maxdd(port):.1%}")
    print(f"    trades {len(all_trades)}   avg duration {np.mean(durs):.1f}h ({np.mean(durs)/24:.1f}d)   "
          f"fee drag {np.mean(fee_fracs)*100:.2f}%/coin   funding {np.mean(fund_fracs)*100:+.2f}%/coin")
    mix = " ".join(f"{k}={v}({v/tot_ex:.0%})" for k, v in reasons.most_common())
    print(f"    exits: {mix}")
    return tot


def main():
    year = 2024
    print("=" * 96)
    print(f"ITEM 9 — 15-MINUTE INTRADAY EXECUTION HARNESS vs DAILY-CLOSE MODEL ({year})")
    print(f"stop -{STOP_PCT:.0%} · trail {TRAIL_PCT:.0%}/+{TRAIL_ACTIVATE:.0%} · no flat TP · fee {FEE:.2%}/side + funding")
    print("=" * 96)
    ir, dr, so = [], [], []
    i_tr, d_tr, i_ff, d_ff, i_fd, d_fd = [], [], [], [], [], []
    i_rs, d_rs = Counter(), Counter()
    used = []
    for coin in UNIVERSE:
        targets = _daily_targets(coin, year)
        if not targets:
            continue
        bars = _fetch_15m(coin)
        bars = bars[(bars.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                    (bars.index <= pd.Timestamp(f"{year}-12-31 23:59", tz="UTC"))]
        if len(bars) < 20000:
            print(f"  (skip {coin}: {len(bars)} 15m bars)"); continue
        fts, frate = _funding_events(coin)
        ri, ti, ffi, fdi, rsi = replay_intraday(bars, targets, fts, frate)
        daily_close = bars["c"].resample("1D").last().dropna()
        rd, td, ffd, fdd, rsd = replay_daily_close(daily_close, targets, fts, frate)
        ir.append(ri.rename(coin)); dr.append(rd.rename(coin))
        so.append(signal_only_returns(daily_close, targets).rename(coin))
        i_tr += ti; d_tr += td; i_ff.append(ffi); d_ff.append(ffd); i_fd.append(fdi); d_fd.append(fdd)
        i_rs += rsi; d_rs += rsd; used.append(coin)
    print(f"\nUniverse: {len(used)} coins {used}")
    port_so = _portfolio(so)
    print("\n  SIGNAL-ONLY (daily, NO stop/trail — pure equal-weight signal edge)")
    print(f"    return {float(np.prod(1 + port_so.to_numpy()) - 1):+.1%}   Sharpe {_sharpe(port_so):.2f}   maxDD {_maxdd(port_so):.1%}")
    _summarize("DAILY-CLOSE MODEL (daily harness: stop/trail checked at close)", _portfolio(dr), d_tr, d_ff, d_fd, d_rs)
    _summarize("15-MINUTE INTRADAY (real intrabar stop/trail)", _portfolio(ir), i_tr, i_ff, i_fd, i_rs)
    print("=" * 96)


if __name__ == "__main__":
    main()
