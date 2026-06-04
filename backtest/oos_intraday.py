"""15-minute intraday execution harness (roadmap item 9), multi-year.

Replays the daily Markov signal on real 15m FUTURES bars with the deployed exit
stack, resolving intraday sequence — the only harness that can test live exit
mechanics. Now models the 1-day post-stop cooldown (StoplossGuard) and runs
2022/2023/2024.

Live config replicated (config.json):
  stop -5% from entry · trailing 8% activating at +10% · no flat TP
  fee 0.05%/side · funding at the 8h marks · 1-day (96-bar) cooldown after a
  HARD STOP (matches StoplossGuard: re-entry blocked for 24h after a losing stop;
  flips and winning trailing exits unaffected — trailing only fires in profit).

Books reported per year (equal-weight): signal-only (no stops), daily-close
(the daily harness), 15m no-cooldown, 15m + 1-day cooldown (DEPLOYED). Universe =
13-coin clean set. NOTE: absolute returns are EQUAL-WEIGHT, not the deployed sized
book (whose 2024 headline was a look-ahead artifact — see project_runportfolio_lookahead).
"""
from __future__ import annotations

import math
import pickle
from collections import Counter

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
FEE = 0.0005
LOOKBACK = 365
COOLDOWN_BARS = 96               # 1 day at 15m — post-hard-stop re-entry lock (StoplossGuard)
CACHE = PROJECT_ROOT / ".cache_15m"
YEARS = [2022, 2023, 2024]


def _fetch_15m(coin: str, start="2021-12-15", end="2025-01-01") -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{coin}_15m.pkl"
    if fp.exists():
        return pickle.load(open(fp, "rb"))
    from binance.enums import HistoricalKlinesType
    kl = _client().get_historical_klines(f"{coin}USDT", "15m", start, end,
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
    rows, start = [], int(pd.Timestamp("2021-12-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
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


def _daily_targets(coin: str, year: int) -> dict:
    es = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=LOOKBACK + 10)
    df = _spot(f"{coin}USDT")
    if df is None:
        return {}
    sl = df[(df.index >= ss) & (df.index <= ee)]
    close = sl["close"].reset_index(drop=True)
    if int((sl.index < es).sum()) < LOOKBACK:
        return {}
    dates = list(sl.index)
    pos = 0.0
    targets = {}
    for t in range(LOOKBACK, len(close)):
        labels = label_regimes(close.iloc[t - LOOKBACK:t], window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
        s = 0.0 if labels.empty else float(signal_from_matrix(build_transition_matrix(labels), int(labels.iloc[-1])))
        desired = 1.0 if s > ENTRY_GATE else (-1.0 if s < -ENTRY_GATE else 0.0)
        if pos == 0.0:
            pos = desired
        elif desired != 0.0 and desired != pos:
            pos = desired if abs(s) >= FLIP else pos
        if dates[t] >= es:
            targets[dates[t].date()] = int(pos)
    return targets


def _bars_arrays(bars: pd.DataFrame):
    """Precompute numpy arrays once per coin (the timestamp loop is the hot path)."""
    ts_ms = np.array([int(t.timestamp() * 1000) for t in bars.index], dtype=np.int64)
    o, h, l, c = bars["o"].to_numpy(), bars["h"].to_numpy(), bars["l"].to_numpy(), bars["c"].to_numpy()
    days = np.array([d.date() for d in bars.index])
    return ts_ms, o, h, l, c, days


def replay_intraday(arr, targets: dict, fund_ts, fund_rate, cooldown_bars: int):
    """15m walk with intrabar stop/trailing + post-hard-stop cooldown."""
    ts_ms, o, h, l, c, days = arr
    n = len(c)
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
                if reason == "hard_stop":
                    cooldown_until = i + cooldown_bars       # lock re-entry after a LOSING stop
                pos = 0
            else:
                equity *= (1 + pos * (c[i] / prev_c - 1)); prev_c = c[i]
                mfe = max(mfe, h[i]) if pos > 0 else min(mfe, l[i])
        desired = targets.get(day, 0)
        if pos == 0:
            if desired != 0 and i >= cooldown_until:
                pos = desired; entry = mfe = prev_c = c[i]; entry_ms = ts_ms[i]
                equity *= (1 - FEE); fee_frac += FEE
        elif desired != 0 and desired != pos:                # flip — never cooled (not a stop)
            equity *= (1 - FEE); fee_frac += FEE
            trades.append(dict(ret=pos * (c[i] / entry - 1), dur=(ts_ms[i] - entry_ms) / 3.6e6, reason="signal_flip"))
            reasons["signal_flip"] += 1
            pos = desired; entry = mfe = prev_c = c[i]; entry_ms = ts_ms[i]
            equity *= (1 - FEE); fee_frac += FEE
        daily_eq[day] = equity
    return _daily_returns(daily_eq), trades, fee_frac, fund_frac, reasons


def replay_daily_close(daily_close: pd.Series, targets: dict, fund_ts, fund_rate):
    idx = list(daily_close.index); px = daily_close.to_numpy()
    ts_ms = np.array([int(pd.Timestamp(d).timestamp() * 1000) for d in idx], dtype=np.int64)
    pos = 0; entry = mfe = prev_c = 0.0; entry_ms = 0; stopped_day = None
    equity = 1.0; fee_frac = 0.0; fund_frac = 0.0
    trades = []; reasons = Counter(); daily_eq = {}; fi = 0
    for i in range(len(px)):
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
            reason = None
            if peak_cum >= TRAIL_ACTIVATE and (peak_cum - cum) >= TRAIL_PCT:
                reason = "trailing_stop"
            elif cum <= -STOP_PCT:
                reason = "hard_stop"
            if reason:
                equity *= (1 - FEE); fee_frac += FEE
                trades.append(dict(ret=cum, dur=(ts_ms[i] - entry_ms) / 3.6e6, reason=reason))
                reasons[reason] += 1; pos = 0; stopped_day = day
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
    ret = daily_close.pct_change().fillna(0.0).to_numpy()
    days = [(d.date() if hasattr(d, "date") else d) for d in daily_close.index]
    pos = np.array([targets.get(d, 0) for d in days], dtype=float)
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pd.Series(pos * ret - cost * turn, index=daily_close.index)


def _daily_returns(daily_eq: dict) -> pd.Series:
    return pd.Series(daily_eq).sort_index().pct_change().fillna(0.0)


def _portfolio(per_coin):
    return pd.concat(per_coin, axis=1).fillna(0.0).mean(axis=1)


def _sharpe(r):
    r = np.asarray(r, float); sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def _maxdd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); rm = np.maximum.accumulate(eq)
    return float(((eq - rm) / rm).min())


def _exitmix(reasons):
    tot = sum(reasons.values()) or 1
    hs, ts_ = reasons.get("hard_stop", 0), reasons.get("trailing_stop", 0)
    return f"hard {hs/tot:.0%} / trail {ts_/tot:.0%} / flip {reasons.get('signal_flip',0)/tot:.0%}"


def _line(label, port, trades=None, fee=None, reasons=None):
    base = f"    {label:<30}{float(np.prod(1+port.to_numpy())-1):>+9.1%}{_sharpe(port):>8.2f}{_maxdd(port):>8.1%}"
    if trades is not None:
        durs = [t["dur"] for t in trades] or [0]
        base += f"   {len(trades):>5}tr  {np.mean(durs)/24:>4.1f}d  fee {np.mean(fee)*100:>4.1f}%  [{_exitmix(reasons)}]"
    print(base)


def run_year(year):
    so, dc, i0, i1 = [], [], [], []
    dtr, i0tr, i1tr = [], [], []
    dfe, i0fe, i1fe = [], [], []
    drs, i0rs, i1rs = Counter(), Counter(), Counter()
    used = []
    for coin in UNIVERSE:
        targets = _daily_targets(coin, year)
        if not targets:
            continue
        bars = _fetch_15m(coin)
        bars = bars[(bars.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                    (bars.index <= pd.Timestamp(f"{year}-12-31 23:59", tz="UTC"))]
        if len(bars) < 20000:
            continue
        fts, frate = _funding_events(coin)
        dclose = bars["c"].resample("1D").last().dropna()
        arr = _bars_arrays(bars)
        so.append(signal_only_returns(dclose, targets).rename(coin))
        r, t, f, _, rs = replay_daily_close(dclose, targets, fts, frate)
        dc.append(r.rename(coin)); dtr += t; dfe.append(f); drs += rs
        r, t, f, _, rs = replay_intraday(arr, targets, fts, frate, cooldown_bars=1)
        i0.append(r.rename(coin)); i0tr += t; i0fe.append(f); i0rs += rs
        r, t, f, _, rs = replay_intraday(arr, targets, fts, frate, cooldown_bars=COOLDOWN_BARS)
        i1.append(r.rename(coin)); i1tr += t; i1fe.append(f); i1rs += rs
        used.append(coin)
    regime = {2022: "BEAR", 2023: "RECOVERY", 2024: "BULL"}[year]
    print(f"\n{year} [{regime}]  ({len(used)} coins)   {' '*7}{'return':>9}{'Sharpe':>8}{'maxDD':>8}")
    print("    " + "-" * 92)
    _line("signal-only (no stops)", _portfolio(so))
    _line("daily-close (daily harness)", _portfolio(dc), dtr, dfe, drs)
    _line("15m, no cooldown", _portfolio(i0), i0tr, i0fe, i0rs)
    _line("15m + 1d cooldown (DEPLOYED)", _portfolio(i1), i1tr, i1fe, i1rs)


def main():
    print("=" * 116)
    print("ITEM 9 — 15m INTRADAY EXECUTION, MULTI-YEAR  |  stop -5% · trail 8%/+10% · no TP · "
          "fee 0.05%/side+funding · 1d cooldown")
    print("  (equal-weight; absolute level is NOT the sized book — see project_runportfolio_lookahead)")
    print("=" * 116)
    for y in YEARS:
        run_year(y)
    print("=" * 116)


if __name__ == "__main__":
    main()
