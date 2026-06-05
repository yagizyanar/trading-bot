"""Pre-add gate for candidate universe coins: momentum signal quality + min-notional.

(A) SIGNAL QUALITY — regenerate the bot's daily 20-day Markov long/short signal
    (365d transition lookback, ENTRY_GATE 0.2, FLIP off) over ALL available daily
    FUTURES history through END, and report signal-only (light-cost) Sharpe, daily
    directional hit-rate, annualized return, #trades, coverage, and per-year. The
    CURRENT-universe coins are run as the 'works well' bar.

(B) MIN-NOTIONAL — Binance USDS-M MIN_NOTIONAL + LOT_SIZE per coin, the resulting
    min order notional, and where the bot's smallest stake clears it by account size.
"""
from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
from binance.enums import HistoricalKlinesType

from config.settings import MARKOV_WINDOW, MARKOV_THRESHOLD, PROJECT_ROOT
from markov.regime_detector import label_regimes, build_transition_matrix, signal_from_matrix
from backtest.oos_multiyear import _client

CANDIDATES = ["WLD", "XRP", "FIL", "UNI", "ENA"]
REFERENCE = ["SOL", "SUI", "FET", "NEAR", "ATOM"]    # current universe = the bar
LOOKBACK, GATE, FEE = 365, 0.2, 0.0005
END = "2026-06-06"
CACHE = PROJECT_ROOT / ".cache_sigq"


def _daily(sym):
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{sym}.pkl"
    if fp.exists():
        return pickle.load(open(fp, "rb"))
    try:
        kl = _client().get_historical_klines(f"{sym}USDT", "1d", "2020-01-01", END,
                                             klines_type=HistoricalKlinesType.FUTURES)
    except Exception:
        kl = []
    if not kl:
        pickle.dump(None, open(fp, "wb"))
        return None
    df = pd.DataFrame(kl, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"])
    idx = pd.DatetimeIndex(pd.to_datetime(df["t"], unit="ms", utc=True).dt.normalize())
    s = pd.Series(pd.to_numeric(df["c"], errors="coerce").to_numpy(), index=idx)
    s = s[~s.index.duplicated(keep="last")]
    pickle.dump(s, open(fp, "wb"))
    return s


def signal_frame(close):
    c = close.to_numpy()
    dates = list(close.index)
    cser = pd.Series(c)
    pos = 0.0
    rows = []
    for t in range(LOOKBACK, len(c)):
        labels = label_regimes(cser.iloc[t - LOOKBACK:t], window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
        s = 0.0 if labels.empty else float(signal_from_matrix(build_transition_matrix(labels), int(labels.iloc[-1])))
        desired = 1.0 if s > GATE else (-1.0 if s < -GATE else 0.0)
        if pos == 0.0:
            pos = desired
        elif desired != 0.0 and desired != pos:
            pos = desired
        rows.append((dates[t], pos, c[t] / c[t - 1] - 1.0))
    df = pd.DataFrame(rows, columns=["date", "pos", "ret"]).set_index("date")
    turn = df["pos"].diff()
    turn.iloc[0] = df["pos"].iloc[0]
    df["net"] = df["pos"] * df["ret"] - FEE * turn.abs()
    return df


def _sharpe(r):
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def _hit(d):
    h = d[d["pos"] != 0]
    return float((h["pos"] * h["ret"] > 0).mean()) if len(h) else float("nan")


def _stats(df):
    turn = df["pos"].diff()
    turn.iloc[0] = df["pos"].iloc[0]
    return dict(
        sharpe=_sharpe(df["net"]), hit=_hit(df),
        ann=float(np.prod(1 + df["net"].to_numpy()) ** (365 / max(len(df), 1)) - 1),
        ntr=int((turn.abs() > 0).sum()), n=len(df), held=int((df["pos"] != 0).sum()),
    )


def _show(coins, tag):
    print(f"\n-- {tag} --")
    out = {}
    for sym in coins:
        s = _daily(sym)
        if s is None or len(s) < LOOKBACK + 60:
            print(f"  {sym:<7}  insufficient futures history ({0 if s is None else len(s)} bars)")
            continue
        df = signal_frame(s)
        st = _stats(df)
        out[sym] = st
        yrs = []
        for y in range(2022, 2027):
            d = df[df.index.year == y]
            if len(d) >= 40:
                yrs.append(f"{y}:{_sharpe(d['net']):+.1f}({_hit(d) * 100:.0f})")
        print(f"  {sym:<7}{st['sharpe']:>7.2f}{st['hit'] * 100:>6.0f}%{st['ann']:>+8.0%}{st['ntr']:>7}tr"
              f"{st['n']:>6}d {s.index.min():%Y-%m}  " + " ".join(yrs))
    return out


def run():
    print("=" * 104)
    print("MOMENTUM SIGNAL QUALITY — 20d Markov long/short, signal-only (fee 0.05%/turn), full futures hist thru " + END)
    print("365d lookback · gate 0.2 · FLIP off   |   columns: Sharpe  hit%  annRet  trades  days  since  per-year Sharpe(hit%)")
    print("=" * 104)
    cand = _show(CANDIDATES, "CANDIDATES")
    ref = _show(REFERENCE, "REFERENCE — current universe (the 'works well' bar)")
    if ref:
        rs = np.mean([v["sharpe"] for v in ref.values()])
        rh = np.mean([v["hit"] for v in ref.values()])
        print(f"\n  >>> reference mean: Sharpe {rs:+.2f}, hit {rh * 100:.0f}%   (a candidate 'works' if it's near/above this)")

    print("\n" + "=" * 104)
    print("MIN-NOTIONAL CHECK (Binance USDS-M futures)")
    print("=" * 104)
    info = _client().futures_exchange_info()
    bysym = {s["symbol"]: s for s in info["symbols"]}
    px = {}
    try:
        for t in _client().futures_mark_price():
            px[t["symbol"]] = float(t["markPrice"])
    except Exception as e:
        print("(mark price fetch failed:", e, ")")
    print(f"  {'coin':<7}{'minNotnl':>9}{'minQty':>11}{'stepSize':>11}{'markPx':>11}{'minOrder$':>11}{'futSince':>10}  flags")
    print("  " + "-" * 86)
    for sym in CANDIDATES + ["LINK"]:
        s = bysym.get(f"{sym}USDT")
        if s is None:
            print(f"  {sym:<7}  NOT LISTED on Binance futures")
            continue
        mn = minq = step = None
        for f in s["filters"]:
            if f["filterType"] == "MIN_NOTIONAL":
                mn = float(f.get("notional", f.get("minNotional", 0)))
            if f["filterType"] == "LOT_SIZE":
                minq = float(f["minQty"]); step = float(f["stepSize"])
        price = px.get(f"{sym}USDT", float("nan"))
        minorder = max(mn or 0, (minq or 0) * price)
        onboard = pd.to_datetime(s.get("onboardDate", 0), unit="ms").strftime("%Y-%m") if s.get("onboardDate") else "?"
        flags = "HIGH-MIN" if (mn and mn > 10) else ""
        print(f"  {sym:<7}{(mn or 0):>9.1f}{(minq or 0):>11.4g}{(step or 0):>11.4g}{price:>11.5g}{minorder:>11.2f}{onboard:>10}  {flags}")

    print("\n  Bot smallest-stake clearance — smallest position ~= 1% base x ~0.5 mult / 3x lev = 0.17% of capital:")
    print(f"  {'account$':>9}{'1% tier$':>10}{'~smallest$':>12}   clears $5 floor?")
    for acct in (300, 1000, 5000, 10000):
        small = 0.01 * 0.5 / 3 * acct
        print(f"  {acct:>9}{0.01 * acct:>10.2f}{small:>12.2f}   {'YES' if small >= 5 else 'NO (too small)'}")
    print("  (deployed: dry_run_wallet $10000, tradable_ratio 0.50 -> ~$5000 sizing capital -> smallest pos ~$8-17)")
    print("=" * 104)


if __name__ == "__main__":
    run()
