"""Research: BTC correlation of liquid Binance USDT-perp futures NOT in our universe.

For each candidate: daily-return Pearson correlation to BTCUSDT-perp per calendar
year (2022/23/24) and full 2022-24 overlap, 24h quote volume (liquidity proxy), and
first futures-bar date (listing / coverage). Surfaces diversifiers (corr < 0.6),
grouped by narrative. A few CURRENT-universe coins are correlated as a baseline.

Correlation is on FUTURES daily closes — matches the trading venue AND confirms
availability (no klines returned => not listed on Binance futures).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from binance.enums import HistoricalKlinesType

from config.settings import PROJECT_ROOT
from backtest.oos_multiyear import _client

CACHE = PROJECT_ROOT / ".cache_corr"
START, END = "2021-12-20", "2025-01-01"
YEARS = [2022, 2023, 2024]

NARRATIVES = {
    "AI/DePIN/Data": ["GRT", "WLD", "ARKM", "AKT", "AR", "FIL", "THETA", "IO", "JASMY", "GRASS", "VIRTUAL"],
    "DeFi":          ["UNI", "AAVE", "MKR", "CRV", "LDO", "PENDLE", "RUNE", "ENA", "JUP", "CAKE", "SNX", "COMP"],
    "Gaming/Meta":   ["GALA", "IMX", "APE", "ENJ", "GMT", "MAGIC", "BEAM", "PIXEL", "ILV", "FLOW"],
    "Meme":          ["DOGE", "1000SHIB", "1000FLOKI", "POPCAT", "NEIRO"],
    "L1/L2/Modular": ["ADA", "XRP", "TRX", "LTC", "BCH", "TIA", "SEI", "HBAR", "ALGO", "ETC", "STX", "EGLD", "ICP", "KAVA"],
    "Oracle/Interop": ["LINK", "PYTH", "BAND", "ZRO", "W", "AXL"],
    "Exchange/Pay/OG": ["BNB", "XLM", "NEO", "DASH", "ZEC", "IOTA", "EOS", "XTZ"],
}
BASELINE = ["SOL", "AVAX", "ARB", "FET", "SUI", "NEAR", "RENDER"]   # current universe, for contrast


def _fut_daily(sym):
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{sym}.pkl"
    if fp.exists():
        return pickle.load(open(fp, "rb"))
    try:
        kl = _client().get_historical_klines(f"{sym}USDT", "1d", START, END,
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


def _ret(s):
    return s.pct_change().dropna().rename("coin")


def _corr(cr, br, year=None):
    a, b = cr, br.rename("btc")
    if year is not None:
        a = a[a.index.year == year]
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if year is None:
        j = j[(j.index.year >= 2022) & (j.index.year <= 2024)]
    if len(j) < 40:
        return None
    return float(j["coin"].corr(j["btc"]))


def _fmt(x):
    return f"{x:+.2f}" if x is not None else " n/a"


def _measure(sym, br):
    s = _fut_daily(sym)
    if s is None:
        return None
    cr = _ret(s)
    cy = {y: _corr(cr, br, y) for y in YEARS}
    cfull = _corr(cr, br)
    vals = [v for v in cy.values() if v is not None]
    avg = float(np.mean(vals)) if vals else None
    start = s.index.min().strftime("%Y-%m")
    return dict(sym=sym, c22=cy[2022], c23=cy[2023], c24=cy[2024], full=cfull, avg=avg, start=start)


def run():
    btc = _fut_daily("BTC")
    br = _ret(btc)
    vol = {}
    try:
        for t in _client().futures_ticker():
            vol[t["symbol"]] = float(t.get("quoteVolume", 0))
    except Exception as e:
        print("(volume fetch failed:", e, ")")

    print("=" * 96)
    print("BTC-CORRELATION SCAN — Binance USDT-perp futures NOT in our universe (daily returns)")
    print("=" * 96)
    hdr = f"  {'sym':<11}{'2022':>7}{'2023':>7}{'2024':>7}{'FULL':>7}{'avg':>7}{'$vol/d':>9}{'since':>9}"
    allrows = []
    for narr, syms in NARRATIVES.items():
        print(f"\n{narr}")
        print(hdr)
        print("  " + "-" * 80)
        measured = []
        for sym in syms:
            m = _measure(sym, br)
            if m is None:
                print(f"  {sym:<11}{'— not on Binance futures —':>40}")
                continue
            vM = vol.get(f"{sym}USDT", 0) / 1e6
            m["vol"] = vM
            measured.append(m)
            allrows.append((narr, m))
        for m in sorted(measured, key=lambda x: (x["full"] is None, x["full"] if x["full"] is not None else 9)):
            print(f"  {m['sym']:<11}{_fmt(m['c22']):>7}{_fmt(m['c23']):>7}{_fmt(m['c24']):>7}"
                  f"{_fmt(m['full']):>7}{_fmt(m['avg']):>7}{m['vol']:>7,.0f}M{m['start']:>9}")

    print("\n" + "=" * 96)
    print("BASELINE — a few CURRENT-universe coins (for contrast)")
    print(hdr)
    print("  " + "-" * 80)
    base_full = []
    for sym in BASELINE:
        m = _measure(sym, br)
        if m is None:
            continue
        vM = vol.get(f"{sym}USDT", 0) / 1e6
        if m["full"] is not None:
            base_full.append(m["full"])
        print(f"  {sym:<11}{_fmt(m['c22']):>7}{_fmt(m['c23']):>7}{_fmt(m['c24']):>7}"
              f"{_fmt(m['full']):>7}{_fmt(m['avg']):>7}{vM:>7,.0f}M{m['start']:>9}")
    if base_full:
        print(f"  >>> current-universe sample mean FULL corr = {np.mean(base_full):+.2f}  (this is the BTC-beta we want to dilute)")

    # shortlist: full corr < 0.65, full history (since <= 2022), vol >= 50M
    print("\n" + "=" * 96)
    print("DIVERSIFIER SHORTLIST — full-history (since 2022), 24h vol >= $50M, sorted by FULL corr asc")
    print(hdr + "   narrative")
    print("  " + "-" * 92)
    short = [(n, m) for (n, m) in allrows
             if m["full"] is not None and m["start"] <= "2022-02" and m.get("vol", 0) >= 50]
    for n, m in sorted(short, key=lambda x: x[1]["full"])[:18]:
        print(f"  {m['sym']:<11}{_fmt(m['c22']):>7}{_fmt(m['c23']):>7}{_fmt(m['c24']):>7}"
              f"{_fmt(m['full']):>7}{_fmt(m['avg']):>7}{m['vol']:>7,.0f}M{m['start']:>9}   {n}")

    print("\n" + "=" * 96)
    print("NEWER NARRATIVE NAMES (listed after 2022 — partial history, corr less reliable)")
    print(hdr + "   narrative")
    print("  " + "-" * 92)
    newer = [(n, m) for (n, m) in allrows
             if m["full"] is not None and m["start"] > "2022-02" and m.get("vol", 0) >= 50]
    for n, m in sorted(newer, key=lambda x: x[1]["full"])[:12]:
        print(f"  {m['sym']:<11}{_fmt(m['c22']):>7}{_fmt(m['c23']):>7}{_fmt(m['c24']):>7}"
              f"{_fmt(m['full']):>7}{_fmt(m['avg']):>7}{m['vol']:>7,.0f}M{m['start']:>9}   {n}")
    print("=" * 96)


if __name__ == "__main__":
    run()
