"""2024 out-of-sample walk-forward backtest + sensitivity analyses.

Validation run (2026-06-03). WHY this and not `freqtrade backtesting`:
the live strategy's entries come from a Postgres `signal_log` row written by the
live routine — `populate_indicators` broadcasts the *latest* decision to every
candle, so a Freqtrade backtest over 2024 would stamp today's signal on 2024
bars (look-ahead garbage). FreqAI is not in the decision path (no model exists;
its predictions are never read). The valid tool is the daily walk-forward
harness, which re-estimates the Markov signal each day from a trailing 365-day
window (no look-ahead — causality is unit-tested) and applies costs + the −5%
daily stop + the position cap.

UNIVERSE: only coins with a full 2024 + >=365d pre-2024 lookback (so 2024 is
truly out-of-sample). 11 coins are excluded — they didn't exist (S, listed
2025), rebranded mid-2024 (POL), listed in 2024 (WIF/TAO/RENDER), or lacked
365d of pre-2024 history (ARB/SUI/GMX/1000PEPE/FET/1000BONK). Excluding them
also removes most 2024 high-fliers, so this is LESS survivorship-inflated.

LIMITATION: daily simulation. The −5% stop is approximated on daily bars; the
intraday +15% TP, 2% trailing, 15m execution, and the live CIRCUIT BREAKERS
(10% drawdown lock, 5% daily / 8% weekly loss) are NOT modelled. This validates
the DIRECTIONAL/signal edge + cost drag, not the live system's achievable P&L.

ANALYSES PRINTED:
  1. Baseline   — cost 0.07%/side (0.05% fee + 0.02% slip), both sides, + monthly
  2. Fee sens.  — cost 0.15%/side (mimics 15m-execution churn), both sides
  3. Side split — long-only vs short-only (standalone), to attribute the edge
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from sentiment.binance_data import fetch_binance_ohlcv
from backtest.daily_walk_forward import simulate_coin

# ---- Parameters (current deployed) ----------------------------------------
UNIVERSE = ["SOL", "AVAX", "LINK", "DOT", "INJ", "OP", "APT",
            "NEAR", "ATOM", "SAND", "MANA", "AXS", "DYDX"]
IN_SAMPLE = 365          # live regime window
GATE = 0.2               # markov direction gate (harness default = live)
STOP = 0.05              # hard −5% stop (daily approximation)
CAP = 10                 # max_open_trades
START_CAP = 300.0

COST_BASE = 0.0007       # 0.05% fee + 0.02% slippage per side
COST_HIGH = 0.0015       # 0.15%/side — mimics heavier 15m-execution churn

EVAL_START = pd.Timestamp("2024-01-01", tz="UTC")
EVAL_END = pd.Timestamp("2024-12-31", tz="UTC")
LOOKBACK_START = pd.Timestamp("2023-01-01", tz="UTC")  # ~365d before 2024


def fetch_close(sym: str):
    df = fetch_binance_ohlcv(sym, interval="1d", limit=1500)
    if df is None or df.empty:
        return None
    return df["close"].astype(float)


def trade_runs(pos: np.ndarray, ret: np.ndarray, cost: float) -> list[tuple[float, int]]:
    """(net_return, duration_days) for each contiguous same-sign position run."""
    trades: list[tuple[float, int]] = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 0.0:
            i += 1
            continue
        sign = pos[i]
        fac, dur, j = 1.0, 0, i
        while j < n and pos[j] == sign:
            fac *= (1.0 + ret[j] * sign)
            dur += 1
            j += 1
        fac *= (1.0 - cost) ** 2          # entry + exit
        trades.append((fac - 1.0, dur))
        i = j
    return trades


def run_scenario(coins, common, POS, GROSS, MOM, per, cost: float, side: str) -> dict:
    """Run the equal-weight capped portfolio for a cost + side ('both'/'long'/'short')."""
    if side == "long":
        POSm = np.where(POS > 0, POS, 0.0)
    elif side == "short":
        POSm = np.where(POS < 0, POS, 0.0)
    else:
        POSm = POS

    ndays = len(common)
    net = np.zeros(ndays)
    eq = START_CAP
    eq_curve = np.zeros(ndays)
    fee_dollars = 0.0
    prev_w = np.zeros(len(coins))
    active_counts = []
    for d in range(ndays):
        active = np.where(POSm[:, d] != 0.0)[0]
        if len(active) > CAP:
            active = active[np.argsort(-MOM[active, d])][:CAP]
        active_counts.append(len(active))
        w = np.zeros(len(coins))
        g = 0.0
        if len(active) > 0:
            wt = 1.0 / len(active)
            for i in active:
                w[i] = wt * POSm[i, d]
                g += wt * GROSS[i, d]
        turnover = np.abs(w - prev_w).sum()
        c = cost * turnover
        net[d] = g - c
        fee_dollars += eq * c
        eq *= (1.0 + net[d])
        eq_curve[d] = eq
        prev_w = w

    net_s = pd.Series(net, index=common)
    sd = net_s.std(ddof=1)
    sharpe = float(net_s.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0
    runmax = np.maximum.accumulate(eq_curve)
    max_dd = float(((eq_curve - runmax) / runmax).min())

    all_trades: list[tuple[float, int]] = []
    for ci, c in enumerate(coins):
        p = POSm[ci]
        r = per[c].reindex(common)["ret"].fillna(0.0).to_numpy()
        all_trades += trade_runs(p, r, cost)
    n_trades = len(all_trades)
    wins = [t for t, _ in all_trades if t > 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0
    avg_dur = float(np.mean([d for _, d in all_trades])) if all_trades else 0.0

    monthly = net_s.groupby([net_s.index.year, net_s.index.month]).apply(
        lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0)
    )
    return dict(
        total_return=eq / START_CAP - 1.0, final=eq, sharpe=sharpe, max_dd=max_dd,
        win_rate=win_rate, n_wins=len(wins), n_trades=n_trades, avg_dur=avg_dur,
        fee_dollars=fee_dollars, avg_active=float(np.mean(active_counts)), monthly=monthly,
    )


def main() -> None:
    print("=" * 80)
    print("2024 OUT-OF-SAMPLE WALK-FORWARD  |  13-coin clean universe  |  365d lookback")
    print(f"cap={CAP}  stop=-{STOP:.0%}  start=${START_CAP:.0f}  lev=1x   "
          "(daily model — no intraday TP/trailing/circuit-breakers)")
    print("=" * 80)

    # ---- fetch + per-coin walk-forward (ONCE; reused by all scenarios) -----
    closes = {}
    for c in UNIVERSE:
        s = fetch_close(f"{c}USDT")
        if s is None:
            print(f"  WARN: no data for {c}")
            continue
        closes[c] = s[(s.index >= LOOKBACK_START) & (s.index <= EVAL_END)].dropna()

    btc = fetch_close("BTCUSDT")
    btc24 = btc[(btc.index >= EVAL_START) & (btc.index <= EVAL_END)].dropna()
    btc_hodl = float(btc24.iloc[-1] / btc24.iloc[0] - 1.0)

    per = {}
    for c, s in closes.items():
        res = simulate_coin(s, coin=c, in_sample=IN_SAMPLE, cost_per_side=0.0, stop=STOP, gate=GATE)
        if res.positions.empty:
            continue
        oos_dates = s.index[IN_SAMPLE:]
        L = min(len(oos_dates), len(res.positions))
        ret_full = s.pct_change().fillna(0.0).to_numpy()
        dfc = pd.DataFrame(
            {"pos": res.positions.to_numpy()[:L],
             "gross": np.nan_to_num(res.gross_daily.to_numpy()[:L]),
             "ret": ret_full[IN_SAMPLE:IN_SAMPLE + L]},
            index=oos_dates[:L],
        )
        per[c] = dfc[(dfc.index >= EVAL_START) & (dfc.index <= EVAL_END)]

    coins = list(per.keys())
    common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()
    POS = np.vstack([per[c].reindex(common)["pos"].fillna(0.0).to_numpy() for c in coins])
    GROSS = np.vstack([per[c].reindex(common)["gross"].fillna(0.0).to_numpy() for c in coins])
    MOM = np.vstack([closes[c].pct_change(20).reindex(common).abs().fillna(0.0).to_numpy() for c in coins])
    print(f"Universe: {len(coins)} coins  |  2024 days: {len(common)} "
          f"({common.min().date()} → {common.max().date()})")

    base = run_scenario(coins, common, POS, GROSS, MOM, per, COST_BASE, "both")
    highf = run_scenario(coins, common, POS, GROSS, MOM, per, COST_HIGH, "both")
    longo = run_scenario(coins, common, POS, GROSS, MOM, per, COST_BASE, "long")
    shorto = run_scenario(coins, common, POS, GROSS, MOM, per, COST_BASE, "short")

    # ---- 1. baseline headline + monthly -----------------------------------
    print("\n" + "-" * 80)
    print(f"1) BASELINE  (cost {COST_BASE:.4f}/side = 0.05% fee + 0.02% slip, both sides)")
    print("-" * 80)
    print(f"  Total return {base['total_return']:+.2%}   final ${base['final']:,.2f}   "
          f"Sharpe {base['sharpe']:.2f}   maxDD {base['max_dd']:.2%}")
    print(f"  Win rate {base['win_rate']:.1%} ({base['n_wins']}/{base['n_trades']})   "
          f"avg dur {base['avg_dur']:.1f}d   fee+slip ${base['fee_dollars']:,.2f}   "
          f"avg pos {base['avg_active']:.1f}/{CAP}")
    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cells = "  ".join(f"{mnames[mo-1]} {v:+.1%}" for (_, mo), v in base["monthly"].items())
    print("  Monthly: " + cells)
    print(f"  Positive months: {int((base['monthly'] > 0).sum())}/{len(base['monthly'])}")

    # ---- 2. fee sensitivity ------------------------------------------------
    print("\n" + "-" * 80)
    print("2) FEE SENSITIVITY  (both sides)")
    print("-" * 80)
    print(f"  {'cost/side':<12}{'total ret':>12}{'final $':>12}{'Sharpe':>9}{'maxDD':>9}{'fee $':>10}{'trades':>8}")
    for label, r, cost in [("0.07% (base)", base, COST_BASE), ("0.15% (15m)", highf, COST_HIGH)]:
        print(f"  {label:<12}{r['total_return']:>+12.1%}{r['final']:>12,.0f}"
              f"{r['sharpe']:>9.2f}{r['max_dd']:>9.1%}{r['fee_dollars']:>10,.0f}{r['n_trades']:>8}")

    # ---- 3. long vs short split (standalone) ------------------------------
    print("\n" + "-" * 80)
    print(f"3) LONG vs SHORT  (standalone, cost {COST_BASE:.4f}/side)")
    print("-" * 80)
    print(f"  {'side':<14}{'total ret':>12}{'Sharpe':>9}{'maxDD':>9}{'win%':>8}{'trades':>8}{'avg pos':>9}")
    for label, r in [("both", base), ("long-only", longo), ("short-only", shorto)]:
        print(f"  {label:<14}{r['total_return']:>+12.1%}{r['sharpe']:>9.2f}{r['max_dd']:>9.1%}"
              f"{r['win_rate']:>8.0%}{r['n_trades']:>8}{r['avg_active']:>9.1f}")
    print("  (standalone runs weight only their own side, so long+short do NOT sum to 'both')")

    # ---- benchmarks --------------------------------------------------------
    ew = np.vstack([per[c].reindex(common)["ret"].fillna(0.0).to_numpy() for c in coins]).mean(axis=0)
    print("\n  BENCHMARKS 2024:  "
          f"strategy {base['total_return']:+.1%}   BTC HODL {btc_hodl:+.1%}   "
          f"13-coin EW HODL {float(np.prod(1.0 + ew) - 1.0):+.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
