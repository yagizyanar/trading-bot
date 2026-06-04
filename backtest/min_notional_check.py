"""Item 3: min-notional probe. Fetch Binance USDⓈ-M min-notional per traded coin
and show the expected per-position stake at a given account size."""
from __future__ import annotations

import sys

from binance.client import Client

from config.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY, TARGET_COINS

ACCOUNT = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0


def _min_notional(symbol_filters):
    for f in symbol_filters:
        if f["filterType"] == "MIN_NOTIONAL":
            return float(f.get("notional", f.get("minNotional", 0.0)))
    return None


def main():
    cli = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    info = cli.futures_exchange_info()
    by_sym = {s["symbol"]: s for s in info["symbols"]}

    print(f"Account = {ACCOUNT:.0f} USDT, 1x leverage (notional = stake)")
    print(f"{'coin':<10}{'min_notional':>14}{'5% stake':>11}{'3%':>8}{'1%':>8}  flags")
    print("-" * 70)
    base_tiers = {"5%": 0.05, "3%": 0.03, "1%": 0.01}
    worst = []
    for coin in TARGET_COINS:
        sym = f"{coin}USDT"
        s = by_sym.get(sym)
        if s is None:
            print(f"{coin:<10}{'NOT LISTED':>14}")
            continue
        mn = _min_notional(s["filters"])
        stakes = {k: ACCOUNT * v for k, v in base_tiers.items()}
        flags = []
        if mn is not None:
            if stakes["1%"] < mn:
                flags.append("1%<min")
            if stakes["3%"] < mn:
                flags.append("3%<min")
            if stakes["5%"] < mn:
                flags.append("5%<min")
            if mn > 10:
                flags.append(f"HIGH-MIN({mn:.0f})")
        if flags:
            worst.append(coin)
        print(f"{coin:<10}{(mn if mn is not None else 0):>14.2f}"
              f"{stakes['5%']:>11.2f}{stakes['3%']:>8.2f}{stakes['1%']:>8.2f}  {','.join(flags)}")
    print("-" * 70)
    print(f"Coins with any sub-min-notional tier or high min: {worst}")


if __name__ == "__main__":
    main()
