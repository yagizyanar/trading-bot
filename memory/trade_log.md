# Trade Log

Append-only chronological log of every trade and every circuit-breaker event.
Format per trade:

```
## YYYY-MM-DD HH:MM UTC
Coin: SOL/USDT
Direction: LONG | SHORT
Entry: $X.XX
Exit: $X.XX  (or OPEN)
Quantity: X.XX
Leverage: 1x | 2x
P&L: +$X.XX (+X.XX%)
Reason In: [signal description]
Reason Out: [exit description]
Outcome: WIN | LOSS | OPEN
```

Circuit-breaker events use prefix `[CIRCUIT BREAKER]` and include level, trigger, equity.

---
