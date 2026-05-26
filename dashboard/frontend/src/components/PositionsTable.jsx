import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const STABLE_SUFFIXED = new Set(["USDT", "USDC", "DAI", "FDUSD", "BUSD"]);
function fmtMoney(n, sym = "USDT", digits = 2) {
  if (n == null || isNaN(n)) return "—";
  if (STABLE_SUFFIXED.has(sym)) {
    return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  try {
    return n.toLocaleString("en-US", { style: "currency", currency: sym, maximumFractionDigits: digits });
  } catch {
    return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
}

function fmtPrice(n) {
  if (n == null || isNaN(n)) return "—";
  // Pick precision based on magnitude — crypto prices vary widely
  if (n >= 100) return n.toFixed(2);
  if (n >= 1)   return n.toFixed(4);
  return n.toFixed(6);
}

function fmtQty(n) {
  if (n == null || isNaN(n)) return "—";
  if (n >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (n >= 1)    return n.toFixed(2);
  return n.toFixed(4);
}

function pnlClass(n) {
  if (n == null || isNaN(n) || Math.abs(n) < 1e-6) return "text-slate-300";
  return n > 0 ? "text-emerald-400" : "text-rose-400";
}

function sideClass(side) {
  return side === "LONG" ? "text-emerald-300" : side === "SHORT" ? "text-rose-300" : "text-slate-300";
}

const FALLBACK_POLL_MS = 30_000;  // safety poll when WS hasn't delivered yet

export default function PositionsTable({ kind, title, live }) {
  const [rows, setRows] = useState([]);
  const [haveWsData, setHaveWsData] = useState(false);

  // Drive open positions from the WebSocket live payload when present
  useEffect(() => {
    if (kind !== "open") return;
    if (live && Array.isArray(live.positions)) {
      setRows(live.positions);
      setHaveWsData(true);
    }
  }, [kind, live]);

  // Fallback polling: closed positions always poll; open positions only poll
  // while we're waiting for the first WS message (or as a safety net).
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      if (kind === "open" && haveWsData) return;   // WS owns it
      const p = kind === "open" ? api.positionsOpen() : api.positionsClosed(50);
      p.then((r) => { if (!cancelled) setRows(r); }).catch(() => {});
    };
    load();
    const id = setInterval(load, FALLBACK_POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [kind, haveWsData]);

  const isOpen = kind === "open";
  const refreshHint = isOpen
    ? (haveWsData ? "live · 5s" : "polling")
    : "polling · 30s";

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-slate-400">{title}</div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500">{refreshHint}</div>
      </div>

      <div className="overflow-x-auto">
        {isOpen ? <OpenTable rows={rows} /> : <ClosedTable rows={rows} />}
      </div>
    </div>
  );
}


// ---- Open positions: full detail set ------------------------------------------------
function OpenTable({ rows }) {
  return (
    <table className="min-w-full text-sm">
      <thead className="text-slate-400 text-xs">
        <tr>
          <th className="text-left py-1 px-2">Coin</th>
          <th className="text-left py-1 px-2">Side</th>
          <th className="text-right py-1 px-2">Lev</th>
          <th className="text-right py-1 px-2">Entry</th>
          <th className="text-right py-1 px-2">Current</th>
          <th className="text-right py-1 px-2">Qty</th>
          <th className="text-right py-1 px-2">Size (USDT)</th>
          <th className="text-right py-1 px-2">P&amp;L (USDT)</th>
          <th className="text-right py-1 px-2">P&amp;L %</th>
          <th className="text-right py-1 px-2">SL</th>
          <th className="text-right py-1 px-2">TP</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id ?? `${r.coin}-${r.entry_ts}`} className="border-t border-slate-700">
            <td className="py-1 px-2 font-mono font-semibold">{r.coin}</td>
            <td className={`py-1 px-2 font-medium ${sideClass(r.side)}`}>{r.side}</td>
            <td className="py-1 px-2 text-right">{r.leverage ?? 1}x</td>
            <td className="py-1 px-2 text-right font-mono">{fmtPrice(r.entry_price)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtPrice(r.current_price)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtQty(r.quantity)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtMoney(r.size_usdt)}</td>
            <td className={`py-1 px-2 text-right font-mono ${pnlClass(r.pnl_usd)}`}>
              {r.pnl_usd != null ? `${r.pnl_usd >= 0 ? "+" : ""}${fmtMoney(r.pnl_usd)}` : "—"}
            </td>
            <td className={`py-1 px-2 text-right font-mono ${pnlClass(r.pnl_pct)}`}>
              {r.pnl_pct != null ? `${r.pnl_pct >= 0 ? "+" : ""}${(r.pnl_pct * 100).toFixed(2)}%` : "—"}
            </td>
            <td className="py-1 px-2 text-right font-mono text-rose-300/80">{fmtPrice(r.stop_loss_price)}</td>
            <td className="py-1 px-2 text-right font-mono text-emerald-300/80">{fmtPrice(r.take_profit_price)}</td>
          </tr>
        ))}
        {rows.length === 0 && (
          <tr><td colSpan="11" className="py-3 px-2 text-slate-500 italic">no open positions</td></tr>
        )}
      </tbody>
    </table>
  );
}


// ---- Closed positions: compact ------------------------------------------------------
function ClosedTable({ rows }) {
  return (
    <table className="min-w-full text-sm">
      <thead className="text-slate-400 text-xs">
        <tr>
          <th className="text-left py-1 px-2">Coin</th>
          <th className="text-left py-1 px-2">Side</th>
          <th className="text-right py-1 px-2">Entry</th>
          <th className="text-right py-1 px-2">Exit</th>
          <th className="text-right py-1 px-2">P&amp;L %</th>
          <th className="text-left py-1 px-2">Outcome</th>
          <th className="text-left py-1 px-2">Reason</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.id ?? `${r.coin}-${r.exit_ts}`} className="border-t border-slate-700">
            <td className="py-1 px-2 font-mono">{r.coin}</td>
            <td className={`py-1 px-2 ${sideClass(r.side)}`}>{r.side}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtPrice(r.entry_price)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtPrice(r.exit_price)}</td>
            <td className={`py-1 px-2 text-right font-mono ${pnlClass(r.pnl_pct)}`}>
              {r.pnl_pct != null ? `${(r.pnl_pct * 100).toFixed(2)}%` : "—"}
            </td>
            <td className="py-1 px-2">{r.outcome}</td>
            <td className="py-1 px-2 text-slate-400 text-xs">{r.reason_out || "—"}</td>
          </tr>
        ))}
        {rows.length === 0 && (
          <tr><td colSpan="7" className="py-3 px-2 text-slate-500 italic">no closed positions</td></tr>
        )}
      </tbody>
    </table>
  );
}
