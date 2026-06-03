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

// Format a Freqtrade timestamp as "DD.MM HH:MM" in UTC.
// Freqtrade serializes timestamps as "YYYY-MM-DD HH:MM:SS" without an
// explicit timezone but always in UTC; we normalize by appending "Z"
// when no tz marker is present so Date() doesn't interpret it as local.
// DB-fallback rows use isoformat() which may already carry "+00:00".
function fmtTs(s) {
  if (!s) return "—";
  let iso = String(s).replace(" ", "T");
  if (!/Z|[+-]\d{2}:?\d{2}$/.test(iso)) iso += "Z";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  const dd = String(d.getUTCDate()).padStart(2, "0");
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mi = String(d.getUTCMinutes()).padStart(2, "0");
  return `${dd}.${mm} ${hh}:${mi}`;
}

function pnlClass(n) {
  if (n == null || isNaN(n) || Math.abs(n) < 1e-6) return "text-slate-300";
  return n > 0 ? "text-emerald-400" : "text-rose-400";
}

function sideClass(side) {
  return side === "LONG" ? "text-emerald-300" : side === "SHORT" ? "text-rose-300" : "text-slate-300";
}

const FALLBACK_POLL_MS = 30_000;  // safety poll when WS hasn't delivered yet
const CLOSED_PAGE = 50;           // closed trades loaded per page

// Merge two trade lists keyed by unique id (Freqtrade trade_id), newest-first.
// Lets the 30s top-refresh and "Load more" pages combine without duplicates.
function mergeById(existing, incoming) {
  const m = new Map();
  const key = (r) => r.id ?? `${r.coin}-${r.exit_ts}`;
  for (const r of existing) m.set(key(r), r);
  for (const r of incoming) m.set(key(r), r);
  return [...m.values()].sort(
    (a, b) => String(b.exit_ts || "").localeCompare(String(a.exit_ts || ""))
  );
}

export default function PositionsTable({ kind, title, live }) {
  const isOpen = kind === "open";
  const [rows, setRows] = useState([]);
  const [haveWsData, setHaveWsData] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  // OPEN: drive from the WebSocket live payload when present
  useEffect(() => {
    if (!isOpen) return;
    if (live && Array.isArray(live.positions)) {
      setRows(live.positions);
      setHaveWsData(true);
    }
  }, [isOpen, live]);

  // OPEN: fallback poll only while waiting for the first WS message
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const load = () => {
      if (haveWsData) return;
      api.positionsOpen().then((r) => { if (!cancelled) setRows(r); }).catch(() => {});
    };
    load();
    const id = setInterval(load, FALLBACK_POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [isOpen, haveWsData]);

  // CLOSED: paginated. Initial newest page + 30s refresh of the top page so
  // new closes appear; "Load more" appends older pages by offset. Dedup-by-id
  // keeps the top-refresh and load-more pages from colliding.
  useEffect(() => {
    if (isOpen) return;
    let cancelled = false;
    api.positionsClosed(CLOSED_PAGE, 0).then((r) => {
      if (cancelled) return;
      setRows(mergeById([], r));
      setHasMore(r.length === CLOSED_PAGE);
    }).catch(() => {});
    const id = setInterval(() => {
      api.positionsClosed(CLOSED_PAGE, 0)
        .then((r) => { if (!cancelled) setRows((prev) => mergeById(prev, r)); })
        .catch(() => {});
    }, FALLBACK_POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [isOpen]);

  const loadMore = () => {
    setLoadingMore(true);
    api.positionsClosed(CLOSED_PAGE, rows.length).then((r) => {
      setRows((prev) => mergeById(prev, r));
      setHasMore(r.length === CLOSED_PAGE);
    }).catch(() => {}).finally(() => setLoadingMore(false));
  };

  const refreshHint = isOpen
    ? (haveWsData ? "live · 5s" : "polling")
    : `${rows.length} loaded · 30s`;

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-slate-400">{title}</div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500">{refreshHint}</div>
      </div>

      <div className={isOpen ? "overflow-x-auto" : "overflow-x-auto overflow-y-auto max-h-[560px]"}>
        {isOpen ? <OpenTable rows={rows} /> : <ClosedTable rows={rows} />}
      </div>

      {!isOpen && (
        <div className="mt-3 flex items-center justify-center gap-3 text-xs">
          <span className="text-slate-500">{rows.length} closed trades loaded</span>
          {hasMore ? (
            <button
              type="button"
              onClick={loadMore}
              disabled={loadingMore}
              className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200 disabled:opacity-50"
            >
              {loadingMore ? "Loading…" : "Load more"}
            </button>
          ) : (
            <span className="text-slate-600">— all loaded —</span>
          )}
        </div>
      )}
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
          <th className="text-left py-1 px-2">Opened</th>
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
            <td className="py-1 px-2 font-mono text-slate-300 text-xs">{fmtTs(r.entry_ts)}</td>
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
          <tr><td colSpan="12" className="py-3 px-2 text-slate-500 italic">no open positions</td></tr>
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
          <th className="text-right py-1 px-2">Lev</th>
          <th className="text-left py-1 px-2">Opened</th>
          <th className="text-left py-1 px-2">Closed</th>
          <th className="text-right py-1 px-2">Entry</th>
          <th className="text-right py-1 px-2">Exit</th>
          <th className="text-right py-1 px-2">Size (USDT)</th>
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
            <td className="py-1 px-2 text-right">{r.leverage ?? 1}x</td>
            <td className="py-1 px-2 font-mono text-slate-300 text-xs">{fmtTs(r.entry_ts)}</td>
            <td className="py-1 px-2 font-mono text-slate-300 text-xs">{fmtTs(r.exit_ts)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtPrice(r.entry_price)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtPrice(r.exit_price)}</td>
            <td className="py-1 px-2 text-right font-mono">{fmtMoney(r.size_usdt)}</td>
            <td className={`py-1 px-2 text-right font-mono ${pnlClass(r.pnl_pct)}`}>
              {r.pnl_pct != null ? `${(r.pnl_pct * 100).toFixed(2)}%` : "—"}
            </td>
            <td className="py-1 px-2">{r.outcome}</td>
            <td className="py-1 px-2 text-slate-400 text-xs">{r.reason_out || "—"}</td>
          </tr>
        ))}
        {rows.length === 0 && (
          <tr><td colSpan="11" className="py-3 px-2 text-slate-500 italic">no closed positions</td></tr>
        )}
      </tbody>
    </table>
  );
}
