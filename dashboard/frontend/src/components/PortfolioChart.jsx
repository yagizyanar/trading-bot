import React, { useEffect, useRef, useState } from "react";
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { api } from "../api/client.js";

// Rolling buffer. Sized for 30 days of 5-min DB snapshots (≈8640 points)
// plus headroom for the WS-pushed live tail at 5s cadence.
const BUFFER_MAX = 10_000;
const HISTORY_DAYS = 30;
const FALLBACK_POLL_MS = 30_000;

// Stablecoins display as suffixed (USDT, USDC etc.); fiat as native currency.
const STABLE_SUFFIXED = new Set(["USDT", "USDC", "DAI", "FDUSD", "BUSD"]);
function fmtMoney(n, symbol) {
  if (n == null || isNaN(n)) return "—";
  const sym = symbol || "USD";
  if (STABLE_SUFFIXED.has(sym)) {
    return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " " + sym;
  }
  try {
    return n.toLocaleString("en-US", { style: "currency", currency: sym, maximumFractionDigits: 2 });
  } catch {
    return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " " + sym;
  }
}

function pnlTone(n) {
  if (n == null || isNaN(n) || Math.abs(n) < 0.005) return "text-slate-300";
  return n > 0 ? "text-emerald-400" : "text-rose-400";
}

function SourceBadge({ source, dryRun }) {
  if (source === "freqtrade") {
    return (
      <span className="inline-flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 rounded px-2 py-0.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        live{dryRun ? " · paper" : ""}
      </span>
    );
  }
  if (source === "snapshot") {
    return (
      <span className="text-[10px] font-medium uppercase tracking-wider text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-0.5">
        snapshot
      </span>
    );
  }
  return (
    <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400 bg-slate-700 rounded px-2 py-0.5">
      no data
    </span>
  );
}

function fmtClock(d) {
  // Same-day points: HH:MM:SS (live-tail granularity).
  // Older points: MM-DD HH:MM so the X-axis stays unambiguous across days.
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
}

export default function PortfolioChart({ live }) {
  // The single source of truth for "latest values": the most recent WS push
  // (or polled /latest snapshot when WS hasn't delivered yet). Big number,
  // peak line, and chart "now" point all read from the same object.
  const [latest, setLatest] = useState(null);
  const [haveWsData, setHaveWsData] = useState(false);
  const [err, setErr] = useState(null);

  // Buffer holds the combined historical DB snapshots (pre-populated on mount
  // from /api/performance/history) AND the WS-pushed live tail. Each entry:
  //   { tsLabel, tsEpoch, equity, daily_pnl, weekly_pnl }
  const bufferRef = useRef([]);
  const [buffer, setBuffer] = useState([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);

  // On mount, fetch 30 days of performance_snapshots and seed the buffer
  // so the chart isn't blank on page refresh. Merges with any WS pushes
  // that arrived during the fetch.
  useEffect(() => {
    let cancelled = false;
    api.perfHistory(HISTORY_DAYS).then((rows) => {
      if (cancelled) return;
      const seeded = (rows || []).map((r) => {
        const d = new Date(r.ts);
        return {
          tsLabel:    fmtClock(d),
          tsEpoch:    d.getTime(),
          equity:     r.total_equity,
          daily_pnl:  r.daily_pnl_usd,
          weekly_pnl: r.weekly_pnl_usd,
        };
      });
      // Merge with any WS pushes that landed before history arrived.
      const all = [...seeded, ...bufferRef.current].sort((a, b) => a.tsEpoch - b.tsEpoch);
      // Dedupe points within 1 second of each other (history's last row may
      // overlap with the first WS push).
      const deduped = [];
      for (const p of all) {
        const last = deduped[deduped.length - 1];
        if (last && Math.abs(last.tsEpoch - p.tsEpoch) < 1000) {
          deduped[deduped.length - 1] = p;
        } else {
          deduped.push(p);
        }
      }
      const trimmed = deduped.slice(-BUFFER_MAX);
      bufferRef.current = trimmed;
      setBuffer(trimmed);
      setHistoryLoaded(true);
    }).catch((e) => {
      if (cancelled) return;
      setErr(`history: ${e.message}`);
      setHistoryLoaded(true);  // unblock UI even on failure
    });
    return () => { cancelled = true; };
  }, []);

  // On every live update, append a buffer point and store latest
  useEffect(() => {
    const perf = live?.performance;
    if (!perf || perf.total_equity == null) return;
    setLatest(perf);
    setHaveWsData(true);
    const now = new Date(perf.ts || Date.now());
    const point = {
      tsLabel: fmtClock(now),
      tsEpoch: now.getTime(),
      equity:     perf.total_equity,
      daily_pnl:  perf.daily_pnl_usd,
      weekly_pnl: perf.weekly_pnl_usd,
    };
    // Dedup against the most recent entry (same timestamp → replace, not duplicate)
    const buf = bufferRef.current;
    const last = buf[buf.length - 1];
    let next;
    if (last && Math.abs(last.tsEpoch - point.tsEpoch) < 1000) {
      next = [...buf.slice(0, -1), point];
    } else {
      next = [...buf, point];
    }
    if (next.length > BUFFER_MAX) next = next.slice(-BUFFER_MAX);
    bufferRef.current = next;
    setBuffer(next);
  }, [live]);

  // Fallback poll for /api/performance/latest when WS hasn't delivered yet
  useEffect(() => {
    let cancelled = false;
    const loadLatest = () => {
      if (haveWsData) return;
      api.perfLatest().then((l) => {
        if (cancelled) return;
        setLatest(l);
        setErr(null);
      }).catch((e) => !cancelled && setErr(e.message));
    };
    loadLatest();
    const id = setInterval(loadLatest, FALLBACK_POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [haveWsData]);

  const sym = latest?.currency_symbol || "USDT";

  // Y-axis domains — equity gets its own scale; daily/weekly share the right axis.
  // Only POSITIVE, finite equities define the scale, so a transient 0/negative
  // reconcile glitch (or any stale point) can't blow the Y-range out to [0, balance]
  // and flatten the real line. Pad is a small % of the range (min $2) for sensitivity.
  const equityValues = buffer.map((b) => b.equity).filter((v) => v != null && isFinite(v) && v > 0);
  const equityMin = equityValues.length ? Math.min(...equityValues) : 0;
  const equityMax = equityValues.length ? Math.max(...equityValues) : 0;
  const equityRange = equityMax - equityMin;
  const equityPad = Math.max(equityRange * 0.1, 2);
  const equityDomain = equityValues.length
    ? [equityMin - equityPad, equityMax + equityPad]
    : ["auto", "auto"];

  // Plot data: null out any non-positive/invalid equity so the line gaps over a
  // transient reconcile glitch rather than spiking down to 0.
  const chartData = buffer.map((b) => ({
    ...b,
    equity: (b.equity != null && isFinite(b.equity) && b.equity > 0) ? b.equity : null,
  }));

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400">Portfolio value</div>
          <div className="flex items-baseline gap-3 mt-1">
            <div className="text-3xl font-bold text-slate-100">
              {fmtMoney(latest?.total_equity, sym)}
            </div>
            <SourceBadge source={latest?.source} dryRun={latest?.dry_run} />
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400 mt-1">
            <span>peak {fmtMoney(latest?.peak_equity, sym)}</span>
            <span className={pnlTone(latest?.daily_pnl_usd)}>
              daily {latest?.daily_pnl_usd != null
                ? `${latest.daily_pnl_usd >= 0 ? "+" : ""}${fmtMoney(latest.daily_pnl_usd, sym)} (${(latest.daily_pnl_pct * 100).toFixed(2)}%)`
                : "—"}
            </span>
            <span className={pnlTone(latest?.weekly_pnl_usd)}>
              weekly {latest?.weekly_pnl_usd != null
                ? `${latest.weekly_pnl_usd >= 0 ? "+" : ""}${fmtMoney(latest.weekly_pnl_usd, sym)} (${(latest.weekly_pnl_pct * 100).toFixed(2)}%)`
                : "—"}
            </span>
            <span>open {latest?.open_positions ?? 0}</span>
          </div>
        </div>
        <div className="text-xs text-slate-500 text-right">
          {haveWsData ? "live · 5s" : "polling"} · last {buffer.length}× pts
          {err && <div className="text-rose-400 mt-1">err: {err}</div>}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={chartData} margin={{ left: 4, right: 4, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="tsLabel" stroke="#94a3b8" fontSize={11} minTickGap={40} />
          <YAxis yAxisId="eq"  orientation="left"  stroke="#10b981" fontSize={11} domain={equityDomain}
                 tickFormatter={(v) => v.toFixed(0)} width={70} />
          <YAxis yAxisId="pnl" orientation="right" stroke="#94a3b8" fontSize={11} width={56}
                 tickFormatter={(v) => (v >= 0 ? "+" : "") + v.toFixed(0)} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
            formatter={(v, name) => [typeof v === "number" ? fmtMoney(v, sym) : v, name]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line yAxisId="eq"  type="monotone" dataKey="equity"     stroke="#10b981" strokeWidth={2} dot={false} name="equity" />
          <Line yAxisId="pnl" type="monotone" dataKey="daily_pnl"  stroke="#38bdf8" strokeWidth={1.5} dot={false} name="daily P&L" />
          <Line yAxisId="pnl" type="monotone" dataKey="weekly_pnl" stroke="#a78bfa" strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="weekly P&L" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
