import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../api/client.js";

const REFRESH_MS = 60_000;

function fmtUSD(n) {
  if (n == null || isNaN(n)) return "—";
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
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

export default function PortfolioChart() {
  const [latest, setLatest] = useState(null);
  const [data, setData] = useState([]);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [l, h] = await Promise.all([api.perfLatest(), api.perfHistory(30)]);
        if (cancelled) return;
        setLatest(l);
        setData(h.map((r) => ({
          ts: new Date(r.ts).toLocaleDateString(),
          equity: r.total_equity,
          peak: r.peak_equity,
        })));
        setErr(null);
      } catch (e) {
        if (!cancelled) setErr(e.message);
      }
    };
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Append the latest live point so the chart's tail shows it even before day_close persists a row.
  const chartData = (() => {
    if (!latest || latest.total_equity == null) return data;
    const liveTs = latest.source === "freqtrade" ? "now" : new Date(latest.ts).toLocaleDateString();
    const last = data[data.length - 1];
    if (last && last.ts === liveTs) return data;
    return [...data, { ts: liveTs, equity: latest.total_equity, peak: latest.peak_equity }];
  })();

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400">Portfolio value</div>
          <div className="flex items-baseline gap-3 mt-1">
            <div className="text-3xl font-bold text-slate-100">
              {fmtUSD(latest?.total_equity)}
            </div>
            <SourceBadge source={latest?.source} dryRun={latest?.dry_run} />
          </div>
          <div className="flex gap-4 text-xs text-slate-400 mt-1">
            <span>peak {fmtUSD(latest?.peak_equity)}</span>
            <span className={pnlTone(latest?.daily_pnl_usd)}>
              {latest?.daily_pnl_usd != null
                ? `${latest.daily_pnl_usd >= 0 ? "+" : ""}${fmtUSD(latest.daily_pnl_usd)} (${(latest.daily_pnl_pct * 100).toFixed(2)}%)`
                : "—"}
            </span>
            <span>open {latest?.open_positions ?? 0}</span>
          </div>
        </div>
        <div className="text-xs text-slate-500 text-right">
          30d history · 60s refresh
          {err && <div className="text-rose-400 mt-1">err: {err}</div>}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="ts" stroke="#94a3b8" fontSize={12} />
          <YAxis stroke="#94a3b8" fontSize={12} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
            formatter={(v) => (typeof v === "number" ? fmtUSD(v) : v)}
          />
          <Line type="monotone" dataKey="equity" stroke="#10b981" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="peak"   stroke="#64748b" dot={false} strokeDasharray="4 4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
