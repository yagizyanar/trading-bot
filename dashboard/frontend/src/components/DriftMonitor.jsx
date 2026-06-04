import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const TONE = {
  good: "text-emerald-400",
  warn: "text-amber-400",
  bad: "text-rose-400",
  neutral: "text-slate-200",
};

function Metric({ label, value, sub, tone }) {
  return (
    <div className="flex-1 min-w-[110px]">
      <div className="text-xs uppercase tracking-wider text-slate-400">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${TONE[tone] || TONE.neutral}`}>{value}</div>
      {sub && <div className="text-[11px] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}

// Live-vs-backtest drift (roadmap item 8). Reads /api/drift/latest, written daily
// by routines.drift_monitor.
export default function DriftMonitor() {
  const [d, setD] = useState(null);

  useEffect(() => {
    const load = () => api.drift().then(setD).catch(() => {});
    load();
    const id = setInterval(load, 60000);
    return () => clearInterval(id);
  }, []);

  if (!d || d.available === false) {
    return (
      <div className="rounded-xl bg-slate-800 p-4">
        <div className="text-xs uppercase tracking-wider text-slate-400">Live vs backtest drift</div>
        <div className="mt-2 text-sm text-slate-500">No snapshot yet — runs daily 16:30 UTC.</div>
      </div>
    );
  }

  const t = d.thresholds || {};
  const sharpe = d.rolling_sharpe;
  const wr = d.win_rate;
  const ap = d.avg_profit_per_trade;
  const alerts = d.alerts || [];

  const sharpeTone = sharpe == null ? "neutral"
    : sharpe < t.sharpe_alert ? "bad" : sharpe < t.backtest_sharpe_baseline ? "warn" : "good";
  const wrTone = wr == null ? "neutral"
    : wr < t.winrate_alert ? "bad" : wr < 0.45 ? "warn" : "good";
  const apTone = ap == null ? "neutral" : ap < 0 ? "bad" : "good";

  return (
    <div className={`rounded-xl bg-slate-800 p-4 ${alerts.length ? "ring-1 ring-rose-700" : ""}`}>
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-slate-400">
          Live vs backtest drift · {d.trades} trades / {d.window_days}d
        </div>
        <div className="text-xs text-slate-500">
          {d.ts ? new Date(d.ts).toLocaleDateString() : ""}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-4">
        <Metric label="30d Sharpe" tone={sharpeTone}
          value={sharpe == null ? "n/a" : sharpe.toFixed(2)}
          sub={`alert <${t.sharpe_alert} · bt ~${t.backtest_sharpe_baseline}`} />
        <Metric label="Win rate" tone={wrTone}
          value={wr == null ? "n/a" : `${(wr * 100).toFixed(0)}%`}
          sub={`alert <${(t.winrate_alert * 100).toFixed(0)}%`} />
        <Metric label="Avg / trade" tone={apTone}
          value={ap == null ? "n/a" : `${(ap * 100).toFixed(2)}%`}
          sub={`neg streak ${d.consecutive_negative_days}d (alert ${t.neg_streak_alert})`} />
        <Metric label="Fee drag" tone="neutral"
          value={d.actual_cost_bps == null ? "n/a" : `${d.actual_cost_bps.toFixed(1)} bps`}
          sub={`expected ${d.expected_cost_bps?.toFixed(0)} bps`} />
      </div>

      {alerts.length > 0 && (
        <div className="mt-3 rounded-lg bg-rose-950/50 border border-rose-800 p-2 space-y-1">
          {alerts.map((a, i) => (
            <div key={i} className="text-xs text-rose-300">⚠️ {a}</div>
          ))}
        </div>
      )}
    </div>
  );
}
