import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

function color(p) {
  if (p === null || p === undefined) return "bg-slate-500";
  if (p < 0.03) return "bg-emerald-500";
  if (p < 0.07) return "bg-amber-400";
  return "bg-rose-500";
}

export default function DrawdownIndicator({ live }) {
  const [perf, setPerf] = useState(null);
  useEffect(() => {
    api.perfLatest().then(setPerf).catch(() => {});
    const id = setInterval(() => api.perfLatest().then(setPerf).catch(() => {}), 30000);
    return () => clearInterval(id);
  }, []);

  const dd = live?.drawdown_pct ?? perf?.drawdown_pct ?? 0;
  const pct = Math.min(1, Math.max(0, dd));

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400">Drawdown from peak</div>
      <div className="mt-2 text-3xl font-bold">{(pct * 100).toFixed(2)}%</div>
      <div className="mt-3 h-2 rounded bg-slate-700 overflow-hidden">
        <div
          className={`h-full ${color(pct)}`}
          style={{ width: `${Math.min(100, pct * 1000 / 1).toFixed(1)}%` }}
        />
      </div>
      <div className="text-xs text-slate-400 mt-1">lockfile at 10%</div>
    </div>
  );
}
