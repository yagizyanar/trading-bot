import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

function tone(ratePct) {
  if (ratePct == null) return "bg-slate-700 text-slate-300";
  if (ratePct >= 0.01) return "bg-rose-500/70 text-white";
  if (ratePct <= -0.01) return "bg-emerald-500/70 text-white";
  return "bg-slate-600 text-slate-100";
}

export default function FundingRateGrid() {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    const load = () => api.funding().then(setRows).catch(() => setRows([]));
    load();
    const id = setInterval(load, 60000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-3">
        Funding rate (Binance Futures)
      </div>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        {rows.map((r) => (
          <div key={r.coin} className={`p-2 rounded ${tone(r.rate_pct)} text-sm`}>
            <div className="font-mono font-semibold">{r.coin}</div>
            <div className="text-xl font-mono">
              {r.rate_pct != null ? `${r.rate_pct.toFixed(4)}%` : "—"}
            </div>
            <div className="text-xs opacity-80">{r.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
