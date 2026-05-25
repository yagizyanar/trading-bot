import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

function tone(score) {
  if (score == null) return "bg-slate-700 text-slate-300";
  if (score >= 0.5) return "bg-emerald-500/80 text-white";
  if (score >= 0.2) return "bg-emerald-500/40 text-white";
  if (score <= -0.5) return "bg-rose-500/80 text-white";
  if (score <= -0.2) return "bg-rose-500/40 text-white";
  return "bg-slate-600 text-slate-100";
}

export default function HyperliquidSentiment() {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    const load = () => api.hyperliquid().then(setRows).catch(() => setRows([]));
    load();
    const id = setInterval(load, 120000);
    return () => clearInterval(id);
  }, []);

  const bullish = rows.filter((r) => r.label === "BULLISH").length;
  const bearish = rows.filter((r) => r.label === "BEARISH").length;
  const haveData = rows.some((r) => r.score != null);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-slate-400">
          Hyperliquid top-trader sentiment
        </div>
        <div className="text-xs text-slate-400">
          {haveData ? `bull ${bullish} / bear ${bearish}` : "awaiting first sample"}
        </div>
      </div>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        {rows.map((r) => (
          <div key={r.coin} className={`p-2 rounded ${tone(r.score)} text-sm`}>
            <div className="font-mono font-semibold">{r.coin}</div>
            <div className="text-xl font-mono">
              {r.score != null ? r.score.toFixed(2) : "—"}
            </div>
            <div className="text-xs opacity-80">{r.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
