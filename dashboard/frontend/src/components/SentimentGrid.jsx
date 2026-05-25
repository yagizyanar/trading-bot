import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

function color(unified) {
  if (unified == null) return "bg-slate-700";
  if (unified > 0.5) return "bg-emerald-500/80";
  if (unified > 0.2) return "bg-emerald-500/40";
  if (unified < -0.5) return "bg-rose-500/80";
  if (unified < -0.2) return "bg-rose-500/40";
  return "bg-slate-600";
}

export default function SentimentGrid() {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    api.sentimentLatest().then(setRows).catch(() => setRows([]));
    const id = setInterval(() => api.sentimentLatest().then(setRows).catch(() => {}), 30000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-3">Sentiment per coin</div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
        {rows.map((r) => (
          <div key={r.coin} className={`p-3 rounded-lg ${color(r.unified)}`}>
            <div className="font-semibold">{r.coin}</div>
            <div className="text-2xl font-mono">{r.unified?.toFixed(2)}</div>
            <div className="text-xs">{r.signal}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
