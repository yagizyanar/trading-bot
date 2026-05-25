import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const COLOR = {
  Bull:     "text-emerald-400",
  Bear:     "text-rose-400",
  Sideways: "text-slate-300",
  Crash:    "text-rose-600",
  Euphoria: "text-amber-300",
  unknown:  "text-slate-500",
};

export default function RegimeDisplay() {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    api.regimeLatest().then(setRows).catch(() => setRows([]));
    const id = setInterval(() => api.regimeLatest().then(setRows).catch(() => {}), 60000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-3">Markov regime per coin</div>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
        {rows.map((r) => (
          <div key={r.coin} className="p-2 rounded bg-slate-900/60 flex items-center justify-between">
            <span className="font-mono font-semibold">{r.coin}</span>
            <span className={`${COLOR[r.regime] ?? COLOR.unknown}`}>{r.regime}</span>
            <span className="text-xs text-slate-400">{(r.confidence * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
