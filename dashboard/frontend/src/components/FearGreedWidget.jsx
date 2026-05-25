import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

function color(v) {
  if (v === null || v === undefined) return "text-slate-400";
  if (v <= 25) return "text-rose-500";
  if (v <= 45) return "text-orange-400";
  if (v <= 55) return "text-slate-200";
  if (v <= 75) return "text-emerald-400";
  return "text-emerald-300";
}

export default function FearGreedWidget() {
  const [d, setD] = useState(null);
  useEffect(() => {
    api.fearGreed().then(setD).catch(() => setD({ value: null, label: "unavailable" }));
    const id = setInterval(() => api.fearGreed().then(setD).catch(() => {}), 60000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400">Fear &amp; Greed</div>
      <div className={`mt-2 text-4xl font-bold ${color(d?.value)}`}>
        {d?.value ?? "—"}
      </div>
      <div className="text-sm text-slate-300 mt-1">{d?.label}</div>
    </div>
  );
}
