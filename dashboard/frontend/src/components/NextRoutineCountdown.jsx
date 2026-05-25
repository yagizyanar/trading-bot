import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

function fmtDelta(ms) {
  if (ms <= 0) return "now";
  const sec = Math.floor(ms / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return `${h}h ${m}m ${s}s`;
}

export default function NextRoutineCountdown() {
  const [next, setNext] = useState(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    api.status().then((s) => setNext(s.next_routine)).catch(() => {});
    const tickStatus = setInterval(() => api.status().then((s) => setNext(s.next_routine)).catch(() => {}), 60000);
    const tickNow = setInterval(() => setNow(Date.now()), 1000);
    return () => { clearInterval(tickStatus); clearInterval(tickNow); };
  }, []);

  const remaining = next?.next_run ? new Date(next.next_run).getTime() - now : null;

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400">Next routine</div>
      <div className="text-2xl font-semibold mt-1">{next?.name ?? "—"}</div>
      <div className="text-sm text-slate-300 mt-1">in {remaining !== null ? fmtDelta(remaining) : "—"}</div>
    </div>
  );
}
