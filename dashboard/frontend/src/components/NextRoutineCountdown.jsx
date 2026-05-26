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
  const [upcoming, setUpcoming] = useState([]);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const refresh = () => api.status().then((s) => setUpcoming(s.upcoming || [])).catch(() => {});
    refresh();
    const tickStatus = setInterval(refresh, 60000);
    const tickNow = setInterval(() => setNow(Date.now()), 1000);
    return () => { clearInterval(tickStatus); clearInterval(tickNow); };
  }, []);

  const earliest = upcoming[0]?.next_run ?? null;
  const concurrent = earliest ? upcoming.filter((u) => u.next_run === earliest) : [];
  const remaining = earliest ? new Date(earliest).getTime() - now : null;
  const label = concurrent.length > 1 ? `Next routines (${concurrent.length})` : "Next routine";

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400">{label}</div>
      {concurrent.length === 0 ? (
        <div className="text-2xl font-semibold mt-1">—</div>
      ) : (
        <ul className="mt-1 space-y-0.5">
          {concurrent.map((u) => (
            <li key={u.name} className="text-2xl font-semibold leading-tight">{u.name}</li>
          ))}
        </ul>
      )}
      <div className="text-sm text-slate-300 mt-1">in {remaining !== null ? fmtDelta(remaining) : "—"}</div>
    </div>
  );
}
