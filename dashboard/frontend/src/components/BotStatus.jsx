import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const COLOR = {
  RUNNING: "bg-emerald-500",
  LOCKED:  "bg-rose-500",
  ERROR:   "bg-amber-500",
  STOPPED: "bg-slate-500",
};

export default function BotStatus() {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const tick = () => api.status().then(setS).catch((e) => !cancelled && setErr(e.message));
    tick();
    const id = setInterval(tick, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400">Bot Status</div>
      {err && <div className="text-rose-400 text-sm">error: {err}</div>}
      {s && (
        <>
          <div className="mt-1 flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full ${COLOR[s.state] || "bg-slate-500"}`} />
            <span className="text-lg font-semibold">{s.state}</span>
          </div>
          <div className="mt-2 text-sm text-slate-300">
            mode: <span className="font-mono">{s.dry_run ? "paper" : "LIVE"}</span>
          </div>
          {s.locked_reason && (
            <pre className="mt-2 text-xs text-rose-300 whitespace-pre-wrap">{s.locked_reason}</pre>
          )}
        </>
      )}
    </div>
  );
}
