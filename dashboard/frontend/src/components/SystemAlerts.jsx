import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const SEVERITY_STYLE = {
  red:    { dot: "bg-rose-500",    box: "bg-rose-500/10 border-rose-500/30",     label: "text-rose-300" },
  yellow: { dot: "bg-amber-400",   box: "bg-amber-500/10 border-amber-500/30",   label: "text-amber-300" },
  green:  { dot: "bg-emerald-500", box: "bg-emerald-500/10 border-emerald-500/30", label: "text-emerald-300" },
};

const CATEGORY_LABEL = {
  trading:   "Trading",
  sentiment: "Sentiment",
  data:      "Data",
  system:    "System",
};

function fmtSince(iso) {
  if (!iso) return "";
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 0 || isNaN(ms)) return "";
  const min = Math.floor(ms / 60000);
  if (min < 1) return "just now";
  if (min < 60) return `${min}m ago`;
  const h = Math.floor(min / 60);
  const m = min % 60;
  return `${h}h ${m}m ago`;
}

export default function SystemAlerts({ live }) {
  const [alerts, setAlerts] = useState(null);
  const [open, setOpen] = useState(false);
  const [autoExpanded, setAutoExpanded] = useState(false);
  const [haveWsData, setHaveWsData] = useState(false);

  // Prefer WS-pushed alerts; fall back to /api poll
  useEffect(() => {
    if (live?.alerts) {
      setAlerts(live.alerts);
      setHaveWsData(true);
    }
  }, [live]);

  useEffect(() => {
    let cancelled = false;
    if (haveWsData) return;
    const load = () => api.alerts().then(d => {
      if (!cancelled) setAlerts(d.alerts || []);
    }).catch(() => {});
    load();
    const id = setInterval(load, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [haveWsData]);

  // Auto-expand once on first non-green snapshot
  useEffect(() => {
    if (!alerts || autoExpanded) return;
    const hasIssue = alerts.some(a => a.severity === "red" || a.severity === "yellow");
    if (hasIssue) setOpen(true);
    setAutoExpanded(true);
  }, [alerts, autoExpanded]);

  if (!alerts) {
    return (
      <div className="rounded-xl bg-slate-800 p-4">
        <div className="text-xs uppercase tracking-wider text-slate-400">System alerts</div>
        <div className="text-sm text-slate-400 mt-2">loading…</div>
      </div>
    );
  }

  const reds    = alerts.filter(a => a.severity === "red");
  const yellows = alerts.filter(a => a.severity === "yellow");
  const greens  = alerts.filter(a => a.severity === "green");
  const topSeverity = reds.length ? "red" : yellows.length ? "yellow" : "green";

  const summaryText = (reds.length || yellows.length)
    ? `${reds.length} alert${reds.length !== 1 ? "s" : ""}, ${yellows.length} warning${yellows.length !== 1 ? "s" : ""} (${greens.length}/${alerts.length} ok)`
    : `All ${greens.length} checks normal`;

  // What to render in the body: when expanded, group everything by category;
  // when collapsed, hide the body completely.
  const grouped = {};
  for (const a of alerts) {
    (grouped[a.category] ||= []).push(a);
  }

  return (
    <div className="rounded-xl bg-slate-800 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-slate-700/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className={`w-3 h-3 rounded-full ${SEVERITY_STYLE[topSeverity].dot} ${topSeverity === "red" ? "animate-pulse" : ""}`} />
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-400">System alerts</div>
            <div className={`text-sm font-medium ${SEVERITY_STYLE[topSeverity].label}`}>{summaryText}</div>
          </div>
        </div>
        <div className="text-xs text-slate-500 select-none">{open ? "▼ collapse" : "▶ expand"}</div>
      </button>

      {open && (
        <div className="border-t border-slate-700 p-4 space-y-4">
          {Object.entries(grouped).map(([cat, items]) => (
            <div key={cat}>
              <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
                {CATEGORY_LABEL[cat] || cat}
              </div>
              <ul className="space-y-2">
                {items.map(a => {
                  const st = SEVERITY_STYLE[a.severity] || SEVERITY_STYLE.green;
                  return (
                    <li key={a.id} className={`rounded border px-3 py-2 ${st.box}`}>
                      <div className="flex items-start gap-2">
                        <span className={`w-2 h-2 rounded-full ${st.dot} mt-1.5 flex-shrink-0 ${a.severity === "red" ? "animate-pulse" : ""}`} />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm text-slate-100">{a.title}</div>
                          {a.detail && (
                            <div className="text-xs text-slate-400 mt-0.5 break-words">{a.detail}</div>
                          )}
                          {(a.started_at || a.suggested_action) && (
                            <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1 text-[10px] text-slate-500">
                              {a.started_at && a.severity !== "green" && (
                                <span className={st.label}>since {fmtSince(a.started_at)}</span>
                              )}
                              {a.suggested_action && (
                                <span className="italic text-slate-500">→ {a.suggested_action}</span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
