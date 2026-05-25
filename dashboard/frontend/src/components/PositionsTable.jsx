import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

export default function PositionsTable({ kind, title }) {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    const load = () =>
      (kind === "open" ? api.positionsOpen() : api.positionsClosed(50))
        .then(setRows).catch(() => setRows([]));
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [kind]);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-3">{title}</div>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-slate-400">
            <tr>
              <th className="text-left py-1 px-2">Coin</th>
              <th className="text-left py-1 px-2">Side</th>
              <th className="text-right py-1 px-2">Entry</th>
              <th className="text-right py-1 px-2">Exit</th>
              <th className="text-right py-1 px-2">P&amp;L</th>
              <th className="text-left py-1 px-2">Lev</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-slate-700">
                <td className="py-1 px-2 font-mono">{r.coin}</td>
                <td className="py-1 px-2">{r.side}</td>
                <td className="py-1 px-2 text-right font-mono">{r.entry_price?.toFixed(4)}</td>
                <td className="py-1 px-2 text-right font-mono">{r.exit_price?.toFixed(4) ?? "—"}</td>
                <td className={`py-1 px-2 text-right font-mono ${
                  (r.pnl_pct ?? 0) > 0 ? "text-emerald-400"
                  : (r.pnl_pct ?? 0) < 0 ? "text-rose-400" : "text-slate-300"
                }`}>
                  {r.pnl_pct != null ? `${(r.pnl_pct * 100).toFixed(2)}%` : "—"}
                </td>
                <td className="py-1 px-2">{r.leverage}x</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan="6" className="py-3 px-2 text-slate-500 italic">no rows</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
