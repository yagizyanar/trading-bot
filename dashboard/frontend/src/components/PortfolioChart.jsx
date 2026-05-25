import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../api/client.js";

export default function PortfolioChart() {
  const [data, setData] = useState([]);
  useEffect(() => {
    api.perfHistory(30).then((rows) =>
      setData(rows.map((r) => ({
        ts: new Date(r.ts).toLocaleDateString(),
        equity: r.total_equity,
        peak: r.peak_equity,
      })))
    ).catch(() => setData([]));
    const id = setInterval(() => {
      api.perfHistory(30).then((rows) => setData(rows.map((r) => ({
        ts: new Date(r.ts).toLocaleDateString(),
        equity: r.total_equity,
        peak: r.peak_equity,
      })))).catch(() => {});
    }, 60000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Portfolio value (30d)</div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="ts" stroke="#94a3b8" fontSize={12} />
          <YAxis stroke="#94a3b8" fontSize={12} />
          <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }} />
          <Line type="monotone" dataKey="equity" stroke="#10b981" dot={false} />
          <Line type="monotone" dataKey="peak"   stroke="#64748b" dot={false} strokeDasharray="4 4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
