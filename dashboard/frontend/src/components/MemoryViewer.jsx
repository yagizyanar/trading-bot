import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

const NAMES = ["market_context", "trade_log", "lessons_learned", "strategy_notes"];

export default function MemoryViewer() {
  const [active, setActive] = useState("market_context");
  const [files, setFiles] = useState({});

  useEffect(() => {
    Promise.all(NAMES.map(async (n) => [n, await api.memoryFile(n)]))
      .then((pairs) => setFiles(Object.fromEntries(pairs)))
      .catch(() => setFiles({}));
  }, [active]);

  return (
    <div className="rounded-xl bg-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-slate-400">Memory</div>
        <div className="flex gap-2">
          {NAMES.map((n) => (
            <button
              key={n}
              onClick={() => setActive(n)}
              className={`px-2 py-1 rounded text-xs ${
                active === n ? "bg-slate-600 text-white" : "bg-slate-700 text-slate-300"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>
      <pre className="text-xs text-slate-200 whitespace-pre-wrap font-mono max-h-96 overflow-y-auto bg-slate-900/60 p-3 rounded">
{files[active]?.content ?? "loading..."}
      </pre>
      {files[active]?.mtime && (
        <div className="text-xs text-slate-500 mt-2">
          last modified: {new Date(files[active].mtime * 1000).toLocaleString()}
        </div>
      )}
    </div>
  );
}
