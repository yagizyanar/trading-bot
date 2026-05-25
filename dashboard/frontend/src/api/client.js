const BASE = "/api";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  health:           () => get("/health"),
  status:           () => get("/status/"),
  fearGreed:        () => get("/fear-greed/"),
  positionsOpen:    () => get("/positions/open"),
  positionsClosed:  (limit = 100) => get(`/positions/closed?limit=${limit}`),
  sentimentLatest:  () => get("/sentiment/latest"),
  regimeLatest:     () => get("/regime/latest"),
  perfLatest:       () => get("/performance/latest"),
  perfHistory:      (days = 30) => get(`/performance/history?days=${days}`),
  memoryList:       () => get("/memory/"),
  memoryFile:       (name) => get(`/memory/${name}`),
  longShort:        () => get("/market/long-short"),
  funding:          () => get("/market/funding"),
  hyperliquid:      () => get("/market/hyperliquid"),
  marketSummary:    () => get("/market/summary"),
};

export function openWebsocket(onMessage) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch { /* swallow */ }
  };
  return ws;
}
