import React, { useEffect, useState } from "react";
import BotStatus from "./components/BotStatus.jsx";
import FearGreedWidget from "./components/FearGreedWidget.jsx";
import NextRoutineCountdown from "./components/NextRoutineCountdown.jsx";
import DrawdownIndicator from "./components/DrawdownIndicator.jsx";
import PortfolioChart from "./components/PortfolioChart.jsx";
import PositionsTable from "./components/PositionsTable.jsx";
import SentimentGrid from "./components/SentimentGrid.jsx";
import RegimeDisplay from "./components/RegimeDisplay.jsx";
import MemoryViewer from "./components/MemoryViewer.jsx";
import LongShortGrid from "./components/LongShortGrid.jsx";
import FundingRateGrid from "./components/FundingRateGrid.jsx";
import HyperliquidSentiment from "./components/HyperliquidSentiment.jsx";
import { openWebsocket } from "./api/client.js";

export default function App() {
  const [live, setLive] = useState(null);

  useEffect(() => {
    const ws = openWebsocket(setLive);
    return () => ws.close();
  }, []);

  return (
    <div className="max-w-screen-2xl mx-auto p-6 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          trade-sentiment-markov
        </h1>
        <span className="text-xs text-slate-400">
          {live?.ts ? `live tick: ${new Date(live.ts).toLocaleTimeString()}` : "connecting..."}
        </span>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <BotStatus />
        <FearGreedWidget />
        <NextRoutineCountdown />
        <DrawdownIndicator live={live} />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PortfolioChart live={live} />
        <RegimeDisplay />
      </section>

      <section>
        <SentimentGrid />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LongShortGrid />
        <FundingRateGrid />
      </section>

      <section>
        <HyperliquidSentiment />
      </section>

      <section className="grid grid-cols-1 gap-4">
        <PositionsTable kind="open"   title="Open positions"   live={live} />
        <PositionsTable kind="closed" title="Closed positions" live={live} />
      </section>

      <section>
        <MemoryViewer />
      </section>
    </div>
  );
}
