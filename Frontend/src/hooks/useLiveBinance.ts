import { useEffect, useRef, useState } from "react";

/**
 * Stream live trade prices for many symbols via a single Binance combined stream.
 * symbols: array of coin tickers like ["BTC","ETH","XRP"]
 * Debounces UI updates to avoid re-render storms.
 */
export function useBinanceMulti(symbols: string[], debounceMs = 250) {
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const latestRef = useRef<Record<string, number>>({});
  const tRef = useRef<number | null>(null);

  useEffect(() => {
    // Build combined stream URL: btcusdt@trade/ethusdt@trade/...
    const streams = symbols
      .filter(Boolean)
      .map((s) => `${s.toLowerCase()}usdt@trade`)
      .join("/");
    if (!streams) return;

    const url = `wss://stream.binance.com:9443/stream?streams=${streams}`;
    let stopped = false;

    // Close previous
    try { wsRef.current?.close(); } catch {}
    wsRef.current = null;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (e) => {
      // Combined payload: { stream: "btcusdt@trade", data: { p: "price", ... } }
      const msg = JSON.parse(e.data);
      const stream: string = msg?.stream || "";
      const data = msg?.data || {};
      const p = Number(data?.p);
      if (!Number.isFinite(p)) return;

      // Extract symbol from "btcusdt@trade" -> BTC
      const base = stream.split("@")[0].replace("usdt", "").toUpperCase();
      latestRef.current = { ...latestRef.current, [base]: p };

      // Debounce state publish
      if (tRef.current) window.clearTimeout(tRef.current);
      tRef.current = window.setTimeout(() => {
        setPrices((prev) => ({ ...prev, ...latestRef.current }));
      }, debounceMs) as any;
    };

    ws.onclose = () => setConnected(false);
    ws.onerror = () => {
      try { ws.close(); } catch {}
    };

    // Pause the socket when tab hidden (optional)
    const onVis = () => {
      if (document.visibilityState !== "visible") {
        try { ws.close(); } catch {}
      }
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", onVis);
      if (tRef.current) window.clearTimeout(tRef.current);
      try { ws.close(); } catch {}
      if (!stopped) wsRef.current = null;
    };
  }, [symbols.join("|"), debounceMs]); // re-open when list changes

  return { prices, connected };
}