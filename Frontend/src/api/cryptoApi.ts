import { CryptoPriceInfo } from "./csvDataService";
// Simple in-memory cache with TTL
const cache = new Map<string, { t: number; v: any }>();
const now = () => Date.now();

export async function getCoingeckoPriceUSD(coinId = "bitcoin", ttlMs = 15_000): Promise<number> {
  const key = `cg:${coinId}`;
  const cached = cache.get(key);
  if (cached && now() - cached.t < ttlMs) return cached.v;

  const url = `https://api.coingecko.com/api/v3/simple/price?ids=${encodeURIComponent(
    coinId
  )}&vs_currencies=usd`;

  const res = await fetch(url, { headers: { "accept": "application/json" } });
  if (!res.ok) throw new Error(`CoinGecko ${res.status}`);
  const json = await res.json();
  const price = Number(json?.[coinId]?.usd);
  if (!Number.isFinite(price)) throw new Error("Invalid CoinGecko response");
  cache.set(key, { t: now(), v: price });
  return price;
}

export async function getBinanceSnapshot(symbol = "BTCUSDT"): Promise<number> {
  const url = `https://api.binance.com/api/v3/ticker/price?symbol=${encodeURIComponent(symbol)}`;
  const r = await fetch(url, { headers: { "accept": "application/json" } });
  if (!r.ok) throw new Error(`Binance ${r.status}`);
  const j = await r.json();
  const price = Number(j?.price);
  if (!Number.isFinite(price)) throw new Error("Invalid Binance response");
  return price;
}

export async function fetchMarketData(): Promise<CryptoPriceInfo[]> {
  const url =
    "https://api.coingecko.com/api/v3/coins/markets" +
    "?vs_currency=usd&ids=bitcoin,ethereum,ripple";
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch market data");
  const data = await res.json();

  // Convert CoinGecko response into your class
  const mapped: CryptoPriceInfo[] = data.map((c: any) => ({
    symbol: String(c.symbol || "").toUpperCase(),
    name: c.name ?? "",
    price: Number(c.current_price ?? 0),
    changePercent24h: Number(c.price_change_percentage_24h ?? 0),
    changePercent7d: Number(c.price_change_percentage_7d_in_currency ?? 0),
    volume: String(c.total_volume ?? "0"),
    icon: String(c.image || ""), // helper for nice icons
  }));

  return mapped;
}




