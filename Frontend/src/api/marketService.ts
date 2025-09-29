/**
 * Market service for live crypto data from CoinGecko
 */

export interface LivePrice {
  id: string;
  symbol: string;
  name: string;
  current_price: number;
  price_change_percentage_24h: number;
  market_cap: number;
  total_volume: number;
  last_updated: string;
}

export interface MarketSummary {
  prices: number[][];
  market_caps: number[][];
  total_volumes: number[][];
}

export interface CryptoContext {
  live: Record<string, LivePrice>;
  predictionSummary?: {
    nextPrediction: number;
    trend: 'up' | 'down' | 'stable';
    confidence: number;
  };
  last7d: Record<string, {
    priceChange: number;
    volume: number;
  }>;
}

// Cache for API responses (30 seconds TTL)
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 30 * 1000; // 30 seconds

const getCachedData = (key: string) => {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  return null;
};

const setCachedData = (key: string, data: any) => {
  cache.set(key, { data, timestamp: Date.now() });
};

/**
 * Get live prices for multiple cryptocurrencies
 */
export const getSimplePrices = async (ids: string[], vsCurrency = 'usd'): Promise<Record<string, LivePrice>> => {
  const cacheKey = `prices_${ids.join(',')}_${vsCurrency}`;
  const cached = getCachedData(cacheKey);
  if (cached) return cached;

  try {
    const idsParam = ids.join(',');
    const response = await fetch(`/coingecko/simple/price?ids=${idsParam}&vs_currencies=${vsCurrency}&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true&include_last_updated_at=true`);
    
    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.status}`);
    }

    const data = await response.json();
    const result: Record<string, LivePrice> = {};

    // Transform the response to our interface
    Object.entries(data).forEach(([id, coinData]: [string, any]) => {
      result[id] = {
        id,
        symbol: id,
        name: id.charAt(0).toUpperCase() + id.slice(1),
        current_price: coinData[vsCurrency],
        price_change_percentage_24h: coinData[`${vsCurrency}_24h_change`] || 0,
        market_cap: coinData[`${vsCurrency}_market_cap`] || 0,
        total_volume: coinData[`${vsCurrency}_24h_vol`] || 0,
        last_updated: new Date(coinData.last_updated_at * 1000).toISOString()
      };
    });

    setCachedData(cacheKey, result);
    return result;
  } catch (error) {
    console.error('Error fetching live prices:', error);
    throw error;
  }
};

/**
 * Get market summary for a specific coin
 */
export const getMarketSummary = async (id: string, days = 7): Promise<MarketSummary> => {
  const cacheKey = `market_${id}_${days}`;
  const cached = getCachedData(cacheKey);
  if (cached) return cached;

  try {
    const response = await fetch(`/coingecko/coins/${id}/market_chart?vs_currency=usd&days=${days}`);
    
    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.status}`);
    }

    const data = await response.json();
    setCachedData(cacheKey, data);
    return data;
  } catch (error) {
    console.error('Error fetching market summary:', error);
    throw error;
  }
};

/**
 * Detect crypto-related intent in user message
 */
export const detectCryptoIntent = (message: string): string[] => {
  const lowerMessage = message.toLowerCase();
  const detectedCoins: string[] = [];
  
  // Comprehensive crypto terms and their CoinGecko IDs
  const cryptoMap: Record<string, string> = {
    // Major cryptocurrencies
    'bitcoin': 'bitcoin',
    'btc': 'bitcoin',
    'ethereum': 'ethereum', 
    'eth': 'ethereum',
    'binance': 'binancecoin',
    'bnb': 'binancecoin',
    'solana': 'solana',
    'sol': 'solana',
    'ripple': 'ripple',
    'xrp': 'ripple',
    'cardano': 'cardano',
    'ada': 'cardano',
    'polkadot': 'polkadot',
    'dot': 'polkadot',
    'chainlink': 'chainlink',
    'link': 'chainlink',
    
    // DeFi tokens
    'uniswap': 'uniswap',
    'uni': 'uniswap',
    'aave': 'aave',
    'compound': 'compound-governance-token',
    'comp': 'compound-governance-token',
    'maker': 'maker',
    'mkr': 'maker',
    'sushi': 'sushiswap',
    'sushi': 'sushiswap',
    
    // Layer 2 & Scaling
    'polygon': 'matic-network',
    'matic': 'matic-network',
    'arbitrum': 'arbitrum',
    'optimism': 'optimism',
    'op': 'optimism',
    'avalanche': 'avalanche-2',
    'avax': 'avalanche-2',
    
    // Meme coins
    'dogecoin': 'dogecoin',
    'doge': 'dogecoin',
    'shiba': 'shiba-inu',
    'shib': 'shiba-inu',
    'pepe': 'pepe',
    
    // Stablecoins
    'tether': 'tether',
    'usdt': 'tether',
    'usd coin': 'usd-coin',
    'usdc': 'usd-coin',
    'dai': 'dai',
    'busd': 'binance-usd',
    
    // Exchange tokens
    'ftx': 'ftx-token',
    'ftt': 'ftx-token',
    'crypto.com': 'crypto-com-chain',
    'cro': 'crypto-com-chain',
    'kucoin': 'kucoin-shares',
    'kcs': 'kucoin-shares',
    
    // Gaming & NFT
    'axie': 'axie-infinity',
    'axs': 'axie-infinity',
    'sandbox': 'the-sandbox',
    'sand': 'the-sandbox',
    'decentraland': 'decentraland',
    'mana': 'decentraland',
    
    // Privacy coins
    'monero': 'monero',
    'xmr': 'monero',
    'zcash': 'zcash',
    'zec': 'zcash',
    
    // Smart contract platforms
    'cosmos': 'cosmos',
    'atom': 'cosmos',
    'algorand': 'algorand',
    'algo': 'algorand',
    'tezos': 'tezos',
    'xtz': 'tezos',
    'near': 'near',
    'near': 'near',
    'fantom': 'fantom',
    'ftm': 'fantom'
  };

  // Price/market related keywords (expanded)
  const priceKeywords = [
    'price', 'giá', 'tăng', 'giảm', 'market', 'thị trường', 'so sánh', 'compare', 
    'analysis', 'phân tích', 'coin', 'crypto', 'cryptocurrency', 'tiền điện tử',
    'tất cả', 'all', 'top', 'hàng đầu', 'phổ biến', 'popular', 'trending',
    'giá cả', 'pricing', 'thống kê', 'statistics', 'báo cáo', 'report'
  ];
  
  const hasPriceIntent = priceKeywords.some(keyword => lowerMessage.includes(keyword));
  
  if (hasPriceIntent) {
    // Find mentioned cryptocurrencies
    Object.entries(cryptoMap).forEach(([term, coinId]) => {
      if (lowerMessage.includes(term) && !detectedCoins.includes(coinId)) {
        detectedCoins.push(coinId);
      }
    });
  }

  // If asking for "all coins" or no specific coins mentioned but has price intent
  const askingForAll = lowerMessage.includes('tất cả') || lowerMessage.includes('all') || 
                      lowerMessage.includes('top') || lowerMessage.includes('hàng đầu') ||
                      lowerMessage.includes('phổ biến') || lowerMessage.includes('popular');
  
  if (hasPriceIntent && (detectedCoins.length === 0 || askingForAll)) {
    // Return top 20 most popular cryptocurrencies
    return [
      'bitcoin', 'ethereum', 'binancecoin', 'solana', 'ripple', 'cardano', 
      'dogecoin', 'polygon', 'polkadot', 'chainlink', 'avalanche-2', 'uniswap',
      'litecoin', 'bitcoin-cash', 'stellar', 'monero', 'ethereum-classic',
      'cosmos', 'algorand', 'vechain'
    ];
  }

  return detectedCoins;
};

/**
 * Build crypto context for AI Chat
 */
export const buildCryptoContext = async (userMessage: string): Promise<CryptoContext | null> => {
  const detectedCoins = detectCryptoIntent(userMessage);
  
  if (detectedCoins.length === 0) {
    return null;
  }

  try {
    const livePrices = await getSimplePrices(detectedCoins);
    
    // Get 7-day data for trend analysis
    const last7d: Record<string, { priceChange: number; volume: number }> = {};
    
    for (const coinId of detectedCoins) {
      try {
        const marketData = await getMarketSummary(coinId, 7);
        if (marketData.prices.length >= 2) {
          const currentPrice = marketData.prices[marketData.prices.length - 1][1];
          const weekAgoPrice = marketData.prices[0][1];
          const priceChange = ((currentPrice - weekAgoPrice) / weekAgoPrice) * 100;
          const avgVolume = marketData.total_volumes.reduce((sum, vol) => sum + vol[1], 0) / marketData.total_volumes.length;
          
          last7d[coinId] = {
            priceChange,
            volume: avgVolume
          };
        }
      } catch (error) {
        console.warn(`Failed to get 7-day data for ${coinId}:`, error);
      }
    }

    return {
      live: livePrices,
      last7d
    };
  } catch (error) {
    console.error('Error building crypto context:', error);
    return null;
  }
};

/**
 * Format crypto context for AI consumption
 */
export const formatCryptoContext = (context: CryptoContext): string => {
  const lines: string[] = [];
  
  lines.push("=== LIVE CRYPTO MARKET DATA ===");
  
  // Sort by market cap (highest first)
  const sortedEntries = Object.entries(context.live).sort((a, b) => b[1].market_cap - a[1].market_cap);
  
  // Group into sections for better readability
  const majorCoins = sortedEntries.slice(0, 10);
  const otherCoins = sortedEntries.slice(10);
  
  if (majorCoins.length > 0) {
    lines.push("🏆 TOP CRYPTOCURRENCIES:");
    lines.push("");
    
    majorCoins.forEach(([coinId, data]) => {
      const change24h = data.price_change_percentage_24h;
      const changeEmoji = change24h > 0 ? '📈' : change24h < 0 ? '📉' : '➡️';
      
      lines.push(`${changeEmoji} **${data.name.toUpperCase()}** (${data.symbol.toUpperCase()}):`);
      lines.push(`  **Price:** $${data.current_price.toLocaleString()}`);
      lines.push(`  **24h Change:** ${change24h > 0 ? '+' : ''}${change24h.toFixed(2)}%`);
      lines.push(`  **Market Cap:** $${(data.market_cap / 1e9).toFixed(2)}B`);
      
      if (context.last7d[coinId]) {
        const weekChange = context.last7d[coinId].priceChange;
        lines.push(`  **7d Change:** ${weekChange > 0 ? '+' : ''}${weekChange.toFixed(2)}%`);
      }
      lines.push('');
    });
  }
  
  if (otherCoins.length > 0) {
    lines.push("OTHER CRYPTOCURRENCIES:");
    lines.push("");
    
    // Show other coins in a more compact format
    otherCoins.forEach(([coinId, data]) => {
      const change24h = data.price_change_percentage_24h;
      const changeEmoji = change24h > 0 ? '📈' : change24h < 0 ? '📉' : '➡️';
      
      lines.push(`${changeEmoji} **${data.name.toUpperCase()}**: $${data.current_price.toLocaleString()} (${change24h > 0 ? '+' : ''}${change24h.toFixed(2)}%)`);
    });
    lines.push('');
  }
  
  // Add market summary
  const totalMarketCap = Object.values(context.live).reduce((sum, coin) => sum + coin.market_cap, 0);
  const avgChange24h = Object.values(context.live).reduce((sum, coin) => sum + coin.price_change_percentage_24h, 0) / Object.keys(context.live).length;
  
  lines.push("📈 **MARKET SUMMARY:**");
  lines.push(`  💎 **Total Market Cap:** $${(totalMarketCap / 1e12).toFixed(2)}T`);
  lines.push(`  📊 **Average 24h Change:** ${avgChange24h > 0 ? '+' : ''}${avgChange24h.toFixed(2)}%`);
  lines.push(`  🔢 **Total Coins Tracked:** ${Object.keys(context.live).length}`);
  
  return lines.join('\n');
};
