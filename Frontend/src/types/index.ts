export interface PredictionResponse {
  [key: string]: {
    dates: string[];
    predicted_prices: number[];
  };
}

export type TimeFrame = '1m' | '6m' | '1y' | '3y' | 'custom';

export interface TimeFrameMapping {
  [key: string]: number;
}

export interface ChartData {
  dates: string[];
  prices: number[];
}

export interface DateRangeSelection {
  startDate: string;
  endDate: string;
}

export interface PredictionData {
  date: string;
  price: number;
}

// Trading Signals Types
export type SignalType = 'BUY' | 'SELL' | 'HOLD';
export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';
export type TimeHorizon = 'SHORT_TERM' | 'MEDIUM_TERM' | 'LONG_TERM';

export interface TradingSignal {
  id: string;
  type: SignalType;
  confidence: ConfidenceLevel;
  timeHorizon: TimeHorizon;
  currentPrice: number;
  targetPrice?: number;
  stopLoss?: number;
  reasoning: string;
  technicalIndicators: {
    trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    momentum: number; // -100 to 100
    volatility: 'HIGH' | 'MEDIUM' | 'LOW';
    supportLevel?: number;
    resistanceLevel?: number;
    movingAverage?: {
      short: number; // 7-day MA
      medium: number; // 30-day MA
      long: number; // 90-day MA
    };
  };
  timestamp: string;
  validUntil: string;
}

export interface SignalAnalysis {
  signals: TradingSignal[];
  marketSummary: {
    overallTrend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    volatility: 'HIGH' | 'MEDIUM' | 'LOW';
    priceChange24h: number;
    priceChange7d: number;
    priceChange30d: number;
  };
  lastUpdated: string;
}