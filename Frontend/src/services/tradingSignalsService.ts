import { PredictionData, TradingSignal, SignalAnalysis, SignalType, ConfidenceLevel, TimeHorizon } from '../types';

/**
 * Trading Signals Analysis Service
 * Analyzes Bitcoin prediction data to generate trading signals
 */

export class TradingSignalsService {
  
  /**
   * Generate trading signals from prediction data
   */
  static async generateSignals(predictions: PredictionData[]): Promise<SignalAnalysis> {
    if (predictions.length < 30) {
      throw new Error('Insufficient data for signal analysis. Need at least 30 data points.');
    }

    const signals: TradingSignal[] = [];
    const currentPrice = predictions[0].price;
    const technicalIndicators = this.calculateTechnicalIndicators(predictions);
    
    // Generate different types of signals
    const trendSignal = this.generateTrendSignal(predictions, technicalIndicators);
    const momentumSignal = this.generateMomentumSignal(predictions, technicalIndicators);
    const supportResistanceSignal = this.generateSupportResistanceSignal(predictions, technicalIndicators);
    
    if (trendSignal) signals.push(trendSignal);
    if (momentumSignal) signals.push(momentumSignal);
    if (supportResistanceSignal) signals.push(supportResistanceSignal);

    const marketSummary = this.generateMarketSummary(predictions, technicalIndicators);

    return {
      signals,
      marketSummary,
      lastUpdated: new Date().toISOString()
    };
  }

  /**
   * Calculate technical indicators from prediction data
   */
  private static calculateTechnicalIndicators(predictions: PredictionData[]) {
    const prices = predictions.map(p => p.price);
    
    // Moving averages
    const ma7 = this.calculateMovingAverage(prices, 7);
    const ma30 = this.calculateMovingAverage(prices, 30);
    const ma90 = this.calculateMovingAverage(prices, 90);
    
    // Price changes
    const priceChange24h = this.calculatePriceChange(prices, 1);
    const priceChange7d = this.calculatePriceChange(prices, 7);
    const priceChange30d = this.calculatePriceChange(prices, 30);
    
    // Volatility
    const volatility = this.calculateVolatility(prices.slice(0, 30));
    
    // Support and resistance levels
    const { support, resistance } = this.calculateSupportResistance(prices.slice(0, 60));
    
    // Trend analysis
    const trend = this.analyzeTrend(prices.slice(0, 30));
    
    // Momentum (RSI-like indicator)
    const momentum = this.calculateMomentum(prices.slice(0, 14));

    return {
      movingAverages: { ma7, ma30, ma90 },
      priceChanges: { priceChange24h, priceChange7d, priceChange30d },
      volatility,
      support,
      resistance,
      trend,
      momentum
    };
  }

  /**
   * Generate trend-based signal
   */
  private static generateTrendSignal(predictions: PredictionData[], indicators: any): TradingSignal | null {
    const currentPrice = predictions[0].price;
    const { movingAverages, trend, priceChanges } = indicators;
    
    let signalType: SignalType;
    let confidence: ConfidenceLevel;
    let reasoning: string;
    let targetPrice: number | undefined;
    let stopLoss: number | undefined;

    // Determine signal based on trend and moving averages
    if (trend === 'BULLISH' && currentPrice > movingAverages.ma7 && movingAverages.ma7 > movingAverages.ma30) {
      signalType = 'BUY';
      confidence = priceChanges.priceChange7d > 5 ? 'HIGH' : 'MEDIUM';
      reasoning = `Strong bullish trend detected. Price above short-term MA (${movingAverages.ma7.toFixed(2)}) and 7-day MA above 30-day MA. 7-day change: ${priceChanges.priceChange7d.toFixed(2)}%`;
      targetPrice = currentPrice * 1.08; // 8% target
      stopLoss = currentPrice * 0.95; // 5% stop loss
    } else if (trend === 'BEARISH' && currentPrice < movingAverages.ma7 && movingAverages.ma7 < movingAverages.ma30) {
      signalType = 'SELL';
      confidence = priceChanges.priceChange7d < -5 ? 'HIGH' : 'MEDIUM';
      reasoning = `Strong bearish trend detected. Price below short-term MA (${movingAverages.ma7.toFixed(2)}) and 7-day MA below 30-day MA. 7-day change: ${priceChanges.priceChange7d.toFixed(2)}%`;
      targetPrice = currentPrice * 0.92; // 8% target down
      stopLoss = currentPrice * 1.05; // 5% stop loss
    } else {
      signalType = 'HOLD';
      confidence = 'MEDIUM';
      reasoning = `Mixed signals detected. Current trend: ${trend}. Price: $${currentPrice.toFixed(2)}, 7-day MA: $${movingAverages.ma7.toFixed(2)}`;
    }

    return {
      id: `trend-${Date.now()}`,
      type: signalType,
      confidence,
      timeHorizon: 'MEDIUM_TERM',
      currentPrice,
      targetPrice,
      stopLoss,
      reasoning,
      technicalIndicators: {
        trend: trend,
        momentum: indicators.momentum,
        volatility: indicators.volatility,
        supportLevel: indicators.support,
        resistanceLevel: indicators.resistance,
        movingAverage: {
          short: movingAverages.ma7,
          medium: movingAverages.ma30,
          long: movingAverages.ma90
        }
      },
      timestamp: new Date().toISOString(),
      validUntil: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString() // Valid for 24 hours
    };
  }

  /**
   * Generate momentum-based signal
   */
  private static generateMomentumSignal(predictions: PredictionData[], indicators: any): TradingSignal | null {
    const currentPrice = predictions[0].price;
    const { momentum, volatility, support, resistance } = indicators;

    let signalType: SignalType;
    let confidence: ConfidenceLevel;
    let reasoning: string;
    let targetPrice: number | undefined;
    let stopLoss: number | undefined;

    if (momentum > 70) {
      signalType = 'SELL';
      confidence = volatility === 'HIGH' ? 'MEDIUM' : 'HIGH';
      reasoning = `Overbought conditions detected. Momentum indicator: ${momentum.toFixed(1)}. Consider taking profits.`;

      // Short-term sell targets - more conservative for quick trades
      if (resistance && resistance > currentPrice) {
        targetPrice = currentPrice * 0.97; // 3% down target for short-term
      } else {
        targetPrice = currentPrice * 0.95; // 5% down target if no resistance data
      }
      stopLoss = currentPrice * 1.02; // 2% stop loss above current price

    } else if (momentum < 30) {
      signalType = 'BUY';
      confidence = volatility === 'HIGH' ? 'MEDIUM' : 'HIGH';
      reasoning = `Oversold conditions detected. Momentum indicator: ${momentum.toFixed(1)}. Potential buying opportunity.`;

      // Short-term buy targets - more conservative for quick trades
      if (support && support < currentPrice) {
        targetPrice = currentPrice * 1.03; // 3% up target for short-term
      } else {
        targetPrice = currentPrice * 1.05; // 5% up target if no support data
      }
      stopLoss = currentPrice * 0.98; // 2% stop loss below current price

    } else {
      signalType = 'HOLD';
      confidence = 'LOW';
      reasoning = `Neutral momentum. Momentum indicator: ${momentum.toFixed(1)}. Wait for clearer signals.`;
      // No target price or stop loss for HOLD signals
    }

    return {
      id: `momentum-${Date.now()}`,
      type: signalType,
      confidence,
      timeHorizon: 'SHORT_TERM',
      currentPrice,
      targetPrice,
      stopLoss,
      reasoning,
      technicalIndicators: {
        trend: indicators.trend,
        momentum: indicators.momentum,
        volatility: indicators.volatility,
        supportLevel: indicators.support,
        resistanceLevel: indicators.resistance,
        movingAverage: {
          short: indicators.movingAverages.ma7,
          medium: indicators.movingAverages.ma30,
          long: indicators.movingAverages.ma90
        }
      },
      timestamp: new Date().toISOString(),
      validUntil: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString() // Valid for 8 hours
    };
  }

  /**
   * Generate support/resistance based signal
   */
  private static generateSupportResistanceSignal(predictions: PredictionData[], indicators: any): TradingSignal | null {
    const currentPrice = predictions[0].price;
    const { support, resistance } = indicators;
    
    if (!support || !resistance) return null;

    let signalType: SignalType;
    let confidence: ConfidenceLevel;
    let reasoning: string;
    let targetPrice: number | undefined;
    let stopLoss: number | undefined;

    const distanceToSupport = ((currentPrice - support) / support) * 100;
    const distanceToResistance = ((resistance - currentPrice) / currentPrice) * 100;

    if (distanceToSupport < 2) {
      signalType = 'BUY';
      confidence = 'HIGH';
      reasoning = `Price near strong support level at $${support.toFixed(2)}. Distance to support: ${distanceToSupport.toFixed(1)}%`;
      targetPrice = resistance * 0.95; // Target near resistance
      stopLoss = support * 0.98; // Stop just below support
    } else if (distanceToResistance < 2) {
      signalType = 'SELL';
      confidence = 'HIGH';
      reasoning = `Price near strong resistance level at $${resistance.toFixed(2)}. Distance to resistance: ${distanceToResistance.toFixed(1)}%`;
      targetPrice = support * 1.05; // Target near support
      stopLoss = resistance * 1.02; // Stop just above resistance
    } else {
      signalType = 'HOLD';
      confidence = 'MEDIUM';
      reasoning = `Price in middle range. Support: $${support.toFixed(2)}, Resistance: $${resistance.toFixed(2)}`;
    }

    return {
      id: `support-resistance-${Date.now()}`,
      type: signalType,
      confidence,
      timeHorizon: 'LONG_TERM',
      currentPrice,
      targetPrice,
      stopLoss,
      reasoning,
      technicalIndicators: {
        trend: indicators.trend,
        momentum: indicators.momentum,
        volatility: indicators.volatility,
        supportLevel: support,
        resistanceLevel: resistance,
        movingAverage: {
          short: indicators.movingAverages.ma7,
          medium: indicators.movingAverages.ma30,
          long: indicators.movingAverages.ma90
        }
      },
      timestamp: new Date().toISOString(),
      validUntil: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString() // Valid for 7 days
    };
  }

  /**
   * Generate market summary
   */
  private static generateMarketSummary(predictions: PredictionData[], indicators: any) {
    const { trend, volatility, priceChanges } = indicators;
    
    return {
      overallTrend: trend,
      volatility,
      priceChange24h: priceChanges.priceChange24h,
      priceChange7d: priceChanges.priceChange7d,
      priceChange30d: priceChanges.priceChange30d
    };
  }

  /**
   * Calculate moving average
   */
  private static calculateMovingAverage(prices: number[], period: number): number {
    if (prices.length < period) return prices[0] || 0;
    
    const sum = prices.slice(0, period).reduce((acc, price) => acc + price, 0);
    return sum / period;
  }

  /**
   * Calculate price change percentage
   */
  private static calculatePriceChange(prices: number[], days: number): number {
    if (prices.length <= days) return 0;
    
    const currentPrice = prices[0];
    const previousPrice = prices[days];
    
    return ((currentPrice - previousPrice) / previousPrice) * 100;
  }

  /**
   * Calculate volatility
   */
  private static calculateVolatility(prices: number[]): 'HIGH' | 'MEDIUM' | 'LOW' {
    if (prices.length < 2) return 'LOW';
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i-1] - prices[i]) / prices[i]);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    // Classify volatility based on standard deviation
    if (stdDev > 0.05) return 'HIGH';
    if (stdDev > 0.02) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Calculate support and resistance levels
   */
  private static calculateSupportResistance(prices: number[]): { support: number; resistance: number } {
    const sortedPrices = [...prices].sort((a, b) => a - b);
    const length = sortedPrices.length;
    
    // Support: 20th percentile
    const supportIndex = Math.floor(length * 0.2);
    const support = sortedPrices[supportIndex];
    
    // Resistance: 80th percentile
    const resistanceIndex = Math.floor(length * 0.8);
    const resistance = sortedPrices[resistanceIndex];
    
    return { support, resistance };
  }

  /**
   * Analyze trend
   */
  private static analyzeTrend(prices: number[]): 'BULLISH' | 'BEARISH' | 'NEUTRAL' {
    if (prices.length < 10) return 'NEUTRAL';
    
    const recentPrices = prices.slice(0, 10);
    const olderPrices = prices.slice(10, 20);
    
    const recentAvg = recentPrices.reduce((sum, price) => sum + price, 0) / recentPrices.length;
    const olderAvg = olderPrices.reduce((sum, price) => sum + price, 0) / olderPrices.length;
    
    const trendStrength = ((recentAvg - olderAvg) / olderAvg) * 100;
    
    if (trendStrength > 2) return 'BULLISH';
    if (trendStrength < -2) return 'BEARISH';
    return 'NEUTRAL';
  }

  /**
   * Calculate momentum (RSI-like indicator)
   */
  private static calculateMomentum(prices: number[]): number {
    if (prices.length < 14) return 50; // Neutral
    
    let gains = 0;
    let losses = 0;
    
    for (let i = 1; i < 14; i++) {
      const change = prices[i-1] - prices[i];
      if (change > 0) {
        gains += change;
      } else {
        losses += Math.abs(change);
      }
    }
    
    const avgGain = gains / 13;
    const avgLoss = losses / 13;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    
    return rsi;
  }
}
