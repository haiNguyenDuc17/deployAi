import React, { useState, useEffect } from 'react';
import { Card } from 'primereact/card';
import { InputNumber } from 'primereact/inputnumber';

import { Button } from 'primereact/button';
import { Message } from 'primereact/message';
import { Badge } from 'primereact/badge';
import { Skeleton } from 'primereact/skeleton';
import { loadPredictionData, PredictionData } from '../api/csvDataService';

interface InvestmentAnalysisProps {
  className?: string;
  style?: React.CSSProperties;
}

interface InvestmentData {
  quantity: number;
  averagePurchasePrice: number;
}

interface AnalysisResult {
  initialInvestment: number;
  currentValue: number;
  currentProfit: number;
  currentProfitPercent: number;
  projectedValue: number;
  projectedProfit: number;
  projectedProfitPercent: number;
  recommendation: 'hold' | 'sell' | 'buy' | 'neutral';
  recommendationText: string;
  currentBtcPrice: number;
  projectedBtcPrice: number;
}

const InvestmentAnalysis: React.FC<InvestmentAnalysisProps> = ({ className, style }) => {
  const [investment, setInvestment] = useState<InvestmentData>({
    quantity: 0,
    averagePurchasePrice: 0
  });
  
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dataLoading, setDataLoading] = useState(true);

  // Load prediction data on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        setDataLoading(true);
        const data = await loadPredictionData();
        setPredictionData(data);
      } catch (err: any) {
        setError('Failed to load prediction data. Please ensure the model has been trained.');
        console.error('Error loading prediction data:', err);
      } finally {
        setDataLoading(false);
      }
    };

    loadData();
  }, []);

  const getCurrentBitcoinPrice = (): number => {
    // Get the most recent prediction as current price approximation
    if (predictionData.length === 0) return 0;
    return predictionData[0].price;
  };

  const getProjectedPrice = (months: number = 6): number => {
    // Get price projection for specified months ahead
    if (predictionData.length === 0) return 0;
    
    const targetDate = new Date();
    targetDate.setMonth(targetDate.getMonth() + months);
    const targetDateStr = targetDate.toISOString().split('T')[0];
    
    // Find closest prediction date
    const closestPrediction = predictionData.find(pred => pred.date >= targetDateStr) || 
                             predictionData[predictionData.length - 1];
    
    return closestPrediction.price;
  };

  const generateRecommendation = (result: AnalysisResult): { recommendation: 'hold' | 'sell' | 'buy' | 'neutral', text: string } => {
    const { currentProfitPercent, projectedProfitPercent, currentBtcPrice, projectedBtcPrice } = result;
    
    // Strong growth expected
    if (projectedBtcPrice > currentBtcPrice * 1.15) {
      if (currentProfitPercent < 0) {
        return { recommendation: 'hold', text: 'Hold - Strong recovery expected. Current losses may be recovered with projected 15%+ growth.' };
      } else {
        return { recommendation: 'hold', text: 'Hold - Strong growth projected. Consider holding for additional gains.' };
      }
    }
    
    // Moderate growth expected
    if (projectedBtcPrice > currentBtcPrice * 1.05) {
      return { recommendation: 'hold', text: 'Hold - Moderate growth expected. Maintain current position.' };
    }
    
    // Decline expected
    if (projectedBtcPrice < currentBtcPrice * 0.9) {
      if (currentProfitPercent > 20) {
        return { recommendation: 'sell', text: 'Consider Selling - Take profits before potential 10%+ decline.' };
      } else {
        return { recommendation: 'hold', text: 'Hold - Despite projected decline, consider long-term potential.' };
      }
    }
    
    // Current price below projected future value
    if (currentBtcPrice < projectedBtcPrice * 0.95) {
      return { recommendation: 'buy', text: 'Consider Buying More - Current price is below projected value. Good accumulation opportunity.' };
    }
    
    return { recommendation: 'neutral', text: 'Neutral - Market appears fairly valued. Monitor for better entry/exit points.' };
  };

  const calculateAnalysis = (): AnalysisResult | null => {
    if (!investment.quantity || !investment.averagePurchasePrice || predictionData.length === 0) {
      return null;
    }

    const currentBtcPrice = getCurrentBitcoinPrice();
    const projectedBtcPrice = getProjectedPrice(6); // 6 months projection

    const initialInvestment = investment.quantity * investment.averagePurchasePrice;
    const currentValue = investment.quantity * currentBtcPrice;
    const projectedValue = investment.quantity * projectedBtcPrice;
    
    const currentProfit = currentValue - initialInvestment;
    const currentProfitPercent = (currentProfit / initialInvestment) * 100;
    
    const projectedProfit = projectedValue - initialInvestment;
    const projectedProfitPercent = (projectedProfit / initialInvestment) * 100;
    
    const result: AnalysisResult = {
      initialInvestment,
      currentValue,
      currentProfit,
      currentProfitPercent,
      projectedValue,
      projectedProfit,
      projectedProfitPercent,
      currentBtcPrice,
      projectedBtcPrice,
      recommendation: 'neutral',
      recommendationText: ''
    };
    
    const recommendation = generateRecommendation(result);
    result.recommendation = recommendation.recommendation;
    result.recommendationText = recommendation.text;
    
    return result;
  };

  const handleAnalyze = () => {
    setLoading(true);
    setError(null);
    
    try {
      // Validate inputs
      if (!investment.quantity || investment.quantity <= 0) {
        throw new Error('Please enter a valid Bitcoin quantity');
      }

      if (!investment.averagePurchasePrice || investment.averagePurchasePrice <= 0) {
        throw new Error('Please enter a valid average purchase price');
      }
      
      if (predictionData.length === 0) {
        throw new Error('Prediction data not available. Please ensure the AI model has been trained.');
      }
      
      const result = calculateAnalysis();
      setAnalysis(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const formatPercent = (percent: number): string => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  const getRecommendationColor = (recommendation: string): string => {
    switch (recommendation) {
      case 'buy': return '#22c55e';
      case 'hold': return '#ff8c00';
      case 'sell': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getRecommendationIcon = (recommendation: string): string => {
    switch (recommendation) {
      case 'buy': return 'pi pi-arrow-up';
      case 'hold': return 'pi pi-minus';
      case 'sell': return 'pi pi-arrow-down';
      default: return 'pi pi-info-circle';
    }
  };

  if (dataLoading) {
    return (
      <Card 
        className={className}
        style={{ 
          backgroundColor: '#2a2a2a', 
          border: '1px solid #404040',
          borderRadius: '12px',
          ...style
        }}
      >
        <div className="mb-3">
          <h3 className="m-0 text-white">Investment Analysis</h3>
          <p className="text-500 m-0 text-sm">Loading prediction data...</p>
        </div>
        <Skeleton height="200px" />
      </Card>
    );
  }

  return (
    <Card 
      className={className}
      style={{ 
        backgroundColor: '#2a2a2a', 
        border: '1px solid #404040',
        borderRadius: '12px',
        ...style
      }}
    >
      {/* Header */}
      <div className="mb-4">
        <h3 className="m-0 text-white">Investment Analysis</h3>
        <p className="text-500 m-0 text-sm">Analyze your Bitcoin investment performance and get AI-powered recommendations</p>
      </div>

      {/* Input Form */}
      <div className="grid mb-4">
        <div className="col-12 md:col-6">
          <label className="block text-white mb-2">Bitcoin Quantity</label>
          <InputNumber
            value={investment.quantity}
            onValueChange={(e) => setInvestment(prev => ({ ...prev, quantity: e.value || 0 }))}
            placeholder="0.00"
            minFractionDigits={0}
            maxFractionDigits={8}
            className="w-full"
            style={{ backgroundColor: '#1a1a1a' }}
          />
        </div>

        <div className="col-12 md:col-6">
          <label className="block text-white mb-2">Average Purchase Price (USD)</label>
          <InputNumber
            value={investment.averagePurchasePrice}
            onValueChange={(e) => setInvestment(prev => ({ ...prev, averagePurchasePrice: e.value || 0 }))}
            placeholder="0.00"
            mode="currency"
            currency="USD"
            locale="en-US"
            className="w-full"
            style={{ backgroundColor: '#1a1a1a' }}
          />
        </div>
      </div>

      {/* Analyze Button */}
      <div className="mb-4">
        <Button
          label="Analyze Investment"
          icon="pi pi-chart-line"
          onClick={handleAnalyze}
          loading={loading}
          className="p-button-warning"
          style={{ backgroundColor: '#ff8c00', border: 'none' }}
          disabled={!investment.quantity || !investment.averagePurchasePrice}
        />
      </div>

      {/* Error Message */}
      {error && (
        <Message
          severity="error"
          text={error}
          className="mb-4"
          style={{ backgroundColor: '#2a2a2a' }}
        />
      )}

      {/* Analysis Results */}
      {analysis && (
        <div className="grid">
          {/* Current Performance */}
          <div className="col-12 md:col-6">
            <Card style={{ backgroundColor: '#1a1a1a', border: '1px solid #404040' }}>
              <h4 className="text-white mb-3">Current Performance</h4>
              <div className="mb-2">
                <span className="text-500">Initial Investment: </span>
                <span className="text-white font-semibold">{formatCurrency(analysis.initialInvestment)}</span>
              </div>
              <div className="mb-2">
                <span className="text-500">Current Value: </span>
                <span className="text-white font-semibold">{formatCurrency(analysis.currentValue)}</span>
              </div>
              <div className="mb-2">
                <span className="text-500">Profit/Loss: </span>
                <span className={`font-semibold ${analysis.currentProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(analysis.currentProfit)} ({formatPercent(analysis.currentProfitPercent)})
                </span>
              </div>
              <div>
                <span className="text-500">Current BTC Price: </span>
                <span className="text-white font-semibold">{formatCurrency(analysis.currentBtcPrice)}</span>
              </div>
            </Card>
          </div>

          {/* Projected Performance */}
          <div className="col-12 md:col-6">
            <Card style={{ backgroundColor: '#1a1a1a', border: '1px solid #404040' }}>
              <h4 className="text-white mb-3">6-Month Projection</h4>
              <div className="mb-2">
                <span className="text-500">Projected Value: </span>
                <span className="text-white font-semibold">{formatCurrency(analysis.projectedValue)}</span>
              </div>
              <div className="mb-2">
                <span className="text-500">Projected Profit/Loss: </span>
                <span className={`font-semibold ${analysis.projectedProfit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(analysis.projectedProfit)} ({formatPercent(analysis.projectedProfitPercent)})
                </span>
              </div>
              <div>
                <span className="text-500">Projected BTC Price: </span>
                <span className="text-white font-semibold">{formatCurrency(analysis.projectedBtcPrice)}</span>
              </div>
            </Card>
          </div>

          {/* Recommendation */}
          <div className="col-12">
            <Card 
              style={{ 
                backgroundColor: '#1a1a1a', 
                border: '1px solid #404040',
                borderLeft: `4px solid ${getRecommendationColor(analysis.recommendation)}`
              }}
            >
              <div className="flex align-items-center mb-3">
                <i 
                  className={getRecommendationIcon(analysis.recommendation)}
                  style={{ 
                    color: getRecommendationColor(analysis.recommendation), 
                    fontSize: '1.5rem',
                    marginRight: '0.5rem'
                  }}
                />
                <h4 className="text-white m-0">AI Recommendation</h4>
                <Badge
                  value={analysis.recommendation.toUpperCase()}
                  style={{ 
                    backgroundColor: getRecommendationColor(analysis.recommendation),
                    marginLeft: '1rem'
                  }}
                />
              </div>
              <p className="text-300 m-0 line-height-3">{analysis.recommendationText}</p>
            </Card>
          </div>
        </div>
      )}
    </Card>
  );
};

export default InvestmentAnalysis;
