import React from 'react';
import { Card } from 'primereact/card';
import { Badge } from 'primereact/badge';
import { Button } from 'primereact/button';
import { Skeleton } from 'primereact/skeleton';
import { useNavigate } from 'react-router-dom';
import { useTradingSignalsSnapshot } from '../hooks/useTradingSignals';
import { TradingSignal, SignalType, ConfidenceLevel } from '../types';

interface TradingSignalsWidgetProps {
  className?: string;
  style?: React.CSSProperties;
}

const TradingSignalsWidget: React.FC<TradingSignalsWidgetProps> = ({ className, style }) => {
  const { signalAnalysis, loading, error } = useTradingSignalsSnapshot();
  const navigate = useNavigate();

  const getSignalIcon = (type: SignalType): string => {
    switch (type) {
      case 'BUY': return 'pi pi-arrow-up';
      case 'SELL': return 'pi pi-arrow-down';
      case 'HOLD': return 'pi pi-minus';
      default: return 'pi pi-question';
    }
  };

  const getSignalColor = (type: SignalType): string => {
    switch (type) {
      case 'BUY': return '#22c55e';
      case 'SELL': return '#ef4444';
      case 'HOLD': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  const getConfidenceBadge = (confidence: ConfidenceLevel) => {
    const severity = confidence === 'HIGH' ? 'success' : confidence === 'MEDIUM' ? 'warning' : 'info';
    return <Badge value={confidence} severity={severity} />;
  };

  const getTrendIcon = (trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL'): string => {
    switch (trend) {
      case 'BULLISH': return 'pi pi-trending-up';
      case 'BEARISH': return 'pi pi-trending-down';
      case 'NEUTRAL': return 'pi pi-minus';
      default: return 'pi pi-question';
    }
  };

  const getTrendColor = (trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL'): string => {
    switch (trend) {
      case 'BULLISH': return '#22c55e';
      case 'BEARISH': return '#ef4444';
      case 'NEUTRAL': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  const formatPrice = (price: number): string => {
    return `$${price.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  };

  if (loading) {
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
        <Skeleton height="200px" />
      </Card>
    );
  }

  if (error || !signalAnalysis) {
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
        <div className="text-center p-4">
          <i className="pi pi-exclamation-triangle text-yellow-500 text-2xl mb-2"></i>
          <p className="text-500 m-0">Unable to load trading signals</p>
          <Button
            label="View Signals"
            icon="pi pi-external-link"
            onClick={() => navigate('/trading-signals')}
            className="p-button-text p-button-sm mt-2"
            style={{ color: '#ff8c00' }}
          />
        </div>
      </Card>
    );
  }

  // Get the most important signal (highest confidence BUY or SELL, or first signal)
  const primarySignal = signalAnalysis.signals.find(s => 
    (s.type === 'BUY' || s.type === 'SELL') && s.confidence === 'HIGH'
  ) || signalAnalysis.signals[0];

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
      <div className="flex justify-content-between align-items-center mb-3">
        <div>
          <h3 className="m-0 text-white">Trading Signals</h3>
          <p className="text-500 m-0 text-sm">AI-powered recommendations</p>
        </div>
        <Button
          icon="pi pi-external-link"
          className="p-button-text p-button-sm"
          style={{ color: '#ff8c00' }}
          onClick={() => navigate('/trading-signals')}
          tooltip="View all signals"
        />
      </div>

      {/* Market Summary */}
      <div className="grid mb-3">
        <div className="col-6">
          <div className="flex align-items-center">
            <i 
              className={getTrendIcon(signalAnalysis.marketSummary.overallTrend)}
              style={{ 
                color: getTrendColor(signalAnalysis.marketSummary.overallTrend), 
                fontSize: '1.2rem',
                marginRight: '0.5rem'
              }}
            />
            <div>
              <div className="text-white font-semibold text-sm">
                {signalAnalysis.marketSummary.overallTrend}
              </div>
              <small className="text-500">Market Trend</small>
            </div>
          </div>
        </div>
        <div className="col-6">
          <div className="text-right">
            <div className={`font-semibold text-sm ${signalAnalysis.marketSummary.priceChange24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {signalAnalysis.marketSummary.priceChange24h >= 0 ? '+' : ''}{signalAnalysis.marketSummary.priceChange24h.toFixed(1)}%
            </div>
            <small className="text-500">24h Change</small>
          </div>
        </div>
      </div>

      {/* Primary Signal */}
      {primarySignal && (
        <div 
          className="p-3 mb-3"
          style={{ 
            backgroundColor: '#1a1a1a', 
            borderRadius: '8px',
            borderLeft: `3px solid ${getSignalColor(primarySignal.type)}`
          }}
        >
          <div className="flex justify-content-between align-items-start mb-2">
            <div className="flex align-items-center">
              <i 
                className={getSignalIcon(primarySignal.type)}
                style={{ 
                  color: getSignalColor(primarySignal.type), 
                  fontSize: '1.2rem',
                  marginRight: '0.5rem'
                }}
              />
              <div>
                <span className="text-white font-semibold text-sm">{primarySignal.type}</span>
                <div className="mt-1">
                  {getConfidenceBadge(primarySignal.confidence)}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-white font-semibold text-sm">
                {formatPrice(primarySignal.currentPrice)}
              </div>
              <small className="text-500">Current</small>
            </div>
          </div>
          
          <p className="text-300 m-0 text-sm line-height-3">
            {primarySignal.reasoning.length > 100 
              ? `${primarySignal.reasoning.substring(0, 100)}...` 
              : primarySignal.reasoning
            }
          </p>

          {primarySignal.targetPrice && (
            <div className="flex justify-content-between align-items-center mt-2 pt-2" style={{ borderTop: '1px solid #404040' }}>
              <small className="text-500">Target: {formatPrice(primarySignal.targetPrice)}</small>
              {primarySignal.stopLoss && (
                <small className="text-500">Stop: {formatPrice(primarySignal.stopLoss)}</small>
              )}
            </div>
          )}
        </div>
      )}

      {/* Signal Summary */}
      <div className="grid">
        <div className="col-4">
          <div className="text-center">
            <div className="text-green-400 font-semibold">
              {signalAnalysis.signals.filter(s => s.type === 'BUY').length}
            </div>
            <small className="text-500">BUY</small>
          </div>
        </div>
        <div className="col-4">
          <div className="text-center">
            <div className="text-yellow-400 font-semibold">
              {signalAnalysis.signals.filter(s => s.type === 'HOLD').length}
            </div>
            <small className="text-500">HOLD</small>
          </div>
        </div>
        <div className="col-4">
          <div className="text-center">
            <div className="text-red-400 font-semibold">
              {signalAnalysis.signals.filter(s => s.type === 'SELL').length}
            </div>
            <small className="text-500">SELL</small>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-3 pt-2" style={{ borderTop: '1px solid #404040' }}>
        <div className="flex justify-content-between align-items-center">
          <small className="text-500">
            {signalAnalysis.signals.length} active signal{signalAnalysis.signals.length !== 1 ? 's' : ''}
          </small>
          <Button
            label="View All"
            className="p-button-text p-button-sm"
            style={{ color: '#ff8c00', padding: '0.25rem 0.5rem' }}
            onClick={() => navigate('/trading-signals')}
          />
        </div>
      </div>
    </Card>
  );
};

export default TradingSignalsWidget;
