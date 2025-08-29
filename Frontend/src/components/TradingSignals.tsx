import React from 'react';
import { Card } from 'primereact/card';
import { Badge } from 'primereact/badge';
import { Button } from 'primereact/button';
import { Skeleton } from 'primereact/skeleton';
import { Message } from 'primereact/message';
import { Divider } from 'primereact/divider';
import { ToggleButton } from 'primereact/togglebutton';
import { Tooltip } from 'primereact/tooltip';
import { useTradingSignals } from '../hooks/useTradingSignals';
import { SignalType, ConfidenceLevel, TimeHorizon } from '../types';


const TradingSignals: React.FC = () => {
  const {
    signalAnalysis,
    loading,
    error,
    lastUpdated,
    refresh,
    startAutoRefresh,
    stopAutoRefresh,
    isAutoRefreshActive
  } = useTradingSignals({
    autoRefresh: true,
    refreshInterval: 5 * 60 * 1000, // 5 minutes
    onError: (error) => {
      console.error('Trading signals error:', error);
    },
    onUpdate: (analysis) => {
      console.log('Trading signals updated:', analysis.signals.length, 'signals');
    }
  });

  const handleAutoRefreshToggle = (enabled: boolean) => {
    if (enabled) {
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }
  };

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
      case 'BUY': return '#22c55e'; // Green
      case 'SELL': return '#ef4444'; // Red
      case 'HOLD': return '#f59e0b'; // Yellow
      default: return '#6b7280'; // Gray
    }
  };

  const getConfidenceBadge = (confidence: ConfidenceLevel) => {
    const severity = confidence === 'HIGH' ? 'success' : confidence === 'MEDIUM' ? 'warning' : 'info';
    return <Badge value={confidence} severity={severity} />;
  };

  const getTimeHorizonTooltip = (timeHorizon: TimeHorizon): string => {
    switch (timeHorizon) {
      case 'SHORT_TERM':
        return 'Minutes to days (typically 1 minute to 1 week) - Quick trades with tight stops';
      case 'MEDIUM_TERM':
        return 'Days to months (typically 1 week to 6 months) - Swing trading approach';
      case 'LONG_TERM':
        return 'Months to years (typically 6 months to several years) - Position trading strategy';
      default:
        return 'Trading time horizon information';
    }
  };

  const getTimeHorizonBadge = (timeHorizon: TimeHorizon, signalId: string) => {
    const label = timeHorizon.replace('_', ' ');
    const tooltipId = `time-horizon-${signalId}`;

    return (
      <>
        <Badge
          value={label}
          severity="secondary"
          data-pr-tooltip={getTimeHorizonTooltip(timeHorizon)}
          data-pr-position="top"
          data-pr-at="center top-8"
          className={tooltipId}
        />
        <Tooltip
          target={`.${tooltipId}`}
          position="top"
          showDelay={300}
          hideDelay={100}
          style={{
            backgroundColor: 'rgba(42, 42, 42, 0.95)',
            color: '#ffffff',
            border: '1px solid #404040',
            borderRadius: '6px',
            fontSize: '0.875rem',
            maxWidth: '300px',
            padding: '8px 12px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
          }}
        />
      </>
    );
  };

  const formatPrice = (price: number): string => {
    return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleString();
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

  if (loading) {
    return (
      <div className="trading-signals-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh' }}>
        <div className="grid">
          <div className="col-12">
            <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
              <Skeleton height="200px" />
            </Card>
          </div>
          <div className="col-12 md:col-6">
            <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
              <Skeleton height="300px" />
            </Card>
          </div>
          <div className="col-12 md:col-6">
            <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
              <Skeleton height="300px" />
            </Card>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="trading-signals-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh' }}>
        <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
          <Message severity="error" text={error} />
          <div className="mt-3">
            <Button
              label="Retry"
              icon="pi pi-refresh"
              onClick={refresh}
              className="p-button-outlined"
              style={{ borderColor: '#ff8c00', color: '#ff8c00' }}
            />
          </div>
        </Card>
      </div>
    );
  }

  if (!signalAnalysis) {
    return (
      <div className="trading-signals-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh' }}>
        <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
          <Message severity="info" text="No trading signals available" />
        </Card>
      </div>
    );
  }

  return (
    <div className="trading-signals-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh', color: '#ffffff' }}>
      {/* Header */}
      <div className="flex justify-content-between align-items-center mb-4">
        <div>
          <h2 className="m-0 text-white">Trading Signals</h2>
          <p className="text-500 m-0 mt-1">AI-powered Bitcoin trading recommendations</p>
          {lastUpdated && (
            <small className="text-400">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </small>
          )}
        </div>
        <div className="flex align-items-center gap-3">
          <div className="flex align-items-center gap-2">
            <ToggleButton
              checked={isAutoRefreshActive}
              onChange={(e) => handleAutoRefreshToggle(e.value)}
              onLabel="Auto-refresh ON"
              offLabel="Auto-refresh OFF"
              onIcon="pi pi-check"
              offIcon="pi pi-times"
              className="p-button-sm"
              style={{
                backgroundColor: isAutoRefreshActive ? '#22c55e' : '#6b7280',
                borderColor: isAutoRefreshActive ? '#22c55e' : '#6b7280'
              }}
            />
          </div>
          <Button
            label="Refresh"
            icon="pi pi-refresh"
            onClick={refresh}
            className="p-button-outlined"
            style={{ borderColor: '#ff8c00', color: '#ff8c00' }}
            loading={loading}
          />
        </div>
      </div>





      {/* Trading Signals */}
      <div className="grid">
        {signalAnalysis.signals.map((signal, index) => (
          <div key={signal.id} className="col-12 lg:col-6">
            <Card 
              style={{ 
                backgroundColor: '#2a2a2a', 
                border: '1px solid #404040',
                borderRadius: '12px',
                borderLeft: `4px solid ${getSignalColor(signal.type)}`
              }}
            >
              {/* Signal Header */}
              <div className="flex justify-content-between align-items-start mb-3">
                <div className="flex align-items-center">
                  <i 
                    className={getSignalIcon(signal.type)}
                    style={{ 
                      color: getSignalColor(signal.type), 
                      fontSize: '1.5rem',
                      marginRight: '0.5rem'
                    }}
                  />
                  <div>
                    <h4 className="m-0 text-white">{signal.type} Signal</h4>
                    <div className="flex gap-2 mt-1">
                      {getConfidenceBadge(signal.confidence)}
                      {getTimeHorizonBadge(signal.timeHorizon, signal.id)}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-white font-semibold">
                    {formatPrice(signal.currentPrice)}
                  </div>
                  <small className="text-500">Current Price</small>
                </div>
              </div>

              {/* Signal Details */}
              <div className="mb-3">
                <p className="text-300 m-0 line-height-3">{signal.reasoning}</p>
              </div>

              {/* Price Targets */}
              {(signal.targetPrice || signal.stopLoss) && (
                <div className="grid mb-3">
                  {signal.targetPrice && (
                    <div className="col-6">
                      <div className="text-center p-2" style={{ backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
                        <div className="text-green-400 font-semibold">
                          {formatPrice(signal.targetPrice)}
                        </div>
                        <small className="text-500">Target Price</small>
                      </div>
                    </div>
                  )}
                  {signal.stopLoss && (
                    <div className="col-6">
                      <div className="text-center p-2" style={{ backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
                        <div className="text-red-400 font-semibold">
                          {formatPrice(signal.stopLoss)}
                        </div>
                        <small className="text-500">Stop Loss</small>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Technical Indicators */}
              <Divider />
              <div className="grid">
                <div className="col-4">
                  <div className="text-center">
                    <div className="text-white font-semibold">
                      {signal.technicalIndicators.momentum.toFixed(1)}
                    </div>
                    <small className="text-500">Momentum</small>
                  </div>
                </div>
                <div className="col-4">
                  <div className="text-center">
                    <div className="text-white font-semibold">
                      {signal.technicalIndicators.volatility}
                    </div>
                    <small className="text-500">Volatility</small>
                  </div>
                </div>
                <div className="col-4">
                  <div className="text-center">
                    <div 
                      className="font-semibold"
                      style={{ color: getTrendColor(signal.technicalIndicators.trend) }}
                    >
                      {signal.technicalIndicators.trend}
                    </div>
                    <small className="text-500">Trend</small>
                  </div>
                </div>
              </div>

              {/* Validity */}
              <div className="mt-3 pt-2" style={{ borderTop: '1px solid #404040' }}>
                <div className="flex justify-content-between align-items-center">
                  <small className="text-500">
                    Generated: {formatTimestamp(signal.timestamp)}
                  </small>
                  <small className="text-500">
                    Valid until: {formatTimestamp(signal.validUntil)}
                  </small>
                </div>
              </div>
            </Card>
          </div>
        ))}
      </div>

      {signalAnalysis.signals.length === 0 && (
        <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
          <Message severity="info" text="No active trading signals at this time. Check back later for new recommendations." />
        </Card>
      )}
    </div>
  );
};

export default TradingSignals;
