import React, { useState, useEffect } from 'react';
import { Card } from 'primereact/card';
import { Badge } from 'primereact/badge';
import { Button } from 'primereact/button';
import { Skeleton } from 'primereact/skeleton';
import { Tooltip } from 'primereact/tooltip';
import { useNavigate } from 'react-router-dom';
import { ModelPerformanceService, ModelPerformanceSummary } from '../services/modelPerformanceService';

interface ModelPerformanceWidgetProps {
  className?: string;
  style?: React.CSSProperties;
}

const ModelPerformanceWidget: React.FC<ModelPerformanceWidgetProps> = ({ className, style }) => {
  const [summary, setSummary] = useState<ModelPerformanceSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadModelPerformance();
  }, []);

  const loadModelPerformance = async () => {
    try {
      setLoading(true);
      setError(null);

      const isAvailable = await ModelPerformanceService.isModelMetricsAvailable();
      if (!isAvailable) {
        setError('Model metrics not available');
        return;
      }

      const summaryData = await ModelPerformanceService.getModelPerformanceSummary();
      setSummary(summaryData);
    } catch (err: any) {
      console.error('Error loading model performance widget:', err);
      setError('Unable to load metrics');
    } finally {
      setLoading(false);
    }
  };

  const getReliabilityBadge = (reliability: 'HIGH' | 'MEDIUM' | 'LOW') => {
    const severity = reliability === 'HIGH' ? 'success' : reliability === 'MEDIUM' ? 'warning' : 'danger';
    return <Badge value={reliability} severity={severity} />;
  };

  const getReliabilityColor = (reliability: 'HIGH' | 'MEDIUM' | 'LOW'): string => {
    switch (reliability) {
      case 'HIGH': return '#22c55e';
      case 'MEDIUM': return '#f59e0b';
      case 'LOW': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const formatPercentage = (value: number): string => {
    return `${value.toFixed(1)}%`;
  };

  const formatCurrency = (value: number): string => {
    return `$${value.toLocaleString()}`;
  };

  // Tooltip content for metrics
  const getMetricTooltip = (metric: string): string => {
    switch (metric) {
      case 'reliability':
        return 'Model confidence level based on recent performance. HIGH/MEDIUM/LOW rating.';
      case 'rmse':
        return 'Measures prediction accuracy - lower values mean better predictions. Shows average prediction error in dollars.';
      case 'mae':
        return 'Average difference between predicted and actual prices. Lower is better. Easy to interpret in dollar terms.';
      case 'r2':
        return 'Shows how well the model explains price movements (0-100%). Higher percentages mean better predictions.';
      case 'mape':
        return 'Average prediction error as a percentage. Lower percentages indicate more accurate predictions.';
      default:
        return 'Model performance metric';
    }
  };

  const tooltipStyle = {
    backgroundColor: 'rgba(42, 42, 42, 0.95)',
    color: '#ffffff',
    border: '1px solid #404040',
    borderRadius: '6px',
    fontSize: '0.875rem',
    maxWidth: '300px',
    padding: '8px 12px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
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

  if (error || !summary) {
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
          <i className="pi pi-info-circle text-blue-500 text-2xl mb-2"></i>
          <p className="text-500 m-0">Model metrics unavailable</p>
          <small className="text-400">Train the AI model to see performance data</small>
          <div className="mt-3">
            <Button
              label="View Trading Signals"
              className="p-button-text p-button-sm"
              style={{ color: '#ff8c00' }}
              onClick={() => navigate('/trading-signals')}
            />
          </div>
        </div>
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
        borderLeft: `4px solid ${getReliabilityColor(summary.reliability)}`,
        ...style
      }}
    >
      {/* Header */}
      <div className="flex justify-content-between align-items-center mb-3">
        <div>
          <h3 className="m-0 text-white">AI Model Performance</h3>
          <p className="text-500 m-0 text-sm">Prediction accuracy & reliability</p>
        </div>
        <Button
          icon="pi pi-external-link"
          className="p-button-text p-button-sm"
          style={{ color: '#ff8c00' }}
          onClick={() => navigate('/model-performance')}
        />
      </div>

      {/* Key Performance Indicator */}
      <div 
        className="p-3 mb-3"
        style={{ 
          backgroundColor: '#1a1a1a', 
          borderRadius: '8px',
          borderLeft: `3px solid ${getReliabilityColor(summary.reliability)}`
        }}
      >
        <div className="flex justify-content-between align-items-center mb-2">
          <div>
            <h4 className="text-white m-0">Model Performance</h4>
            <small className="text-500">Key metrics overview</small>
          </div>
          <div className="text-right">
            {getReliabilityBadge(summary.reliability)}
            <div className="mt-1">
              <small className="text-500 reliability-tooltip">
                Reliability
                <i className="pi pi-info-circle ml-1" style={{ fontSize: '0.75rem', color: '#6b7280' }}></i>
              </small>
              <Tooltip
                target=".reliability-tooltip"
                content={getMetricTooltip('reliability')}
                position="top"
                showDelay={300}
                hideDelay={100}
                style={tooltipStyle}
              />
            </div>
          </div>
        </div>
        
        <div className="grid mt-3">
          <div className="col-6">
            <div className="text-center">
              <div className="text-blue-400 font-semibold">
                {summary.keyMetrics.r2Score.toFixed(3)}
              </div>
              <small className="text-500 r2-tooltip">
                R² Score
                <i className="pi pi-info-circle ml-1" style={{ fontSize: '0.75rem', color: '#6b7280' }}></i>
              </small>
              <Tooltip
                target=".r2-tooltip"
                content={getMetricTooltip('r2')}
                position="top"
                showDelay={300}
                hideDelay={100}
                style={tooltipStyle}
              />
            </div>
          </div>
          <div className="col-6">
            <div className="text-center">
              <div className="text-orange-400 font-semibold">
                {formatCurrency(summary.keyMetrics.rmse)}
              </div>
              <small className="text-500 rmse-tooltip">
                RMSE
                <i className="pi pi-info-circle ml-1" style={{ fontSize: '0.75rem', color: '#6b7280' }}></i>
              </small>
              <Tooltip
                target=".rmse-tooltip"
                content={getMetricTooltip('rmse')}
                position="top"
                showDelay={300}
                hideDelay={100}
                style={tooltipStyle}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Training Summary */}
      <div className="grid mb-3">
        <div className="col-4">
          <div className="text-center">
            <div className="text-white font-semibold">
              {summary.trainingInfo.epochsTrained}
            </div>
            <small className="text-500">Epochs</small>
          </div>
        </div>
        <div className="col-4">
          <div className="text-center">
            <div className="text-white font-semibold">
              {(summary.trainingInfo.datasetSize / 1000).toFixed(1)}K
            </div>
            <small className="text-500">Data Points</small>
          </div>
        </div>
        <div className="col-4">
          <div className="text-center">
            <div className="text-white font-semibold">
              {summary.trainingInfo.features.join(', ')}
            </div>
            <small className="text-500">Features</small>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="flex justify-content-between align-items-center pt-2" style={{ borderTop: '1px solid #404040' }}>
        <small className="text-500">
          {summary.modelVersion}
        </small>
        <small className="text-500">
          Trained: {summary.lastTrainingDate}
        </small>
      </div>

      {/* Performance Indicators */}
      <div className="mt-2">
        <div className="flex align-items-center justify-content-between">
          <small className="text-500 mae-tooltip">
            MAE: {formatCurrency(summary.keyMetrics.mae)}
            <i className="pi pi-info-circle ml-1" style={{ fontSize: '0.75rem', color: '#6b7280' }}></i>
          </small>
          <small className="text-500 mape-tooltip">
            MAPE: {formatPercentage(summary.keyMetrics.mape)}
            <i className="pi pi-info-circle ml-1" style={{ fontSize: '0.75rem', color: '#6b7280' }}></i>
          </small>
          <Tooltip
            target=".mae-tooltip"
            content={getMetricTooltip('mae')}
            position="top"
            showDelay={300}
            hideDelay={100}
            style={tooltipStyle}
          />
          <Tooltip
            target=".mape-tooltip"
            content={getMetricTooltip('mape')}
            position="top"
            showDelay={300}
            hideDelay={100}
            style={tooltipStyle}
          />
        </div>
      </div>

      {/* Footer */}
      <div className="mt-3 pt-2" style={{ borderTop: '1px solid #404040' }}>
        <div className="flex justify-content-between align-items-center">
          <small className="text-500">
            Model performance metrics
          </small>
          <Button
            label="View Details"
            className="p-button-text p-button-sm"
            style={{ color: '#ff8c00', padding: '0.25rem 0.5rem' }}
            onClick={() => navigate('/trading-signals')}
          />
        </div>
      </div>
    </Card>
  );
};

export default ModelPerformanceWidget;
