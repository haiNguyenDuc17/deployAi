import React, { useState, useEffect } from 'react';
import { Card } from 'primereact/card';
import { Badge } from 'primereact/badge';
import { Button } from 'primereact/button';
import { Skeleton } from 'primereact/skeleton';
import { Message } from 'primereact/message';
import { Chart } from 'primereact/chart';
import { Divider } from 'primereact/divider';
import { ModelPerformanceService, ModelPerformanceSummary } from '../services/modelPerformanceService';

interface ModelPerformanceProps {
  className?: string;
  style?: React.CSSProperties;
  showDetailedView?: boolean;
}

const ModelPerformance: React.FC<ModelPerformanceProps> = ({ 
  className, 
  style, 
  showDetailedView = false 
}) => {
  const [summary, setSummary] = useState<ModelPerformanceSummary | null>(null);
  const [detailedMetrics, setDetailedMetrics] = useState<any>(null);
  const [trainingHistory, setTrainingHistory] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModelPerformance();
  }, [showDetailedView]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadModelPerformance = async () => {
    try {
      setLoading(true);
      setError(null);

      // Check if metrics are available
      const isAvailable = await ModelPerformanceService.isModelMetricsAvailable();
      if (!isAvailable) {
        throw new Error('Model performance metrics not available. Please train the model first.');
      }

      // Load summary
      const summaryData = await ModelPerformanceService.getModelPerformanceSummary();
      setSummary(summaryData);

      // Load detailed metrics if requested
      if (showDetailedView) {
        const [detailedData, historyData] = await Promise.all([
          ModelPerformanceService.getDetailedMetrics(),
          ModelPerformanceService.getTrainingHistory()
        ]);
        setDetailedMetrics(detailedData);
        setTrainingHistory(historyData);
      }
    } catch (err: any) {
      console.error('Error loading model performance:', err);
      setError(err.message || 'Failed to load model performance metrics');
    } finally {
      setLoading(false);
    }
  };

  const getReliabilityBadge = (reliability: 'HIGH' | 'MEDIUM' | 'LOW') => {
    const severity = reliability === 'HIGH' ? 'success' : reliability === 'MEDIUM' ? 'warning' : 'danger';
    return <Badge value={reliability} severity={severity} />;
  };

  const getConvergenceBadge = (convergence: 'GOOD' | 'FAIR' | 'POOR') => {
    const severity = convergence === 'GOOD' ? 'success' : convergence === 'FAIR' ? 'warning' : 'danger';
    return <Badge value={convergence} severity={severity} />;
  };

  const formatPercentage = (value: number): string => {
    return `${value.toFixed(1)}%`;
  };

  const formatCurrency = (value: number): string => {
    return `$${value.toLocaleString()}`;
  };

  const getTrainingLossChartData = () => {
    if (!trainingHistory) return null;

    return {
      labels: trainingHistory.epochs,
      datasets: [
        {
          label: 'Training Loss',
          data: trainingHistory.trainingLoss,
          borderColor: '#ff8c00',
          backgroundColor: 'rgba(255, 140, 0, 0.1)',
          tension: 0.4,
          fill: false
        },
        {
          label: 'Validation Loss',
          data: trainingHistory.validationLoss,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          tension: 0.4,
          fill: false
        }
      ]
    };
  };

  const getChartOptions = () => {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#ffffff'
          }
        },
        title: {
          display: true,
          text: 'Training Loss Over Epochs',
          color: '#ffffff'
        }
      },
      scales: {
        x: {
          ticks: {
            color: '#ffffff'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        },
        y: {
          ticks: {
            color: '#ffffff'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        }
      }
    };
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

  if (error) {
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
        <Message severity="warn" text={error} />
        <div className="mt-3">
          <Button
            label="Retry"
            icon="pi pi-refresh"
            onClick={loadModelPerformance}
            className="p-button-outlined"
            style={{ borderColor: '#ff8c00', color: '#ff8c00' }}
          />
        </div>
      </Card>
    );
  }

  if (!summary) {
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
        <Message severity="info" text="No model performance data available" />
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
      <div className="flex justify-content-between align-items-center mb-3">
        <div>
          <h3 className="m-0 text-white">AI Model Performance</h3>
          <p className="text-500 m-0 text-sm">
            {summary.modelVersion} • Last trained: {summary.lastTrainingDate}
          </p>
        </div>
        <div className="flex align-items-center gap-2">
          {getReliabilityBadge(summary.reliability)}
          <Button
            icon="pi pi-refresh"
            className="p-button-text p-button-sm"
            style={{ color: '#ff8c00' }}
            onClick={loadModelPerformance}
            tooltip="Refresh metrics"
          />
        </div>
      </div>

      {/* Key Metrics Summary */}
      <div className="grid mb-3">
        <div className="col-6 md:col-3">
          <div className="text-center p-2" style={{ backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
            <div className="text-green-400 font-semibold text-lg">
              {formatPercentage(summary.accuracy)}
            </div>
            <small className="text-500">Accuracy</small>
          </div>
        </div>
        <div className="col-6 md:col-3">
          <div className="text-center p-2" style={{ backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
            <div className="text-blue-400 font-semibold text-lg">
              {summary.keyMetrics.r2Score.toFixed(3)}
            </div>
            <small className="text-500">R² Score</small>
          </div>
        </div>
        <div className="col-6 md:col-3">
          <div className="text-center p-2" style={{ backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
            <div className="text-orange-400 font-semibold text-lg">
              {formatCurrency(summary.keyMetrics.rmse)}
            </div>
            <small className="text-500">RMSE</small>
          </div>
        </div>
        <div className="col-6 md:col-3">
          <div className="text-center p-2" style={{ backgroundColor: '#1a1a1a', borderRadius: '8px' }}>
            <div className="text-purple-400 font-semibold text-lg">
              {formatPercentage(summary.keyMetrics.mape)}
            </div>
            <small className="text-500">MAPE</small>
          </div>
        </div>
      </div>

      {/* Training Info */}
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
              {summary.trainingInfo.datasetSize.toLocaleString()}
            </div>
            <small className="text-500">Data Points</small>
          </div>
        </div>
        <div className="col-4">
          <div className="text-center">
            <div className="text-white font-semibold">
              {summary.trainingInfo.features.length}
            </div>
            <small className="text-500">Features</small>
          </div>
        </div>
      </div>

      {/* Detailed View */}
      {showDetailedView && detailedMetrics && (
        <>
          <Divider />
          
          {/* Training Loss Chart */}
          {trainingHistory && getTrainingLossChartData() && (
            <div className="mb-4">
              <h4 className="text-white mb-3">Training Progress</h4>
              <div style={{ height: '300px' }}>
                <Chart
                  type="line"
                  data={getTrainingLossChartData()!}
                  options={getChartOptions()}
                  style={{ height: '100%' }}
                />
              </div>
            </div>
          )}

          {/* Detailed Metrics */}
          <div className="grid">
            <div className="col-12 md:col-6">
              <h4 className="text-white mb-3">Performance Details</h4>
              <div className="grid">
                <div className="col-6">
                  <small className="text-500">MAE</small>
                  <div className="text-white font-semibold">
                    {formatCurrency(detailedMetrics.performance.mae)}
                  </div>
                </div>
                <div className="col-6">
                  <small className="text-500">MAPE</small>
                  <div className="text-white font-semibold">
                    {formatPercentage(detailedMetrics.performance.mape)}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="col-12 md:col-6">
              <h4 className="text-white mb-3">Training Quality</h4>
              <div className="flex align-items-center justify-content-between">
                <span className="text-500">Convergence</span>
                {getConvergenceBadge(detailedMetrics.training.convergence)}
              </div>
              <div className="flex align-items-center justify-content-between mt-2">
                <span className="text-500">Final Loss</span>
                <span className="text-white font-semibold">
                  {detailedMetrics.training.finalLoss.toExponential(3)}
                </span>
              </div>
            </div>
          </div>
        </>
      )}
    </Card>
  );
};

export default ModelPerformance;
