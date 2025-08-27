import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import { Button } from 'primereact/button';
import { Calendar } from 'primereact/calendar';
import { Message } from 'primereact/message';
import { loadPredictionData, filterPredictionsByDays, filterPredictionsByDateRange, PredictionData } from '../api/csvDataService';
import { TimeFrame, TimeFrameMapping, DateRangeSelection } from '../types';

const timeFrameMap: TimeFrameMapping = {
  '1m': 30,
  '6m': 180,
  '1y': 365,
  '3y': 1095,
};

const BitcoinPriceChart: React.FC = () => {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>('1m');
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [filteredData, setFilteredData] = useState<PredictionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showCustomRange, setShowCustomRange] = useState<boolean>(false);
  const [customStartDate, setCustomStartDate] = useState<Date | null>(null);
  const [customEndDate, setCustomEndDate] = useState<Date | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);

  useEffect(() => {
    const loadPredictions = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await loadPredictionData();
        setPredictionData(data);

        // Set initial filtered data for default timeframe
        const days = timeFrameMap[selectedTimeFrame];
        const filtered = filterPredictionsByDays(data, days);
        setFilteredData(filtered);
      } catch (err: any) {
        console.error('Error loading prediction data:', err);
        setError(err.message || 'Failed to load Bitcoin prediction data. Please ensure the CSV file exists.');
      } finally {
        setLoading(false);
      }
    };

    loadPredictions();
  }, []);

  // Update filtered data when timeframe changes
  useEffect(() => {
    if (predictionData.length === 0 || selectedTimeFrame === 'custom') return;

    const days = timeFrameMap[selectedTimeFrame];
    const filtered = filterPredictionsByDays(predictionData, days);
    setFilteredData(filtered);
  }, [selectedTimeFrame, predictionData]);

  const handleTimeFrameChange = (timeFrame: TimeFrame) => {
    setSelectedTimeFrame(timeFrame);
    setCustomError(null);

    if (timeFrame === 'custom') {
      setShowCustomRange(true);
    } else {
      setShowCustomRange(false);
      // Data will be filtered by useEffect
    }
  };

  const handleCustomDateRangeApply = () => {
    if (!customStartDate || !customEndDate) {
      setCustomError('Please select both start and end dates.');
      return;
    }

    if (customStartDate >= customEndDate) {
      setCustomError('Start date must be before end date.');
      return;
    }

    // Check if date range exceeds 3 years
    const diffTime = Math.abs(customEndDate.getTime() - customStartDate.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays > 1095) {
      setCustomError('Date range cannot exceed 3 years (1095 days).');
      return;
    }

    setCustomError(null);

    const startDateStr = customStartDate.toISOString().split('T')[0];
    const endDateStr = customEndDate.toISOString().split('T')[0];

    try {
      const filtered = filterPredictionsByDateRange(predictionData, startDateStr, endDateStr);

      if (filtered.length === 0) {
        setCustomError('No prediction data available for the selected date range.');
        return;
      }

      setFilteredData(filtered);
    } catch (err: any) {
      setCustomError('Error applying date range filter.');
    }
  };

  const getOption = () => {
    if (filteredData.length === 0) return {};

    const dates = filteredData.map(item => item.date);
    const prices = filteredData.map(item => item.price);

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(42, 42, 42, 0.95)',
        borderColor: '#404040',
        textStyle: {
          color: '#ffffff',
        },
        formatter: (params: any) => {
          const dataIndex = params[0].dataIndex;
          const date = dates[dataIndex];
          const price = prices[dataIndex].toLocaleString('en-US', {
            style: 'currency',
            currency: 'USD',
          });
          return `<strong>${date}</strong><br/>Predicted Price: ${price}`;
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '10%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: {
          lineStyle: {
            color: '#404040',
          },
        },
        axisLabel: {
          color: '#888888',
          formatter: (value: string) => {
            const date = new Date(value);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          },
        },
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        axisLine: {
          lineStyle: {
            color: '#404040',
          },
        },
        splitLine: {
          lineStyle: {
            color: '#333333',
          },
        },
        axisLabel: {
          color: '#888888',
          formatter: (value: number) => {
            return '$' + value.toLocaleString();
          },
        },
        scale: true,
      },
      series: [
        {
          name: 'BTC Prediction',
          type: 'line',
          data: prices,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: {
            color: '#ff8c00',
            width: 3,
          },
          itemStyle: {
            color: '#ff8c00',
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                {
                  offset: 0,
                  color: 'rgba(255, 140, 0, 0.3)',
                },
                {
                  offset: 1,
                  color: 'rgba(255, 140, 0, 0.05)',
                },
              ],
            },
          },
        },
      ],
    };
  };

  if (loading) {
    return (
      <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#ffffff' }}>Loading Bitcoin prediction data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ height: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
        <div style={{ color: '#ff6b6b', textAlign: 'center', marginBottom: '20px' }}>{error}</div>
        <Button
          label="Retry"
          onClick={() => window.location.reload()}
          className="p-button-outlined"
          style={{ borderColor: '#ff8c00', color: '#ff8c00' }}
        />
      </div>
    );
  }

  return (
    <div style={{ width: '100%' }}>
      {/* Time Frame Filter Buttons */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '20px',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        <Button
          label="1M"
          onClick={() => handleTimeFrameChange('1m')}
          className={selectedTimeFrame === '1m' ? 'p-button-warning' : 'p-button-outlined'}
          size="small"
          style={selectedTimeFrame === '1m' ?
            { backgroundColor: '#ff8c00', border: 'none' } :
            { borderColor: '#404040', color: '#ffffff', backgroundColor: 'transparent' }
          }
        />
        <Button
          label="6M"
          onClick={() => handleTimeFrameChange('6m')}
          className={selectedTimeFrame === '6m' ? 'p-button-warning' : 'p-button-outlined'}
          size="small"
          style={selectedTimeFrame === '6m' ?
            { backgroundColor: '#ff8c00', border: 'none' } :
            { borderColor: '#404040', color: '#ffffff', backgroundColor: 'transparent' }
          }
        />
        <Button
          label="1Y"
          onClick={() => handleTimeFrameChange('1y')}
          className={selectedTimeFrame === '1y' ? 'p-button-warning' : 'p-button-outlined'}
          size="small"
          style={selectedTimeFrame === '1y' ?
            { backgroundColor: '#ff8c00', border: 'none' } :
            { borderColor: '#404040', color: '#ffffff', backgroundColor: 'transparent' }
          }
        />
        <Button
          label="3Y"
          onClick={() => handleTimeFrameChange('3y')}
          className={selectedTimeFrame === '3y' ? 'p-button-warning' : 'p-button-outlined'}
          size="small"
          style={selectedTimeFrame === '3y' ?
            { backgroundColor: '#ff8c00', border: 'none' } :
            { borderColor: '#404040', color: '#ffffff', backgroundColor: 'transparent' }
          }
        />
        <Button
          label="Custom"
          onClick={() => handleTimeFrameChange('custom')}
          className={selectedTimeFrame === 'custom' ? 'p-button-warning' : 'p-button-outlined'}
          size="small"
          style={selectedTimeFrame === 'custom' ?
            { backgroundColor: '#ff8c00', border: 'none' } :
            { borderColor: '#404040', color: '#ffffff', backgroundColor: 'transparent' }
          }
        />

        {/* Data Points Info */}
        <div style={{
          marginLeft: 'auto',
          color: '#888888',
          fontSize: '12px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span>Showing {filteredData.length} data points</span>
          {selectedTimeFrame === 'custom' && customStartDate && customEndDate && (
            <span>
              ({customStartDate.toLocaleDateString()} - {customEndDate.toLocaleDateString()})
            </span>
          )}
        </div>
      </div>

      {/* Custom Date Range Selector */}
      {showCustomRange && (
        <div style={{
          backgroundColor: '#2a2a2a',
          border: '1px solid #404040',
          borderRadius: '8px',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <h4 style={{ color: '#ffffff', margin: '0 0 15px 0' }}>Custom Date Range</h4>
          <div style={{ display: 'flex', gap: '15px', alignItems: 'end', flexWrap: 'wrap' }}>
            <div>
              <label style={{ color: '#888888', fontSize: '12px', display: 'block', marginBottom: '5px' }}>
                Start Date
              </label>
              <Calendar
                value={customStartDate}
                onChange={(e) => {
                  setCustomStartDate(e.value as Date);
                  setCustomError(null);
                }}
                showIcon
                dateFormat="yy-mm-dd"
                placeholder="Select start date"
                style={{ backgroundColor: '#1a1a1a' }}
              />
            </div>
            <div>
              <label style={{ color: '#888888', fontSize: '12px', display: 'block', marginBottom: '5px' }}>
                End Date
              </label>
              <Calendar
                value={customEndDate}
                onChange={(e) => {
                  setCustomEndDate(e.value as Date);
                  setCustomError(null);
                }}
                showIcon
                dateFormat="yy-mm-dd"
                placeholder="Select end date"
                style={{ backgroundColor: '#1a1a1a' }}
              />
            </div>
            <Button
              label="Apply Range"
              onClick={handleCustomDateRangeApply}
              className="p-button-warning"
              style={{ backgroundColor: '#ff8c00', border: 'none' }}
              disabled={!customStartDate || !customEndDate}
            />
          </div>
          {customError && (
            <Message
              severity="error"
              text={customError}
              style={{ marginTop: '10px', backgroundColor: '#2a2a2a' }}
            />
          )}
        </div>
      )}

      {/* Chart */}
      <div style={{ height: '350px', width: '100%' }}>
        <ReactECharts
          option={getOption()}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'canvas' }}
          notMerge={true}
          lazyUpdate={true}
        />
      </div>

      {/* Chart Info */}
      <div style={{
        marginTop: '10px',
        color: '#888888',
        fontSize: '11px',
        textAlign: 'center'
      }}>
        * This chart shows Bitcoin price predictions and should not be used as financial advice.
      </div>
    </div>
  );
};

export default BitcoinPriceChart; 