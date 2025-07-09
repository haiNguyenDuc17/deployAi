import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  ChartContainer,
  ChartHeader,
  ChartTitle,
  TimeFrameSelector,
  TimeFrameButton,
  Disclaimer,
  LoadingSpinner
} from '../styles/StyledComponents';
import {
  loadPredictionData,
  filterPredictionsByDays,
  filterPredictionsByDateRange,
  convertToChartData,
  PredictionData
} from '../api/csvDataService';
import { TimeFrame, TimeFrameMapping, ChartData, DateRangeSelection } from '../types';
import DateRangeSelector from './DateRangeSelector';

const timeFrameMap: TimeFrameMapping = {
  '1m': 30,
  '6m': 180,
  '1y': 365,
  '3y': 1095,
};

const BitcoinPriceChart: React.FC = () => {
  const [selectedTimeFrame, setSelectedTimeFrame] = useState<TimeFrame>('1m');
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [allPredictions, setAllPredictions] = useState<PredictionData[]>([]);
  const [showCustomRange, setShowCustomRange] = useState<boolean>(false);

  // Load all predictions on component mount
  useEffect(() => {
    const loadAllPredictions = async () => {
      try {
        setLoading(true);
        setError(null);

        console.log('Loading CSV prediction data...');
        const predictions = await loadPredictionData();
        setAllPredictions(predictions);

        // Set initial data for default timeframe
        const days = timeFrameMap[selectedTimeFrame];
        const filteredPredictions = filterPredictionsByDays(predictions, days);
        const chartData = convertToChartData(filteredPredictions);
        setChartData(chartData);

        console.log(`Successfully loaded ${predictions.length} total predictions`);
        console.log(`Showing ${filteredPredictions.length} predictions for ${selectedTimeFrame}`);

      } catch (err: any) {
        console.error('Error loading CSV data:', err);
        setError(err.message || 'Failed to load prediction data. Please ensure the model has been trained and CSV file exists.');
      } finally {
        setLoading(false);
      }
    };

    loadAllPredictions();
  }, []);

  // Update chart data when timeframe changes
  useEffect(() => {
    if (allPredictions.length === 0 || selectedTimeFrame === 'custom') return;

    const days = timeFrameMap[selectedTimeFrame];
    const filteredPredictions = filterPredictionsByDays(allPredictions, days);
    const chartData = convertToChartData(filteredPredictions);
    setChartData(chartData);

    console.log(`Updated chart for ${selectedTimeFrame} (${days} days): ${filteredPredictions.length} predictions`);
  }, [selectedTimeFrame, allPredictions]);

  const getOption = () => {
    if (!chartData) return {};

    return {
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(13, 17, 23, 0.9)',
        borderColor: '#30363d',
        textStyle: {
          color: '#e6e8ea',
        },
        formatter: (params: any) => {
          const dataIndex = params[0].dataIndex;
          const date = chartData.dates[dataIndex];
          const price = chartData.prices[dataIndex].toLocaleString('en-US', {
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
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: chartData.dates,
        axisLine: {
          lineStyle: {
            color: '#30363d',
          },
        },
        axisLabel: {
          color: '#8b949e',
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
            color: '#30363d',
          },
        },
        splitLine: {
          lineStyle: {
            color: '#21262d',
          },
        },
        axisLabel: {
          color: '#8b949e',
          formatter: (value: number) => {
            return '$' + value.toLocaleString();
          },
        },
        scale: true,
        min: (value: { min: number }) => {
          // Set min to ~20% below the minimum value
          return Math.floor(value.min * 0.8);
        },
        max: (value: { max: number }) => {
          // Set max to ~20% above the maximum value
          return Math.ceil(value.max * 1.2);
        },
        splitNumber: 20,
        minInterval: 500
      },
      series: [
        {
          name: 'BTC Price Prediction',
          type: 'line',
          data: chartData.prices,
          smooth: true,
          symbol: 'none',
          lineStyle: {
            color: '#f7931a',
            width: 3,
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
                  color: 'rgba(247, 147, 26, 0.5)',
                },
                {
                  offset: 1,
                  color: 'rgba(247, 147, 26, 0.05)',
                },
              ],
            },
          },
        },
      ],
    };
  };

  const handleTimeFrameChange = (timeFrame: TimeFrame) => {
    setSelectedTimeFrame(timeFrame);
    if (timeFrame === 'custom') {
      setShowCustomRange(true);
    } else {
      setShowCustomRange(false);
    }
  };

  const handleDateRangeChange = (dateRange: DateRangeSelection) => {
    if (allPredictions.length === 0) return;

    try {
      const filteredPredictions = filterPredictionsByDateRange(
        allPredictions,
        dateRange.startDate,
        dateRange.endDate
      );

      if (filteredPredictions.length === 0) {
        setError('No predictions available for the selected date range.');
        return;
      }

      const chartData = convertToChartData(filteredPredictions);
      setChartData(chartData);
      setError(null);

      console.log(`Custom date range applied: ${dateRange.startDate} to ${dateRange.endDate}`);
      console.log(`Showing ${filteredPredictions.length} predictions`);

    } catch (err: any) {
      console.error('Error applying date range:', err);
      setError('Failed to apply date range filter.');
    }
  };

  const renderErrorMessage = () => {
    const isCsvError = error?.includes('CSV') || error?.includes('model has been trained');
    
    return (
      <div style={{ 
        height: '400px', 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center', 
        color: '#f85149',
        textAlign: 'center',
        padding: '0 20px'
      }}>
        <div style={{ fontSize: '1.2rem', marginBottom: '10px' }}>
          {error}
        </div>
        
        {isCsvError && (
          <div style={{
            color: '#8b949e',
            fontSize: '0.9rem',
            marginTop: '20px',
            maxWidth: '600px',
            textAlign: 'left',
            background: 'rgba(255,255,255,0.05)',
            padding: '15px',
            borderRadius: '5px'
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>To fix this issue:</div>
            <ol style={{ paddingLeft: '20px', margin: 0 }}>
              <li>Make sure you have trained the model first</li>
              <li>Run: <code style={{ background: '#21262d', padding: '3px 5px', borderRadius: '3px' }}>python AI/bitcoin_price_prediction_using_lstm.py</code></li>
              <li>Wait for the training to complete and CSV file to be generated</li>
              <li>Ensure <code style={{ background: '#21262d', padding: '3px 5px', borderRadius: '3px' }}>Data/bitcoin_predictions.csv</code> exists</li>
              <li>Refresh this page after the CSV file is created</li>
            </ol>
          </div>
        )}
        
        <button 
          onClick={() => handleTimeFrameChange(selectedTimeFrame)} 
          style={{ 
            marginTop: '15px', 
            padding: '8px 15px', 
            backgroundColor: 'var(--accent)',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Retry
        </button>
      </div>
    );
  };

  return (
    <ChartContainer>
      <ChartHeader>
        <ChartTitle>Bitcoin Price Prediction</ChartTitle>
        <TimeFrameSelector>
          <TimeFrameButton
            active={selectedTimeFrame === '1m'}
            onClick={() => handleTimeFrameChange('1m')}
          >
            1M
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === '6m'}
            onClick={() => handleTimeFrameChange('6m')}
          >
            6M
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === '1y'}
            onClick={() => handleTimeFrameChange('1y')}
          >
            1Y
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === '3y'}
            onClick={() => handleTimeFrameChange('3y')}
          >
            3Y
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === 'custom'}
            onClick={() => handleTimeFrameChange('custom')}
          >
            Custom
          </TimeFrameButton>
        </TimeFrameSelector>
      </ChartHeader>

      {showCustomRange && !loading && !error && (
        <DateRangeSelector
          onDateRangeChange={handleDateRangeChange}
          disabled={loading}
        />
      )}

      {loading ? (
        <LoadingSpinner />
      ) : error ? (
        renderErrorMessage()
      ) : (
        <ReactECharts
          option={getOption()}
          style={{ height: '600px' }}
          opts={{ renderer: 'canvas' }}
          notMerge={true}
          lazyUpdate={true}
        />
      )}
      <Disclaimer>
        * This is a prediction model and should not be used as financial advice.
      </Disclaimer>
    </ChartContainer>
  );
};

export default BitcoinPriceChart; 