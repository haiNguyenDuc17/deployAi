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
import { fetchPrediction } from '../api/predictionApi';
import { TimeFrame, TimeFrameMapping, ChartData } from '../types';

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

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const days = timeFrameMap[selectedTimeFrame];
        console.log(`Loading data for ${selectedTimeFrame} (${days} days)`);
        
        console.log('Fetching data from API');
        const data = await fetchPrediction(days);
        
        const key = `${days}_days`;
        
        if (data && data[key]) {
          console.log(`Successfully fetched data for ${days} days:`, data[key]);
          setChartData({
            dates: data[key].dates,
            prices: data[key].predicted_prices,
          });
        } else {
          console.error('Invalid data format:', data);
          throw new Error('Invalid data format received from API');
        }
      } catch (err: any) {
        console.error('Error loading chart data:', err);
        setError(err.message || 'Failed to load prediction data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedTimeFrame]);

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
  };

  const renderErrorMessage = () => {
    const isApiConnectionError = error?.includes('Network error') || error?.includes('404');
    
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
        
        {isApiConnectionError && (
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
            <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Diagnostic Steps:</div>
            <ol style={{ paddingLeft: '20px', margin: 0 }}>
              <li>Make sure your Flask API server is running</li>
              <li>Open browser DevTools (F12)</li>
              <li>In the console tab, run: <code style={{ background: '#21262d', padding: '3px 5px', borderRadius: '3px' }}>testApiEndpoints()</code></li>
              <li>Look for successful endpoint connections in the results</li>
              <li>Update the API configuration in the code accordingly</li>
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
            1 M
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === '6m'}
            onClick={() => handleTimeFrameChange('6m')}
          >
            6 M
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === '1y'}
            onClick={() => handleTimeFrameChange('1y')}
          >
            1 Y
          </TimeFrameButton>
          <TimeFrameButton
            active={selectedTimeFrame === '3y'}
            onClick={() => handleTimeFrameChange('3y')}
          >
            3 Y
          </TimeFrameButton>
        </TimeFrameSelector>
      </ChartHeader>

      {loading ? (
        <LoadingSpinner />
      ) : error ? (
        renderErrorMessage()
      ) : (
        <ReactECharts
          option={getOption()}
          style={{ height: '400px', width: '100%' }}
          opts={{ renderer: 'canvas' }}
          notMerge={true}
          lazyUpdate={true}
        />
      )}

      <Disclaimer>
        Disclaimer: The information on this site is for reference only and does not constitute financial advice. 
        We accept no liability for investment decisions made based on these predictions.
      </Disclaimer>
    </ChartContainer>
  );
};

export default BitcoinPriceChart; 