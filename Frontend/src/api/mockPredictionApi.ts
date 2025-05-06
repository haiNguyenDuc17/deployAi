import { PredictionResponse } from '../types';

// Mock data generator function
const generateMockData = (days: number): PredictionResponse => {
  const key = `${days}_days`;
  const dates: string[] = [];
  const predicted_prices: number[] = [];
  
  // Current date
  const currentDate = new Date();
  
  // Start with a base price around current BTC value
  let price = 64000;
  
  // Generate data for each day
  for (let i = 0; i < days; i++) {
    // Add dates
    const date = new Date(currentDate);
    date.setDate(date.getDate() + i);
    dates.push(date.toISOString().split('T')[0]);
    
    // Generate a random price change between -5% and +5% with some trend
    const trend = Math.sin(i / 30) * 0.5; // Add a sine wave pattern
    const randomChange = (Math.random() - 0.5) * 0.05; // Random -2.5% to +2.5%
    const change = price * (randomChange + trend * 0.01);
    
    price += change;
    // Ensure price doesn't go below a reasonable value
    if (price < 10000) price = 10000 + Math.random() * 5000;
    
    predicted_prices.push(Math.round(price * 100) / 100);
  }
  
  return {
    [key]: {
      dates,
      predicted_prices
    }
  };
};

// Mock API function
export const fetchMockPrediction = async (days: number): Promise<PredictionResponse> => {
  return new Promise((resolve) => {
    // Simulate network delay
    setTimeout(() => {
      const mockData = generateMockData(days);
      resolve(mockData);
    }, 800);
  });
}; 