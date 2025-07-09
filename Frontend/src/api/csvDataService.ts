import { ChartData } from '../types';

export interface PredictionData {
  date: string;
  price: number;
}

export interface DateRange {
  startDate: string;
  endDate: string;
}

/**
 * Load and parse CSV prediction data
 */
export const loadPredictionData = async (): Promise<PredictionData[]> => {
  try {
    console.log('Loading CSV prediction data...');
    
    // Fetch the CSV file from the public directory
    const response = await fetch('/Data/bitcoin_predictions.csv');
    
    if (!response.ok) {
      throw new Error(`Failed to load CSV file: ${response.status} ${response.statusText}`);
    }
    
    const csvText = await response.text();
    
    // Parse CSV data
    const lines = csvText.trim().split('\n');
    
    if (lines.length < 2) {
      throw new Error('CSV file appears to be empty or invalid');
    }
    
    // Skip header row and parse data
    const predictions: PredictionData[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      
      const [dateStr, priceStr] = line.split(',');
      
      if (!dateStr || !priceStr) {
        console.warn(`Skipping invalid line ${i}: ${line}`);
        continue;
      }
      
      const date = dateStr.trim();
      const price = parseFloat(priceStr.trim());
      
      if (isNaN(price)) {
        console.warn(`Skipping line ${i} with invalid price: ${priceStr}`);
        continue;
      }
      
      predictions.push({ date, price });
    }
    
    console.log(`Successfully loaded ${predictions.length} predictions`);
    console.log(`Date range: ${predictions[0]?.date} to ${predictions[predictions.length - 1]?.date}`);
    
    return predictions;
    
  } catch (error) {
    console.error('Error loading CSV data:', error);
    throw new Error(`Failed to load prediction data: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

/**
 * Filter predictions by date range
 */
export const filterPredictionsByDateRange = (
  predictions: PredictionData[],
  startDate: string,
  endDate: string
): PredictionData[] => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  
  return predictions.filter(prediction => {
    const predDate = new Date(prediction.date);
    return predDate >= start && predDate <= end;
  });
};

/**
 * Filter predictions by number of days from start
 */
export const filterPredictionsByDays = (
  predictions: PredictionData[],
  days: number
): PredictionData[] => {
  return predictions.slice(0, days);
};

/**
 * Convert prediction data to chart format
 */
export const convertToChartData = (predictions: PredictionData[]): ChartData => {
  return {
    dates: predictions.map(p => p.date),
    prices: predictions.map(p => p.price)
  };
};

/**
 * Get available date range from predictions
 */
export const getAvailableDateRange = (predictions: PredictionData[]): DateRange | null => {
  if (predictions.length === 0) return null;
  
  return {
    startDate: predictions[0].date,
    endDate: predictions[predictions.length - 1].date
  };
};

/**
 * Get tomorrow's date in YYYY-MM-DD format
 */
export const getTomorrowDate = (): string => {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  return tomorrow.toISOString().split('T')[0];
};

/**
 * Get date N days from today in YYYY-MM-DD format
 */
export const getDateFromToday = (days: number): string => {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return date.toISOString().split('T')[0];
};

/**
 * Validate date range
 */
export const validateDateRange = (startDate: string, endDate: string): string | null => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const tomorrow = new Date(getTomorrowDate());
  const maxDate = new Date(getDateFromToday(1095));
  
  if (start < tomorrow) {
    return 'Start date cannot be earlier than tomorrow';
  }
  
  if (end > maxDate) {
    return 'End date cannot be more than 3 years from today';
  }
  
  if (start >= end) {
    return 'Start date must be before end date';
  }
  
  return null; // Valid
};
