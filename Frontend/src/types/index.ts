export interface PredictionResponse {
  [key: string]: {
    dates: string[];
    predicted_prices: number[];
  };
}

export type TimeFrame = '1m' | '6m' | '1y' | '3y' | 'custom';

export interface TimeFrameMapping {
  [key: string]: number;
}

export interface ChartData {
  dates: string[];
  prices: number[];
}

export interface DateRangeSelection {
  startDate: string;
  endDate: string;
}

export interface PredictionData {
  date: string;
  price: number;
}