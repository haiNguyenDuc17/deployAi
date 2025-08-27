import { ChartData } from '../types';

export interface PredictionData {
  date: string;
  price: number;
}

export interface DateRange {
  startDate: string;
  endDate: string;
}

export interface CryptoData {
  date: string;
  price: number;
  open: number;
  high: number;
  low: number;
  volume: string;
  changePercent: number;
}

export interface CryptoPriceInfo {
  symbol: string;
  name: string;
  price: number;
  changePercent24h: number;
  changePercent7d: number;
  volume: string;
  icon: string;
}

const parseCSVLine = (csvText: string): any[] => {
  const lines = csvText.split('\n');
  const headers = parseCSVRow(lines[0]);
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim()) {
      const values = parseCSVRow(lines[i]);
      const row: any = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || '';
      });
      data.push(row);
    }
  }

  return data;
};

const parseCSVRow = (row: string): string[] => {
  const result = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < row.length; i++) {
    const char = row[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
};

const fetchCryptoData = async (filename: string): Promise<CryptoData[]> => {
  try {
    const response = await fetch(`/Data/${filename}`);
    const csvText = await response.text();
    const data = parseCSVLine(csvText);

    return data.map((row: any) => ({
      date: row.Date,
      price: parseFloat(row.Price.replace(/,/g, '')),
      open: parseFloat(row.Open.replace(/,/g, '')),
      high: parseFloat(row.High.replace(/,/g, '')),
      low: parseFloat(row.Low.replace(/,/g, '')),
      volume: row['Vol.'] || row.Vol || 'N/A',
      changePercent: parseFloat((row['Change %'] || row['Change%'] || '0').replace('%', ''))
    }));
  } catch (error) {
    console.error(`Error fetching ${filename}:`, error);
    throw error;
  }
};

const calculatePriceChange = (currentPrice: number, previousPrice: number): number => {
  if (previousPrice === 0) return 0;
  return ((currentPrice - previousPrice) / previousPrice) * 100;
};

const getPriceChangeForDays = (data: CryptoData[], days: number): number => {
  if (data.length <= days) return 0;

  // Data is ordered from most recent (index 0) to oldest
  // So for 24h change: compare data[0] (today) with data[1] (yesterday)
  // For 7d change: compare data[0] (today) with data[7] (7 days ago)
  const currentPrice = data[0].price;
  const previousPrice = data[days].price;

  console.log(`Price change calculation for ${days} days:`, {
    currentPrice,
    previousPrice,
    change: calculatePriceChange(currentPrice, previousPrice)
  });

  return calculatePriceChange(currentPrice, previousPrice);
};

export const fetchBitcoinData = async (): Promise<CryptoData[]> => {
  return fetchCryptoData('Bitcoin Historical Data.csv');
};

export const fetchEthereumData = async (): Promise<CryptoData[]> => {
  return fetchCryptoData('Ethereum Historical Data.csv');
};

export const fetchXRPData = async (): Promise<CryptoData[]> => {
  return fetchCryptoData('XRP Historical Data.csv');
};

export const fetchSolanaData = async (): Promise<CryptoData[]> => {
  return fetchCryptoData('Solana Historical Data.csv');
};

export const fetchBNBData = async (): Promise<CryptoData[]> => {
  return fetchCryptoData('BNB Historical Data.csv');
};

export const fetchAllCryptoPrices = async (): Promise<CryptoPriceInfo[]> => {
  try {
    const [btcData, ethData, xrpData, solData, bnbData] = await Promise.all([
      fetchBitcoinData(),
      fetchEthereumData(),
      fetchXRPData(),
      fetchSolanaData(),
      fetchBNBData()
    ]);

    return [
      {
        symbol: 'BTC',
        name: 'Bitcoin',
        price: btcData[0]?.price || 0,
        changePercent24h: getPriceChangeForDays(btcData, 1),
        changePercent7d: getPriceChangeForDays(btcData, 7),
        volume: btcData[0]?.volume || 'N/A',
        icon: 'pi pi-bitcoin'
      },
      {
        symbol: 'ETH',
        name: 'Ethereum',
        price: ethData[0]?.price || 0,
        changePercent24h: getPriceChangeForDays(ethData, 1),
        changePercent7d: getPriceChangeForDays(ethData, 7),
        volume: ethData[0]?.volume || 'N/A',
        icon: 'pi pi-ethereum'
      },
      {
        symbol: 'XRP',
        name: 'XRP',
        price: xrpData[0]?.price || 0,
        changePercent24h: getPriceChangeForDays(xrpData, 1),
        changePercent7d: getPriceChangeForDays(xrpData, 7),
        volume: xrpData[0]?.volume || 'N/A',
        icon: 'pi pi-circle-fill'
      },
      {
        symbol: 'SOL',
        name: 'Solana',
        price: solData[0]?.price || 0,
        changePercent24h: getPriceChangeForDays(solData, 1),
        changePercent7d: getPriceChangeForDays(solData, 7),
        volume: solData[0]?.volume || 'N/A',
        icon: 'pi pi-sun'
      },
      {
        symbol: 'BNB',
        name: 'BNB',
        price: bnbData[0]?.price || 0,
        changePercent24h: getPriceChangeForDays(bnbData, 1),
        changePercent7d: getPriceChangeForDays(bnbData, 7),
        volume: bnbData[0]?.volume || 'N/A',
        icon: 'pi pi-circle'
      }
    ];
  } catch (error) {
    console.error('Error fetching crypto prices:', error);
    return [];
  }
};

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
