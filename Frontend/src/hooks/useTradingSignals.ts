import { useState, useEffect, useCallback, useRef } from 'react';
import { TradingSignalsService } from '../services/tradingSignalsService';
import { loadPredictionData } from '../api/csvDataService';
import { SignalAnalysis } from '../types';

interface UseTradingSignalsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
  onError?: (error: string) => void;
  onUpdate?: (analysis: SignalAnalysis) => void;
}

interface UseTradingSignalsReturn {
  signalAnalysis: SignalAnalysis | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
  startAutoRefresh: () => void;
  stopAutoRefresh: () => void;
  isAutoRefreshActive: boolean;
}

/**
 * Custom hook for managing trading signals with real-time updates
 */
export const useTradingSignals = (options: UseTradingSignalsOptions = {}): UseTradingSignalsReturn => {
  const {
    autoRefresh = false,
    refreshInterval = 5 * 60 * 1000, // 5 minutes default
    onError,
    onUpdate
  } = options;

  const [signalAnalysis, setSignalAnalysis] = useState<SignalAnalysis | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isAutoRefreshActive, setIsAutoRefreshActive] = useState<boolean>(autoRefresh);

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isLoadingRef = useRef<boolean>(false);

  const loadSignals = useCallback(async () => {
    // Prevent multiple simultaneous requests
    if (isLoadingRef.current) return;

    try {
      isLoadingRef.current = true;
      setLoading(true);
      setError(null);

      const predictionData = await loadPredictionData();
      const analysis = await TradingSignalsService.generateSignals(predictionData);

      setSignalAnalysis(analysis);
      setLastUpdated(new Date());

      // Call onUpdate callback if provided
      if (onUpdate) {
        onUpdate(analysis);
      }
    } catch (err: any) {
      const errorMessage = err.message || 'Failed to load trading signals';
      console.error('Error loading trading signals:', err);
      setError(errorMessage);

      // Call onError callback if provided
      if (onError) {
        onError(errorMessage);
      }
    } finally {
      setLoading(false);
      isLoadingRef.current = false;
    }
  }, []); // Remove onError and onUpdate from dependencies to prevent infinite loops

  const startAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setIsAutoRefreshActive(true);

    intervalRef.current = setInterval(() => {
      loadSignals();
    }, refreshInterval);
  }, [refreshInterval]); // Remove loadSignals dependency to prevent recreation

  const stopAutoRefresh = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsAutoRefreshActive(false);
  }, []);

  const refresh = useCallback(async () => {
    await loadSignals();
  }, []); // Remove loadSignals dependency

  // Initial load
  useEffect(() => {
    loadSignals();
  }, []); // Remove loadSignals dependency for initial load only

  // Auto-refresh setup
  useEffect(() => {
    if (autoRefresh) {
      startAutoRefresh();
    }

    return () => {
      stopAutoRefresh();
    };
  }, [autoRefresh]); // Remove function dependencies to prevent recreation

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Handle visibility change to pause/resume auto-refresh when tab is not visible
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Tab is not visible, pause auto-refresh
        if (isAutoRefreshActive && intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } else {
        // Tab is visible, resume auto-refresh if it was active
        if (isAutoRefreshActive && !intervalRef.current) {
          startAutoRefresh();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isAutoRefreshActive]); // Remove startAutoRefresh dependency

  return {
    signalAnalysis,
    loading,
    error,
    lastUpdated,
    refresh,
    startAutoRefresh,
    stopAutoRefresh,
    isAutoRefreshActive
  };
};

/**
 * Hook for getting cached trading signals without auto-refresh
 * Useful for components that just need to display signals without managing updates
 */
export const useTradingSignalsSnapshot = () => {
  const [signalAnalysis, setSignalAnalysis] = useState<SignalAnalysis | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadSignals = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const predictionData = await loadPredictionData();
        const analysis = await TradingSignalsService.generateSignals(predictionData);
        
        setSignalAnalysis(analysis);
      } catch (err: any) {
        console.error('Error loading trading signals snapshot:', err);
        setError(err.message || 'Failed to load trading signals');
      } finally {
        setLoading(false);
      }
    };

    loadSignals();
  }, []);

  return {
    signalAnalysis,
    loading,
    error
  };
};


