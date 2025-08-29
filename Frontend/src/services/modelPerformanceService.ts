/**
 * Model Performance Service
 * Handles loading and processing AI model performance metrics
 */

export interface ModelPerformanceMetrics {
  validation_metrics: {
    rmse_scaled: number;
    mae_scaled: number;
    rmse_original: number;
    mae_original: number;
    data_period: string;
  };
  final_model_metrics: {
    rmse_scaled: number;
    mae_scaled: number;
    rmse_original: number;
    mae_original: number;
    mape: number;
    r2_scaled: number;
    r2_original: number;
    test_loss: number;
    accuracy_percentage: number;
    data_period: string;
  };
  training_history: {
    validation_loss: number[];
    validation_val_loss: number[];
    epochs_trained: number;
    final_loss: number;
    final_val_loss: number;
    best_loss: number;
    best_val_loss: number;
  };
  model_architecture: {
    window_size: number;
    features: string[];
    lstm_layers: number;
    lstm_units: number;
    dropout_rate: number;
    optimizer: string;
    loss_function: string;
  };
  dataset_info: {
    total_samples: number;
    training_samples: number;
    test_samples: number;
    train_test_split: string;
    date_range: {
      start_date: string;
      end_date: string;
    };
  };
  export_timestamp: string;
  model_version: string;
}

export interface ModelPerformanceSummary {
  accuracy: number;
  reliability: 'HIGH' | 'MEDIUM' | 'LOW';
  lastTrainingDate: string;
  modelVersion: string;
  keyMetrics: {
    rmse: number;
    mae: number;
    r2Score: number;
    mape: number;
  };
  trainingInfo: {
    epochsTrained: number;
    datasetSize: number;
    features: string[];
  };
}

export class ModelPerformanceService {
  private static readonly METRICS_FILE_PATH = '/Data/model_performance_metrics.json';
  private static cachedMetrics: ModelPerformanceMetrics | null = null;
  private static lastFetchTime: number = 0;
  private static readonly CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

  /**
   * Load model performance metrics from JSON file
   */
  static async loadModelPerformanceMetrics(): Promise<ModelPerformanceMetrics> {
    const now = Date.now();
    
    // Return cached data if still valid
    if (this.cachedMetrics && (now - this.lastFetchTime) < this.CACHE_DURATION) {
      return this.cachedMetrics;
    }

    try {
      const response = await fetch(this.METRICS_FILE_PATH);
      
      if (!response.ok) {
        throw new Error(`Failed to load model performance metrics: ${response.status} ${response.statusText}`);
      }

      const metrics: ModelPerformanceMetrics = await response.json();
      
      // Validate the metrics structure
      this.validateMetricsStructure(metrics);
      
      // Cache the metrics
      this.cachedMetrics = metrics;
      this.lastFetchTime = now;
      
      return metrics;
    } catch (error) {
      console.error('Error loading model performance metrics:', error);
      throw new Error(`Unable to load model performance metrics: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Get a simplified summary of model performance
   */
  static async getModelPerformanceSummary(): Promise<ModelPerformanceSummary> {
    const metrics = await this.loadModelPerformanceMetrics();
    
    const accuracy = metrics.final_model_metrics.accuracy_percentage;
    const reliability = this.calculateReliability(metrics);
    
    return {
      accuracy: Math.round(accuracy * 100) / 100,
      reliability,
      lastTrainingDate: new Date(metrics.export_timestamp).toLocaleDateString(),
      modelVersion: metrics.model_version,
      keyMetrics: {
        rmse: Math.round(metrics.final_model_metrics.rmse_original * 100) / 100,
        mae: Math.round(metrics.final_model_metrics.mae_original * 100) / 100,
        r2Score: Math.round(metrics.final_model_metrics.r2_original * 1000) / 1000,
        mape: Math.round(metrics.final_model_metrics.mape * 100) / 100
      },
      trainingInfo: {
        epochsTrained: metrics.training_history.epochs_trained,
        datasetSize: metrics.dataset_info.total_samples,
        features: metrics.model_architecture.features
      }
    };
  }

  /**
   * Get training history data for charts
   */
  static async getTrainingHistory(): Promise<{
    epochs: number[];
    trainingLoss: number[];
    validationLoss: number[];
  }> {
    const metrics = await this.loadModelPerformanceMetrics();
    
    const epochs = Array.from({ length: metrics.training_history.epochs_trained }, (_, i) => i + 1);
    
    return {
      epochs,
      trainingLoss: metrics.training_history.validation_loss,
      validationLoss: metrics.training_history.validation_val_loss
    };
  }

  /**
   * Get detailed metrics for display
   */
  static async getDetailedMetrics(): Promise<{
    performance: {
      accuracy: number;
      rmse: number;
      mae: number;
      mape: number;
      r2Score: number;
    };
    training: {
      epochs: number;
      finalLoss: number;
      bestLoss: number;
      convergence: 'GOOD' | 'FAIR' | 'POOR';
    };
    dataset: {
      totalSamples: number;
      trainingSamples: number;
      testSamples: number;
      dateRange: string;
    };
    architecture: {
      windowSize: number;
      features: string[];
      layers: string;
      optimizer: string;
    };
  }> {
    const metrics = await this.loadModelPerformanceMetrics();
    
    const convergence = this.assessConvergence(metrics.training_history);
    
    return {
      performance: {
        accuracy: Math.round(metrics.final_model_metrics.accuracy_percentage * 100) / 100,
        rmse: Math.round(metrics.final_model_metrics.rmse_original * 100) / 100,
        mae: Math.round(metrics.final_model_metrics.mae_original * 100) / 100,
        mape: Math.round(metrics.final_model_metrics.mape * 100) / 100,
        r2Score: Math.round(metrics.final_model_metrics.r2_original * 1000) / 1000
      },
      training: {
        epochs: metrics.training_history.epochs_trained,
        finalLoss: Math.round(metrics.training_history.final_loss * 1000000) / 1000000,
        bestLoss: Math.round(metrics.training_history.best_loss * 1000000) / 1000000,
        convergence
      },
      dataset: {
        totalSamples: metrics.dataset_info.total_samples,
        trainingSamples: metrics.dataset_info.training_samples,
        testSamples: metrics.dataset_info.test_samples,
        dateRange: `${metrics.dataset_info.date_range.start_date} to ${metrics.dataset_info.date_range.end_date}`
      },
      architecture: {
        windowSize: metrics.model_architecture.window_size,
        features: metrics.model_architecture.features,
        layers: `${metrics.model_architecture.lstm_layers} LSTM layers (${metrics.model_architecture.lstm_units} units each)`,
        optimizer: metrics.model_architecture.optimizer
      }
    };
  }

  /**
   * Check if model metrics are available
   */
  static async isModelMetricsAvailable(): Promise<boolean> {
    try {
      await this.loadModelPerformanceMetrics();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Clear cached metrics (force refresh on next load)
   */
  static clearCache(): void {
    this.cachedMetrics = null;
    this.lastFetchTime = 0;
  }

  /**
   * Validate the structure of loaded metrics
   */
  private static validateMetricsStructure(metrics: any): void {
    const requiredFields = [
      'validation_metrics',
      'final_model_metrics',
      'training_history',
      'model_architecture',
      'dataset_info'
    ];

    for (const field of requiredFields) {
      if (!metrics[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }

    // Validate final_model_metrics has required performance indicators
    const requiredPerformanceFields = ['rmse_original', 'mae_original', 'mape', 'r2_original', 'accuracy_percentage'];
    for (const field of requiredPerformanceFields) {
      if (typeof metrics.final_model_metrics[field] !== 'number') {
        throw new Error(`Missing or invalid performance metric: ${field}`);
      }
    }
  }

  /**
   * Calculate model reliability based on performance metrics
   */
  private static calculateReliability(metrics: ModelPerformanceMetrics): 'HIGH' | 'MEDIUM' | 'LOW' {
    const accuracy = metrics.final_model_metrics.accuracy_percentage;
    const r2Score = metrics.final_model_metrics.r2_original;
    const mape = metrics.final_model_metrics.mape;

    // High reliability: Good accuracy, high R², low MAPE
    if (accuracy >= 85 && r2Score >= 0.8 && mape <= 10) {
      return 'HIGH';
    }
    
    // Medium reliability: Moderate performance
    if (accuracy >= 70 && r2Score >= 0.6 && mape <= 20) {
      return 'MEDIUM';
    }
    
    // Low reliability: Poor performance
    return 'LOW';
  }

  /**
   * Assess training convergence quality
   */
  private static assessConvergence(trainingHistory: ModelPerformanceMetrics['training_history']): 'GOOD' | 'FAIR' | 'POOR' {
    const { final_loss, best_loss, validation_val_loss } = trainingHistory;
    
    // Check if training converged well (final loss close to best loss)
    const lossRatio = final_loss / best_loss;
    
    // Check for overfitting (validation loss much higher than training loss)
    const overfittingRatio = validation_val_loss[validation_val_loss.length - 1] / final_loss;
    
    if (lossRatio <= 1.1 && overfittingRatio <= 2.0) {
      return 'GOOD';
    } else if (lossRatio <= 1.5 && overfittingRatio <= 3.0) {
      return 'FAIR';
    } else {
      return 'POOR';
    }
  }
}
