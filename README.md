# Bitcoin Price Prediction Project

A comprehensive application for predicting Bitcoin prices using LSTM (Long Short-Term Memory) neural networks with a professional dark-themed frontend visualization. This project combines machine learning with a responsive web interface to provide Bitcoin price predictions for various timeframes.

## Project Overview

This application consists of:
1. An Advanced Multivariate LSTM model for Bitcoin price prediction
2. CSV-based data flow for predictions
3. A React frontend with interactive charts for data visualization

## Project Structure

```
hackthon_hnt_team/
├── AI/
│   ├── bitcoin_price_prediction_using_lstm.py  # LSTM model training script with CSV generation
│   └── model/                                 # Trained LSTM model files
│       ├── bitcoin_advanced_multivariate_lstm.keras
│       ├── bitcoin_price_scaler.save
│       ├── bitcoin_volume_scaler.save
│       ├── training_loss_curves.png
│       ├── advanced_lstm_analysis.png
│       └── price_volume_analysis.png
├── Data/
│   ├── Bitcoin Historical Data.csv            # Historical Bitcoin price data
│   └── bitcoin_predictions.csv               # Generated predictions (created after training)
├── Frontend/                                  # React frontend application
└── README.md                                  # This file
```

## Features

### AI Model
- Advanced Multivariate LSTM with Price + Volume features
- Bitcoin price prediction for 3 years (1095 days)
- 3-layer LSTM architecture with 64 units each
- Dropout regularization for overfitting prevention
- Nadam optimizer with early stopping and learning rate reduction
- Comprehensive evaluation and visualization
- Direct CSV output generation

### Frontend
- Interactive price prediction chart using ECharts
- Custom date range selection (tomorrow to 3 years)
- CSV-based data loading (no backend server required)
- Responsive design for desktop, tablet, and mobile
- Dark mode UI with Bitcoin-themed styling
- Real-time chart updates based on selected date range

## Tech Stack

### AI Model
- Python 3.8+ (recommended: 3.11 or lower)
- TensorFlow 2.x for Advanced Multivariate LSTM model
- NumPy, pandas for data manipulation
- scikit-learn for model evaluation
- joblib for model persistence
- matplotlib for visualization

### Frontend
- React with TypeScript
- ECharts for data visualization
- Styled Components for styling
- CSV file reading for data loading

## Getting Started

### Prerequisites

1. Python 3.8+ (recommended: 3.11 or lower)
2. Node.js and npm
3. Git

### Installation

#### Clone the Repository
```bash
git clone <repository-url>
cd hackthon_hnt_team
```

#### AI Model Setup
1. Install Python dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn joblib matplotlib
```

2. Train the model and generate predictions:
```bash
cd AI
python bitcoin_price_prediction_using_lstm.py
```
This will:
- Train the Advanced Multivariate LSTM model on all historical data
- Generate 3-year predictions (1095 days)
- Save predictions to `Data/bitcoin_predictions.csv`
- Create comprehensive model analysis visualizations

#### Frontend Setup
1. Install Node.js dependencies:
```bash
cd Frontend
npm install
```

2. Start the development server:
```bash
npm start
```
The frontend will be available at http://localhost:3000

**Note:** No backend server is required. The frontend reads predictions directly from the CSV file.

## Data Flow

### CSV Structure

The generated `Data/bitcoin_predictions.csv` file contains:

```csv
Date,Predicted_Price
2024-01-01,67250.85
2024-01-02,67450.32
2024-01-03,67680.12
...
```

### Frontend Data Loading

The frontend application:
1. Reads the CSV file directly from the `Data/` directory
2. Allows users to select custom date ranges (tomorrow to 3 years)
3. Filters and displays the corresponding predictions
4. Updates charts in real-time based on user selection

### Example CSV Reading

```javascript
// Example of reading CSV data in the frontend
async function loadPredictions() {
  try {
    const response = await fetch('/Data/bitcoin_predictions.csv');
    const csvText = await response.text();

    // Parse CSV data
    const lines = csvText.split('\n');
    const predictions = lines.slice(1).map(line => {
      const [date, price] = line.split(',');
      return { date, price: parseFloat(price) };
    });

    return predictions;
  } catch (error) {
    console.error('Error loading predictions:', error);
  }
}
```

## Model Architecture

The Advanced Multivariate LSTM model architecture includes:
- Input: 60-day lookback period with Price + Volume features
- LSTM Layer 1: 64 units, return_sequences=True
- Dropout: 0.2 (regularization)
- LSTM Layer 2: 64 units, return_sequences=True
- Dropout: 0.2 (regularization)
- LSTM Layer 3: 64 units
- Dropout: 0.2 (regularization)
- Dense: 32 units, ReLU activation
- Output: 1 unit (price prediction)

## Model Training Strategy

1. **Validation Phase**: Test on latest 3 years of data
2. **Final Training**: Train on complete historical dataset
3. **Features**: Price + Volume (multivariate input)
4. **Optimization**: Nadam optimizer with early stopping and learning rate reduction
5. **Evaluation**: RMSE, MAE, and comprehensive visualizations

## Model Evaluation

The model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Comprehensive visualization plots saved in `AI/model/` directory:
  - Training loss curves
  - Advanced LSTM analysis
  - Price-volume correlation analysis

## Workflow

1. **Train Model**: Run `python AI/bitcoin_price_prediction_using_lstm.py`
2. **Generate Predictions**: Script automatically creates `Data/bitcoin_predictions.csv`
3. **View Results**: Open frontend at `http://localhost:3000`
4. **Select Date Range**: Choose custom prediction timeframe
5. **Analyze Charts**: Interactive visualization with ECharts

## Troubleshooting

### Model Training Issues
- Ensure you have sufficient historical data in the Data directory
- Check that TensorFlow is properly installed and compatible with your system
- For GPU acceleration, ensure CUDA and cuDNN are properly configured

### Frontend Issues
- Ensure `Data/bitcoin_predictions.csv` exists (run model training first)
- Check that the CSV file is properly formatted
- Verify that the frontend can access the Data directory

## License

This project is for hackathon purposes only.

## Authors

HNT Team
