# Bitcoin Price Prediction Project

This project provides Bitcoin price predictions for various timeframes using LSTM (Long Short-Term Memory) neural networks. The API allows users to get Bitcoin price predictions for different time frames and returns data formatted for easy integration with frontend applications.

## Project Structure

```
hackthon_hnt_team/
├── AI/
│   ├── bitcoin_price_prediction_using_lstm.py  # Enhanced LSTM model with 60/40 train/test split
│   └── model/                                 # Trained LSTM model files
│       ├── bitcoin_lstm_model.keras
│       ├── bitcoin_price_scaler.save
│       ├── loss_curves.png
│       └── actual_vs_predicted.png
├── Backend/
│   ├── bitcoin_price_prediction_api.py        # API for Bitcoin price predictions
│   └── example_api_usage.py                   # Example of how to use the API
├── Data/
│   └── Bitcoin Historical Data.csv            # Historical Bitcoin price data
├── Frontend/                                  # Frontend code for the application
└── README.md                                  # This file
```

## Features

- Bitcoin price prediction for:
  - 1 month (30 days)
  - 6 months (180 days)
  - 1 year (365 days)
  - 3 years (1095 days)
- Machine learning model with 60% training data and 40% testing data
- Accuracy evaluation metrics (RMSE, MAE, R² score)
- REST API for retrieving predictions
- Example code for integrating with the API

## System Overview

1. **LSTM Model Training** (`bitcoin_price_prediction_using_lstm.py`): 
   - Processes historical Bitcoin price data
   - Trains an LSTM model
   - Saves the trained model and scaler for future use

2. **Prediction API** (`bitcoin_price_prediction_api.py`):
   - Loads the saved model
   - Provides endpoints for making predictions
   - Returns data formatted for visualization

3. **API Usage Example** (`example_api_usage.py`):
   - Demonstrates how to call the API
   - Shows how to process the returned data

## Getting Started

### Prerequisites

- Python 3.8+ (recommended: 3.11 or lower)
- Required Python packages:
  - TensorFlow 2.x
  - Flask
  - NumPy
  - pandas
  - scikit-learn
  - joblib
  - matplotlib
  - requests

### Installation

```bash
pip install tensorflow numpy pandas scikit-learn joblib flask matplotlib requests
```

### Training the Model

```bash
cd AI
python bitcoin_price_prediction_using_lstm.py
```

When prompted, select option 1 to train a new model with all historical data.

### Running the API Server

```bash
cd Backend
python bitcoin_price_prediction_api.py
```

When prompted, select option 3 to start the API server.

## API Endpoints

### 1. Get Predictions for Standard Timeframes

```
GET /predict
```

Returns predictions for all standard timeframes (30, 180, 365, 1095 days).

#### Example API Call

```bash
curl http://localhost:5000/predict
```

#### Example Response

```json
{
  "30_days": {
    "dates": [
      "2023-05-01", "2023-05-02", "2023-05-03", "..."
    ],
    "predicted_prices": [
      67250.85, 67450.32, 67680.12, "..."
    ]
  },
  "180_days": {
    "dates": ["..."],
    "predicted_prices": ["..."]
  },
  "365_days": {
    "dates": ["..."],
    "predicted_prices": ["..."]
  },
  "1095_days": {
    "dates": ["..."],
    "predicted_prices": ["..."]
  }
}
```

### 2. Get Prediction for Custom Timeframe

```
GET /predict/<days>
```

Returns prediction for a custom timeframe specified by the number of days.

#### Example API Call

```bash
curl http://localhost:5000/predict/30
```

#### Example Response

```json
{
  "30_days": {
    "dates": [
      "2023-05-01", "2023-05-02", "...", "2023-06-29"
    ],
    "predicted_prices": [
      67250.85, 67450.32, "...", 70520.45
    ]
  }
}
```

## Model Architecture

The LSTM (Long Short-Term Memory) model architecture includes:
- Input layer with 15-day lookback period
- LSTM layers with dropout to prevent overfitting
- Dense output layer
- Model trained on 60% of historical data, validated on 40%

## Model Evaluation

The model is evaluated using:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (coefficient of determination)
- Mean Absolute Percentage Error (MAPE)
- Prediction Accuracy Percentage

Visualization of model performance is saved in `AI/model/` directory.

## Frontend Integration

The API responses are designed to be easily integrated with frontend visualization libraries like echarts. The prediction data includes dates and corresponding predicted prices that can be directly used in chart components.

Example of using the API with JavaScript:

```javascript
// Example of fetching predictions
async function fetchPredictions() {
  try {
    const response = await fetch('http://localhost:5000/predict');
    const data = await response.json();
    
    // Process the 30-day prediction data
    const thirtyDayPrediction = data['30_days'];
    const dates = thirtyDayPrediction.dates;
    const prices = thirtyDayPrediction.predicted_prices;
    
    // Now you can use this data with your favorite charting library
    console.log(`First prediction: ${dates[0]} - $${prices[0]}`);
    
    return data;
  } catch (error) {
    console.error('Error fetching predictions:', error);
  }
}
```

## License

This project is for hackathon purposes only.

## Authors

HNT Team 