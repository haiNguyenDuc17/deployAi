# Bitcoin Price Prediction Project

A comprehensive application for predicting Bitcoin prices using LSTM (Long Short-Term Memory) neural networks with a professional dark-themed frontend visualization. This project combines machine learning with a responsive web interface to provide Bitcoin price predictions for various timeframes.

## Project Overview

This application consists of:
1. A backend LSTM model for Bitcoin price prediction
2. A Flask API to serve predictions
3. A React frontend with interactive charts for data visualization

## Project Structure

```
hackthon_hnt_team/
├── AI/
│   ├── bitcoin_price_prediction_using_lstm.py  # LSTM model training script
│   └── model/                                 # Trained LSTM model files
│       ├── bitcoin_lstm_model.keras
│       ├── bitcoin_price_scaler.save
│       ├── loss_curves.png
│       └── actual_vs_predicted.png
├── Backend/
│   └── api_server.py                          # API server for predictions
├── Data/
│   └── Bitcoin Historical Data.csv            # Historical Bitcoin price data
├── Frontend/                                  # React frontend application
└── README.md                                  # This file
```

## Features

### Backend (AI)
- Bitcoin price prediction for multiple timeframes:
  - 1 month (30 days)
  - 6 months (180 days)
  - 1 year (365 days)
  - 3 years (1095 days)
- LSTM model with 60% training data and 40% testing data
- Accuracy evaluation metrics (RMSE, MAE, R² score)
- REST API for retrieving predictions

### Frontend
- Interactive price prediction chart using ECharts
- Time-frame selection between 1 month, 6 months, 1 year, and 3 years
- Real-time data fetching from the Flask API
- Responsive design for desktop, tablet, and mobile
- Dark mode UI with Bitcoin-themed styling
- Fallback mechanism when API is unavailable

## Tech Stack

### Backend
- Python 3.8+ (recommended: 3.11 or lower)
- TensorFlow 2.x for LSTM model
- Flask for API server
- NumPy, pandas for data manipulation
- scikit-learn for model evaluation
- joblib for model persistence

### Frontend
- React with TypeScript
- ECharts for data visualization
- Styled Components for styling
- Axios for API requests

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

#### Backend Setup
1. Install Python dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn joblib flask matplotlib requests
```

2. Train the model (optional - pre-trained model included):
```bash
cd AI
python bitcoin_price_prediction_using_lstm.py
```
This will automatically train the model with all historical data.

3. Start the API server:
```bash
cd Backend
python api_server.py
```
The API server will run at http://localhost:5000

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

## API Documentation

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
- Visualization of model performance is saved in `AI/model/` directory

## Frontend Integration Example

```javascript
// Example of fetching predictions with JavaScript
async function fetchPredictions(days = 30) {
  try {
    const response = await fetch(`http://localhost:5000/predict/${days}`);
    const data = await response.json();

    // Process the prediction data
    const prediction = data[`${days}_days`];
    const dates = prediction.dates;
    const prices = prediction.predicted_prices;

    // Now you can use this data with your favorite charting library
    console.log(`First prediction: ${dates[0]} - $${prices[0]}`);

    return prediction;
  } catch (error) {
    console.error('Error fetching predictions:', error);
  }
}
```

## Troubleshooting

### API Connection Issues
- Ensure the Flask API is running at http://localhost:5000
- Check for CORS issues if accessing from a different domain
- Verify that the correct endpoint is being called (/predict or /predict/<days>)

### Model Training Issues
- Ensure you have sufficient historical data in the Data directory
- Check that TensorFlow is properly installed and compatible with your system
- For GPU acceleration, ensure CUDA and cuDNN are properly configured

## License

This project is for hackathon purposes only.

## Authors

HNT Team
