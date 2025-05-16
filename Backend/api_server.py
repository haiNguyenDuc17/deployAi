from flask import Flask, jsonify
import os
import sys
import pandas as pd
import numpy as np
import math
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create Flask API
app = Flask(__name__)

# Function to clean up the price values (remove commas and quotes)
def clean_price(price_str):
    if isinstance(price_str, str):
        return float(price_str.replace('"', '').replace(',', ''))
    return price_str

# Function to process volume data with K, M, B suffixes
def process_volume(vol_str):
    if isinstance(vol_str, str):
        if 'K' in vol_str:
            return float(vol_str.replace('K', '')) * 1000
        elif 'M' in vol_str:
            return float(vol_str.replace('M', '')) * 1000000
        elif 'B' in vol_str:
            return float(vol_str.replace('B', '')) * 1000000000
        else:
            return float(vol_str)
    return vol_str

# Function to load and preprocess data
def load_data():
    # Load our dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'Data', 'Bitcoin Historical Data.csv')
    maindf = pd.read_csv(data_path)

    # Clean numeric columns - they have commas and quotes
    numeric_columns = ['Price', 'Open', 'High', 'Low']
    for col in numeric_columns:
        maindf[col] = maindf[col].apply(clean_price)

    # Handle the 'Vol.' column
    maindf['Volume'] = maindf['Vol.'].apply(process_volume)

    # Convert Date to datetime
    maindf['Date'] = pd.to_datetime(maindf['Date'], format='%m/%d/%Y')

    # Since the data is in reverse chronological order (newest first), sort it chronologically
    maindf = maindf.sort_values('Date')

    print('Total number of days present in the dataset: ', maindf.shape[0])
    print('Total number of fields present in the dataset: ', maindf.shape[1])

    return maindf

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to predict future values
def predict_future(model, scaler, data, window_size, days_to_predict):
    x_input = data[len(data)-window_size:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # Calculate historical volatility from the original data
    historical_data = scaler.inverse_transform(data[-30:])  # Use last 30 days for volatility calculation
    daily_returns = np.diff(historical_data.flatten()) / historical_data[:-1].flatten()
    historical_volatility = np.std(daily_returns)
    
    # Use historical volatility as the baseline for simulation
    # Scale factor can be adjusted based on desired volatility level
    volatility_scale = 0.7  # Adjust this value to control volatility intensity

    lst_output = []
    n_steps = window_size
    i = 0

    while i < days_to_predict:
        if len(temp_input) > window_size:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            
            # Add realistic market volatility based on historical patterns
            pred_value = yhat[0][0]
            scaled_value = pred_value
            noise = np.random.normal(0, historical_volatility * volatility_scale * abs(scaled_value), 1)[0]
            yhat[0][0] = scaled_value + noise
            
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = np.array(temp_input).reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            pred_value = yhat[0][0]
            scaled_value = pred_value
            noise = np.random.normal(0, historical_volatility * volatility_scale * abs(scaled_value), 1)[0]
            yhat[0][0] = scaled_value + noise
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    # Transform the predictions back to original scale
    predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    return predictions

def make_predictions(timeframes=[30, 180, 365, 1095]):
    """
    Make predictions for different timeframes
    timeframes: list of days to predict [1 month, 6 months, 1 year, 3 years]
    """
    # Check if model exists, if not train a new one
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'AI', 'model', 'bitcoin_lstm_model.keras')
    scaler_path = os.path.join(base_dir, 'AI', 'model', 'bitcoin_price_scaler.save')
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model path absolute: {os.path.abspath(model_path)}")
    print(f"Model path exists: {os.path.exists(model_path)}")
    print(f"Scaler path exists: {os.path.exists(scaler_path)}")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading existing model and scaler...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Load the latest data
        maindf = load_data()
        closedf = maindf[['Date','Price']].copy()
        closedf['Date'] = pd.to_datetime(closedf['Date'])
        closedf = closedf.sort_values(by='Date', ascending=True).reset_index(drop=True)
        closedf['Price'] = closedf['Price'].astype('float64')
        dates = closedf['Date']
        price = closedf['Price']
        window_size = 60
        # Prepare the input for prediction: last window_size days
        data = scaler.transform(price.values.reshape(-1,1))
    else:
        print("No model found. Cannot make predictions.")
        return {"error": "Model not found"}

    # Make predictions for each timeframe
    predictions = {}
    for days in timeframes:
        print(f"Predicting for next {days} days...")
        preds = predict_future(model, scaler, data, window_size, days)

        # Create dates for predictions
        last_date = dates.iloc[-1]
        future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(preds))]

        # Store predictions with dates
        predictions[f"{days}_days"] = {
            "dates": future_dates,
            "predicted_prices": [round(price, 2) for price in preds]
        }

    return predictions

@app.route('/predict', methods=['GET'])
def predict_api():
    predictions = make_predictions()
    return jsonify(predictions)

@app.route('/predict/<int:days>', methods=['GET'])
def predict_custom(days):
    predictions = make_predictions([days])
    return jsonify(predictions)

if __name__ == "__main__":
    print("Starting Bitcoin Price Prediction API server...")
    app.run(host='0.0.0.0', port=5000)
