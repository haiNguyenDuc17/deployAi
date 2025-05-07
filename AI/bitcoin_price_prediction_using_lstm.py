# First we will import the necessary Library

import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

# For Evalution we will use these library

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# For PLotting we will use these library

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

"""# 3. Loading Dataset"""

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
    maindf = pd.read_csv('Data/Bitcoin Historical Data.csv')

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

# Function to build and train LSTM model
def build_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    # Increasing the number of LSTM units for better accuracy
    model.add(LSTM(50, input_shape=(None, 1), activation="relu", return_sequences=True))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(LSTM(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=200, batch_size=32, verbose=1)

    return model, history

# Function to make predictions for different time frames
def predict_future(model, scaler, data, time_step, days_to_predict):
    x_input = data[len(data)-time_step:].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = time_step
    i = 0

    while i < days_to_predict:
        if len(temp_input) > time_step:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i += 1

    # Transform the predictions back to original scale
    predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    return predictions

def evaluate_model(model, X, y, scaler, name=""):
    """Evaluate model performance with various metrics"""
    predictions = model.predict(X)

    # Transform back to original form
    predictions = scaler.inverse_transform(predictions)
    original_y = scaler.inverse_transform(y.reshape(-1,1))

    # Calculate various metrics
    rmse = math.sqrt(mean_squared_error(original_y, predictions))
    mae = mean_absolute_error(original_y, predictions)
    r2 = r2_score(original_y, predictions)

    print(f"\n{name} Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    return rmse, mae, r2

def train_and_save_model():
    # Load the data
    maindf = load_data()

    # Extract price data
    closedf = maindf[['Date','Price']]
    print("Shape of price dataframe:", closedf.shape)

    # Create a copy for visualization
    close_stock = closedf.copy()

    # Deleting date column and normalizing using MinMax Scaler
    dates = closedf['Date']  # Save dates for later reference
    del closedf['Date']
    scaler = MinMaxScaler(feature_range=(0,1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1,1))

    # Split into training and testing sets (60% training, 40% testing)
    training_size = int(len(closedf)*0.60)
    test_size = len(closedf)-training_size
    train_data, test_data = closedf[0:training_size,:], closedf[training_size:len(closedf),:1]
    print("train_data: ", train_data.shape)
    print("test_data: ", test_data.shape)

    # Create time series dataset with 15-day lookback
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape data for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train the model
    model, history = build_model(X_train, y_train, X_test, y_test)

    # Save the model and scaler
    model_dir = 'AI/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_save_path = os.path.join(model_dir, 'bitcoin_lstm_model.keras')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    import joblib
    scaler_save_path = os.path.join(model_dir, 'bitcoin_price_scaler.save')
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

    # Plot loss curves
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.savefig(os.path.join(model_dir, 'loss_curves.png'))

    # Evaluate the model on both training and test data
    train_rmse, train_mae, train_r2 = evaluate_model(model, X_train, y_train, scaler, "Training Data")
    test_rmse, test_mae, test_r2 = evaluate_model(model, X_test, y_test, scaler, "Testing Data")

    # Visualize actual vs predicted values on test data
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)

    # Create a DataFrame for visualization
    actual_vs_pred = pd.DataFrame({
        'Actual': scaler.inverse_transform(y_test.reshape(-1,1)).flatten(),
        'Predicted': test_predict.flatten()
    })

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(actual_vs_pred['Actual'], label='Actual Prices')
    plt.plot(actual_vs_pred['Predicted'], label='Predicted Prices')
    plt.title('Actual vs Predicted Bitcoin Prices (Test Data)')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, 'actual_vs_predicted.png'))

    return model, scaler, test_data, time_step, closedf, dates

def make_predictions(timeframes=[30, 180, 365, 1095]):
    """
    Make predictions for different timeframes
    timeframes: list of days to predict [1 month, 6 months, 1 year, 3 years]
    """
    import joblib

    # Check if model exists, if not train a new one
    model_path = 'AI/model/bitcoin_lstm_model.keras'
    scaler_path = 'AI/model/bitcoin_price_scaler.save'

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading existing model and scaler...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Load the latest data
        maindf = load_data()
        closedf = maindf[['Date','Price']]
        dates = closedf['Date']
        del closedf['Date']
        closedf = scaler.transform(np.array(closedf).reshape(-1,1))
        time_step = 15
    else:
        print("No model found. Training new model...")
        model, scaler, _, time_step, closedf, dates = train_and_save_model()

    # Make predictions for each timeframe
    predictions = {}
    for days in timeframes:
        print(f"Predicting for next {days} days...")
        preds = predict_future(model, scaler, closedf, time_step, days)

        # Create dates for predictions
        last_date = dates.iloc[-1]
        future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(preds))]

        # Store predictions with dates
        predictions[f"{days}_days"] = {
            "dates": future_dates,
            "predicted_prices": [round(price, 2) for price in preds]
        }

    return predictions

# Main execution
if __name__ == "__main__":
    print("Bitcoin Price Prediction Model")
    print("Training new model with all historical data...")

    # Train the model by default
    model, scaler, test_data, time_step, closedf, dates = train_and_save_model()

    # Make predictions for different timeframes
    timeframes = [30, 180, 365, 1095]  # 1 month, 6 months, 1 year, 3 years
    predictions = make_predictions(timeframes)

    # Print predictions
    for timeframe, data in predictions.items():
        print(f"\nPredictions for next {timeframe}:")
        for i, (date, price) in enumerate(zip(data['dates'], data['predicted_prices'])):
            if i < 5 or i > len(data['dates']) - 6:  # Show first and last 5 predictions
                print(f"{date}: ${price}")
            elif i == 5:
                print("...")