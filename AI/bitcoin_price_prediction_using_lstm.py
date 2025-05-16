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

    # Prepare DataFrame similar to gold LSTM code
    closedf = maindf[['Date', 'Price']].copy()
    closedf['Date'] = pd.to_datetime(closedf['Date'])
    closedf = closedf.sort_values(by='Date', ascending=True).reset_index(drop=True)
    closedf['Price'] = closedf['Price'].astype('float64')
    dates = closedf['Date']

    # Test set: last year
    test_year = closedf['Date'].dt.year.max() - 1
    test_size = closedf[closedf['Date'].dt.year == test_year].shape[0]
    
    # Prepare train/test split
    price = closedf['Price']
    scaler = MinMaxScaler()
    scaler.fit(price.values.reshape(-1,1))
    window_size = 60

    # Training data
    train_data = price[:-test_size]
    train_data = scaler.transform(train_data.values.reshape(-1,1))
    X_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])

    # Test data
    test_data = price[-test_size-window_size:]
    test_data = scaler.transform(test_data.values.reshape(-1,1))
    X_test, y_test = [], []
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
        y_test.append(test_data[i, 0])

    # Convert to numpy arrays and reshape
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1,1))
    y_test = np.reshape(y_test, (-1,1))

    print('X_train Shape: ', X_train.shape)
    print('y_train Shape: ', y_train.shape)
    print('X_test Shape:  ', X_test.shape)
    print('y_test Shape:  ', y_test.shape)

    # Gold LSTM model architecture
    from keras import Model
    from keras.layers import Input, Dense, Dropout, LSTM
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = 64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)
    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()

    # Train model
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

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
    from sklearn.metrics import mean_absolute_percentage_error
    test_loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    Accuracy = 1 - MAPE
    print("Test Loss:", test_loss)
    print("Test MAPE:", MAPE)
    print("Test Accuracy:", Accuracy)

    # Visualize actual vs predicted values on test data
    y_test_true = scaler.inverse_transform(y_test)
    y_test_pred = scaler.inverse_transform(y_pred)
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'yellow'
    plt.rc('axes',edgecolor='white')
    plt.plot(dates.iloc[:-test_size], scaler.inverse_transform(train_data), color='black', lw=2)
    plt.plot(dates.iloc[-test_size:], y_test_true, color='blue', lw=2)
    plt.plot(dates.iloc[-test_size:], y_test_pred, color='red', lw=2)
    plt.title('Model Performance on Bitcoin Price Prediction', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    plt.savefig(os.path.join(model_dir, 'actual_vs_predicted.png'))

    return model, scaler, test_data, window_size, closedf, dates

# Main execution
if __name__ == "__main__":
    print("Bitcoin Price Prediction Model")
    print("Training new model with all historical data...")

    # Train the model
    train_and_save_model()