# Bitcoin Price Prediction using Advanced Multivariate LSTM

import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
import joblib
import csv
import shutil
import json

# For evaluation and preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# For model building
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam

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

    # Remove any rows with NaN values
    maindf = maindf.dropna()

    # Check for any infinite values and replace with NaN, then drop
    maindf = maindf.replace([np.inf, -np.inf], np.nan).dropna()

    print('Total number of days present in the dataset: ', maindf.shape[0])
    print('Total number of fields present in the dataset: ', maindf.shape[1])

    # Check for any remaining NaN or infinite values
    print('NaN values in Price:', maindf['Price'].isna().sum())
    print('NaN values in Volume:', maindf['Volume'].isna().sum())
    print('Infinite values in Price:', np.isinf(maindf['Price']).sum())
    print('Infinite values in Volume:', np.isinf(maindf['Volume']).sum())

    return maindf

# Advanced Multivariate LSTM model architecture
def build_advanced_lstm_model(window_size, n_features=2):
    """Build advanced LSTM model with functional API supporting multivariate input"""
    input1 = Input(shape=(window_size, n_features))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    dnn_output = Dense(1)(x)
    model = Model(inputs=input1, outputs=[dnn_output])
    return model

def evaluate_model(model, X, y, scaler, name=""):
    """Evaluate model performance with various metrics"""
    predictions = model.predict(X)

    # Transform back to original form
    predictions = scaler.inverse_transform(predictions)
    original_y = scaler.inverse_transform(y.reshape(-1,1))

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(original_y, predictions))
    mae = mean_absolute_error(original_y, predictions)

    print(f"\n{name} Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    return rmse, mae

def export_model_performance_metrics(metrics_data, output_path):
    """
    Export model performance metrics to JSON file for frontend consumption

    Args:
        metrics_data: Dictionary containing all performance metrics
        output_path: Path where to save the JSON file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # Convert all numpy types in metrics_data
        serializable_metrics = convert_numpy_types(metrics_data)

        # Add metadata
        serializable_metrics['export_timestamp'] = dt.datetime.now().isoformat()
        serializable_metrics['model_version'] = 'Advanced_Multivariate_LSTM_v1.0'

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"✓ Model performance metrics exported to: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to export model performance metrics: {str(e)}")
        return False

def create_multivariate_sequences(price_data, volume_data, window_size):
    """Create sequences with both price and volume features"""
    X, y = [], []
    for i in range(window_size, len(price_data)):
        # Combine price and volume features
        price_seq = price_data[i-window_size:i, 0]
        volume_seq = volume_data[i-window_size:i, 0]

        # Check for NaN or infinite values
        if np.any(np.isnan(price_seq)) or np.any(np.isnan(volume_seq)) or \
           np.any(np.isinf(price_seq)) or np.any(np.isinf(volume_seq)) or \
           np.isnan(price_data[i, 0]) or np.isinf(price_data[i, 0]):
            continue

        # Stack features: [price_seq, volume_seq]
        feature_seq = np.column_stack([price_seq, volume_seq])
        X.append(feature_seq)
        y.append(price_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Final check for NaN values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print("Warning: NaN values found in sequences, removing them...")
        valid_indices = ~(np.any(np.isnan(X.reshape(X.shape[0], -1)), axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]

    return X, y

def predict_future_multivariate(model, price_scaler, volume_scaler, price_data, volume_data, window_size, days_to_predict):
    """
    Generate future price predictions using the trained multivariate LSTM model

    Args:
        model: Trained LSTM model
        price_scaler: Fitted MinMaxScaler for price data
        volume_scaler: Fitted MinMaxScaler for volume data
        price_data: Scaled price data array
        volume_data: Scaled volume data array
        window_size: Number of days to look back for prediction
        days_to_predict: Number of future days to predict

    Returns:
        predictions: Array of predicted prices in original scale
    """
    print(f"\n=== Generating Future Predictions for {days_to_predict} Days ===")

    # Get the last window_size days of data for initial input
    last_price_sequence = price_data[-window_size:].flatten()
    last_volume_sequence = volume_data[-window_size:].flatten()

    # Initialize input sequences
    temp_input_price = list(last_price_sequence)
    temp_input_volume = list(last_volume_sequence)

    # Calculate historical volatility for noise generation
    historical_data = price_scaler.inverse_transform(price_data[-30:])  # Use last 30 days
    daily_returns = np.diff(historical_data.flatten()) / historical_data[:-1].flatten()
    historical_volatility = np.std(daily_returns)

    # Volatility scale factor (can be adjusted)
    volatility_scale = 0.7

    lst_output = []
    n_steps = window_size
    i = 0

    print("Generating predictions with progress tracking...")
    print(f"Initial sequence lengths - Price: {len(temp_input_price)}, Volume: {len(temp_input_volume)}")

    while i < days_to_predict:
        if len(temp_input_price) > n_steps:
            # Use the last n_steps for prediction
            price_seq = np.array(temp_input_price[-n_steps:])
            volume_seq = np.array(temp_input_volume[-n_steps:])
        else:
            # Use all available data if less than n_steps
            price_seq = np.array(temp_input_price)
            volume_seq = np.array(temp_input_volume)

        # Ensure we have exactly n_steps
        if len(price_seq) != n_steps:
            print(f"❌ Error: Sequence length mismatch. Expected {n_steps}, got {len(price_seq)}")
            break

        # Combine features for multivariate input
        x_input = np.column_stack([price_seq, volume_seq])
        x_input = x_input.reshape((1, n_steps, 2))

        # Make prediction
        yhat = model.predict(x_input, verbose=0)
        pred_value = yhat[0][0]

        # Add controlled noise based on historical volatility
        scaled_value = pred_value
        noise = np.random.normal(0, historical_volatility * volatility_scale * abs(scaled_value), 1)[0]
        final_pred = scaled_value + noise

        # Update input sequences (remove oldest, add newest)
        temp_input_price.append(final_pred)
        temp_input_volume.append(temp_input_volume[-1])  # Use last volume value

        lst_output.append(final_pred)
        i += 1

        # Progress tracking
        if i % 100 == 0 or i == days_to_predict:
            print(f"Progress: {i}/{days_to_predict} predictions generated ({i/days_to_predict*100:.1f}%)")

    if len(lst_output) != days_to_predict:
        print(f"⚠️  Warning: Generated {len(lst_output)} predictions instead of {days_to_predict}")

    # Transform predictions back to original scale
    predictions = price_scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    print(f"Successfully generated {len(predictions)} predictions")
    print(f"Price range: ${min(predictions):.2f} - ${max(predictions):.2f}")

    return predictions

def generate_and_save_csv_predictions(model, price_scaler, volume_scaler, closedf, window_size):
    """
    Generate 1095-day predictions and save to CSV file

    Args:
        model: Trained LSTM model
        price_scaler: Fitted price scaler
        volume_scaler: Fitted volume scaler
        closedf: DataFrame with historical data
        window_size: Window size used for training
    """
    try:
        print("\n" + "="*80)
        print("GENERATING CSV PREDICTIONS")
        print("="*80)

        # Prepare data for prediction
        price = closedf['Price']
        volume = closedf['Volume']

        print(f"Data shapes before scaling:")
        print(f"  Price: {price.shape}")
        print(f"  Volume: {volume.shape}")

        # Scale the data using the fitted scalers
        price_scaled = price_scaler.transform(price.values.reshape(-1, 1))
        volume_scaled = volume_scaler.transform(volume.values.reshape(-1, 1))

        print(f"Data shapes after scaling:")
        print(f"  Price scaled: {price_scaled.shape}")
        print(f"  Volume scaled: {volume_scaled.shape}")
        print(f"  Window size: {window_size}")

        # Generate predictions for 1095 days (3 years)
        days_to_predict = 1095
        predictions = predict_future_multivariate(
            model, price_scaler, volume_scaler,
            price_scaled, volume_scaled,
            window_size, days_to_predict
        )

        # Generate future dates starting from the day after the last date in dataset
        last_date = closedf['Date'].max()
        future_dates = []

        print(f"\nGenerating dates starting from: {last_date.date()}")

        for i in range(1, days_to_predict + 1):
            future_date = last_date + pd.Timedelta(days=i)
            future_dates.append(future_date.strftime('%Y-%m-%d'))

        # Validate data integrity
        if len(future_dates) != len(predictions):
            raise ValueError(f"Date count ({len(future_dates)}) doesn't match prediction count ({len(predictions)})")

        if len(predictions) != days_to_predict:
            raise ValueError(f"Expected {days_to_predict} predictions, got {len(predictions)}")

        # Create CSV file path
        csv_path = 'Frontend/public/Data/bitcoin_predictions.csv'

        # Ensure Frontend/public/Data directory exists
        os.makedirs('Frontend/public/Data', exist_ok=True)

        # Write to CSV file
        print(f"\nSaving predictions to: {csv_path}")

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Date', 'Predicted_Price'])

            # Write data rows
            for date, price in zip(future_dates, predictions):
                writer.writerow([date, f"{price:.2f}"])

        # Validation: Read back and verify
        print("\nValidating CSV file...")
        validation_df = pd.read_csv(csv_path)

        if len(validation_df) != days_to_predict:
            raise ValueError(f"CSV validation failed: Expected {days_to_predict} rows, found {len(validation_df)}")

        # Check for any missing or invalid data
        if validation_df['Date'].isnull().any():
            raise ValueError("CSV validation failed: Found null dates")

        if validation_df['Predicted_Price'].isnull().any():
            raise ValueError("CSV validation failed: Found null prices")

        # Display summary statistics
        print(f"\n=== CSV Generation Summary ===")
        print(f"File saved: {csv_path}")
        print(f"Total predictions: {len(validation_df)}")
        print(f"Date range: {validation_df['Date'].iloc[0]} to {validation_df['Date'].iloc[-1]}")
        print(f"Price range: ${validation_df['Predicted_Price'].min():.2f} - ${validation_df['Predicted_Price'].max():.2f}")
        print(f"Average predicted price: ${validation_df['Predicted_Price'].mean():.2f}")

        print("\n✓ CSV file generated and validated successfully!")

        return csv_path

    except Exception as e:
        print(f"\n❌ Error generating CSV predictions: {str(e)}")
        print("Please check the model and data integrity.")
        raise

def copy_to_frontend_public(csv_path):
    """Copy the CSV file to the frontend public directory for web access"""
    try:
        frontend_public_dir = 'Frontend/public/Data'
        frontend_csv_path = os.path.join(frontend_public_dir, 'bitcoin_predictions.csv')

        # Create directory if it doesn't exist
        os.makedirs(frontend_public_dir, exist_ok=True)

        # Copy the file
        shutil.copy2(csv_path, frontend_csv_path)

        # Verify copy
        if os.path.exists(frontend_csv_path):
            print(f"✅ CSV copied to frontend: {frontend_csv_path}")
        else:
            print(f"⚠️  Warning: Failed to copy CSV to frontend directory")

    except Exception as e:
        print(f"⚠️  Warning: Could not copy CSV to frontend directory: {str(e)}")
        print("You can manually copy the file or run: python copy_csv_to_public.py")

def train_and_save_model():
    """Train Advanced Multivariate LSTM model with validation on 3 years, then full dataset training"""

    # Load the data
    maindf = load_data()

    # Prepare DataFrame with Price and Volume
    closedf = maindf[['Date', 'Price', 'Volume']].copy()
    closedf['Date'] = pd.to_datetime(closedf['Date'])
    closedf = closedf.sort_values(by='Date', ascending=True).reset_index(drop=True)
    closedf['Price'] = closedf['Price'].astype('float64')
    closedf['Volume'] = closedf['Volume'].astype('float64')
    dates = closedf['Date']

    # Prepare train/test split with multivariate features
    price = closedf['Price']
    volume = closedf['Volume']

    # Separate scalers for price and volume
    price_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()

    price_scaler.fit(price.values.reshape(-1,1))
    volume_scaler.fit(volume.values.reshape(-1,1))

    window_size = 60

    print(f"Total data points: {len(price)}")
    print(f"Window size: {window_size}")
    print(f"Features: Price + Volume (multivariate)")

    # Step 1: Test on latest 3 years of data for validation
    print("\n=== Step 1: Validation on Latest 3 Years ===")

    # Get latest 3 years for validation
    latest_date = closedf['Date'].max()
    three_years_ago = latest_date - pd.DateOffset(years=3)
    validation_data = closedf[closedf['Date'] >= three_years_ago].copy()

    print(f"Validation period: {three_years_ago.date()} to {latest_date.date()}")
    print(f"Validation data points: {len(validation_data)}")

    # Prepare validation data
    val_price = validation_data['Price']
    val_volume = validation_data['Volume']
    val_dates = validation_data['Date']

    # Scale validation data
    val_price_scaled = price_scaler.transform(val_price.values.reshape(-1,1))
    val_volume_scaled = volume_scaler.transform(val_volume.values.reshape(-1,1))

    # Split validation data (80% train, 20% test)
    val_split_idx = int(len(val_price_scaled) * 0.8)

    val_train_price = val_price_scaled[:val_split_idx]
    val_train_volume = val_volume_scaled[:val_split_idx]
    val_test_price = val_price_scaled[val_split_idx-window_size:]
    val_test_volume = val_volume_scaled[val_split_idx-window_size:]

    # Create sequences for validation
    X_val_train, y_val_train = create_multivariate_sequences(val_train_price, val_train_volume, window_size)
    X_val_test, y_val_test = create_multivariate_sequences(val_test_price, val_test_volume, window_size)

    y_val_train = np.reshape(y_val_train, (-1,1))
    y_val_test = np.reshape(y_val_test, (-1,1))

    print(f'Validation X_train Shape: {X_val_train.shape}')
    print(f'Validation y_train Shape: {y_val_train.shape}')
    print(f'Validation X_test Shape: {X_val_test.shape}')
    print(f'Validation y_test Shape: {y_val_test.shape}')

    # Train validation model
    print("\nTraining Advanced Multivariate LSTM on validation data...")

    # Add callbacks for better training
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

    # Build and compile validation model
    val_model = build_advanced_lstm_model(window_size, n_features=2)
    val_model.compile(loss='mean_squared_error', optimizer=Nadam())

    # Train validation model
    val_history = val_model.fit(
        X_val_train, y_val_train,
        validation_data=(X_val_test, y_val_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate validation model
    val_pred = val_model.predict(X_val_test, verbose=0)
    val_rmse = np.sqrt(mean_squared_error(y_val_test, val_pred))
    val_mae = mean_absolute_error(y_val_test, val_pred)

    print(f"\nValidation Results:")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAE: {val_mae:.4f}")

    # Transform back to original scale for better interpretation
    val_pred_original = price_scaler.inverse_transform(val_pred)
    val_test_original = price_scaler.inverse_transform(y_val_test)
    val_rmse_original = np.sqrt(mean_squared_error(val_test_original, val_pred_original))
    val_mae_original = mean_absolute_error(val_test_original, val_pred_original)

    print(f"Original Scale RMSE: ${val_rmse_original:.2f}")
    print(f"Original Scale MAE: ${val_mae_original:.2f}")

    # Step 2: Train final model on complete dataset
    print("\n=== Step 2: Training Final Model on Complete Dataset ===")

    # Prepare full dataset
    full_price_scaled = price_scaler.transform(price.values.reshape(-1,1))
    full_volume_scaled = volume_scaler.transform(volume.values.reshape(-1,1))

    # Create sequences for full dataset (use 80% for training, 20% for validation)
    split_idx = int(len(full_price_scaled) * 0.8)

    train_price_full = full_price_scaled[:split_idx]
    train_volume_full = full_volume_scaled[:split_idx]
    test_price_full = full_price_scaled[split_idx-window_size:]
    test_volume_full = full_volume_scaled[split_idx-window_size:]

    # Create sequences
    X_train_full, y_train_full = create_multivariate_sequences(train_price_full, train_volume_full, window_size)
    X_test_full, y_test_full = create_multivariate_sequences(test_price_full, test_volume_full, window_size)

    y_train_full = np.reshape(y_train_full, (-1,1))
    y_test_full = np.reshape(y_test_full, (-1,1))

    print(f'Full dataset X_train Shape: {X_train_full.shape}')
    print(f'Full dataset y_train Shape: {y_train_full.shape}')
    print(f'Full dataset X_test Shape: {X_test_full.shape}')
    print(f'Full dataset y_test Shape: {y_test_full.shape}')

    # Build and compile final model
    print("\nTraining Advanced Multivariate LSTM on complete dataset...")
    final_model = build_advanced_lstm_model(window_size, n_features=2)
    final_model.compile(loss='mean_squared_error', optimizer=Nadam())

    # Train final model
    final_history = final_model.fit(
        X_train_full, y_train_full,
        validation_data=(X_test_full, y_test_full),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate final model
    print("\n=== Final Model Evaluation ===")

    final_pred = final_model.predict(X_test_full, verbose=0)
    final_rmse = np.sqrt(mean_squared_error(y_test_full, final_pred))
    final_mae = mean_absolute_error(y_test_full, final_pred)

    print(f"Final Model RMSE: {final_rmse:.4f}")
    print(f"Final Model MAE: {final_mae:.4f}")

    # Transform back to original scale for better interpretation
    final_pred_original = price_scaler.inverse_transform(final_pred)
    final_test_original = price_scaler.inverse_transform(y_test_full)
    final_rmse_original = np.sqrt(mean_squared_error(final_test_original, final_pred_original))
    final_mae_original = mean_absolute_error(final_test_original, final_pred_original)

    print(f"Original Scale RMSE: ${final_rmse_original:.2f}")
    print(f"Original Scale MAE: ${final_mae_original:.2f}")

    # Use final model and data for saving and visualization
    model = final_model
    history = final_history
    final_X_test = X_test_full
    final_y_test = y_test_full
    final_pred = final_pred

    # Collect comprehensive performance metrics for export
    performance_metrics = {
        'validation_metrics': {
            'rmse_scaled': float(val_rmse),
            'mae_scaled': float(val_mae),
            'rmse_original': float(val_rmse_original),
            'mae_original': float(val_mae_original),
            'data_period': '3_years_validation'
        },
        'final_model_metrics': {
            'rmse_scaled': float(final_rmse),
            'mae_scaled': float(final_mae),
            'rmse_original': float(final_rmse_original),
            'mae_original': float(final_mae_original),
            'data_period': 'complete_dataset'
        },
        'training_history': {
            'validation_loss': history.history['loss'],
            'validation_val_loss': history.history['val_loss'],
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_loss': float(min(history.history['loss'])),
            'best_val_loss': float(min(history.history['val_loss']))
        },
        'model_architecture': {
            'window_size': window_size,
            'features': ['price', 'volume'],
            'lstm_layers': 3,
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'optimizer': 'Nadam',
            'loss_function': 'mean_squared_error'
        },
        'dataset_info': {
            'total_samples': len(closedf),
            'training_samples': len(X_train_full),
            'test_samples': len(X_test_full),
            'train_test_split': '80/20',
            'date_range': {
                'start_date': dates.iloc[0].strftime('%Y-%m-%d'),
                'end_date': dates.iloc[-1].strftime('%Y-%m-%d')
            }
        }
    }

    # Save the model and scalers
    model_dir = 'AI/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_save_path = os.path.join(model_dir, 'bitcoin_advanced_multivariate_lstm.keras')
    model.save(model_save_path)
    print(f"Advanced Multivariate LSTM model saved to {model_save_path}")

    # Save scalers
    price_scaler_path = os.path.join(model_dir, 'bitcoin_price_scaler.save')
    joblib.dump(price_scaler, price_scaler_path)
    print(f"Price scaler saved to {price_scaler_path}")

    volume_scaler_path = os.path.join(model_dir, 'bitcoin_volume_scaler.save')
    joblib.dump(volume_scaler, volume_scaler_path)
    print(f"Volume scaler saved to {volume_scaler_path}")

    # Plot loss curves
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Advanced Multivariate LSTM - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, 'training_loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Final evaluation with enhanced metrics
    test_loss = model.evaluate(final_X_test, final_y_test, verbose=0)

    # Use the final prediction
    y_pred = final_pred

    MAPE = mean_absolute_percentage_error(final_y_test, y_pred)

    # Calculate R-squared for both scaled and original values
    from sklearn.metrics import r2_score
    r2_scaled = r2_score(final_y_test, y_pred)

    # Calculate R-squared for original scale values
    y_test_original_r2 = price_scaler.inverse_transform(final_y_test.reshape(-1,1)).flatten()
    y_pred_original_r2 = price_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    r2_original = r2_score(y_test_original_r2, y_pred_original_r2)

    # Add additional metrics to performance_metrics
    performance_metrics['final_model_metrics'].update({
        'mape': float(MAPE * 100),  # Convert to percentage
        'r2_scaled': float(r2_scaled),
        'r2_original': float(r2_original),
        'test_loss': float(test_loss)
    })

    # Calculate accuracy as percentage (100 - MAPE)
    accuracy_percentage = max(0, 100 - (MAPE * 100))
    performance_metrics['final_model_metrics']['accuracy_percentage'] = float(accuracy_percentage)
    Accuracy = 1 - MAPE

    print(f"\n=== Final Model Performance Summary ===")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAPE: {MAPE:.4f}")
    print(f"Test Accuracy: {Accuracy:.4f}")
    print(f"Final RMSE (scaled): {final_rmse:.4f}")
    print(f"Final RMSE (original): ${final_rmse_original:.2f}")
    print(f"Final MAE (original): ${final_mae_original:.2f}")

    # Enhanced visualization
    y_test_true = price_scaler.inverse_transform(final_y_test)
    y_test_pred = price_scaler.inverse_transform(y_pred)

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Plot 1: Price prediction comparison
    split_point = int(len(price) * 0.8)
    train_dates = dates.iloc[:split_point]
    test_dates = dates.iloc[split_point:]

    train_price_original = price_scaler.inverse_transform(full_price_scaled[:split_point])
    ax1.plot(train_dates, train_price_original, color='black', lw=1, label='Training Data', alpha=0.7)
    ax1.plot(test_dates[-len(y_test_true):], y_test_true, color='blue', lw=2, label='Actual Test Data')
    ax1.plot(test_dates[-len(y_test_pred):], y_test_pred, color='red', lw=2, label='Predicted Test Data')
    ax1.set_title('Advanced Multivariate LSTM - Bitcoin Price Prediction', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss curves
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction accuracy scatter
    ax3.scatter(y_test_true, y_test_pred, alpha=0.6, color='green')
    ax3.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Prices ($)')
    ax3.set_ylabel('Predicted Prices ($)')
    ax3.set_title('Actual vs Predicted Prices')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals
    residuals = y_test_true.flatten() - y_test_pred.flatten()
    ax4.scatter(range(len(residuals)), residuals, alpha=0.6, color='purple')
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Residuals ($)')
    ax4.set_title('Prediction Residuals')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'advanced_lstm_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Create volume analysis plot
    plt.figure(figsize=(15, 8))

    # Plot price
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(test_dates[-len(y_test_true):], y_test_true, color='blue', lw=2, label='Actual Price')
    plt.plot(test_dates[-len(y_test_pred):], y_test_pred, color='red', lw=2, label='Predicted Price')
    plt.title('Advanced Multivariate LSTM - Price Prediction with Volume Analysis')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot volume
    ax2 = plt.subplot(2, 1, 2)
    test_volume_original = volume_scaler.inverse_transform(full_volume_scaled[split_point:][-len(y_test_true):])
    plt.plot(test_dates[-len(y_test_true):], test_volume_original, color='orange', lw=2, label='Volume')
    plt.ylabel('Volume')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'price_volume_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization plots saved:")
    print(f"- Training loss curves: {os.path.join(model_dir, 'training_loss_curves.png')}")
    print(f"- Model analysis: {os.path.join(model_dir, 'advanced_lstm_analysis.png')}")
    print(f"- Price-volume analysis: {os.path.join(model_dir, 'price_volume_analysis.png')}")

    # Export performance metrics to JSON files
    print(f"\n=== Exporting Model Performance Metrics ===")

    # Export to AI/model directory
    ai_metrics_path = os.path.join(model_dir, 'model_performance_metrics.json')
    export_model_performance_metrics(performance_metrics, ai_metrics_path)

    # Also export to Frontend/public/Data for easy access by React app
    frontend_data_dir = os.path.join('Frontend', 'public', 'Data')
    frontend_metrics_path = os.path.join(frontend_data_dir, 'model_performance_metrics.json')

    try:
        export_model_performance_metrics(performance_metrics, frontend_metrics_path)
        print(f"✓ Performance metrics also copied to frontend: {frontend_metrics_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not copy metrics to frontend directory: {str(e)}")
        print("You can manually copy the file from AI/model/ to Frontend/public/Data/")

    return model, price_scaler, volume_scaler, window_size, closedf, dates

def print_model_summary():
    """Print summary of the Advanced Multivariate LSTM model"""
    print("\n" + "="*80)
    print("ADVANCED MULTIVARIATE LSTM BITCOIN PRICE PREDICTION MODEL")
    print("="*80)
    print("\nModel Features:")
    print("1. ✓ Advanced LSTM architecture with functional API")
    print("2. ✓ Multivariate input: Price + Volume data")
    print("3. ✓ 3-layer LSTM with 64 units each")
    print("4. ✓ Dropout regularization (0.2) for overfitting prevention")
    print("5. ✓ Dense layer with ReLU activation")
    print("6. ✓ Nadam optimizer for better convergence")
    print("7. ✓ Early stopping and learning rate reduction")
    print("8. ✓ Comprehensive evaluation and visualization")

    print("\nTraining Strategy:")
    print("- Step 1: Validation on latest 3 years of data")
    print("- Step 2: Final training on complete historical dataset")
    print("- Window size: 60 days")
    print("- Features: Price + Volume (normalized)")
    print("- Train/Test split: 80/20")

    print("\nModel Architecture:")
    print("- Input: (window_size, 2) - Price and Volume sequences")
    print("- LSTM Layer 1: 64 units, return_sequences=True")
    print("- Dropout: 0.2")
    print("- LSTM Layer 2: 64 units, return_sequences=True")
    print("- Dropout: 0.2")
    print("- LSTM Layer 3: 64 units")
    print("- Dropout: 0.2")
    print("- Dense: 32 units, ReLU activation")
    print("- Output: 1 unit (price prediction)")
    print("="*80)

# Main execution
if __name__ == "__main__":
    print_model_summary()
    print("\nStarting Advanced Multivariate LSTM training process...")

    # Train the model
    model, price_scaler, volume_scaler, window_size, closedf, dates = train_and_save_model()

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nModel files saved:")
    print("- Advanced LSTM model: AI/model/bitcoin_advanced_multivariate_lstm.keras")
    print("- Price scaler: AI/model/bitcoin_price_scaler.save")
    print("- Volume scaler: AI/model/bitcoin_volume_scaler.save")
    print("- Performance metrics: AI/model/model_performance_metrics.json")
    print("- Performance metrics (Frontend): Frontend/public/Data/model_performance_metrics.json")
    print("- Training loss curves: AI/model/training_loss_curves.png")
    print("- Model analysis: AI/model/advanced_lstm_analysis.png")
    print("- Price-volume analysis: AI/model/price_volume_analysis.png")

    # Generate CSV predictions
    try:
        csv_path = generate_and_save_csv_predictions(model, price_scaler, volume_scaler, closedf, window_size)
        print(f"\n✓ Predictions CSV generated: {csv_path}")
    except Exception as e:
        print(f"\n❌ Failed to generate CSV predictions: {str(e)}")
        print("Model training completed, but CSV generation failed.")

    print("\n" + "="*80)
    print("PROCESS COMPLETED!")
    print("="*80)
    print("\nFiles generated:")
    print("1. Model files in AI/model/")
    print("2. Predictions CSV: Frontend/public/Data/bitcoin_predictions.csv")
    print("\nThe system is ready for frontend use!")
    print("Run the frontend with: cd Frontend && npm start")
    print("="*80)