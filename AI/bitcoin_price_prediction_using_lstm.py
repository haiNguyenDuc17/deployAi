# Bitcoin Price Prediction using Advanced Multivariate LSTM

import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
import joblib

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
    print("- Training loss curves: AI/model/training_loss_curves.png")
    print("- Model analysis: AI/model/advanced_lstm_analysis.png")
    print("- Price-volume analysis: AI/model/price_volume_analysis.png")
    print("\nThe Advanced Multivariate LSTM model is ready for use!")
    print("="*80)