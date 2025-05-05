"""
Example usage of the Bitcoin Price Prediction API

This script demonstrates how to:
1. Fetch recent Bitcoin price data
2. Send it to the prediction API
3. Receive and process the predictions

You can use this as a template for your frontend integration.
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# API URLs
API_URL = "http://localhost:5000/api/predict"
STATUS_URL = "http://localhost:5000/api/status"

# Check if the API is running and model is loaded
def check_api_status():
    try:
        response = requests.get(STATUS_URL)
        status = response.json()
        return status.get('ready', False)
    except Exception as e:
        print(f"Error checking API status: {e}")
        return False

# Function to get recent Bitcoin price data (simulated here)
def get_recent_bitcoin_data(days=30):
    """
    In a real application, you would fetch real Bitcoin data here.
    For this example, we'll generate sample data.
    
    Returns:
        List of recent Bitcoin prices (at least 15 days required)
    """
    # For demo purposes, let's create some sample price data
    # In production, you'd fetch this from a data source like CoinGecko or similar
    
    # Example: Creating a simple upward trend with some noise
    base_price = 65000  # Starting price (example)
    trend = np.linspace(0, 5000, days)  # Upward trend
    noise = np.random.normal(0, 1000, days)  # Random noise
    
    prices = base_price + trend + noise
    
    # Generate dates for reference (not needed for API call)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    print(f"Generated sample data for {days} days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Price range: ${min(prices):.2f} to ${max(prices):.2f}")
    
    return prices.tolist()

# Function to send data to API and get predictions
def get_predictions(prices):
    """
    Send recent price data to the API and get predictions for different timeframes
    
    Args:
        prices: List of recent Bitcoin prices (at least 15 days)
        
    Returns:
        Dictionary containing predictions for different timeframes
    """
    try:
        # Prepare request data
        request_data = {
            'prices': prices
        }
        
        # Send POST request to API
        response = requests.post(API_URL, json=request_data)
        
        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Error calling prediction API: {e}")
        return None

# Function to visualize predictions
def plot_predictions(predictions, timeframe='1month'):
    """
    Plot the predictions for a specific timeframe
    
    Args:
        predictions: Prediction data from API
        timeframe: Which timeframe to plot ('1month', '6months', '1year', '3years')
    """
    if not predictions or 'predictions' not in predictions or timeframe not in predictions['predictions']:
        print(f"No prediction data available for {timeframe}")
        return
    
    # Get prediction data for the specified timeframe
    pred_data = predictions['predictions'][timeframe]
    
    # Extract dates and prices
    dates = pred_data['xAxis']
    prices = pred_data['series'][0]['data']
    
    # Convert dates to datetime objects for plotting
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, 'b-', linewidth=2)
    plt.title(f'Bitcoin Price Prediction - {timeframe}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Main function to demonstrate API usage
def main():
    # Check if API is ready
    if not check_api_status():
        print("API is not ready. Please make sure the server is running and the model is loaded.")
        return
    
    # Get recent Bitcoin price data (minimum 15 days required)
    recent_prices = get_recent_bitcoin_data(days=30)
    
    # Get predictions from API
    predictions = get_predictions(recent_prices)
    
    if predictions and predictions.get('success'):
        print("Successfully received predictions!")
        
        # Available timeframes
        timeframes = ['1month', '6months', '1year', '3years']
        
        # Plot predictions for each timeframe
        for timeframe in timeframes:
            plot_predictions(predictions, timeframe)
            
        # Example: Save the prediction data for frontend use
        with open('predictions_for_frontend.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        print("Saved predictions to 'predictions_for_frontend.json'")
        
    else:
        print("Failed to get predictions")

if __name__ == "__main__":
    main() 