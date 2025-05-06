"""
Simple Flask server that mimics the expected Bitcoin prediction API
Use this to test if your React frontend can connect to a Flask server

To run:
1. Install dependencies: pip install flask flask-cors
2. Run this script: python flask_test_server.py
3. Access http://localhost:5000/predict/30 in your browser to test
"""

from flask import Flask, jsonify
from flask_cors import CORS
import datetime
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict/<int:days>')
def predict(days):
    """Generate mock Bitcoin price prediction data for testing"""
    print(f"Received request for {days} days prediction")
    
    # Current date
    current_date = datetime.datetime.now()
    
    # Generate mock data
    dates = []
    predicted_prices = []
    price = 63000  # Starting price around current BTC value
    
    for i in range(days):
        # Generate date
        date = current_date + datetime.timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Generate random price with slight trend
        change = (random.random() - 0.5) * 0.05  # -2.5% to +2.5%
        price = price * (1 + change)
        predicted_prices.append(round(price, 2))
    
    # Prepare response in the expected format
    response = {
        f"{days}_days": {
            "dates": dates,
            "predicted_prices": predicted_prices
        }
    }
    
    print(f"Sending response with {len(dates)} dates and prices")
    return jsonify(response)

@app.route('/')
def index():
    """Simple index route to verify server is running"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(f"{rule.endpoint}: {rule}")
    
    return jsonify({
        "message": "Bitcoin Price Prediction Test API",
        "available_routes": routes,
        "example": "Try /predict/30 to get 30 days of predictions"
    })

if __name__ == '__main__':
    print("\n=== BITCOIN PREDICTION TEST SERVER ===")
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule}")
    
    print("\nServer running at http://localhost:5000")
    print("Try accessing these URLs in your browser:")
    print("  http://localhost:5000/ - API info")
    print("  http://localhost:5000/predict/30 - 30 days prediction")
    
    app.run(debug=True) 