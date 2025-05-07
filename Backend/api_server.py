from flask import Flask, jsonify
import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to sys.path to import from the bitcoin_price_prediction_using_lstm module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI.bitcoin_price_prediction_using_lstm import make_predictions

# Create Flask API
app = Flask(__name__)

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
