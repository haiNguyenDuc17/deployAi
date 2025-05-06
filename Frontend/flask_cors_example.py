"""
Example of how to enable CORS in your Flask API to allow requests from your React frontend

To fix the CORS issue:
1. Install flask-cors: pip install flask-cors
2. Then update your Flask app as shown below
"""

from flask import Flask, jsonify
from flask_cors import CORS  # Import the CORS extension

# Create Flask app
app = Flask(__name__)

# Enable CORS for all routes and origins (you can restrict this if needed)
CORS(app)

# Example route that returns Bitcoin price predictions
@app.route('/predict/<int:days>')
def predict(days):
    # Your prediction logic here
    # ...
    
    # Example response format
    response = {
        f"{days}_days": {
            "dates": ["2023-05-01", "2023-05-02"],  # Your actual dates
            "predicted_prices": [67250.85, 67500.50]  # Your actual predictions
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

"""
Alternatively, if you want more fine-grained control, you can add CORS headers manually:

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    return response
""" 