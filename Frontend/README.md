# Bitcoin Price Prediction Web App

A dark-themed, professional web application that visualizes Bitcoin price predictions, similar in look and feel to Binance or other leading crypto platforms.

## Features

- Interactive price prediction chart using ECharts
- Time-frame selection: 1 month, 6 months, 1 year, and 3 years
- Real-time data fetching from a Flask API
- Responsive design for desktop, tablet, and mobile
- Dark mode UI with Bitcoin-themed styling

## Tech Stack

- React with TypeScript
- ECharts for data visualization
- Styled Components for styling
- Axios for API requests

## API Integration

The application connects to a local Flask API:

```
GET http://localhost:5000/predict/<days>
```

Example: `http://localhost:5000/predict/30`

Expected JSON structure:

```json
{
  "30_days": {
    "dates": ["2023-05-01", ...],
    "predicted_prices": [67250.85, ...]
  }
}
```

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm start
   ```
4. Ensure the Flask backend is running at `http://localhost:5000`

## Fallback Mechanism

If the Flask API is unavailable, the app will automatically switch to a simulated data mode for demonstration purposes. This ensures the app remains functional even when the backend is down.

## License

This project is part of a hackathon and is for demonstration purposes. 