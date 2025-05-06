import axios from 'axios';
import { PredictionResponse } from '../types';
import { testApiEndpoints } from './directApiTest';

// Configure axios with proxy settings
const api = axios.create({
  baseURL: 'http://localhost:5000',
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
  },
  // Add timeout to avoid long waiting
  timeout: 10000,
  // Proxy will be automatically picked up from package.json proxy setting
});

// Make test function available
// @ts-ignore
window.testApiEndpoints = testApiEndpoints;

// UPDATE THIS BASED ON YOUR TESTING RESULTS
const API_CONFIG = {
  // Set to true to bypass React's proxy and use direct API calls
  useDirectApi: false,
  // Base URL for direct API calls (only used if useDirectApi is true)
  baseUrl: 'http://localhost:5000',
  // Path pattern to use for endpoint
  endpointPattern: '/predict/{days}' // will be replaced with actual days
};

export const fetchPrediction = async (days: number): Promise<PredictionResponse> => {
  try {
    console.log(`Fetching predictions for ${days} days...`);
    
    // Construct the endpoint based on the pattern
    const endpoint = API_CONFIG.endpointPattern.replace('{days}', days.toString());
    console.log('Using endpoint pattern:', API_CONFIG.endpointPattern);
    console.log('Constructed endpoint:', endpoint);
    
    let response;
    
    if (API_CONFIG.useDirectApi) {
      // Use direct API calls
      const fullUrl = `${API_CONFIG.baseUrl}${endpoint}`;
      console.log('Making direct API call to:', fullUrl);
      response = await axios.get(fullUrl);
    } else {
      // Use React's proxy
      console.log('Using React proxy with endpoint:', endpoint);
      response = await axios.get(endpoint);
    }
    
    console.log('Received data:', response.data);
    return response.data;
  } catch (error: any) {
    console.error('Error fetching prediction data:', error);
    
    if (error.response) {
      // The request was made and the server responded with a status code outside of 2xx range
      console.error('Server responded with:', error.response.status, error.response.data);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request);
      console.log('The Flask server might not be running or accessible.');
      console.log('');
      console.log('=== DIAGNOSIS ===');
      console.log('1. Open Chrome DevTools (F12)');
      console.log('2. Go to Console tab');
      console.log('3. Run this command to test all possible endpoints:');
      console.log('   testApiEndpoints()');
      console.log('4. Check the results to find a working endpoint');
      console.log('5. Update API_CONFIG in src/api/predictionApi.ts with the working endpoint');
    } else {
      // Something happened in setting up the request that triggered an error
      console.error('Request error:', error.message);
    }
    
    throw new Error(`Network error: ${error.message}. Use the browser console and run testApiEndpoints() to diagnose.`);
  }
}; 