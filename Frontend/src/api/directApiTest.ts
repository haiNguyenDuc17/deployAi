import axios from 'axios';

// This function will be called from the developer console
// to test various API configurations directly
async function testApiEndpoints() {
  console.log('======= DIRECT API TESTING =======');
  console.log('Testing various endpoints to find the correct one');
  
  // List of possible base URLs
  const baseUrls = [
    'http://localhost:5000',
    'http://127.0.0.1:5000', 
    'http://localhost:8000',   // Common alternative port
    window.location.origin     // Try same origin
  ];
  
  // List of possible endpoint patterns
  const endpointPatterns = [
    '/predict/30',
    '/api/predict/30',
    '/prediction/30',
    '/predict?days=30',
    '/forecast/30'
  ];
  
  // Try all combinations
  for (const baseUrl of baseUrls) {
    console.log(`\nTesting base URL: ${baseUrl}`);
    
    // First try a basic check for server availability
    try {
      console.log(`  Checking if server is reachable at ${baseUrl}`);
      const response = await axios.get(baseUrl, { timeout: 5000 });
      console.log(`  ✅ Server reachable: Status ${response.status}`);
    } catch (error: any) {
      console.log(`  ❌ Cannot reach server: ${error.message}`);
      continue; // Skip further tests for this base URL
    }
    
    // Now try each endpoint pattern
    for (const endpoint of endpointPatterns) {
      const fullUrl = `${baseUrl}${endpoint}`;
      console.log(`  Testing endpoint: ${fullUrl}`);
      
      try {
        const response = await axios.get(fullUrl, { timeout: 5000 });
        console.log(`  ✅ SUCCESS with ${fullUrl}`);
        console.log(`  Response:`, response.data);
        
        // Check if response has expected format
        const key = Object.keys(response.data)[0];
        if (key && response.data[key] && 
            Array.isArray(response.data[key].dates) && 
            Array.isArray(response.data[key].predicted_prices)) {
          console.log(`  ✅ VALID RESPONSE FORMAT: This endpoint works!`);
          console.log(`  Add this to your notes: Working endpoint is ${fullUrl}`);
        } else {
          console.log(`  ❌ Incorrect data format`);
        }
      } catch (error: any) {
        console.log(`  ❌ Failed: ${error.message}`);
      }
    }
  }
  
  console.log('\n======= TESTING COMPLETE =======');
  console.log('If all tests failed, please check:');
  console.log('1. Is your Flask server running?');
  console.log('2. Is your Flask server exposing the API at a different URL?');
  console.log('3. Do you need to enable CORS in your Flask server?');
}

// Export for use in the browser console
// @ts-ignore
window.testApiEndpoints = testApiEndpoints;

export { testApiEndpoints }; 