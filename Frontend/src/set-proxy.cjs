const { ProxyAgent } = require('proxy-agent');
const http = require('http');
const https = require('https');

// Configure via environment or fallback default
const proxyUrl = process.env.DEV_PROXY_URL || 'http://127.0.0.1:3128';

// Export proxy to libraries that read env vars (http-proxy, got, axios when not overridden)
process.env.HTTP_PROXY = process.env.HTTP_PROXY || proxyUrl;
process.env.HTTPS_PROXY = process.env.HTTPS_PROXY || proxyUrl;
process.env.NO_PROXY = process.env.NO_PROXY || 'localhost,127.0.0.1';

// Optional: allow self-signed corporate CAs when TRUST_PROXY_CERT=1
if (process.env.TRUST_PROXY_CERT === '1') {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
}

// Create and bind a global agent so Node core http(s) uses the proxy
const agent = new ProxyAgent(proxyUrl);
http.globalAgent = agent;
https.globalAgent = agent;

console.log(`✅ Using proxy agent for: ${proxyUrl}`);
console.log(`   HTTP_PROXY=${process.env.HTTP_PROXY}`);
console.log(`   HTTPS_PROXY=${process.env.HTTPS_PROXY}`);

// Test the proxy connection with a simple HTTP request
// console.log('Testing proxy connection...');
// https.get('https://httpbin.org/ip', (res) => {
//   let data = '';
//   res.on('data', (chunk) => {
//     data += chunk;
//   });
//   res.on('end', () => {
//     console.log('✅ Proxy test successful:', data);
//   });
// }).on('error', (err) => {
//   console.error('❌ Proxy test failed:', err.message);
// });
