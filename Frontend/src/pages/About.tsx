import React from 'react';
import { Card } from 'primereact/card';
import { Divider } from 'primereact/divider';

const About: React.FC = () => {
  return (
    <div className="about-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh', color: '#ffffff' }}>
      <div className="grid">
        <div className="col-12">
          <Card
            className="mb-4"
            style={{
              backgroundColor: '#2a2a2a',
              border: '1px solid #404040',
              borderRadius: '12px'
            }}
          >
            <div className="text-center mb-4">
              <h1 className="text-white m-0 mb-2">BTC Predict Dashboard</h1>
              <p className="text-500 text-lg">Advanced Bitcoin Price Prediction System</p>
            </div>

            <Divider />

            <div className="grid">
              <div className="col-12 md:col-6">
                <h3 className="text-white mb-3">System Overview</h3>
                <p className="text-300 line-height-3 mb-4">
                  The BTC Predict Dashboard is a comprehensive cryptocurrency analysis platform that leverages
                  advanced machine learning algorithms to provide Bitcoin price predictions. Our system combines
                  historical market data with sophisticated predictive models to offer insights into potential
                  future price movements.
                </p>

                <h3 className="text-white mb-3">Key Features</h3>
                <ul className="text-300 line-height-3 mb-4">
                  <li>Real-time cryptocurrency price tracking</li>
                  <li>Advanced Bitcoin price prediction using LSTM neural networks</li>
                  <li>Interactive charts with multiple timeframe options</li>
                  <li>Historical data analysis and trend visualization</li>
                  <li>Multi-cryptocurrency portfolio overview</li>
                  <li>Responsive design for all devices</li>
                </ul>
              </div>

              <div className="col-12 md:col-6">
                <h3 className="text-white mb-3">Prediction Methodology</h3>
                <p className="text-300 line-height-3 mb-4">
                  Our Bitcoin prediction system utilizes a hybrid VMD-LSTM-ELM (Variational Mode Decomposition -
                  Long Short-Term Memory - Extreme Learning Machine) model. This advanced approach combines:
                </p>
                <ul className="text-300 line-height-3 mb-4">
                  <li><strong>VMD:</strong> Signal decomposition for noise reduction</li>
                  <li><strong>LSTM:</strong> Deep learning for temporal pattern recognition</li>
                  <li><strong>ELM:</strong> Fast learning algorithm for improved accuracy</li>
                  <li><strong>Sentiment Analysis:</strong> Market sentiment integration</li>
                </ul>

                <h3 className="text-white mb-3">Data Sources</h3>
                <p className="text-300 line-height-3 mb-4">
                  Our system processes data from multiple reliable sources:
                </p>
                <ul className="text-300 line-height-3">
                  <li>Historical price data from major exchanges</li>
                  <li>Trading volume and market capitalization metrics</li>
                  <li>Technical indicators and market signals</li>
                  <li>Social media sentiment analysis</li>
                  <li>Economic indicators and news sentiment</li>
                </ul>
              </div>
            </div>

            <Divider />

            <div className="grid">
              <div className="col-12 md:col-6">
                <h3 className="text-white mb-3">Technology Stack</h3>
                <div className="grid">
                  <div className="col-6">
                    <h4 className="text-orange-500 mb-2">Frontend</h4>
                    <ul className="text-300 line-height-3">
                      <li>React 18</li>
                      <li>TypeScript</li>
                      <li>PrimeReact UI Library</li>
                      <li>ECharts for Data Visualization</li>
                      <li>Responsive CSS Grid</li>
                    </ul>
                  </div>
                  <div className="col-6">
                    <h4 className="text-orange-500 mb-2">Backend & AI</h4>
                    <ul className="text-300 line-height-3">
                      <li>Python Machine Learning</li>
                      <li>TensorFlow/Keras</li>
                      <li>LSTM Neural Networks</li>
                      <li>CSV Data Processing</li>
                      <li>Statistical Analysis</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="col-12 md:col-6">
                <h3 className="text-white mb-3">Development Team</h3>
                <p className="text-300 line-height-3 mb-4">
                  This project was developed as part of a hackathon initiative focused on creating innovative
                  financial technology solutions. The team combines expertise in machine learning, web development,
                  and financial analysis to deliver a comprehensive cryptocurrency prediction platform.
                </p>

                <h3 className="text-white mb-3">Project Information</h3>
                <div className="text-300 line-height-3">
                  <p><strong>Version:</strong> 1.0.0</p>
                  <p><strong>Last Updated:</strong> August 2025</p>
                  <p><strong>License:</strong> MIT License</p>
                  <p><strong>Repository:</strong> Hackathon HNT Team</p>
                </div>
              </div>
            </div>

            <Divider />

            <div className="text-center">
              <h3 className="text-white mb-3">Important Disclaimer</h3>
              <div
                className="p-4 border-round-lg mb-4"
                style={{
                  backgroundColor: '#2d1b1b',
                  border: '1px solid #ff6b6b'
                }}
              >
                <i className="pi pi-exclamation-triangle text-red-400 text-2xl mb-3"></i>
                <p className="text-300 line-height-3 m-0">
                  <strong>Financial Disclaimer:</strong> The predictions and analysis provided by this system are for
                  educational and informational purposes only. They should not be considered as financial advice or
                  investment recommendations. Cryptocurrency markets are highly volatile and unpredictable. Always
                  conduct your own research and consult with qualified financial advisors before making any investment
                  decisions. Past performance does not guarantee future results.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default About; 