import React from 'react';
import { Divider } from 'primereact/divider';

interface FooterProps {
  sidebarCollapsed: boolean;
}

const Footer: React.FC<FooterProps> = ({ sidebarCollapsed }) => {
  return (
    <div
      className="footer mt-4"
      style={{
        marginLeft: sidebarCollapsed ? '80px' : '280px',
        transition: 'margin-left 0.3s ease',
        backgroundColor: '#1a1a1a',
        borderTop: '1px solid #404040',
        color: '#ffffff',
        padding: '20px'
      }}
    >
      <div className="grid">
        <div className="col-12 md:col-4">
          <h4 className="text-white mb-3">BTC Predict Dashboard</h4>
          <p className="text-400 text-sm line-height-3">
            Advanced cryptocurrency prediction platform powered by machine learning algorithms
            and real-time market data analysis.
          </p>
        </div>

        <div className="col-12 md:col-4">
          <h4 className="text-white mb-3">Quick Links</h4>
          <ul className="list-none p-0 m-0">
            <li className="mb-2">
              <a href="#" className="text-400 text-sm no-underline hover:text-orange-500">
                Terms of Service
              </a>
            </li>
            <li className="mb-2">
              <a href="#" className="text-400 text-sm no-underline hover:text-orange-500">
                Privacy Policy
              </a>
            </li>
            <li className="mb-2">
              <a href="#" className="text-400 text-sm no-underline hover:text-orange-500">
                Risk Disclaimer
              </a>
            </li>
            <li className="mb-2">
              <a href="#" className="text-400 text-sm no-underline hover:text-orange-500">
                Support Center
              </a>
            </li>
          </ul>
        </div>

        <div className="col-12 md:col-4">
          <h4 className="text-white mb-3">System Information</h4>
          <div className="text-400 text-sm">
            <p className="mb-2"><strong>Version:</strong> 1.0.0</p>
            <p className="mb-2"><strong>Last Updated:</strong> August 27, 2025</p>
            <p className="mb-2"><strong>Data Source:</strong> Historical CSV Files</p>
            <p className="mb-2"><strong>Prediction Model:</strong> VMD-LSTM-ELM</p>
          </div>
        </div>
      </div>

      <Divider />

      <div className="flex flex-column md:flex-row justify-content-between align-items-center">
        <div className="text-400 text-sm mb-2 md:mb-0">
          © 2025 BTC Predict Dashboard. All rights reserved. | Hackathon HNT Team
        </div>
        <div className="text-400 text-sm">
          <i className="pi pi-exclamation-triangle mr-2"></i>
          Not financial advice - For educational purposes only
        </div>
      </div>
    </div>
  );
};

export default Footer; 