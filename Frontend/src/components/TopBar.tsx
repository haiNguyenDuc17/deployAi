import React from 'react';
import { Button } from 'primereact/button';

interface TopBarProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

const TopBar: React.FC<TopBarProps> = ({ sidebarCollapsed, onToggleSidebar }) => {
  return (
    <div
      className="topbar flex align-items-center justify-content-between px-4 py-3 fixed top-0 right-0 z-4"
      style={{
        backgroundColor: '#1a1a1a',
        borderBottom: '1px solid #404040',
        color: '#ffffff',
        left: sidebarCollapsed ? '80px' : '280px',
        height: '70px',
        transition: 'left 0.3s ease'
      }}
    >
      {/* Left Side - Sidebar Toggle */}
      <div className="flex align-items-center gap-4">
        <Button
          icon="pi pi-bars"
          className="p-button-text p-button-rounded"
          style={{ color: '#ffffff' }}
          onClick={onToggleSidebar}
        />
        <span className="text-xl font-bold text-white">BTC Predict Dashboard</span>
      </div>

      {/* Right Side - Buy/Sell Button */}
      <div className="flex align-items-center gap-3">
        <Button
          label="Buy / Sell"
          className="p-button-warning p-button-sm"
          style={{
            backgroundColor: '#ff8c00',
            border: 'none',
            borderRadius: '6px'
          }}
          onClick={() => window.open('https://www.binance.com/', '_blank')}
        />
      </div>
    </div>
  );
};

export default TopBar;
