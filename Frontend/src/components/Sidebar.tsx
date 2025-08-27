import React from 'react';
import { Button } from 'primereact/button';
import { useNavigate, useLocation } from 'react-router-dom';

interface SidebarProps {
  collapsed: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed }) => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <div
      className="sidebar h-screen fixed left-0 top-0 z-5"
      style={{
        width: collapsed ? '80px' : '280px',
        backgroundColor: '#1a1a1a',
        borderRight: '1px solid #404040',
        color: '#ffffff',
        transition: 'width 0.3s ease',
        overflow: 'hidden'
      }}
    >
      {/* Logo Section */}
      <div className="p-4 border-bottom-1 border-400">
        <div className="flex align-items-center gap-2">
          <img
            src="/assets/Logo.png"
            alt="Logo"
            style={{ width: '32px', height: '32px' }}
          />
          {!collapsed && <span className="text-xl font-bold text-white">Dashboard</span>}
        </div>
      </div>

      {/* Navigation Menu */}
      <div className="p-3">
        <div className="mb-4">
          <Button
            icon="pi pi-home"
            label={collapsed ? "" : "Dashboard"}
            className={`w-full ${collapsed ? 'justify-content-center' : 'justify-content-start'} p-button-text text-white`}
            style={{
              backgroundColor: location.pathname === '/' ? '#ff8c00' : 'transparent',
              border: 'none',
              borderRadius: '8px',
              padding: '12px 16px'
            }}
            onClick={() => navigate('/')}
          />
        </div>

        <div className="mb-4">
          <Button
            icon="pi pi-user"
            label={collapsed ? "" : "User"}
            className={`w-full ${collapsed ? 'justify-content-center' : 'justify-content-start'} p-button-text`}
            style={{
              color: '#ffffff',
              backgroundColor: 'transparent',
              border: 'none',
              padding: '12px 16px'
            }}
          />
        </div>

        <div className="mb-4">
          <Button
            icon="pi pi-info-circle"
            label={collapsed ? "" : "About"}
            className={`w-full ${collapsed ? 'justify-content-center' : 'justify-content-start'} p-button-text`}
            style={{
              color: '#ffffff',
              backgroundColor: location.pathname === '/about' ? '#ff8c00' : 'transparent',
              border: 'none',
              padding: '12px 16px'
            }}
            onClick={() => navigate('/about')}
          />
        </div>
      </div>

      {/* Pro Features Section */}
      {!collapsed && (
        <div className="absolute bottom-0 left-0 right-0 p-3">
          <div
            className="p-3 text-center border-round-lg"
            style={{
              backgroundColor: '#2a2a2a',
              border: '1px solid #404040'
            }}
          >
            <div className="mb-3">
              <i className="pi pi-shield text-4xl text-orange-500"></i>
            </div>
            <h4 className="text-white m-0 mb-2">Be more secure</h4>
            <h4 className="text-white m-0 mb-3">with Pro Features</h4>
            <Button
              label="Upgrade Now"
              className="w-full p-button-warning p-button-sm"
              style={{
                backgroundColor: '#ff8c00',
                border: 'none',
                borderRadius: '6px'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
