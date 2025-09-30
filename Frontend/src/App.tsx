import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Footer from './components/Footer';
import Dashboard from './components/Dashboard';
import TradingSignals from './components/TradingSignals';
import ModelPerformancePage from './pages/ModelPerformance';
import About from './pages/About';
import { AiChat } from './components/AiChat/AiChat';


const App: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <Router>
      <div className="app-layout" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh' }}>
        <Sidebar collapsed={sidebarCollapsed} />
        <TopBar sidebarCollapsed={sidebarCollapsed} onToggleSidebar={toggleSidebar} />
        <div
          className="main-content"
          style={{
            marginLeft: sidebarCollapsed ? '80px' : '280px',
            marginTop: '70px',
            minHeight: 'calc(100vh - 70px)',
            transition: 'margin-left 0.3s ease'
          }}
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/model-performance" element={<ModelPerformancePage />} />
            <Route path="/trading-signals" element={<TradingSignals />} />
            <Route path="/ai-chat" element={<AiChat />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
        <Footer sidebarCollapsed={sidebarCollapsed} />
      </div>
    </Router>
  );
};

export default App;