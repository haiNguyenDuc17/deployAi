import React from 'react';
import ModelPerformance from '../components/ModelPerformance';

const ModelPerformancePage: React.FC = () => {
  return (
    <div className="model-performance-page p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh', color: '#ffffff' }}>
      <div className="mb-4">
        <h1 className="text-white text-4xl font-bold mb-2">Model Performance</h1>
        <p className="text-500 text-lg">Comprehensive AI model metrics and performance analysis</p>
      </div>
      
      <ModelPerformance showDetailedView={true} />
    </div>
  );
};

export default ModelPerformancePage;
