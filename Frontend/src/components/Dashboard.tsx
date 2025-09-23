import React, { useState, useEffect } from 'react';
import { Card } from 'primereact/card';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';


import { Badge } from 'primereact/badge';
import { Skeleton } from 'primereact/skeleton';
import { fetchAllCryptoPrices, CryptoPriceInfo } from '../api/csvDataService';
import BitcoinPriceChart from './BitcoinPriceChart';
import InvestmentAnalysis from './InvestmentAnalysis';
import { useBinanceMulti } from '../hooks/useLiveBinance';
import { fetchMarketData } from "../api/cryptoApi"; // returns CoinGecko data mapped to CryptoPriceInfo[]


interface DashboardProps {}
const Dashboard: React.FC<DashboardProps> = () => {
  const [cryptoPrices, setCryptoPrices] = useState<CryptoPriceInfo[]>([]);
  const [symbols, setSymbols] = useState<string[]>(["BTC","ETH","XRP"]); // default; will be replaced after fetch
  const { prices: liveMap, connected } = useBinanceMulti(symbols, 300);  // debounce 300ms
  const [loading, setLoading] = useState(true);


  useEffect(() => {
  const loadData = async () => {
    try {
      setLoading(true);
      const data = await fetchMarketData(); // <- returns CryptoPriceInfo[]
      setCryptoPrices(data);
      // update symbols list from the data you show (BTC, ETH, XRP, etc.)
      setSymbols(data.map((c) => c.symbol.toUpperCase()));
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  loadData();
  // NO dependency on live prices; we only fetch once
}, []);


  const formatPrice = (price: number) => {
    if (price >= 1000) {
      return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    return `$${price.toFixed(2)}`;
  };



  const priceBodyTemplate = (row: CryptoPriceInfo) => (
  <span className="font-semibold">{formatPrice(displayPrice(row))}</span>
);

  const change24hBodyTemplate = (rowData: CryptoPriceInfo) => {
    const isPositive = rowData.changePercent24h >= 0;
    return (
      <Badge
        value={`${isPositive ? '+' : ''}${rowData.changePercent24h.toFixed(2)}%`}
        severity={isPositive ? 'success' : 'danger'}
      />
    );
  };

  const change7dBodyTemplate = (rowData: CryptoPriceInfo) => {
    const isPositive = rowData.changePercent7d >= 0;
    return (
      <Badge
        value={`${isPositive ? '+' : ''}${rowData.changePercent7d.toFixed(2)}%`}
        severity={isPositive ? 'success' : 'danger'}
      />
    );
  };

  const nameBodyTemplate = (rowData: CryptoPriceInfo) => {
    return (
      <div className="flex align-items-center gap-2">
        <i className={`${rowData.icon} text-orange-500`} style={{ fontSize: '1.2rem' }}></i>
        <div>
          <div className="font-semibold">{rowData.name}</div>
          <div className="text-sm text-500">{rowData.symbol}</div>
        </div>
      </div>
    );
  };

const displayPrice = (c: CryptoPriceInfo) =>
  Number.isFinite(liveMap[c.symbol]) ? (liveMap[c.symbol] as number) : c.price;

  const volumeBodyTemplate = (rowData: CryptoPriceInfo) => {
    return <span>{rowData.volume}</span>;
  };

  if (loading) {
    return (
      <div className="dashboard-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh' }}>
        <div className="grid">
          <div className="col-12 md:col-8">
            <Card className="mb-4" style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
              <Skeleton height="300px" />
            </Card>
          </div>
          <div className="col-12 md:col-4">
            <Card style={{ backgroundColor: '#2a2a2a', border: '1px solid #404040' }}>
              <Skeleton height="300px" />
            </Card>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container p-4" style={{ backgroundColor: '#1a1a1a', minHeight: '100vh', color: '#ffffff' }}>
      {/* Top Price Cards */}
      <div className="grid mb-4">
        {cryptoPrices.slice(0, 3).map((crypto) => (
          <div key={crypto.symbol} className="col-12 md:col-4">
            <Card 
              className="text-center"
              style={{ 
                backgroundColor: '#2a2a2a', 
                border: '1px solid #404040',
                borderRadius: '12px'
              }}
            >
              <div className="flex align-items-center justify-content-center gap-2 mb-2">
                {crypto.icon ? (
                  <img
                    src={crypto.icon}
                    alt={crypto.symbol}
                    width={20}
                    height={20}
                    style={{ borderRadius: '50%' }}
                    loading="lazy"
                  />
                ) : (
                  <i className="pi pi-circle-fill" style={{ fontSize: '1.2rem' }} />
                )}
                <span className="text-lg font-semibold text-400">{crypto.symbol}/USDT</span>
              </div>
              <div className="text-2xl font-bold mb-2">{formatPrice(displayPrice(crypto))}</div>
              <Badge
                value={`${crypto.changePercent24h >= 0 ? '+' : ''}${crypto.changePercent24h.toFixed(2)}%`}
                severity={crypto.changePercent24h >= 0 ? 'success' : 'danger'}
              />
            </Card>
          </div>
        ))}
      </div>

      {/* BTC Prediction Chart - Full Width */}
      <Card
        className="mb-4"
        style={{
          backgroundColor: '#2a2a2a',
          border: '1px solid #404040',
          borderRadius: '12px'
        }}
      >
        <div className="flex justify-content-between align-items-center mb-3">
          <div>
            <h3 className="m-0 text-white">BTC Prediction</h3>
            <p className="text-500 m-0"></p>
          </div>
        </div>
        <BitcoinPriceChart />
      </Card>

      {/* Investment Analysis */}
      <InvestmentAnalysis className="mb-4" />

      {/* Recent Prices Table */}
      <Card 
        style={{ 
          backgroundColor: '#2a2a2a', 
          border: '1px solid #404040',
          borderRadius: '12px'
        }}
      >
        <div className="flex justify-content-between align-items-center mb-3">
          <div>
            <h3 className="m-0 text-white">Recent</h3>
            <p className="text-500 m-0"></p>
          </div>
        </div>
        
        <DataTable 
          value={cryptoPrices} 
          className="p-datatable-sm"
          style={{ backgroundColor: 'transparent' }}
        >
          <Column field="name" header="Name" body={nameBodyTemplate} />
          <Column field="price" header="Price" body={priceBodyTemplate} />
          <Column field="changePercent24h" header="24h%" body={change24hBodyTemplate} />
          <Column field="changePercent7d" header="7d%" body={change7dBodyTemplate} />
          <Column field="volume" header="Volume(24h)" body={volumeBodyTemplate} />
        </DataTable>
      </Card>
    </div>
  );
};

export default Dashboard;
