import React, { useState, useEffect } from 'react';
import { Card } from 'primereact/card';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { Badge } from 'primereact/badge';
import { Skeleton } from 'primereact/skeleton';
import { CryptoPriceInfo } from '../api/csvDataService';
import BitcoinPriceChart from './BitcoinPriceChart';
import InvestmentAnalysis from './InvestmentAnalysis';
import { useBinanceMulti } from '../hooks/useLiveBinance';
import { fetchMarketData } from '../api/cryptoApi';

interface DashboardProps {}
const Dashboard: React.FC<DashboardProps> = () => {
  const [cryptoPrices, setCryptoPrices] = useState<CryptoPriceInfo[]>([]);
  const [symbols, setSymbols] = useState<string[]>(["BTC", "ETH", "XRP"]);
  const { prices: liveMap, connected } = useBinanceMulti(symbols, 300);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const data = await fetchMarketData();
        setCryptoPrices(data);
        setSymbols(data.map((c) => c.symbol.toUpperCase()));
      } catch (error) {
        console.error('Error loading dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  const formatPrice = (price: number) => {
    if (price >= 1000) {
      return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    return `$${price.toFixed(2)}`;
  };

  const displayPrice = (c: CryptoPriceInfo) =>
    Number.isFinite(liveMap[c.symbol] as number)
      ? (liveMap[c.symbol] as number)
      : c.price;

  const priceBodyTemplate = (row: CryptoPriceInfo) => (
    <span className="font-semibold">{formatPrice(displayPrice(row))}</span>
  );

  const change24hBodyTemplate = (row: CryptoPriceInfo) => {
    const isPositive = row.changePercent24h >= 0;
    return (
      <Badge
        value={`${isPositive ? '+' : ''}${row.changePercent24h.toFixed(2)}%`}
        severity={isPositive ? 'success' : 'danger'}
      />
    );
  };

  const change7dBodyTemplate = (row: CryptoPriceInfo) => {
    const isPositive = row.changePercent7d >= 0;
    return (
      <Badge
        value={`${isPositive ? '+' : ''}${row.changePercent7d.toFixed(2)}%`}
        severity={isPositive ? 'success' : 'danger'}
      />
    );
  };

  const nameBodyTemplate = (row: CryptoPriceInfo) => (
    <div className="flex align-items-center gap-2">
      {row.icon ? (
        <img
          src={row.icon}
          alt={row.symbol}
          width={20}
          height={20}
          style={{ borderRadius: '50%' }}
          loading="lazy"
        />
      ) : (
        <i className="pi pi-circle-fill" style={{ fontSize: '1.2rem' }} />
      )}
      <div>
        <div className="font-semibold">{row.name}</div>
        <div className="text-sm text-500">{row.symbol}</div>
      </div>
    </div>
  );

  const volumeBodyTemplate = (row: CryptoPriceInfo) => (
    <span>{Number(row.volume).toLocaleString()}</span>
  );

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

      <InvestmentAnalysis className="mb-4" />

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
          value={cryptoPrices.map(c => ({
            ...c,
            price: Number.isFinite(liveMap[c.symbol] as number)
              ? (liveMap[c.symbol] as number)
              : c.price
          }))}
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
