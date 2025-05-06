import React from 'react';
import { Container } from '../styles/StyledComponents';
import BitcoinPriceChart from '../components/BitcoinPriceChart';

const Home: React.FC = () => {
  return (
    <Container>
      <BitcoinPriceChart />
    </Container>
  );
};

export default Home; 