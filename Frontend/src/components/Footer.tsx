import React from 'react';
import { Footer as FooterContainer, Container } from '../styles/StyledComponents';

const Footer: React.FC = () => {
  return (
    <FooterContainer>
      <Container>
        <p>© {new Date().getFullYear()} BTC Predict. All rights reserved.</p>
      </Container>
    </FooterContainer>
  );
};

export default Footer; 