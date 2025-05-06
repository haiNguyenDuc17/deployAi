import React from 'react';
import styled from 'styled-components';
import { Container } from '../styles/StyledComponents';

// Styled components for the About page
const AboutContainer = styled.div`
  background-color: var(--card-bg);
  border-radius: 8px;
  padding: 2rem;
  margin-bottom: 30px;
  border: 1px solid var(--border);
`;

const SectionTitle = styled.h2`
  font-size: 1.8rem;
  margin-bottom: 1.5rem;
  color: var(--text-primary);
`;

const SubTitle = styled.h3`
  font-size: 1.3rem;
  margin: 1.5rem 0 0.75rem;
  color: var(--accent);
`;

const Paragraph = styled.p`
  margin-bottom: 1rem;
  line-height: 1.6;
  color: var(--text-primary);
`;

const FeatureList = styled.ul`
  margin: 1rem 0 1.5rem 1.5rem;
  
  li {
    margin-bottom: 0.5rem;
    line-height: 1.5;
  }
`;

const TechBadge = styled.span`
  display: inline-block;
  background-color: rgba(247, 147, 26, 0.15);
  color: var(--accent);
  padding: 0.35rem 0.75rem;
  border-radius: 4px;
  margin-right: 0.75rem;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
  font-weight: 500;
  border: 1px solid rgba(247, 147, 26, 0.3);
`;

const TechStack = styled.div`
  margin: 1.5rem 0;
`;

const About: React.FC = () => {
  return (
    <Container>
      <AboutContainer>
        <SectionTitle>About BTC Predict</SectionTitle>
        
        <Paragraph>
          BTC Predict is an advanced web application designed to visualize Bitcoin price predictions
          using machine learning algorithms. Our platform provides forecasts for various time horizons,
          helping cryptocurrency investors make more informed decisions.
        </Paragraph>
        
        <SubTitle>Key Features</SubTitle>
        <FeatureList>
          <li>Interactive price prediction chart using the powerful ECharts library</li>
          <li>Flexible time frame selection (1 month, 6 months, 1 year, and 3 years)</li>
          <li>Real-time data fetching from our prediction API</li>
          <li>Professional dark-themed UI inspired by leading crypto platforms</li>
          <li>Responsive design that works on desktop, tablet, and mobile devices</li>
        </FeatureList>
        
        <SubTitle>How It Works</SubTitle>
        <Paragraph>
          Our system uses historical Bitcoin price data to train machine learning models that can
          identify patterns and trends in cryptocurrency markets. These models analyze factors such as
          price momentum, trading volume, market sentiment, and macroeconomic indicators to generate
          price predictions for different time horizons.
        </Paragraph>
        
        <Paragraph>
          For each prediction timeframe, our API generates a series of daily price forecasts, which
          are then visualized on our interactive chart. The longer the prediction horizon, the more
          uncertainty is inherently involved - users should always consider this when viewing long-term
          forecasts.
        </Paragraph>
        
        <SubTitle>Technology Stack</SubTitle>
        <TechStack>
          <TechBadge>React</TechBadge>
          <TechBadge>TypeScript</TechBadge>
          <TechBadge>ECharts</TechBadge>
          <TechBadge>Styled Components</TechBadge>
          <TechBadge>Flask</TechBadge>
          <TechBadge>Python</TechBadge>
          <TechBadge>Machine Learning</TechBadge>
        </TechStack>
        
        <SubTitle>Disclaimer</SubTitle>
        <Paragraph>
          The predictions provided by BTC Predict are based on historical data and mathematical models.
          Cryptocurrency markets are highly volatile and unpredictable, and many factors can influence
          prices that are impossible to forecast. The information on this site is for reference only and
          does not constitute financial advice. We accept no liability for investment decisions made based
          on these predictions.
        </Paragraph>
        
        <Paragraph>
          Always conduct your own research and consider consulting with a qualified financial advisor
          before making investment decisions.
        </Paragraph>
      </AboutContainer>
    </Container>
  );
};

export default About; 