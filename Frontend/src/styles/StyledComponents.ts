import styled from 'styled-components';

export const Container = styled.div`
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 20px;
  width: 100%;
`;

export const AppWrapper = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

export const Header = styled.header`
  height: var(--header-height);
  border-bottom: 1px solid var(--border);
  background-color: var(--background);
  position: sticky;
  top: 0;
  z-index: 10;
  width: 100%;
`;

export const HeaderContent = styled(Container)`
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
`;

export const Logo = styled.div`
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--text-primary);

  span {
    color: var(--accent);
  }
`;

export const Nav = styled.nav`
  display: flex;
  align-items: center;
`;

export const NavLinks = styled.div`
  display: flex;
  gap: 24px;
  margin-right: 24px;

  @media (max-width: 768px) {
    display: none;
  }
`;

export const NavLink = styled.a`
  font-weight: 500;
  &:hover {
    color: var(--accent);
  }
`;

export const Button = styled.a`
  background-color: var(--accent);
  color: #fff;
  padding: 10px 20px;
  border-radius: 4px;
  font-weight: 600;
  border: none;
  cursor: pointer;
  transition: background-color 0.2s ease;

  &:hover {
    background-color: var(--accent-hover);
    color: #fff;
  }
`;

export const MainContent = styled.main`
  flex: 1;
  padding: 40px 0;
`;

export const ChartContainer = styled.div`
  background-color: var(--card-bg);
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 30px;
  border: 1px solid var(--border);
`;

export const ChartHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;

  @media (max-width: 768px) {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
`;

export const ChartTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
`;

export const TimeFrameSelector = styled.div`
  display: flex;
  gap: 8px;

  @media (max-width: 768px) {
    width: 100%;
    overflow-x: auto;
    padding-bottom: 8px;
  }
`;

export const TimeFrameButton = styled.button<{ active: boolean }>`
  background-color: ${props => props.active ? 'var(--accent)' : 'transparent'};
  color: ${props => props.active ? '#fff' : 'var(--text-primary)'};
  border: 1px solid ${props => props.active ? 'var(--accent)' : 'var(--border)'};
  padding: 8px 16px;
  border-radius: 4px;
  font-weight: 500;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${props => props.active ? 'var(--accent-hover)' : 'rgba(247, 147, 26, 0.1)'};
    border-color: var(--accent);
  }
`;

export const Disclaimer = styled.div`
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 16px;
  margin-top: 20px;
  font-size: 0.9rem;
  color: var(--text-secondary);
  line-height: 1.6;
`;

export const Footer = styled.footer`
  border-top: 1px solid var(--border);
  padding: 24px 0;
  color: var(--text-secondary);
  font-size: 0.9rem;
`;

export const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
  
  &:after {
    content: '';
    width: 40px;
    height: 40px;
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-left-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`; 