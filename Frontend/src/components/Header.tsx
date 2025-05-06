import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Header as HeaderContainer, 
  HeaderContent, 
  Logo, 
  Nav, 
  NavLinks, 
  Button 
} from '../styles/StyledComponents';
import styled from 'styled-components';

// Create a styled Link component that looks like our NavLink
const RouterLink = styled(Link)<{ $active?: boolean }>`
  font-weight: 500;
  color: ${props => props.$active ? 'var(--accent)' : 'var(--text-primary)'};
  
  &:hover {
    color: var(--accent);
  }
`;

// Mobile menu components
const MobileMenuButton = styled.button`
  display: none;
  background: transparent;
  border: none;
  color: var(--text-primary);
  font-size: 1.5rem;
  cursor: pointer;
  margin-right: 15px;
  
  @media (max-width: 768px) {
    display: block;
  }
`;

const MobileMenu = styled.div<{ isOpen: boolean }>`
  display: none;
  
  @media (max-width: 768px) {
    display: ${props => props.isOpen ? 'flex' : 'none'};
    position: absolute;
    top: var(--header-height);
    left: 0;
    right: 0;
    background-color: var(--card-bg);
    flex-direction: column;
    padding: 1rem;
    border-bottom: 1px solid var(--border);
    z-index: 100;
  }
`;

const MobileNavLink = styled(Link)<{ $active?: boolean }>`
  font-weight: 500;
  color: ${props => props.$active ? 'var(--accent)' : 'var(--text-primary)'};
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  
  &:last-child {
    border-bottom: none;
  }
  
  &:hover {
    color: var(--accent);
  }
`;

const Header: React.FC = () => {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };
  
  return (
    <HeaderContainer>
      <HeaderContent>
        <Logo>
          <RouterLink to="/">
            BTC <span>Predict</span>
          </RouterLink>
        </Logo>
        <Nav>
          {/* Mobile menu button */}
          <MobileMenuButton onClick={toggleMobileMenu}>
            {mobileMenuOpen ? '✕' : '☰'}
          </MobileMenuButton>
          
          {/* Desktop navigation */}
          <NavLinks>
            <RouterLink to="/" $active={location.pathname === '/'}>Home</RouterLink>
            <RouterLink to="/about" $active={location.pathname === '/about'}>About</RouterLink>
          </NavLinks>
          
          <Button 
            href="https://www.binance.com/" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Buy Bitcoin
          </Button>
        </Nav>
      </HeaderContent>
      
      {/* Mobile navigation menu */}
      <MobileMenu isOpen={mobileMenuOpen}>
        <MobileNavLink to="/" $active={location.pathname === '/'} onClick={() => setMobileMenuOpen(false)}>
          Home
        </MobileNavLink>
        <MobileNavLink to="/about" $active={location.pathname === '/about'} onClick={() => setMobileMenuOpen(false)}>
          About
        </MobileNavLink>
      </MobileMenu>
    </HeaderContainer>
  );
};

export default Header; 