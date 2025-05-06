import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import GlobalStyles from './styles/GlobalStyles';
import { AppWrapper, MainContent } from './styles/StyledComponents';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import About from './pages/About';

const App: React.FC = () => {
  return (
    <Router>
      <GlobalStyles />
      <AppWrapper>
        <Header />
        <MainContent>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </MainContent>
        <Footer />
      </AppWrapper>
    </Router>
  );
};

export default App; 