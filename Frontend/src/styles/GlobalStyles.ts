import { createGlobalStyle } from 'styled-components';

const GlobalStyles = createGlobalStyle`
  :root {
    --background: #0d1117;
    --text-primary: #e6e8ea;
    --text-secondary: #8b949e;
    --accent: #f7931a; /* Bitcoin orange */
    --accent-hover: #f9a942;
    --border: #30363d;
    --card-bg: #161b22;
    --header-height: 70px;
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-color: var(--background);
    color: var(--text-primary);
    line-height: 1.5;
  }

  a {
    color: var(--text-primary);
    text-decoration: none;
    transition: color 0.2s ease;
    
    &:hover {
      color: var(--accent);
    }
  }

  button {
    cursor: pointer;
    font-family: inherit;
  }
`;

export default GlobalStyles; 