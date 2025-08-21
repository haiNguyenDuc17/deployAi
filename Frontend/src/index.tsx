import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import 'primereact/resources/themes/bootstrap4-dark-blue/theme.css';
import 'primereact/resources/primereact.min.css';
import 'primeicons/primeicons.css';
import 'primeflex/primeflex.css';

import { PublicClientApplication, EventType, AccountInfo } from '@azure/msal-browser';
import { MsalProvider } from '@azure/msal-react';
import { msalConfig } from './auth/authConfig'; // <-- use your config

// Create MSAL instance
const pca = new PublicClientApplication(msalConfig);

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

pca.addEventCallback((event) => {
  if (event.eventType === EventType.LOGIN_SUCCESS || event.eventType === EventType.SSO_SILENT_SUCCESS) {
    const account = (event as any).payload?.account as AccountInfo | undefined;
    if (account) {
      pca.setActiveAccount(account);
      console.log("[MSAL] Active account set:", account.username); // dev-only
    }
  }
});

root.render(
  <React.StrictMode>
    <MsalProvider instance={pca}>
      <App />
    </MsalProvider>
  </React.StrictMode>
);
