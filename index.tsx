import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AuthProvider } from './context/AuthContext';
import './index.css';

// --- SYSTEM PATCH: NOISE CANCELLATION PROTOCOL ---
// Intercepts and suppresses benign "Canceled" errors from Monaco Editor workers and other async ops.
window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  
  if (reason && (reason === 'Canceled' || reason.message === 'Canceled' || reason?.name === 'Canceled')) {
    event.preventDefault();
    return;
  }
  
  if (reason && (reason.name === 'RpcError' || (typeof reason.message === 'string' && reason.message.includes('RpcError')))) {
     console.warn("Suppressed RpcError (likely network/cancellation):", reason);
     event.preventDefault();
     return;
  }
});

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <AuthProvider>
    <App />
  </AuthProvider>
);