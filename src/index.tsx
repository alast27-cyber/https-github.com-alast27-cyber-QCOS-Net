
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AuthProvider } from './context/AuthContext';
import './index.css';

// --- SYSTEM PATCH: NOISE CANCELLATION PROTOCOL ---
// Intercepts and suppresses benign "Canceled" errors from Monaco Editor workers and other async ops.
window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  
  // Monaco Editor worker cancellation
  if (reason && (reason === 'Canceled' || reason.message === 'Canceled' || reason?.name === 'Canceled')) {
    event.preventDefault();
    return;
  }
  
  // Suppress "RpcError" which might be network/cancellation related from Gemini SDK
  if (reason && (reason.name === 'RpcError' || (typeof reason.message === 'string' && reason.message.includes('RpcError')))) {
     // Log quietly if needed, but prevent uncaught exception noise if it's just a cancelled request
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
