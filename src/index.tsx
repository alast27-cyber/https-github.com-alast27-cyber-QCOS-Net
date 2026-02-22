import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AuthProvider } from './context/AuthContext';
import './index.css';
import { loader } from "@monaco-editor/react";

// Configure Monaco to use a reliable CDN and handle worker issues
loader.config({
  paths: {
    vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs'
  },
});

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

  // Suppress Monaco worker loading errors as it falls back to main thread
  if (reason && reason.message && (reason.message.includes('importScripts') || reason.message.includes('WorkerGlobalScope'))) {
    console.warn("Suppressed Monaco Worker loading error (falling back to main thread):", reason);
    event.preventDefault();
    return;
  }
});

// Also catch global errors for worker failures
window.addEventListener('error', (event) => {
  if (event.message && (event.message.includes('importScripts') || event.message.includes('WorkerGlobalScope'))) {
    console.warn("Caught and suppressed Monaco Worker network error:", event.message);
    event.preventDefault();
  }
}, true);

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