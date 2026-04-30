import React from 'react'
import ReactDOM from 'react-dom/client'
import StandaloneApp from './StandaloneApp'
import './index.css'

/**
 * Standalone entry point for AgentQ.
 * This file initializes the AgentQ supremacy in an isolated DOM root.
 */
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StandaloneApp />
  </React.StrictMode>,
)
