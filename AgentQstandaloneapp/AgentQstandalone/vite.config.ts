import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Vite Configuration for AgentQ Standalone
 * 
 * This configuration focuses on building the AgentQ interface as an isolated
 * production-ready application while maintaining symbolic links/references 
 * to the parent src/ directory.
 */
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  root: __dirname,
  base: '/',
  build: {
    outDir: '../dist-standalone',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'framer-motion'],
          ui: ['lucide-react', 'recharts'],
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': __dirname,
    },
  },
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
