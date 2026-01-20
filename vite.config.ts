import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'url';
import path from 'path';
import process from 'node:process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.ELEVENLABS_API_KEY': JSON.stringify(env.ELEVENLABS_API_KEY),
        'process.env.LEX_FRIDMAN_VOICE_ID': JSON.stringify(env.LEX_FRIDMAN_VOICE_ID)
      },
      resolve: {
        alias: {
          // Points @ to the src directory for cleaner imports
          '@': path.resolve(__dirname, './src'),
          '@components': path.resolve(__dirname, './src/components'),
          '@hooks': path.resolve(__dirname, './src/hooks'),
          '@utils': path.resolve(__dirname, './src/utils'),
          '@context': path.resolve(__dirname, './src/context'),
          '@services': path.resolve(__dirname, './src/services'),
          '@types': path.resolve(__dirname, './src/types'),
        },
        extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json']
      }
    };
});