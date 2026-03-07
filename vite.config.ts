import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath } from 'url';
import path from 'path';
import process from 'node:process';

import monacoEditorPlugin from 'vite-plugin-monaco-editor';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react(), tailwindcss(), (monacoEditorPlugin as any).default({})],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.ELEVENLABS_API_KEY': JSON.stringify(env.ELEVENLABS_API_KEY),
        'process.env.LEX_FRIDMAN_VOICE_ID': JSON.stringify(env.LEX_FRIDMAN_VOICE_ID)
      },
      resolve: {
        alias: [
          { find: '@', replacement: path.resolve(__dirname, 'src') }
        ],
        extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json']
      }
    };
});