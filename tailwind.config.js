/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Share Tech Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
        baybayin: ['"Noto Sans Tagalog"', 'sans-serif'],
      },
      animation: {
        'spin-slow': 'spin 8s linear infinite',
        'spin-reverse-slow': 'spin-reverse 12s linear infinite',
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.5s ease-out',
        'fade-in-up': 'fadeInUp 0.5s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      colors: {
        cyan: {
          500: '#06b6d4',
          900: '#164e63',
        }
      }
    },
  },
  plugins: [],
}