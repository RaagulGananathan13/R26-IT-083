import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The API runs on :5000 by default. Requests to /api are proxied there so the
// browser never sees a cross-origin request in development.
const API = process.env.VITE_API_TARGET || 'http://127.0.0.1:5000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API, changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
})
