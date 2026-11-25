import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        '/api': {
          target: env.VITE_API_TARGET || 'http://localhost:8012',
          changeOrigin: true,
        },
        // Used by the frontend to detect the gateway's actual API version
        // (see resolveApiVersion() in src/api/client.js) instead of hardcoding it.
        '/health': {
          target: env.VITE_API_TARGET || 'http://localhost:8012',
          changeOrigin: true,
        },
      },
    },
  }
})
