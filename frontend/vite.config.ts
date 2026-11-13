import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, __dirname, ''), ...process.env }
  const port = Number(env.FRONTEND_PORT ?? 5003)
  const shared = process.argv.includes('--host')
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    // `vite --host` (what ./start.sh runs) serves other machines. With the watcher and
    // the HMR websocket on, every file save pushes an update to each browser that still
    // has the page open, which network monitoring reads as this box contacting those
    // IPs. Shared mode therefore serves the files and nothing else: no watcher, no
    // websocket. Edits show up on a manual reload. `npm run dev` (localhost) keeps HMR.
    server: shared
      ? { port, allowedHosts: true, watch: null, hmr: false, ws: false }
      : { port, allowedHosts: true },
    preview: { port },
  }
})
