import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Legacy api-inference.huggingface.co returns 410 Gone — use Inference Providers router.
const hfProxy = {
  '/api-hf': {
    target: 'https://router.huggingface.co',
    changeOrigin: true,
    secure: true,
    rewrite: (path) => path.replace(/^\/api-hf/, ''),
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: hfProxy },
  preview: { proxy: hfProxy },
})
