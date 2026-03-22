import express from 'express'
import { createProxyMiddleware } from 'http-proxy-middleware'
import path from 'path'
import { fileURLToPath } from 'url'
import { existsSync } from 'fs'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const dist = path.join(__dirname, 'dist')
const port = Number(process.env.PORT) || 3000

if (!existsSync(dist)) {
  console.error('Missing dist/. Run npm run build first.')
  process.exit(1)
}

const app = express()

app.use(
  '/api-hf',
  createProxyMiddleware({
    target: 'https://router.huggingface.co',
    changeOrigin: true,
    secure: true,
    pathRewrite: { '^/api-hf': '' },
  })
)

app.use(express.static(dist))
// Express 5 / path-to-regexp v8: '*' is not a valid route pattern (crashes at startup).
app.use((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.status(404).end()
    return
  }
  res.sendFile(path.join(dist, 'index.html'))
})

app.listen(port, '0.0.0.0', () => {
  console.log(`Listening on ${port}`)
})
