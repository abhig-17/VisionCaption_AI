import { useCallback, useRef, useState } from 'react'

const HF_INFERENCE_BASE = (
  import.meta.env.VITE_HF_INFERENCE_BASE_URL || '/api-hf'
).replace(/\/$/, '')
/** Vision model on Inference Providers (OpenAI-compatible chat). Override with VITE_HF_VISION_MODEL. */
const VISION_MODEL =
  import.meta.env.VITE_HF_VISION_MODEL?.trim() || 'Qwen/Qwen2.5-VL-7B-Instruct'
const HF_CHAT_URL = `${HF_INFERENCE_BASE}/v1/chat/completions`
const REQUEST_TIMEOUT_MS = 60_000
const MAX_UPLOAD_BYTES = 1024 * 1024
const MAX_IMAGE_EDGE = 1600

function parseChatCaption(data) {
  const msg = data?.choices?.[0]?.message
  if (!msg) return null
  const c = msg.content
  if (typeof c === 'string') {
    const t = c.trim()
    return t || null
  }
  if (Array.isArray(c)) {
    const text = c
      .map((p) => {
        if (typeof p === 'string') return p
        if (p?.type === 'text' && p.text) return String(p.text)
        return ''
      })
      .join('')
    const t = text.trim()
    return t || null
  }
  return null
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

function looksLikeHtml(body) {
  const s = body.trim().toLowerCase()
  return s.startsWith('<!doctype') || s.startsWith('<html')
}

/** Never surface raw HTML error pages in the UI. */
async function readErrorMessage(response) {
  const ct = (response.headers.get('content-type') || '').toLowerCase()
  if (ct.includes('text/html')) {
    return 'Network error'
  }

  const text = await response.text()
  if (!text) {
    return response.statusText || 'Network error'
  }
  if (looksLikeHtml(text)) {
    return 'Network error'
  }

  try {
    const json = JSON.parse(text)
    if (json?.error) {
      if (typeof json.error === 'string') return json.error
      if (json.error?.message) return String(json.error.message)
    }
    if (typeof json?.message === 'string') return json.message
  } catch {
    /* not JSON */
  }

  const plain = text.replace(/\s+/g, ' ').trim()
  if (plain.length > 160) {
    return `${plain.slice(0, 160)}…`
  }
  return plain || 'Network error'
}

function parseJsonResponse(raw) {
  if (looksLikeHtml(raw)) {
    return { ok: false, error: 'Network error' }
  }
  try {
    return { ok: true, data: JSON.parse(raw) }
  } catch {
    return { ok: false, error: 'Network error' }
  }
}

/**
 * Resize + JPEG encode until under maxBytes (same visual preview; upload is compressed).
 */
async function compressImageForUpload(file) {
  let bitmap
  try {
    bitmap = await createImageBitmap(file)
  } catch {
    throw new Error('INVALID_IMAGE')
  }

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    bitmap.close?.()
    throw new Error('INVALID_IMAGE')
  }

  let scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(bitmap.width, bitmap.height))
  let quality = 0.92
  let blob = null

  try {
    for (let i = 0; i < 28; i++) {
      const w = Math.max(1, Math.round(bitmap.width * scale))
      const h = Math.max(1, Math.round(bitmap.height * scale))
      canvas.width = w
      canvas.height = h
      ctx.drawImage(bitmap, 0, 0, w, h)

      blob = await new Promise((resolve, reject) => {
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error('INVALID_IMAGE'))),
          'image/jpeg',
          quality
        )
      })

      if (blob.size <= MAX_UPLOAD_BYTES) {
        return { blob, mimeType: 'image/jpeg' }
      }

      if (quality > 0.52) quality -= 0.07
      else scale *= 0.82
    }

    if (blob && blob.size > MAX_UPLOAD_BYTES) {
      throw new Error('INVALID_IMAGE')
    }
    return { blob, mimeType: 'image/jpeg' }
  } finally {
    bitmap.close?.()
  }
}

function LoadingSpinner({ className = '' }) {
  return (
    <span
      className={`relative inline-flex h-5 w-5 shrink-0 items-center justify-center ${className}`}
      aria-hidden
    >
      <span className="spinner-orbit absolute inset-0 rounded-full border-2 border-white/20 border-t-[#7C3AED] border-r-white/60" />
      <span className="absolute inset-1 rounded-full border border-white/10" />
    </span>
  )
}

export default function App() {
  const fileInputRef = useRef(null)
  const inFlightRef = useRef(false)
  const abortRef = useRef(null)

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [caption, setCaption] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [errorRetryable, setErrorRetryable] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [copied, setCopied] = useState(false)
  const [captionAnimNonce, setCaptionAnimNonce] = useState(0)

  const token = import.meta.env.VITE_HF_API_TOKEN?.trim()

  const resetPreview = useCallback(() => {
    if (loading) return
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setFile(null)
    setCaption('')
    setError('')
    setErrorRetryable(false)
  }, [loading])

  const applyFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) {
      setError('Invalid image')
      setErrorRetryable(false)
      return
    }
    setError('')
    setErrorRetryable(false)
    setCaption('')
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(f)
    })
    setFile(f)
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragActive(false)
      const f = e.dataTransfer?.files?.[0]
      applyFile(f)
    },
    [applyFile]
  )

  const runCaptionRequest = useCallback(async () => {
    if (!file || !token) return
    if (inFlightRef.current) return

    inFlightRef.current = true
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac

    setLoading(true)
    setError('')
    setErrorRetryable(false)
    setCaption('')

    const timeoutId = window.setTimeout(() => ac.abort(), REQUEST_TIMEOUT_MS)

    try {
      let payloadJson
      try {
        const out = await compressImageForUpload(file)
        const buf = await out.blob.arrayBuffer()
        const b64 = arrayBufferToBase64(buf)
        const dataUrl = `data:${out.mimeType};base64,${b64}`
        payloadJson = JSON.stringify({
          model: VISION_MODEL,
          max_tokens: 160,
          temperature: 0.3,
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'text',
                  text: 'Write one short, natural sentence describing this image, like a photo caption or alt text. No preamble.',
                },
                {
                  type: 'image_url',
                  image_url: { url: dataUrl },
                },
              ],
            },
          ],
        })
      } catch (e) {
        if (e instanceof Error && e.message === 'INVALID_IMAGE') {
          setError('Invalid image')
          setErrorRetryable(false)
          return
        }
        throw e
      }

      const response = await fetch(HF_CHAT_URL, {
        method: 'POST',
        signal: ac.signal,
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: payloadJson,
      })

      if (response.status === 503) {
        setError('Model loading, try again')
        setErrorRetryable(true)
        return
      }

      if (!response.ok) {
        if (response.status === 401) {
          setError('Invalid or missing API token. Check VITE_HF_API_TOKEN.')
          setErrorRetryable(false)
        } else if (response.status === 403) {
          setError(
            'Token cannot call Inference Providers. Use a fine-grained token with “Make calls to Inference Providers” enabled.'
          )
          setErrorRetryable(false)
        } else if (response.status === 410) {
          setError('Network error')
          setErrorRetryable(true)
        } else {
          const msg = await readErrorMessage(response)
          setError(msg.length > 120 ? `${msg.slice(0, 120)}…` : msg)
          setErrorRetryable(true)
        }
        return
      }

      const rawBody = await response.text()
      const parsed = parseJsonResponse(rawBody)
      if (!parsed.ok) {
        setError(parsed.error)
        setErrorRetryable(true)
        return
      }
      const data = parsed.data
      const text = parseChatCaption(data)
      if (!text) {
        setError('Invalid image')
        setErrorRetryable(false)
        return
      }

      setCaption(text)
      setCaptionAnimNonce((n) => n + 1)
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Network error')
        setErrorRetryable(true)
      } else {
        setError('Network error')
        setErrorRetryable(true)
      }
    } finally {
      window.clearTimeout(timeoutId)
      setLoading(false)
      inFlightRef.current = false
    }
  }, [file, token])

  const onGenerate = () => {
    if (!file) {
      setError('Upload an image first.')
      setErrorRetryable(false)
      return
    }
    if (!token) {
      setError(
        'Add your Hugging Face API token: create a .env file with VITE_HF_API_TOKEN=your_token and restart the dev server.'
      )
      setErrorRetryable(false)
      return
    }
    void runCaptionRequest()
  }

  const copyCaption = async () => {
    if (!caption) return
    try {
      await navigator.clipboard.writeText(caption)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      setError('Could not copy to clipboard.')
      setErrorRetryable(false)
    }
  }

  return (
    <div className="relative min-h-svh overflow-hidden bg-[#0F172A] px-4 py-10 text-slate-200">
      <div
        className="pointer-events-none absolute inset-0 opacity-40"
        style={{
          background:
            'radial-gradient(ellipse 80% 50% at 50% -20%, #7C3AED 0%, transparent 55%), radial-gradient(ellipse 60% 40% at 100% 50%, rgba(124, 58, 237, 0.25) 0%, transparent 50%), radial-gradient(ellipse 50% 40% at 0% 80%, rgba(124, 58, 237, 0.2) 0%, transparent 45%)',
        }}
      />

      <div className="relative mx-auto flex max-w-lg flex-col items-center">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            VisionCaption{' '}
            <span className="bg-gradient-to-r from-[#7C3AED] to-violet-300 bg-clip-text text-transparent">
              AI
            </span>
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Image captions via Hugging Face Inference Providers (vision chat API)
          </p>
        </header>

        <div
          className={`w-full rounded-3xl border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-black/40 backdrop-blur-xl transition-[box-shadow,transform] duration-300 hover:shadow-[0_0_40px_-10px_rgba(124,58,237,0.35)] ${
            dragActive ? 'ring-2 ring-[#7C3AED]/60 ring-offset-2 ring-offset-[#0F172A]' : ''
          }`}
          onDragEnter={(e) => {
            e.preventDefault()
            setDragActive(true)
          }}
          onDragLeave={(e) => {
            e.preventDefault()
            if (!e.currentTarget.contains(e.relatedTarget)) setDragActive(false)
          }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            disabled={loading}
            onChange={(e) => {
              const f = e.target.files?.[0]
              applyFile(f)
              e.target.value = ''
            }}
          />

          <div
            role="button"
            tabIndex={loading ? -1 : 0}
            onKeyDown={(e) => {
              if (loading) return
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                fileInputRef.current?.click()
              }
            }}
            onClick={() => !loading && fileInputRef.current?.click()}
            className={`group flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-[#0F172A]/50 px-6 py-10 transition-all duration-300 hover:border-[#7C3AED]/50 hover:bg-[#0F172A]/70 ${loading ? 'pointer-events-none opacity-60' : 'cursor-pointer'}`}
          >
            <div className="mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#7C3AED]/20 text-[#7C3AED] transition-transform duration-300 group-hover:scale-105 group-hover:bg-[#7C3AED]/30">
              <svg
                className="h-7 w-7"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
                aria-hidden
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
            </div>
            <p className="text-center text-sm font-medium text-slate-200">
              Drag and drop an image here
            </p>
            <p className="mt-1 text-center text-xs text-slate-500">
              or click to browse — PNG, JPG, WebP, GIF
            </p>
          </div>

          {previewUrl && (
            <div className="mt-6 space-y-4">
              <div className="overflow-hidden rounded-2xl border border-white/10 bg-black/20">
                <img
                  src={previewUrl}
                  alt="Uploaded preview"
                  className="mx-auto max-h-72 w-full object-contain"
                />
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={onGenerate}
                  disabled={loading}
                  aria-busy={loading}
                  aria-live="polite"
                  className="flex flex-1 min-w-[140px] items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-[#7C3AED] to-[#6d28d9] px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-[#7C3AED]/25 transition-all duration-300 hover:scale-[1.02] hover:shadow-[#7C3AED]/35 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
                >
                  {loading ? (
                    <>
                      <LoadingSpinner />
                      <span>Generating caption...</span>
                    </>
                  ) : (
                    'Generate Caption'
                  )}
                </button>
                <button
                  type="button"
                  onClick={resetPreview}
                  disabled={loading}
                  className="rounded-xl border border-white/15 bg-white/5 px-5 py-3 text-sm font-medium text-slate-300 transition-all duration-300 hover:border-white/25 hover:bg-white/10 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
                >
                  Clear
                </button>
              </div>

              {error && errorRetryable && !loading && (
                <div className="flex flex-wrap items-center justify-center gap-2">
                  <button
                    type="button"
                    onClick={onGenerate}
                    className="rounded-xl border border-[#7C3AED]/40 bg-[#7C3AED]/15 px-4 py-2 text-sm font-semibold text-[#e9d5ff] transition-all duration-300 hover:border-[#7C3AED]/60 hover:bg-[#7C3AED]/25 active:scale-[0.98]"
                  >
                    Retry
                  </button>
                </div>
              )}

              {caption && (
                <div
                  key={captionAnimNonce}
                  className="caption-reveal rounded-2xl border border-white/10 bg-[#0F172A]/60 p-4 shadow-lg shadow-[#7C3AED]/10 backdrop-blur-sm ring-1 ring-[#7C3AED]/20"
                >
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="text-xs font-medium uppercase tracking-wider text-[#7C3AED]">
                      Caption
                    </span>
                    <button
                      type="button"
                      onClick={copyCaption}
                      disabled={loading}
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-200 transition-all duration-300 hover:border-[#7C3AED]/40 hover:bg-[#7C3AED]/15 hover:text-white active:scale-95 disabled:opacity-50"
                    >
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <p className="text-left text-base leading-relaxed text-slate-100">{caption}</p>
                </div>
              )}
            </div>
          )}

          {error && (
            <div
              className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-left text-sm text-red-200"
              role="alert"
            >
              {error}
            </div>
          )}

          {!token && (
            <p className="mt-4 text-center text-xs text-amber-200/80">
              Set <code className="rounded bg-black/30 px-1 py-0.5">VITE_HF_API_TOKEN</code>{' '}
              in <code className="rounded bg-black/30 px-1 py-0.5">.env</code> (see{' '}
              <code className="rounded bg-black/30 px-1 py-0.5">.env.example</code>). Use a fine-grained token with
              Inference Providers access.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
