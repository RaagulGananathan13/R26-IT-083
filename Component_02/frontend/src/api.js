// Thin API client. Vite proxies /api to the Flask server (see vite.config.js).

const BASE = import.meta.env.VITE_API_BASE || ''

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, opts)
  let body
  try {
    body = await res.json()
  } catch {
    throw new Error(`Server returned ${res.status} (not JSON). Is the backend running?`)
  }
  if (!res.ok) throw new Error(body?.error || `Request failed (${res.status})`)
  return body
}

const theme = () =>
  document.documentElement.classList.contains('dark') ? 'dark' : 'light'

export const getHealth = () => req('/api/health')

export const getPatients = (cls, q = '') =>
  req(`/api/patients/${cls}${q ? `?q=${encodeURIComponent(q)}` : ''}`)

export const analyzeById = (id) =>
  req(`/api/analyze/${id}?theme=${theme()}`, { method: 'POST' })

export const analyzeRandom = () =>
  req(`/api/demo?theme=${theme()}`, { method: 'POST' })

export const analyzeUpload = (datFile, heaFile) => {
  const fd = new FormData()
  fd.append('dat_file', datFile)
  fd.append('hea_file', heaFile)
  return req(`/api/predict?theme=${theme()}`, { method: 'POST', body: fd })
}
