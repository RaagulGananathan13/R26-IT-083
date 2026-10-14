const API = (import.meta.env.VITE_ACS_API_URL || '').trim().replace(/\/$/, '')
export const configured = Boolean(API)
export async function predict(patient) {
  if (!configured) throw new Error('No ACS backend configured. Set VITE_ACS_API_URL to a supported inference adapter URL.')
  const response = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patient) })
  if (!response.ok) { let detail = ''; try { detail = (await response.json()).detail || '' } catch { /* HTTP failure */ } throw new Error(`Prediction failed (${response.status})${detail ? `: ${detail}` : ''}`) }
  const data = await response.json()
  if (!data || !['No_ACS','UA','NSTEMI','STEMI'].includes(data.prediction) || !data.probabilities) throw new Error('Response did not match the Component 04 prediction contract.')
  return data
}
