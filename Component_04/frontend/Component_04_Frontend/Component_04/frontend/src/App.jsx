import { useState } from 'react'
import { configured, predict } from './api.js'
import { scenarios } from './scenarios.js'

const initial = { ...scenarios[0].patient }
const fields = [
  ['age', 'Age', 'years'], ['heartrate', 'Heart rate', 'bpm'], ['sbp', 'Systolic BP', 'mmHg'],
  ['dbp', 'Diastolic BP', 'mmHg'], ['o2sat', 'Oxygen saturation', '%'], ['resprate', 'Respiratory rate', '/min'],
  ['pain', 'Pain score', '0–10'], ['acuity', 'Triage acuity', '1–5'],
  ['troponin_first', 'First troponin', 'ng/mL'], ['troponin_second', 'Second troponin', 'ng/mL']
]
const fmt = (v) => `${(Number(v || 0) * 100).toFixed(1)}%`
const clsText = { No_ACS: 'No ACS', UA: 'Unstable angina', NSTEMI: 'NSTEMI', STEMI: 'STEMI' }

// Plausibility checks for research-form inputs; not diagnostic reference ranges.
const inputLimits = {
  age: { min: 0, max: 120 },
  heartrate: { min: 20, max: 300 },
  sbp: { min: 40, max: 300 },
  dbp: { min: 15, max: 200 },
  o2sat: { min: 40, max: 100 },
  resprate: { min: 4, max: 80 },
  pain: { min: 0, max: 10, integer: true },
  acuity: { min: 1, max: 5, integer: true },
  troponin_first: { min: 0 },
  troponin_second: { min: 0 }
}

function validatePatientInput(patient) {
  for (const [key, label] of fields) {
    const raw = patient[key]
    if (raw === '' || raw == null) {
      if (key === 'age') return `${label} is required.`
      continue // Missing measurements are supported by the model.
    }
    const value = Number(raw)
    const limit = inputLimits[key]
    if (!Number.isFinite(value)) return `${label} must be a valid number.`
    if (value < limit.min || (limit.max != null && value > limit.max)) {
      return `${label} must be ${limit.max == null ? `at least ${limit.min}` : `between ${limit.min} and ${limit.max}`}.`
    }
    if (limit.integer && !Number.isInteger(value)) return `${label} must be a whole number.`
  }
  return ''
} 

function Mark({ children, variant = '' }) { return <span className={`mark ${variant}`}>{children}</span> }
function Section({ title, meta, children }) { return <section className="panel"><div className="panel-title"><h2>{title}</h2>{meta && <span>{meta}</span>}</div>{children}</section> }
function Progress({ value, color }) { const width = Math.max(0, Math.min(100, Number(value || 0) * 100)); return <div className="track"><span style={{ width: `${width}%`, background: color || 'var(--blue)' }} /></div> }

function App() {
  const [patient, setPatient] = useState(initial)
  const [result, setResult] = useState(null)
  const [source, setSource] = useState('')
  const [selected, setSelected] = useState('case-01')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('overview')
  const update = (key, value) => { setPatient(old => ({ ...old, [key]: value })); setResult(null); setSource(''); setError('') }
  const load = (s) => { setSelected(s.id); setPatient({ ...s.patient }); setResult(structuredClone(s.result)); setSource('illustration'); setTab('overview'); setError('') }
  const resetPatientForm = () => {
    setPatient({ ...initial })
    setResult(null)
    setSource('')
    setSelected('')
    setTab('overview')
    setError('')
  }
  async function analyze(e) {
    e.preventDefault(); setError('')
    const validationError = validatePatientInput(patient)
    if (validationError) { setError(validationError); return }
    if (!configured) { setError('Live inference is unavailable until a backend adapter is configured. Choose a synthetic scenario to explore the UI.'); return }
    setBusy(true); setResult(null); setSource('')
    try {
      const payload = { ...patient, age: Number(patient.age), sex: patient.sex, chief_complaint: patient.chief_complaint,
        troponin: [patient.troponin_first, patient.troponin_second].filter(v => v !== '' && v != null).map(Number),
        ecg: { st_elevation: patient.st_elevation, st_depression: patient.st_depression, t_inversion: patient.t_inversion } }
      for (const [key] of fields) { if (key.startsWith('troponin_')) delete payload[key]; else if (payload[key] !== '') payload[key] = Number(payload[key]); else delete payload[key] }
      delete payload.st_elevation; delete payload.st_depression; delete payload.t_inversion
      setResult(await predict(payload)); setSource('backend'); setTab('overview')
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  return <div className="shell">
    <aside className="rail">
      <div className="brand"><div className="brand-icon"><svg viewBox="0 0 38 38" fill="none"><path d="M3 20h9l4-10 6 19 4-9h9" stroke="currentColor" strokeWidth="2.6" strokeLinejoin="round"/></svg></div><div><strong>CARDIO<span>INTEL</span></strong><small>RESEARCH CONSOLE / C04</small></div></div>
      <div className="rail-label">WORKSPACE</div>
      <div className="rail-item selected"><span className="rail-dot"/> ACS classification <span className="rail-counter">01</span></div>
      <div className="rail-item muted"><span className="rail-ring"/> Feature explanations</div>
      <div className="rail-item muted"><span className="rail-ring"/> Evaluation horizons</div>
      <div className="rail-separator" />
      <div className="rail-label">EXAMPLE RECORDS <Mark variant="subtle">SYNTHETIC</Mark></div>
      <div className="cases">{scenarios.map((s,i) => <button key={s.id} className={`case ${selected===s.id && source==='illustration' ? 'active' : ''}`} onClick={() => load(s)}><span className="case-index">0{i+1}</span><span className="case-body"><strong>{s.title}</strong><small>{s.subtitle}</small></span><span className="case-chevron">›</span></button>)}</div>
      <div className="rail-bottom"><div className="small-led"/><span>{configured ? 'API URL configured' : 'Offline demonstration'}</span><small>v1.0 • research prototype</small></div>
    </aside>

    <main className="main">
      <header className="topbar"><div className="breadcrumb">COMPONENT 04 <span>/</span> ACS TRIAGE <span>/</span> PATIENT REVIEW</div><div className="topright"><span className="status-dot"/> {configured ? 'Backend mode available' : 'Demonstration mode'} <div className="avatar">04</div></div></header>
      <div className="content">
        <div className="heading-row"><div><div className="eyebrow">EXPLAINABLE AI • CLINICAL RESEARCH</div><h1>ACS Detection & Classification</h1><p>Review patient features, subtype probabilities, and explainability evidence.</p></div><div className="heading-meta"><span>RESEARCH COMPONENT</span><strong>04 / 04</strong><small>Clinical decision-support prototype</small></div></div>
        <div className="safety"><span className="safety-icon">!</span><div><strong>Research use only — not a diagnosis.</strong> The synthetic scenarios show invented probabilities for interface demonstration. Actual predictions require a separately configured, validated backend. Do not use these results for patient care.</div></div>
        <div className="layout">
          <form className="inputs" onSubmit={analyze}>
            <Section title="Patient information" meta="01 — PROFILE"><div className="panel-body"><div className="input-grid"><label className="field"><span>Sex</span><select value={patient.sex} onChange={e=>update('sex',e.target.value)}><option value="M">Male</option><option value="F">Female</option></select></label><label className="field"><span>Age <em>years</em></span><input type="number" min={inputLimits.age.min} max={inputLimits.age.max} step="1" required value={patient.age} onChange={e=>update('age',e.target.value)}/></label></div><div className="subheading">VITAL SIGNS</div><div className="input-grid">{fields.slice(1,8).map(([key,title,unit])=><label key={key} className="field"><span>{title} <em>{unit}</em></span><input type="number" step={inputLimits[key].integer ? '1' : 'any'} min={inputLimits[key].min} max={inputLimits[key].max} value={patient[key] ?? ''} onChange={e=>update(key,e.target.value)}/></label>)}</div></div></Section>
            <Section title="Clinical evidence" meta="02 — INPUT"><div className="panel-body"><label className="field"><span>Chief complaint <em>free text</em></span><textarea rows="3" value={patient.chief_complaint} onChange={e=>update('chief_complaint',e.target.value)} placeholder="Describe presenting symptoms"/></label><div className="subheading">BIOMARKERS</div><div className="input-grid">{fields.slice(8).map(([key,title,unit])=><label key={key} className="field"><span>{title} <em>{unit}</em></span><input type="number" min="0" step="any" placeholder="Not available" value={patient[key]} onChange={e=>update(key,e.target.value)}/></label>)}</div><div className="subheading">ECG INTERPRETATION FLAGS</div><div className="checks">{[['st_elevation','ST elevation'],['st_depression','ST depression'],['t_inversion','T-wave inversion']].map(([key,title])=><label key={key}><input type="checkbox" checked={Boolean(patient[key])} onChange={e=>update(key,e.target.checked)}/>{title}</label>)}</div></div></Section>
            <div className="input-actions"><button type="submit" className="primary" disabled={busy || !configured}>{busy ? 'Analyzing…' : 'Run ACS analysis'} <span>→</span></button><button type="button" className="secondary" onClick={resetPatientForm} disabled={busy}>Reset patient form</button><small>{configured ? 'Sends input to your configured research API' : 'Connect an API to enable live analysis. Example records work offline.'}</small>{error && <div className="error" role="alert">{error}</div>}</div>
          </form>
          <div className="results">
            {!result ? <div className="empty"><div className="empty-graphic"><svg viewBox="0 0 200 90" fill="none"><path d="M0 45h56l13-16 11 34 14-52 19 61 15-28h72" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"/></svg></div><div className="empty-overline">NO ANALYSIS SELECTED</div><h2>Begin with an example record</h2><p>Select a synthetic case from the left panel to explore the analysis view. Adjusting a field clears the illustrative result until a new case is loaded.</p><button type="button" className="secondary" onClick={()=>load(scenarios[0])}>Open example case <span>→</span></button></div> : <>
              <div className="result-heading"><div><div className="eyebrow">ANALYSIS OUTPUT</div><h2>Classification result</h2></div><Mark variant={source==='illustration'?'warn':'live'}>{source==='illustration'?'SYNTHETIC EXAMPLE':'API RESPONSE'}</Mark></div>
              <div className="summary-cards"><div className="summary-main"><span>CLASSIFICATION</span><strong className={result.prediction === 'No_ACS' ? 'green' : 'red'}>{clsText[result.prediction] || result.prediction}</strong><small>{source==='illustration'?'Illustrative scenario label':'Returned by configured adapter'}</small></div><div className="summary-score"><span>ACS PROBABILITY</span><strong>{fmt(result.p_acs)}</strong><div className="mini-track"><span style={{ width: fmt(result.p_acs) }}/></div><small>{source==='illustration'?'Invented for UI demo':'Model response'}</small></div></div>
              <div className="tabs" role="tablist" aria-label="Result view">{[['overview','Overview'],['explanation','Explainability'],['details','Clinical text']].map(([id,label])=><button type="button" role="tab" aria-selected={tab===id} key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}</div>
              {tab==='overview' && <><Section title="Four-class probability distribution" meta="MODEL OUTPUT"><div className="prob-list">{Object.entries(result.probabilities).map(([name,p])=><div className="prob-row" key={name}><span>{clsText[name]||name}</span><Progress value={p} color={name===result.prediction ? '#2180cc' : '#a4b5c8'}/><strong>{fmt(p)}</strong></div>)}</div></Section><Section title="Interpretation context" meta="REVIEW NOTE"><div className="context"><div><span>Predicted class</span><strong>{clsText[result.prediction]||result.prediction}</strong></div><div><span>Risk descriptor</span><strong>{result.risk_level||'Not provided'}</strong></div><div className="wide"><span>Clinical action</span><p>{result.recommended_action || 'No clinical action supplied; clinician review is required.'}</p></div></div></Section></>}
              {tab==='explanation' && <><Section title="Feature contributions" meta="AVAILABLE WHEN PROVIDED"><div className="explain-list">{result.feature_importance?.length ? result.feature_importance.map((item,i)=><div className="explain-row" key={`${item.feature}-${i}`}><div><span>{item.feature}</span><strong>{fmt(item.value)}</strong></div><Progress value={item.value}/></div>) : <p className="notice">The API response did not provide feature attributions. The CLI output alone does not establish SHAP evidence for this result.</p>}</div></Section><Section title="Evidence by modality" meta="SHARE OF CONTRIBUTION"><div className="modality-list">{result.modality_attribution ? Object.entries(result.modality_attribution).map(([name,p])=><div className="prob-row" key={name}><span>{name}</span><Progress value={p}/><strong>{fmt(p)}</strong></div>) : <p className="notice">Modality attributions were not returned by the API.</p>}</div></Section></>}
              {tab==='details' && <Section title="Chief complaint language evidence" meta="KEYWORD REVIEW"><div className="text-panel"><div className="quote">“{patient.chief_complaint}”</div><div className="subheading">DETECTED CLINICAL TERMS</div>{result.text_attribution?.length ? <div className="tokens">{result.text_attribution.map((t,i)=><div key={i}><strong>{t.term}</strong><span>{t.category || 'keyword'}</span>{t.negated && <Mark variant="warn">NEGATED</Mark>}</div>)}</div> : <p className="notice">No token-level annotations supplied for this record.</p>}</div></Section>}
              <div className="footer-warning">AI-generated research outputs are not validated for direct clinical decisions. Review source data, acquisition times, and limitations.</div>
            </>}
          </div>
        </div>
        <footer className="footer">R26-IT-083 <span>•</span> Component 04 — Explainable Acute Coronary Syndrome Detection <span>•</span> Research prototype</footer>
      </div>
    </main>
  </div>
}
export default App
