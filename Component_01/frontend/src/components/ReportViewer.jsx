import { useState } from 'react'

export default function ReportViewer({ reportText, reportTextRaw, groundTruthReport, classifierPrompt }) {
  const [copied, setCopied] = useState(false)
  const [viewMode, setViewMode] = useState('cleaned') // cleaned | original | ground_truth

  const activeText = viewMode === 'ground_truth' 
    ? (groundTruthReport || 'No ground truth available for this image.') 
    : viewMode === 'original' 
      ? reportTextRaw 
      : reportText

  // Split into Impression (first sentence) and Findings (rest)
  const sentences = activeText.split('. ').filter(s => s.trim().length > 0).map(s => s.trim() + '.')
  
  let impression = ''
  let findings = ''
  
  if (sentences.length <= 2) {
    impression = activeText
  } else {
    impression = sentences[0] + (sentences[1].length < 60 ? ' ' + sentences[1] : '')
    findings = activeText.substring(impression.length).trim()
  }

  const rawNote = viewMode === 'original'
    ? 'Exact decoder output, including BART special tokens. It is almost identical to the AI Report because this system removes artefacts from the training targets rather than from the output - the model never learned to write "compared to the prior study". Fabricated prior-study references: 0.0000 across 4,722 test reports.'
    : null

  const handleCopy = () => {
    navigator.clipboard.writeText(activeText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="card panel report-container">
      <div className="panel-header" style={{ borderBottom: '1px solid var(--line)', paddingBottom: '16px', marginBottom: '20px' }}>
        <div>
          <div className="eyebrow" style={{ color: 'var(--accent)' }}>
            {viewMode === 'ground_truth' ? 'Radiologist Report' : 'Automated Analysis'}
          </div>
          <div className="panel-title" style={{ fontSize: '1.25rem', fontFamily: 'var(--font-display)' }}>
            {viewMode === 'ground_truth' ? 'Ground Truth (CSV)' : 'Radiology Report'}
          </div>
        </div>
        <button onClick={handleCopy} className="ghost-button" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {copied ? (
            <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12" /></svg> Copied</>
          ) : (
            <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg> Copy</>
          )}
        </button>
      </div>

      {/* Toggle: Cleaned / Original / Ground Truth */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', flexWrap: 'wrap' }}>
        <span className="report-label" style={{ margin: 0, fontSize: '0.72rem' }}>View</span>
        <div className="segmented">
          <button
            className={`segmented-btn ${viewMode === 'cleaned' ? 'active' : ''}`}
            onClick={() => setViewMode('cleaned')}
          >
            AI Report
          </button>
          <button
            className={`segmented-btn ${viewMode === 'original' ? 'active' : ''}`}
            onClick={() => setViewMode('original')}
          >
            Raw Output
          </button>
          {groundTruthReport && (
            <button
              className={`segmented-btn ${viewMode === 'ground_truth' ? 'active' : ''}`}
              onClick={() => setViewMode('ground_truth')}
              style={viewMode === 'ground_truth' ? { background: 'rgba(15, 118, 110, 0.1)', color: 'var(--accent)' } : {}}
            >
              ✦ Ground Truth
            </button>
          )}
        </div>
      </div>

      {viewMode === 'ground_truth' && (
        <div style={{ 
          fontSize: '0.75rem', color: 'var(--accent)', background: 'rgba(15,118,110,0.06)', 
          padding: '8px 12px', borderRadius: '8px', marginBottom: '16px' 
        }}>
          This is the original report written by the radiologist (from the training CSV). Use it to compare against the AI-generated report.
        </div>
      )}

      <div className="report-section impression-block">
        <div className="report-label">Impression</div>
        <div className="report-text impression-text">{impression}</div>
      </div>

      {findings && (
        <div className="report-section findings-block">
          <div className="report-label">Findings</div>
          <div className="report-text report-muted findings-text">{findings}</div>
        </div>
      )}
      {rawNote && (
        <p className="panel-note" style={{ marginTop: '14px', lineHeight: 1.55 }}>
          {rawNote}
        </p>
      )}

      {viewMode === 'original' && classifierPrompt && (
        <div style={{ marginTop: '12px' }}>
          <div className="eyebrow" style={{ marginBottom: '6px' }}>Classifier prompt fed to the decoder</div>
          <code style={{
            display: 'block', fontSize: '0.78rem', lineHeight: 1.5,
            background: 'var(--surface-2, #f5f5f7)', color: 'var(--muted)',
            padding: '10px 12px', borderRadius: 'var(--radius-md)',
            border: '1px solid var(--line)', whiteSpace: 'pre-wrap'
          }}>{classifierPrompt}</code>
        </div>
      )}

    </div>
  )
}
