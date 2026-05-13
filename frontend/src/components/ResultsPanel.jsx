import GradCamViewer from './GradCamViewer'
import ReportViewer from './ReportViewer'

export default function ResultsPanel({ result, originalImg }) {
  if (!result) return null

  const { prediction, confidence, gradcam_image, report_text, report_text_raw, ground_truth_report, copathologies } = result
  const hasCardiomegaly = prediction === 'Cardiomegaly'

  // Format confidence to 1 decimal place
  const confPercent = (confidence * 100).toFixed(1)

  const statusText = hasCardiomegaly ? 'Cardiomegaly detected' : 'No cardiomegaly detected'
  const statusNote = hasCardiomegaly
    ? 'Review imaging and clinical context for confirmation.'
    : 'Findings do not indicate cardiomegaly.'

  // Only show Edema and Pleural Effusion as secondary findings
  const showOnly = ['Edema', 'Pleural Effusion']
  const filtered = copathologies?.filter(c => showOnly.includes(c.name)) || []
  const presentFindings = filtered.filter(c => c.status === 'present')
  const absentFindings = filtered.filter(c => c.status === 'absent')

  return (
    <div>
      {/* Main diagnosis card */}
      <div className="card summary-card rise-in-1">
        <div>
          <p className="eyebrow">Primary Diagnosis</p>
          <div className={`summary-title ${hasCardiomegaly ? 'tone-danger' : 'tone-ok'}`}>{statusText}</div>
          <p className="summary-sub">{statusNote}</p>
        </div>
        <div>
          <div className="metric-label">Confidence</div>
          <div className="metric-value">{confPercent}%</div>
          <div className="meter">
            <span
              className={`meter-fill ${hasCardiomegaly ? 'danger' : ''}`}
              style={{ width: `${confPercent}%` }}
            />
          </div>
          <div className="meter-scale">
            <span>0</span>
            <span>50</span>
            <span>100</span>
          </div>
        </div>
      </div>

      {/* Co-pathology findings */}
      {copathologies && copathologies.length > 0 && (
        <div className="card rise-in-1" style={{ marginTop: '16px', padding: '20px' }}>
          <p className="eyebrow" style={{ marginBottom: '14px', color: 'var(--accent)' }}>Secondary Findings (Multi-label Classifier)</p>
          <div className="copathology-grid">
            {presentFindings.map((c) => (
              <div key={c.name} className="copathology-chip present">
                <span className="copathology-dot present" />
                {c.name}
                {c.probability != null && <span className="copathology-prob">{(c.probability * 100).toFixed(0)}%</span>}
              </div>
            ))}
            {absentFindings.map((c) => (
              <div key={c.name} className="copathology-chip absent">
                <span className="copathology-dot absent" />
                {c.name}
                {c.probability != null && <span className="copathology-prob">{(c.probability * 100).toFixed(0)}%</span>}
              </div>
            ))}
          </div>
          {presentFindings.length === 0 && (
            <p style={{ fontSize: '0.82rem', color: 'var(--muted)', marginTop: '8px' }}>
              No additional pathologies detected by the classifier.
            </p>
          )}
        </div>
      )}

      <div className="grid-2 rise-in-2">
        <GradCamViewer originalImg={originalImg} heatmapImg={`data:image/png;base64,${gradcam_image}`} />
        <ReportViewer reportText={report_text} reportTextRaw={report_text_raw} groundTruthReport={ground_truth_report} />
      </div>
    </div>
  )
}
