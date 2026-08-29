/**
 * Reliability notice — surfaces the measured AP/PA performance gap.
 *
 * The classifier scores AUROC 0.8224 on AP (bedside) films versus 0.8864 on PA
 * (standing) — a gap of 0.0639, significant across 7 of 8 pathologies. AP films
 * come from the sickest patients, so the system is weakest exactly where it
 * matters most. We measured that this gap survives every model-side
 * intervention we tried, so the honest thing is to state it rather than hide it.
 */
export default function ReliabilityNotice({ reliability, threshold, thresholdSource }) {
  if (!reliability) return null

  const tone = {
    reduced: { color: 'var(--danger)', label: 'Reduced reliability' },
    standard: { color: 'var(--accent)', label: 'Standard reliability' },
    unknown: { color: 'var(--muted)', label: 'Projection not specified' },
  }[reliability.level] || { color: 'var(--muted)', label: 'Reliability unknown' }

  return (
    <div
      className="card rise-in-1"
      style={{ marginTop: '16px', padding: '18px 20px', borderLeft: `3px solid ${tone.color}` }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <span
          className="status-dot"
          style={{ background: tone.color, width: '9px', height: '9px', borderRadius: '50%', display: 'inline-block' }}
        />
        <span className="eyebrow" style={{ color: tone.color, margin: 0 }}>{tone.label}</span>

        {reliability.view && (
          <span className="pill" style={{ fontSize: '0.72rem' }}>{reliability.view}</span>
        )}
        {reliability.measured_auroc != null && (
          <span style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
            measured AUROC {reliability.measured_auroc.toFixed(4)}
          </span>
        )}
        {threshold != null && (
          <span style={{ fontSize: '0.75rem', color: 'var(--muted)', marginLeft: 'auto' }}>
            operating point {threshold.toFixed(3)}
            {thresholdSource ? ` (${thresholdSource})` : ''}
          </span>
        )}
      </div>

      <p className="panel-note" style={{ marginTop: '10px', marginBottom: 0, lineHeight: 1.55 }}>
        {reliability.message}
      </p>
    </div>
  )
}
