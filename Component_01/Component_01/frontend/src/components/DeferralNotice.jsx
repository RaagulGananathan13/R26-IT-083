/**
 * Deferral notice — Stage 13.
 *
 * Shown when the prediction sits too close to this projection's decision
 * threshold to be actionable. The model's answer is still displayed above; this
 * marks it as not safe to act on rather than hiding it, because a clinician
 * needs to see what the system thought in order to judge the referral.
 *
 * The cutoff is projection-specific by design: AP (bedside) films are deferred
 * at a much higher rate than PA, because the measured AP/PA accuracy gap is
 * information lost at acquisition and cannot be recovered by the model. A single
 * global cutoff leaves that gap intact (6.68 -> 6.28 points); the per-projection
 * policy closes it (6.68 -> 0.78).
 */
export default function DeferralNotice({ deferral }) {
  if (!deferral || !deferral.active || !deferral.defer) return null

  const m = deferral.measured

  return (
    <div
      className="card rise-in-1"
      style={{
        marginTop: '16px',
        padding: '18px 20px',
        borderLeft: '3px solid var(--warning, #D55E00)',
        background: 'color-mix(in srgb, var(--warning, #D55E00) 6%, transparent)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <span aria-hidden="true" style={{ fontSize: '1rem' }}>⚠️</span>
        <span className="eyebrow" style={{ color: 'var(--warning, #D55E00)', margin: 0 }}>
          Uncertain — refer to radiologist
        </span>
        {deferral.view && (
          <span className="pill" style={{ fontSize: '0.72rem' }}>{deferral.view}</span>
        )}
        {deferral.margin != null && (
          <span style={{ fontSize: '0.75rem', color: 'var(--muted)', marginLeft: 'auto' }}>
            margin {deferral.margin.toFixed(3)} &lt; cutoff {deferral.cutoff.toFixed(3)}
          </span>
        )}
      </div>

      <p className="panel-note" style={{ marginTop: '10px', marginBottom: 0, lineHeight: 1.55 }}>
        {deferral.reason} The prediction above is shown for transparency but should
        not be acted on without review.
      </p>

      {m && (
        <p
          className="panel-note"
          style={{
            marginTop: '10px',
            marginBottom: 0,
            fontSize: '0.72rem',
            color: 'var(--muted)',
            lineHeight: 1.5,
          }}
        >
          Measured on {`${m.coverage}%`} coverage: accuracy {m.accuracy}%, sensitivity{' '}
          {m.sensitivity}%, AP/PA gap {m.gap} points. This policy answers{' '}
          {m.coverage_ap}% of AP films and {m.coverage_pa}% of PA films — accuracy
          figures apply only to answered cases.
        </p>
      )}
    </div>
  )
}
