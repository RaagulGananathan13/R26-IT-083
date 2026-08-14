import { TRIAGE, Tag } from './ui'

/**
 * The patient identification band. Every clinical review system has one, it is
 * always at the top, and it never scrolls away — so the reviewer can never be
 * looking at the wrong patient's data.
 */
export default function PatientBanner({ data }) {
  const t = TRIAGE[data.triage] || TRIAGE['REPEAT ECG']
  const q = data.quality || {}
  const pt = data.patient || {}
  const v = data.verification || {}

  return (
    <div className="border-b border-ink-300 dark:border-ink-800 bg-white
      dark:bg-ink-900">
      <div className="flex items-stretch">
        {/* severity spine */}
        <div className={`w-1.5 shrink-0 ${t.bg}`} />

        <div className="flex-1 px-4 py-2.5 flex flex-wrap items-center gap-x-8
          gap-y-2">
          <Field label="Record">
            <span className="font-mono text-[15px] font-semibold">
              {data.patientId ?? '—'}
            </span>
          </Field>
          <Field label="Age / Sex">
            <span className="font-mono">
              {pt.age != null ? `${pt.age} y` : '—'}{'  '}
              {pt.sex || ''}
            </span>
          </Field>
          <Field label="Rate">
            <span className="font-mono">
              {q.heart_rate_bpm ? `${Math.round(q.heart_rate_bpm)} bpm` : '—'}
            </span>
          </Field>
          <Field label="Recording">
            <span className="font-mono">
              {q.duration_s ? `${Number(q.duration_s).toFixed(1)} s` : '—'} ·
              12-lead · 500 Hz
            </span>
          </Field>
          <Field label="Interpretation">
            <span className="font-medium">
              {data.refused ? 'Not interpretable' : (data.headline || '—')}
            </span>
          </Field>

          <div className="ml-auto flex items-center gap-2">
            {data.electrode?.suspected && (
              <Tag tone="crit" title={(data.electrode.reasons || []).join(' ')}>
                {data.electrode.reversal} lead reversal?
              </Tag>
            )}
            {!v.passed && <Tag tone="crit">report withheld</Tag>}
            <div className={`${t.bg} ${t.fg} px-3 py-1.5 text-center`}>
              <div className="text-[15px] font-bold tracking-wide leading-none">
                {data.triage}
              </div>
              <div className="text-[9px] uppercase tracking-[0.1em] opacity-80
                mt-0.5">{t.note}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Field({ label, children }) {
  return (
    <div className="min-w-0">
      <div className="text-[9px] uppercase tracking-[0.1em] text-ink-500
        dark:text-ink-400 font-semibold">{label}</div>
      <div className="text-[13px] mt-0.5 truncate">{children}</div>
    </div>
  )
}
