import { Panel, Tag, ZONE } from './ui'

/**
 * The decision table. Dense, scannable, monospace numerics — the layout a
 * reviewer reads in two seconds, not a set of cards they browse.
 *
 * The probability column shows where the patient sits between the two conformal
 * thresholds, because that position IS the decision.
 */
export function DecisionTable({ classes, delta }) {
  if (!classes?.length) return null
  return (
    <Panel
      title="Diagnostic decision"
      right={delta ? `conformal, PAC δ=${delta}` : 'no guarantee fitted'}
      bodyClass="p-0"
    >
      <table>
        <thead>
          <tr>
            <th className="w-[26%]">Finding</th>
            <th className="w-[13%] text-right">Prob.</th>
            <th className="w-[13%]">Decision</th>
            <th>Position between thresholds</th>
            <th className="w-[10%] text-right">Miss bound</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => {
            const z = ZONE[c.zone] || ZONE.refer
            const out = c.ruleOut ?? 0
            const inn = c.ruleIn ?? 100
            const pos = Math.min(Math.max(c.probability, 0), 100)
            return (
              <tr key={c.name} className="hover:bg-ink-50 dark:hover:bg-ink-800/40">
                <td>
                  <span className="font-semibold">{c.name}</span>
                  <span className="text-ink-500 dark:text-ink-400 ml-2 text-[11px]">
                    {c.fullName}
                  </span>
                </td>
                <td className="num font-semibold">{c.probability.toFixed(1)}%</td>
                <td>
                  <span className={`tag ${z.cls}`}>{z.label}</span>
                </td>
                <td>
                  <div className="relative h-4">
                    <div className="absolute inset-x-0 top-1.5 h-1 flex
                      bg-ink-200 dark:bg-ink-800">
                      <div className="bg-clear/45" style={{ width: `${out}%` }} />
                      <div className="bg-review/45"
                        style={{ width: `${Math.max(inn - out, 0)}%` }} />
                      <div className="bg-crit/45"
                        style={{ width: `${Math.max(100 - inn, 0)}%` }} />
                    </div>
                    {[out, inn].map((x, i) => (
                      <div key={i} className="absolute top-0 h-4 w-px
                        bg-ink-500 dark:bg-ink-400" style={{ left: `${x}%` }} />
                    ))}
                    <div className={`absolute top-0.5 -translate-x-1/2 w-[3px]
                      h-3 ${z.bar}`} style={{ left: `${pos}%` }} />
                  </div>
                  <div className="flex justify-between font-mono text-[9px]
                    text-ink-400 mt-0.5">
                    <span>{out.toFixed(0)}%</span>
                    <span>{inn.toFixed(0)}%</span>
                  </div>
                </td>
                <td className="num text-ink-500 dark:text-ink-400">
                  {c.alpha != null ? `≤ ${(c.alpha * 100).toFixed(0)}%` : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="px-3 py-2 text-[11px] text-ink-500 dark:text-ink-400
        border-t border-ink-200 dark:border-ink-800 leading-relaxed">
        Below the left threshold the finding is <b className="text-clear">ruled
        out</b> with a proven miss-rate bound. Above the right threshold it is
        <b className="text-crit"> ruled in</b>. Between them the system
        <b className="text-review"> declines to decide</b> and refers.
      </p>
    </Panel>
  )
}

/** Measurements + acquisition quality — what a reviewer checks before trusting. */
export function Measurements({ data }) {
  const q = data.quality || {}
  const hr = q.heart_rate_bpm
  const band = hr == null ? '—'
    : hr < 60 ? 'bradycardia' : hr > 100 ? 'tachycardia' : 'normal range'

  return (
    <Panel title="Measurements & acquisition" bodyClass="p-3">
      <dl className="space-y-0">
        <Row k="Ventricular rate" v={hr ? `${Math.round(hr)} bpm` : '—'} note={band} />
        <Row k="Complexes analysed" v={q.n_beats ?? '—'} />
        <Row k="Duration" v={q.duration_s ? `${Number(q.duration_s).toFixed(1)} s` : '—'} />
        <Row k="Signal quality index" v={Number(q.sqi ?? 0).toFixed(2)}
             note={q.sqi >= 0.8 ? 'good' : q.sqi >= 0.5 ? 'reduced' : 'poor'} />
        <Row k="Leads usable"
             v={`${12 - (q.flat_leads?.length || 0)} / 12`}
             note={q.flat_leads?.length ? q.flat_leads.join(', ') : null} />
        <Row k="Gain correction"
             v={q.gain_correction && q.gain_correction !== 1
                ? `×${q.gain_correction}` : 'none'} />
        <Row k="Intervals (PR/QRS/QT)" v="not measured" />
        <Row k="QRS axis" v="not measured" />
      </dl>

      {data.scope?.outOfScope && (
        <div className="mt-3 border border-crit/40 bg-crit/8 px-2.5 py-2">
          <div className="text-[11px] font-bold text-crit uppercase tracking-wide">
            Rhythm outside the model's five classes
          </div>
          <p className="text-[11px] mt-1 leading-relaxed text-ink-700
            dark:text-ink-300">
            {(data.scope.detail || []).join(' ')}
          </p>
          <p className="text-[11px] mt-1.5 font-semibold text-crit">
            Statistical guarantees withheld — no conformal bound covers a class
            outside the label space. An arrhythmia has NOT been excluded.
          </p>
        </div>
      )}

      {data.electrode?.suspected && (
        <div className="mt-3 border border-crit/40 bg-crit/8 px-2.5 py-2">
          <div className="text-[11px] font-bold text-crit uppercase tracking-wide">
            Possible {data.electrode.reversal} electrode reversal
          </div>
          <p className="text-[11px] mt-1 leading-relaxed text-ink-700
            dark:text-ink-300">
            {(data.electrode.reasons || []).join(' ')}
          </p>
          <p className="text-[11px] mt-1.5 font-semibold text-crit">
            Statistical guarantees are suspended for this record. Verify
            electrode placement and repeat.
          </p>
        </div>
      )}

      {(q.warnings || []).length > 0 && (
        <ul className="mt-3 space-y-1">
          {q.warnings.map((w, i) => (
            <li key={i} className="text-[11px] text-review leading-relaxed">
              • {w}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}

function Row({ k, v, note }) {
  return (
    <div className="kv border-b border-ink-100 dark:border-ink-800/70 last:border-0">
      <dt>{k}</dt>
      <dd>
        {v}
        {note && <span className="text-ink-400 ml-2 font-sans text-[10px]">
          {note}</span>}
      </dd>
    </div>
  )
}

/** Signed lead attribution, compact. Sign is preserved — see src/xai.py. */
export function LeadEvidence({ leads, explanation }) {
  if (!leads?.length) {
    return (
      <Panel title="Lead evidence" bodyClass="p-3">
        <p className="text-[12px] text-ink-500">No explanation generated.</p>
      </Panel>
    )
  }
  const max = Math.max(1, ...leads.map((l) => Math.abs(l.signed)))
  return (
    <Panel title="Lead evidence"
           right={explanation?.target ? `target: ${explanation.target}` : null}
           bodyClass="p-3">
      {explanation?.territory && (
        <div className="mb-2.5 pb-2.5 border-b border-ink-200 dark:border-ink-800">
          <div className="text-[11px] text-ink-500 dark:text-ink-400">
            Distribution
          </div>
          <div className="text-[13px] font-semibold capitalize">
            {explanation.territory}
          </div>
          <div className="text-[11px] text-ink-500 dark:text-ink-400">
            {explanation.artery}
          </div>
        </div>
      )}
      <div className="space-y-[3px]">
        {leads.map((l) => {
          const w = (Math.abs(l.signed) / max) * 50
          const pos = l.signed >= 0
          return (
            <div key={l.name} className="flex items-center gap-2">
              <span className="w-8 text-right font-mono text-[10px]
                text-ink-500 dark:text-ink-400">{l.name}</span>
              <div className="flex-1 h-3 relative">
                <div className="absolute inset-y-0 left-1/2 w-px
                  bg-ink-300 dark:bg-ink-700" />
                <div className={`absolute top-0.5 h-2 ${pos ? 'bg-crit' : 'bg-clear'}`}
                  style={pos ? { left: '50%', width: `${w}%` }
                             : { right: '50%', width: `${w}%` }} />
              </div>
              <span className={`w-11 num text-[10px] ${pos ? 'text-crit' : 'text-clear'}`}>
                {pos ? '+' : ''}{l.signed}%
              </span>
            </div>
          )
        })}
      </div>
      <div className="flex gap-3 mt-2.5 pt-2 border-t border-ink-200
        dark:border-ink-800 text-[10px] text-ink-500 dark:text-ink-400">
        <span><span className="inline-block w-2 h-2 bg-crit mr-1" />supports</span>
        <span><span className="inline-block w-2 h-2 bg-clear mr-1" />argues against</span>
      </div>
    </Panel>
  )
}
