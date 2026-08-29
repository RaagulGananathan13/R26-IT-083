import { useState } from 'react'
import { Panel, Tag, ZONE } from './ui'

/**
 * Laid out the way a real ECG report reads: a single interpretation line, then
 * the findings that justify it, then the caveats. Not a chat transcript.
 */
export default function ReportCard({ data }) {
  const [raw, setRaw] = useState(false)
  const v = data.verification || {}
  const findings = data.findings || []
  const ruledIn = findings.filter((f) => f.zone === 'rule_in')
  const referred = findings.filter((f) => f.zone === 'refer')

  return (
    <Panel
      title="Clinical report"
      right={
        <button onClick={() => setRaw((s) => !s)}
          className="underline underline-offset-2 hover:text-ink-900
            dark:hover:text-ink-100">
          {raw ? 'structured' : 'plain text'}
        </button>
      }
      bodyClass="p-0"
    >
      {!v.passed && (
        <div className="px-3 py-2 bg-crit/10 border-b border-crit/30 text-[12px]
          text-crit">
          <b>Report withheld.</b> Failed automated verification:{' '}
          {(v.errors || []).join('; ')}
        </div>
      )}

      {raw ? (
        <pre className="whitespace-pre-wrap font-mono text-[11.5px] leading-[1.7]
          p-4 overflow-x-auto">{data.reportText}</pre>
      ) : (
        <div className="divide-y divide-ink-200 dark:divide-ink-800">
          {data.refused ? (
            <p className="p-4 text-[13px] leading-relaxed text-crit">
              {data.reportText}
            </p>
          ) : (
            <>
              <Block label="Interpretation">
                {ruledIn.length === 0 && referred.length === 0 ? (
                  <p className="text-[12px] text-ink-500">
                    No finding reached its rule-in threshold.
                  </p>
                ) : (
                  <ol className="space-y-2.5">
                    {[...ruledIn, ...referred].map((f, i) => {
                      const z = ZONE[f.zone] || ZONE.refer
                      return (
                        <li key={i} className="flex gap-2.5">
                          <span className={`tag ${z.cls} shrink-0 mt-px`}>
                            {z.label}
                          </span>
                          <div className="min-w-0">
                            <p className="text-[12.5px] leading-relaxed">
                              {f.sentence}
                            </p>
                            {f.evidence?.length > 0 && (
                              <ul className="mt-1">
                                {f.evidence.map((e, j) => (
                                  <li key={j} className="text-[10.5px] font-mono
                                    text-ink-500 dark:text-ink-400">— {e}</li>
                                ))}
                              </ul>
                            )}
                          </div>
                        </li>
                      )
                    })}
                  </ol>
                )}
              </Block>

              {data.ruledOut?.length > 0 && (
                <Block label="Excluded">
                  <div className="flex flex-wrap gap-1.5">
                    {data.ruledOut.map((c) => (
                      <Tag key={c} tone="clear">
                        {data.classFullNames?.[c] || c}
                      </Tag>
                    ))}
                  </div>
                </Block>
              )}

              {data.guarantees?.length > 0 && (
                <Block label="Statistical basis">
                  <ul className="space-y-1">
                    {data.guarantees.map((g, i) => (
                      <li key={i} className={`text-[11.5px] leading-relaxed
                        ${g.startsWith('STATISTICAL GUARANTEES SUSPENDED')
                          ? 'text-crit font-medium' : 'text-ink-600 dark:text-ink-300'}`}>
                        {g}
                      </li>
                    ))}
                  </ul>
                </Block>
              )}

              <Block label="Limitations">
                <ul className="space-y-0.5">
                  {(data.limitations || []).map((l, i) => (
                    <li key={i} className="text-[10.5px] leading-relaxed
                      text-ink-500 dark:text-ink-400">— {l}</li>
                  ))}
                </ul>
              </Block>
            </>
          )}
        </div>
      )}

      {(data.referenceReport || data.groundTruth) && (
        <div className="px-4 py-3 bg-ink-50 dark:bg-ink-950/50 border-t
          border-ink-200 dark:border-ink-800 space-y-1.5">
          <div className="text-[9px] uppercase tracking-[0.1em] font-bold
            text-ink-500">Reference (dataset ground truth — not shown in deployment)</div>
          {data.referenceReport && (
            <p className="text-[11.5px] text-ink-600 dark:text-ink-300 italic">
              “{data.referenceReport}”
            </p>
          )}
          {data.groundTruth && (
            <div className="flex flex-wrap items-center gap-1.5">
              {data.groundTruth.length === 0
                ? <Tag>none</Tag>
                : data.groundTruth.map((c) => <Tag key={c} tone="crit">{c}</Tag>)}
            </div>
          )}
        </div>
      )}

      <p className="px-4 py-2 text-[10px] leading-relaxed text-ink-500
        border-t border-ink-200 dark:border-ink-800">
        {data.disclaimer}
      </p>
    </Panel>
  )
}

function Block({ label, children }) {
  return (
    <div className="px-4 py-3">
      <h3 className="text-[9px] font-bold uppercase tracking-[0.11em]
        text-ink-500 mb-1.5">{label}</h3>
      {children}
    </div>
  )
}
