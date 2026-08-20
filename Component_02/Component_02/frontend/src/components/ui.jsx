// Presentational primitives. No emoji, no gradients — colour means severity.

export const ZONE = {
  rule_out: { label: 'RULE OUT', cls: 'text-clear border-clear/40 bg-clear/8',
              bar: 'bg-clear' },
  refer:    { label: 'REFER',    cls: 'text-review border-review/40 bg-review/10',
              bar: 'bg-review' },
  rule_in:  { label: 'RULE IN',  cls: 'text-crit border-crit/40 bg-crit/10',
              bar: 'bg-crit' },
}

// Triage tiers use the same vocabulary an ED board would.
export const TRIAGE = {
  IMMEDIATE:     { bg: 'bg-crit',   fg: 'text-white', note: 'act now' },
  URGENT:        { bg: 'bg-urgent', fg: 'text-white', note: 'this episode of care' },
  PRIORITY:      { bg: 'bg-review', fg: 'text-white', note: 'same-day review' },
  ROUTINE:       { bg: 'bg-clear',  fg: 'text-white', note: 'no ECG-driven action' },
  'REPEAT ECG':  { bg: 'bg-ink-600', fg: 'text-white', note: 'not interpretable' },
}

export function Panel({ title, right, children, className = '', bodyClass = 'p-3' }) {
  return (
    <section className={`panel ${className}`}>
      {title && (
        <header className="panel-head">
          <span>{title}</span>
          {right && <span className="font-normal normal-case tracking-normal
            text-ink-500 dark:text-ink-400">{right}</span>}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  )
}

export function Tag({ children, tone = 'neutral', title }) {
  const tones = {
    neutral: 'border-ink-300 dark:border-ink-700 text-ink-600 dark:text-ink-300',
    clear:   'border-clear/40 text-clear bg-clear/8',
    review:  'border-review/40 text-review bg-review/10',
    crit:    'border-crit/40 text-crit bg-crit/10',
  }
  return <span className={`tag ${tones[tone]}`} title={title}>{children}</span>
}

export function Spinner({ className = 'size-4' }) {
  return (
    <span className={`${className} inline-block rounded-full border-2
      border-ink-300 dark:border-ink-700 border-t-ink-700 dark:border-t-ink-300
      animate-spin align-[-2px]`} />
  )
}

export function Empty({ title, children }) {
  return (
    <div className="flex flex-col items-center justify-center text-center gap-2
      py-28 px-6">
      <h2 className="text-[15px] font-semibold text-ink-700 dark:text-ink-200">
        {title}
      </h2>
      <p className="text-[12px] text-ink-500 dark:text-ink-400 max-w-md
        leading-relaxed">{children}</p>
    </div>
  )
}
