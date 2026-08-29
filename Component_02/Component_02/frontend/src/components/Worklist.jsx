import { useEffect, useRef, useState } from 'react'
import { Spinner, Tag } from './ui'

const ORDER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

/**
 * A worklist, not a navigation menu. Modelled on the study list in a PACS or
 * ECG management system: filter on the left, dense rows, one click to open.
 */
export default function Worklist({
  health, activeClass, onSelectClass, patients, loading,
  activeId, onSelect, onRandom, onUpload, busy,
}) {
  const [q, setQ] = useState('')
  const [files, setFiles] = useState({ dat: null, hea: null })
  const [drag, setDrag] = useState(false)
  const input = useRef(null)

  useEffect(() => setQ(''), [activeClass])

  const rows = patients.filter(
    (p) => !q || String(p.ecg_id).includes(q) ||
           (p.report_en || '').toLowerCase().includes(q.toLowerCase()))

  function take(list) {
    const n = { ...files }
    for (const f of list) {
      const s = f.name.toLowerCase()
      if (s.endsWith('.dat')) n.dat = f
      else if (s.endsWith('.hea')) n.hea = f
    }
    setFiles(n)
  }

  return (
    <aside className="w-[300px] shrink-0 flex flex-col h-screen border-r
      border-ink-300 dark:border-ink-800 bg-white dark:bg-ink-900">
      <div className="px-3 py-2.5 border-b border-ink-300 dark:border-ink-800">
        <div className="text-[12px] font-bold tracking-tight">
          ECG Review — Component 02
        </div>
        <div className="text-[10px] text-ink-500 dark:text-ink-400 mt-0.5">
          {health ? `${health.nTest} studies · ${health.model}` : 'connecting…'}
        </div>
      </div>

      <div className="panel-head border-t-0">Worklist filter</div>
      <div className="p-2 grid grid-cols-1 gap-px bg-ink-200 dark:bg-ink-800">
        {ORDER.map((c) => {
          const on = activeClass === c
          return (
            <button key={c} onClick={() => onSelectClass(c)}
              className={`flex items-center justify-between px-2.5 py-1.5
                text-left text-[12px] transition
                ${on ? 'bg-ink-800 dark:bg-ink-100 text-white dark:text-ink-900 font-semibold'
                     : 'bg-white dark:bg-ink-900 hover:bg-ink-50 dark:hover:bg-ink-800'}`}>
              <span className="truncate">
                <span className="font-mono font-semibold mr-2">{c}</span>
                <span className={on ? 'opacity-80' : 'text-ink-500 dark:text-ink-400'}>
                  {health?.classFullNames?.[c]}
                </span>
              </span>
              <span className="num text-[10px] opacity-80">
                {health?.counts?.[c] ?? '—'}
              </span>
            </button>
          )
        })}
      </div>

      {activeClass && (
        <div className="px-2 py-1.5 border-b border-ink-200 dark:border-ink-800">
          <input value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Filter by ID or report…"
            className="w-full text-[11px] px-2 py-1 bg-ink-50 dark:bg-ink-950
              border border-ink-300 dark:border-ink-700 outline-none
              focus:border-ink-600" />
        </div>
      )}

      <div className="flex-1 overflow-y-auto min-h-0">
        {!activeClass && (
          <p className="text-[11px] text-ink-500 text-center py-8 px-4">
            Select a finding above to load the worklist.
          </p>
        )}
        {loading && (
          <p className="text-[11px] text-ink-500 text-center py-6">
            <Spinner /> loading
          </p>
        )}
        {rows.map((p) => {
          const on = String(activeId) === String(p.ecg_id)
          return (
            <button key={p.ecg_id} onClick={() => onSelect(p.ecg_id)} disabled={busy}
              className={`w-full text-left px-2.5 py-1.5 border-b border-ink-100
                dark:border-ink-800/70 disabled:opacity-50
                ${on ? 'bg-ink-100 dark:bg-ink-800'
                     : 'hover:bg-ink-50 dark:hover:bg-ink-800/50'}`}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[12px] font-semibold">
                  {p.ecg_id}
                </span>
                <span className="font-mono text-[10px] text-ink-500">
                  {p.age ? `${p.age}y` : '—'} {p.sex || ''}
                </span>
              </div>
              <div className="text-[10.5px] text-ink-500 dark:text-ink-400
                truncate">{p.report_en || '—'}</div>
            </button>
          )
        })}
      </div>

      <div className="border-t border-ink-300 dark:border-ink-800 p-2 space-y-1.5">
        <button onClick={onRandom} disabled={busy}
          className="w-full text-[11px] py-1.5 border border-ink-300
            dark:border-ink-700 hover:border-ink-600 disabled:opacity-50">
          Open a random study
        </button>

        <div onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); take(e.dataTransfer.files) }}
          onClick={() => input.current?.click()}
          className={`border border-dashed px-2 py-3 text-center cursor-pointer
            ${drag ? 'border-ink-600 bg-ink-50 dark:bg-ink-800'
                   : 'border-ink-300 dark:border-ink-700 hover:border-ink-500'}`}>
          <p className="text-[11px]">Import WFDB study</p>
          <p className="text-[9.5px] text-ink-500 mt-0.5">.dat + .hea</p>
          <input ref={input} type="file" multiple accept=".dat,.hea"
            className="hidden" onChange={(e) => take(e.target.files)} />
          <div className="flex gap-1 justify-center flex-wrap mt-1.5">
            {files.dat && <Tag>{files.dat.name}</Tag>}
            {files.hea && <Tag>{files.hea.name}</Tag>}
          </div>
        </div>

        {files.dat && files.hea && (
          <button onClick={() => onUpload(files.dat, files.hea)} disabled={busy}
            className="w-full text-[11px] py-1.5 bg-ink-800 dark:bg-ink-100
              text-white dark:text-ink-900 font-semibold disabled:opacity-50">
            Analyse imported study
          </button>
        )}
      </div>
    </aside>
  )
}
