import { useCallback, useEffect, useState } from 'react'
import { analyzeById, analyzeRandom, analyzeUpload, getHealth, getPatients } from './api'
import Worklist from './components/Worklist'
import PatientBanner from './components/PatientBanner'
import { DecisionTable, LeadEvidence, Measurements } from './components/Interpretation'
import ReportCard from './components/ReportCard'
import EcgViewer from './components/EcgViewer'
import { Empty, Spinner } from './components/ui'

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthErr, setHealthErr] = useState(null)
  const [cls, setCls] = useState(null)
  const [patients, setPatients] = useState([])
  const [loadingList, setLoadingList] = useState(false)
  const [activeId, setActiveId] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  // Light by default. Clinical review happens under room lighting on a
  // calibrated display; dark mode is a preference, not the identity.
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setHealthErr(e.message))
  }, [])

  const selectClass = useCallback(async (c) => {
    setCls(c); setLoadingList(true)
    try {
      setPatients((await getPatients(c)).patients || [])
    } catch (e) {
      setErr(e.message); setPatients([])
    } finally {
      setLoadingList(false)
    }
  }, [])

  async function run(fn, id = null) {
    setBusy(true); setErr(null); setActiveId(id)
    try {
      const d = await fn()
      setResult(d); setActiveId(d.patientId ?? id)
    } catch (e) {
      setErr(e.message); setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Worklist
        health={health} activeClass={cls} onSelectClass={selectClass}
        patients={patients} loading={loadingList} activeId={activeId}
        onSelect={(id) => run(() => analyzeById(id), id)}
        onRandom={() => run(analyzeRandom)}
        onUpload={(d, h) => run(() => analyzeUpload(d, h))}
        busy={busy}
      />

      <main className="flex-1 overflow-y-auto flex flex-col">
        {result && !busy && <div className="sticky top-0 z-20">
          <PatientBanner data={result} />
        </div>}

        <div className="px-4 py-3 flex-1">
          {health?.disclaimer && (
            <div className="mb-3 border-l-2 border-review bg-review/8 px-3 py-1.5
              text-[11px] text-ink-700 dark:text-ink-300">
              {health.disclaimer}
            </div>
          )}

          {healthErr && (
            <Empty title="Cannot reach the analysis service">
              {healthErr}
              <br />
              <span className="font-mono text-[11px]">
                python -X utf8 backend/server.py
              </span>
            </Empty>
          )}

          {err && (
            <div className="mb-3 border-l-2 border-crit bg-crit/8 px-3 py-1.5
              text-[11.5px] text-crit">{err}</div>
          )}

          {busy && (
            <div className="flex flex-col items-center gap-2 py-28">
              <Spinner className="size-5" />
              <p className="text-[11.5px] text-ink-500">
                Quality gate → classification → calibration → conformal triage →
                explanation → report → verification
              </p>
            </div>
          )}

          {!busy && !result && !healthErr && (
            <Empty title="No study open">
              Select a finding in the worklist to load studies from the held-out
              test fold, open a random study, or import a WFDB <b>.dat</b> +{' '}
              <b>.hea</b> pair. Studies that fail acquisition quality control are
              refused rather than interpreted.
            </Empty>
          )}

          {!busy && result && (
            <div className="space-y-3">
              {/* The trace comes first — it is the primary clinical artefact. */}
              <EcgViewer image={result.ecgImage} target={result.gradcamTarget}
                peaks={result.explanation?.peaksSeconds} />

              {!result.refused && (
                <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-3">
                  <DecisionTable classes={result.classes} delta={health?.delta} />
                  <Measurements data={result} />
                </div>
              )}

              <div className="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-3">
                <ReportCard data={result} />
                {!result.refused && (
                  <LeadEvidence leads={result.leads}
                    explanation={result.explanation} />
                )}
              </div>
            </div>
          )}
        </div>

        <footer className="px-4 py-1.5 border-t border-ink-200 dark:border-ink-800
          flex items-center gap-4 text-[10px] text-ink-500 dark:text-ink-400">
          <span>
            {health ? `${health.checkpoint} · conformal δ=${health.delta}` : ''}
          </span>
          <button onClick={() => setDark((d) => !d)}
            className="ml-auto underline underline-offset-2
              hover:text-ink-900 dark:hover:text-ink-100">
            {dark ? 'light theme' : 'dark theme'}
          </button>
        </footer>
      </main>
    </div>
  )
}
