import { useState } from 'react'
import { Panel } from './ui'

/**
 * The primary artefact. A cardiologist reads the trace first and the AI's
 * opinion second, so this gets the most space and sits above the interpretation.
 * Rendered server-side on standard ECG paper (25 mm/s, 10 mm/mV).
 */
export default function EcgViewer({ image, target, peaks }) {
  const [zoom, setZoom] = useState(false)
  if (!image) return null
  const src = `data:image/png;base64,${image}`

  return (
    <Panel
      title="12-lead electrocardiogram"
      right={
        <span className="flex items-center gap-3">
          {target && <span>attention overlay: {target}</span>}
          <button onClick={() => setZoom((z) => !z)}
            className="underline underline-offset-2 hover:text-ink-900
              dark:hover:text-ink-100">
            {zoom ? 'fit width' : 'full size'}
          </button>
          <a href={src} download="ecg.png"
            className="underline underline-offset-2 hover:text-ink-900
              dark:hover:text-ink-100">download</a>
        </span>
      }
      bodyClass="p-0"
    >
      <div className={zoom ? 'overflow-auto max-h-[80vh]' : 'overflow-x-auto'}>
        <img src={src} alt="12-lead ECG on standard paper"
          className={zoom ? 'max-w-none' : 'w-full min-w-250 block'} />
      </div>
      <div className="px-3 py-1.5 border-t border-ink-200 dark:border-ink-800
        flex flex-wrap justify-between gap-2 text-[10.5px] text-ink-500
        dark:text-ink-400">
        <span>
          Standard 3×4 layout with lead II rhythm strip · 25 mm/s · 10 mm/mV ·
          0.5–40 Hz · red shading marks the intervals that drove the decision
        </span>
        {peaks?.length > 0 && (
          <span className="font-mono">
            peak attention: {peaks.map((p) => `${p}s`).join(' · ')}
          </span>
        )}
      </div>
    </Panel>
  )
}
