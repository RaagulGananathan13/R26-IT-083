"use client";

/**
 * Load a study from the curated demo set.
 *
 * WHY THIS EXISTS
 * ---------------
 * The demo set was chosen so each file has a known answer: a clip picked to be
 * severe, a trace picked to be MI, a film picked to show cardiomegaly. Uploading
 * an unlabelled file demonstrates that the component produces *an* answer;
 * loading a labelled one demonstrates that it produces the *right* one, and a
 * reviewer can tell the difference without leaving the page.
 *
 * Samples are grouped by that expected class, because the class is the reason to
 * pick one. A flat list of forty filenames would make the reviewer do the
 * grouping in their head.
 *
 * The loaded bytes become a real `File` and travel the ordinary upload path, so
 * this exercises the same code an uploaded study does rather than a shortcut
 * around it.
 */
import { useState } from "react";

import { Button } from "@/components/ui";
import { classLabel, loadDemoSample, type DemoSample } from "@/lib/demo";
import { cn } from "@/lib/format";

export function SamplePicker({
  samples,
  onPick,
  disabled,
  emptyNote = "No demo set on disk. Build it with: python backend/scripts/build_demo_set.py",
}: {
  samples: DemoSample[];
  onPick: (files: File[], sample: DemoSample) => void;
  disabled?: boolean;
  emptyNote?: string;
}) {
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  if (samples.length === 0) {
    return <p className="font-mono text-2xs text-ink-faint">{emptyNote}</p>;
  }

  // Grouped by expected class, in the order the classes first appear, so the
  // grouping follows the demo set's own ordering rather than alphabetising it.
  const groups = new Map<string, DemoSample[]>();
  for (const sample of samples) {
    const list = groups.get(sample.klass) ?? [];
    list.push(sample);
    groups.set(sample.klass, list);
  }

  async function pick(sample: DemoSample) {
    setLoadingId(sample.id);
    setFailed(null);
    try {
      onPick(await loadDemoSample(sample), sample);
    } catch (cause) {
      setFailed(cause instanceof Error ? cause.message : "Could not load that sample.");
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="space-y-2.5">
      {[...groups.entries()].map(([klass, list]) => (
        <div key={klass} className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
          <span className="w-40 flex-none font-mono text-2xs uppercase tracking-wider text-ink-faint">
            {classLabel(klass)}
          </span>
          {list.map((sample) => (
            <Button
              key={sample.id}
              size="sm"
              variant="secondary"
              disabled={disabled || loadingId !== null}
              loading={loadingId === sample.id}
              onClick={() => pick(sample)}
              className={cn("font-mono text-2xs", loadingId === sample.id && "opacity-70")}
            >
              {sample.label.split("·").slice(1).join("·").trim() || sample.label}
            </Button>
          ))}
          {/* Stated up front so an early stop reads as the pathway working
              rather than as something having gone wrong. */}
          {list[0]?.expect && (
            <span className="w-full pl-40 text-2xs leading-relaxed text-ink-muted">
              {list[0].expect}
            </span>
          )}
        </div>
      ))}
      {failed && <p className="text-xs text-verdict-withheld">{failed}</p>}
    </div>
  );
}
