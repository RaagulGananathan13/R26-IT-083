"use client";

import { useState } from "react";

import { Card, CardBody, CardHeader } from "@/components/ui";
import { humanise } from "@/lib/format";
import type { ModelCard } from "@/lib/types";

/**
 * Provenance and limits, carried with every result.
 *
 * Limitations are rendered open by default rather than hidden behind a link.
 * Every component in this project publishes what it cannot do, and a console
 * that tucks that away undoes the work.
 */
export function ModelCardPanel({ model }: { model: ModelCard }) {
  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <Card>
      <CardHeader
        title="Model card"
        description={`${model.component_name} — ${model.owner}`}
      />
      <CardBody className="space-y-4">
        <dl className="grid gap-x-6 gap-y-2 text-xs sm:grid-cols-2">
          <Row label="Modality" value={model.modality} />
          <Row label="Task" value={model.task} />
          <Row label="Dataset" value={model.dataset} />
          <Row label="Architecture" value={model.architecture} />
          {model.decision_rule && (
            <div className="sm:col-span-2">
              <dt className="eyebrow">Decision rule applied</dt>
              <dd className="mt-0.5 leading-relaxed text-ink-muted">{model.decision_rule}</dd>
            </div>
          )}
        </dl>

        <div>
          <p className="eyebrow mb-1.5">Stated limitations</p>
          <ul className="space-y-1.5">
            {model.limitations.map((limitation, index) => (
              <li key={index} className="flex gap-2 text-xs leading-relaxed text-ink-muted">
                <span aria-hidden className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-faint" />
                <span>{limitation}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowMetrics((open) => !open)}
            className="text-xs font-medium text-ink-muted underline underline-offset-2 hover:text-ink"
          >
            {showMetrics ? "Hide" : "Show"} published test-set metrics
          </button>
          {showMetrics && (
            <pre className="mt-2 max-h-72 overflow-auto scrollbar-thin rounded-lg bg-surface-2 p-3 text-2xs leading-relaxed text-ink-muted">
              {JSON.stringify(model.metrics, null, 2)}
            </pre>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="eyebrow">{humanise(label)}</dt>
      <dd className="mt-0.5 leading-relaxed text-ink-muted">{value}</dd>
    </div>
  );
}
