"use client";

import { Card, CardBody, CardHeader } from "@/components/ui";
import { cn, decimal } from "@/lib/format";
import type { Envelope } from "@/lib/types";
import { isActionable } from "@/lib/types";

const BOUNDARIES = [30, 40, 55];
const BANDS = [
  { from: 0, to: 30, label: "Severe" },
  { from: 30, to: 40, label: "Moderate" },
  { from: 40, to: 55, label: "Mild" },
  { from: 55, to: 100, label: "Normal" },
];

/**
 * Ejection fraction on its clinical scale.
 *
 * Component 03 produces no saliency map — there is nothing spatial to attribute
 * — so its explanation is the interval and where it falls against the severity
 * boundaries. That is the right thing to draw, because boundary proximity is
 * this component's whole story: over a third of studies sit within one MAE of a
 * threshold, and the single crowded boundary at EF 55 produces 68 % of all its
 * errors. A number alone hides that; a number on the scale shows it.
 */
export function EchoViewer({
  result,
  videoUrl,
}: {
  result: Envelope;
  /** Object URL of the uploaded clip, when the browser can play it. */
  videoUrl: string | null;
}) {
  const raw = result.raw ?? {};
  const ef = Number(raw.ef_calibrated ?? 0);
  const interval = (raw.ef_interval_95 ?? null) as number[] | null;
  const grade = String(raw.severity_class ?? "");
  const clinical = String(raw.clinical_reference_class ?? "");
  const uncertainty = raw.uncertainty ?? {};
  const members = (raw.ensemble ?? []) as { run: string; ef: number }[];
  const muted = !isActionable(result.reliability.actionability);
  // Present only for bundled EchoNet studies, where a human tracing exists.
  const truth = raw.ground_truth_ef as
    | { ef: number; predicted_ef: number; absolute_error: number;
        within_interval: boolean; grade?: string; source?: string }
    | undefined;
  const position = (value: number) => `${Math.min(100, Math.max(0, value))}%`;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader
          title="Ejection fraction"
          description="Operational grade against the clinical reference. The two answer different questions: published boundaries score higher overall accuracy while abandoning the minority classes."
        />
        <CardBody className={cn("space-y-6", muted && "not-actionable")}>
          {/* The measured EF beside the estimated one, for bundled EchoNet
              studies. Shown only where a human tracing exists -- an uploaded
              clip has no reference, and a number beside a prediction is read as
              the answer whether or not it belongs to that patient. */}
          {truth && (
            <div className="grid gap-3 rounded-xl border border-line bg-surface-2 px-4 py-3 sm:grid-cols-3">
              <div>
                <p className="eyebrow">Measured EF</p>
                <p className="tabular mt-0.5 text-lg font-semibold text-ink">
                  {decimal(truth.ef, 1)} %
                </p>
              </div>
              <div className="sm:border-l sm:border-line sm:pl-4">
                <p className="eyebrow">Model estimate</p>
                <p className="tabular mt-0.5 text-lg font-semibold text-ink">
                  {decimal(truth.predicted_ef, 1)} %
                </p>
              </div>
              <div className="sm:border-l sm:border-line sm:pl-4">
                <p className="eyebrow">Difference</p>
                <p
                  className={cn(
                    "tabular mt-0.5 text-lg font-semibold",
                    truth.within_interval ? "text-verdict-actionable" : "text-verdict-caution",
                  )}
                >
                  {decimal(truth.absolute_error, 1)} pts
                </p>
                <p className="mt-0.5 text-2xs text-ink-faint">
                  {truth.within_interval
                    ? "inside the 95 % interval"
                    : "outside the 95 % interval"}
                </p>
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-baseline gap-3">
            <span className="tabular display text-5xl leading-none text-ink">
              {decimal(ef, 1)}
            </span>
            <span className="text-lg text-ink-muted">%</span>
            {interval && interval.length === 2 && (
              <span className="tabular text-xs text-ink-faint">
                95 % interval {decimal(interval[0], 1)} – {decimal(interval[1], 1)}
              </span>
            )}
          </div>

          <div>
            <div className="relative h-12">
              {/* Severity bands, so the reader sees which side of a line they are on. */}
              <div className="absolute inset-x-0 top-4 flex h-3 overflow-hidden rounded-full">
                {BANDS.map((band, index) => (
                  <span
                    key={band.label}
                    className={cn(
                      "h-full",
                      index === 0 && "bg-verdict-withheld/25",
                      index === 1 && "bg-verdict-caution/25",
                      index === 2 && "bg-accent/20",
                      index === 3 && "bg-accent/35",
                    )}
                    style={{ width: `${band.to - band.from}%` }}
                    title={band.label}
                  />
                ))}
              </div>

              {interval && interval.length === 2 && (
                <div
                  className="absolute top-4 h-3 rounded-full bg-ink/30"
                  style={{
                    left: position(interval[0]!),
                    width: position(Math.max(0.5, interval[1]! - interval[0]!)),
                  }}
                  aria-hidden
                />
              )}

              {BOUNDARIES.map((boundary) => (
                <div
                  key={boundary}
                  className="absolute top-1"
                  style={{ left: position(boundary) }}
                >
                  <div className="h-9 w-px bg-ink/25" aria-hidden />
                  <span className="absolute -left-2 top-9 text-2xs text-ink-faint">
                    {boundary}
                  </span>
                </div>
              ))}

              <div
                className="absolute top-2 h-7 w-1 -translate-x-1/2 rounded-full bg-ink shadow"
                style={{ left: position(ef) }}
                aria-hidden
              />
            </div>
            <div className="mt-5 flex justify-between text-2xs text-ink-faint">
              {BANDS.map((band) => (
                <span key={band.label}>{band.label}</span>
              ))}
            </div>
          </div>

          <dl className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-accent/30 bg-accent/5 px-4 py-3">
              <dt className="eyebrow">Operational grade</dt>
              <dd className="mt-1 text-sm font-semibold text-ink">{grade}</dd>
            </div>
            <div className="rounded-xl border border-line px-4 py-3">
              <dt className="eyebrow">Clinical reference</dt>
              <dd className="mt-1 text-sm text-ink-muted">{clinical}</dd>
            </div>
          </dl>
        </CardBody>
      </Card>

      {videoUrl && (
        <Card>
          <CardHeader
            title="Study"
            description="Playback of the uploaded recording. The model samples clips across the whole window, label-free — the annotated end-diastole and end-systole frames are never consulted."
          />
          <CardBody>
            <video
              src={videoUrl}
              controls
              loop
              muted
              playsInline
              className="mx-auto block max-h-[26rem] w-full max-w-[36rem] rounded-xl border border-line bg-black object-contain"
            />
          </CardBody>
        </Card>
      )}

      <Card>
        <CardHeader
          title="Uncertainty"
          description={String(uncertainty.note ?? "")}
        />
        <CardBody className="space-y-4">
          <dl className="grid gap-4 sm:grid-cols-3">
            <Stat label="Aleatoric σ" value={uncertainty.aleatoric_ef_std} />
            <Stat label="Epistemic σ" value={uncertainty.epistemic_ef_std} />
            <Stat label="Total σ" value={uncertainty.total_ef_std} />
          </dl>
          {members.length > 0 && (
            <div className="border-t border-line pt-3">
              <p className="eyebrow mb-2">
                Ensemble members · {raw.tta_clips} clips each
              </p>
              <div className="flex flex-wrap gap-2">
                {members.map((member) => (
                  <span key={member.run} className="pill tabular">
                    <span className="font-mono text-2xs text-ink-faint">{member.run}</span>
                    {decimal(member.ef, 1)} %
                  </span>
                ))}
              </div>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="tabular mt-1 text-xl font-semibold text-ink">
        {typeof value === "number" ? decimal(value, 2) : "—"}
        <span className="ml-1 text-2xs font-normal text-ink-faint">EF pts</span>
      </dd>
    </div>
  );
}
