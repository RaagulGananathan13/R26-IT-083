"use client";

import { useEffect, useState } from "react";

import { Card, CardBody, CardHeader, Hero } from "@/components/ui";
import { useHealth } from "@/hooks/useHealth";
import { getCohorts } from "@/lib/api";
import { cn, humanise } from "@/lib/format";
import type { CohortReport } from "@/lib/types";

/**
 * Why the multi-modal view aggregates rather than fuses.
 *
 * The claim "no cohort carries all four modalities for the same patient" is
 * load-bearing, so it is measured on the running install and shown here rather
 * than asserted in prose.
 */
export default function AboutPage() {
  const [report, setReport] = useState<CohortReport | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    getCohorts().then(setReport).catch(setError);
  }, []);

  return (
    <div className="mx-auto max-w-4xl">
      <Hero
        eyebrow="Method"
        title="Method and cohorts"
        subtitle="Four components, four datasets, four independent validations. This page records what that does and does not permit."
        className="mb-8"
      />

      <div className="space-y-6">
        <Card>
          <CardHeader title="What integration means here" />
          <CardBody className="space-y-3 text-sm leading-relaxed text-ink-muted">
            <p>
              One service, one response contract, and one reliability verdict spanning four
              modalities. Each component keeps its own weights, its own frozen decision
              rule and its own published figures; the service reproduces them unchanged and
              adds no probabilities of its own.
            </p>
            <p>
              The multi-modal view therefore <span className="font-medium text-ink">aggregates</span>.
              It runs the modalities supplied, takes the worst verdict, and reports agreement
              and disagreement as traceable observations. No joint model was trained across
              modalities and no combined accuracy is claimed.
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Why fusion is not merely undone but impossible"
            description={report?.source ? `Source: ${report.source}` : undefined}
          />
          <CardBody className="space-y-4">
            {error ? (
              <p className="text-sm text-ink-muted">
                Could not reach the backend to load the measurement.
              </p>
            ) : null}
            {report && (
              <>
                <p className="text-sm leading-relaxed text-ink-muted">{report.conclusion}</p>

                {report.pairs && (
                  <div className="overflow-x-auto scrollbar-thin">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-line text-left">
                          <th className="eyebrow py-2 pr-3 font-semibold">Pair</th>
                          <th className="eyebrow py-2 pr-3 font-semibold">Linkable</th>
                          <th className="eyebrow py-2 pr-3 text-right font-semibold">
                            Shared patients
                          </th>
                          <th className="eyebrow py-2 font-semibold">Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(report.pairs).map(([pair, detail]) => (
                          <tr key={pair} className="border-b border-line/60 last:border-0">
                            <td className="py-2 pr-3 font-medium text-ink">
                              {pair.replace("+", " + ")}
                            </td>
                            <td className="py-2 pr-3">
                              <span
                                className={cn(
                                  "text-xs font-semibold",
                                  detail.linkable
                                    ? "text-verdict-actionable"
                                    : "text-ink-faint",
                                )}
                              >
                                {detail.linkable ? "yes" : "no"}
                              </span>
                            </td>
                            <td className="tabular py-2 pr-3 text-right text-ink-muted">
                              {detail.shared_patients.toLocaleString()}
                            </td>
                            <td className="py-2 text-xs leading-relaxed text-ink-faint">
                              {detail.reason}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {report.pairs?.["cxr+triage"]?.caveat && (
                  <div className="rounded-lg border border-verdict-caution/30 bg-verdict-caution/10 px-3 py-2.5">
                    <p className="eyebrow mb-1">On the one linkable pair</p>
                    <p className="text-xs leading-relaxed text-ink-muted">
                      {report.pairs["cxr+triage"].caveat}
                    </p>
                  </div>
                )}
              </>
            )}
          </CardBody>
        </Card>

        {report?.cohorts && (
          <Card>
            <CardHeader title="Dataset provenance" />
            <CardBody>
              <div className="grid gap-4 sm:grid-cols-2">
                {Object.entries(report.cohorts).map(([id, cohort]) => (
                  <div key={id} className="rounded-lg border border-line px-3 py-2.5">
                    <p className="text-sm font-semibold text-ink">
                      {String(cohort.dataset ?? id)}
                    </p>
                    <dl className="mt-1.5 space-y-0.5">
                      {Object.entries(cohort)
                        .filter(([key]) => key !== "dataset")
                        .map(([key, value]) => (
                          <div key={key} className="flex gap-2 text-2xs">
                            <dt className="shrink-0 text-ink-faint">{humanise(key)}</dt>
                            <dd className="text-ink-muted">
                              {value === null ? "not published" : String(value)}
                            </dd>
                          </div>
                        ))}
                    </dl>
                  </div>
                ))}
              </div>
            </CardBody>
          </Card>
        )}

        <RuntimeCard />

        <Card>
          <CardHeader title="Data use" />
          <CardBody className="text-sm leading-relaxed text-ink-muted">
            MIMIC-CXR, MIMIC-IV-ED and PTB-XL are credentialed under PhysioNet data use
            agreements; EchoNet-Dynamic and CAMUS carry their own terms. No images,
            waveforms, videos, reports or derived datasets are distributed with this code,
            and model weights derived from credentialed data are not redistributable either.
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

/**
 * Where the service is actually running.
 *
 * Deliberately here and not in the header. Which GPU answered a request is
 * operator telemetry -- it tells a clinician nothing about whether a result can
 * be trusted -- but it is exactly what someone reproducing a figure or
 * diagnosing a slow response needs, so it belongs on the provenance page.
 */
function RuntimeCard() {
  const { health, error } = useHealth();
  if (error || !health) return null;

  const ready = health.components.filter(
    (component) => component.status === "ready",
  ).length;

  return (
    <Card>
      <CardHeader
        title="Runtime"
        description="The environment serving this session. Recorded for reproducibility, not for clinical interpretation."
      />
      <CardBody>
        <dl className="grid gap-x-8 sm:grid-cols-2">
          <div className="readout">
            <dt>Service</dt>
            <dd>{health.service} {health.version}</dd>
          </div>
          <div className="readout">
            <dt>Compute device</dt>
            <dd className="truncate" title={health.device}>{health.device}</dd>
          </div>
          <div className="readout">
            <dt>Components loaded</dt>
            <dd>{ready} of {health.components.length}</dd>
          </div>
          <div className="readout">
            <dt>Uptime</dt>
            <dd>{Math.floor(health.uptime_s / 60)} min</dd>
          </div>
        </dl>
      </CardBody>
    </Card>
  );
}

