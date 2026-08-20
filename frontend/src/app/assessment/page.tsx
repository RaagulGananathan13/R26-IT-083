"use client";

import { useCallback, useState } from "react";

import { ErrorNotice, VerdictBanner } from "@/components/clinical";
import {
  Badge,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Field,
  FileDrop,
  Hero,
  Input,
  ResultSkeleton,
  Select,
  Textarea,
} from "@/components/ui";
import { useAnalysis } from "@/hooks/useAnalysis";
import { runAssessment, type AssessmentInput } from "@/lib/api";
import { cn, COMPONENTS, duration, VERDICT } from "@/lib/format";
import type { AssessmentResponse, ComponentId, CrossModalObservation } from "@/lib/types";

const OBSERVATION_STYLE: Record<CrossModalObservation["kind"], string> = {
  concordance: "border-verdict-actionable/30 bg-verdict-actionable/10",
  discordance: "border-verdict-caution/30 bg-verdict-caution/10",
  context: "border-line bg-surface-2",
};

export default function AssessmentPage() {
  const [patientId, setPatientId] = useState("DEMO-001");
  const [cxrFile, setCxrFile] = useState<File | null>(null);
  const [cxrView, setCxrView] = useState("");
  const [ecgDat, setEcgDat] = useState<File | null>(null);
  const [ecgHea, setEcgHea] = useState<File | null>(null);
  const [echoFile, setEchoFile] = useState<File | null>(null);
  const [triageJson, setTriageJson] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);

  const run = useCallback((input: AssessmentInput) => runAssessment(input), []);
  const { result, error, pending, execute } = useAnalysis<AssessmentResponse, AssessmentInput>(run);

  const hasAny = Boolean(cxrFile || (ecgDat && ecgHea) || echoFile || triageJson.trim());

  function submit() {
    let triage: Record<string, unknown> | null = null;
    if (triageJson.trim()) {
      try {
        triage = JSON.parse(triageJson);
        setJsonError(null);
      } catch (cause) {
        setJsonError(cause instanceof Error ? cause.message : "Invalid JSON");
        return;
      }
    }
    execute({
      patientId,
      cxrFile,
      cxrView: cxrView || null,
      ecgDat,
      ecgHea,
      echoFile,
      triage,
    });
  }

  return (
    <div className="mx-auto max-w-[1200px] px-6 pb-20 pt-6">
      <Hero
        eyebrow="Across components"
        title="Multi-modal assessment"
        subtitle="Runs whichever modalities you supply for one patient and reduces their verdicts to the worst case, because a chain of evidence is no stronger than its weakest link. A modality that is absent, unavailable or failing is reported rather than allowed to fail the whole request."
        className="mb-8"
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card>
            <CardHeader title="Patient" />
            <CardBody>
              <Field label="Local identifier" hint="Never send a real medical record number.">
                <Input
                  value={patientId}
                  onChange={(event) => setPatientId(event.target.value)}
                  disabled={pending}
                />
              </Field>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Studies" description="Supply any subset." />
            <CardBody className="space-y-3">
              <FileDrop
                label="Chest radiograph"
                accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp"
                file={cxrFile}
                onFile={setCxrFile}
                disabled={pending}
                compact
              />
              {cxrFile && (
                <Select
                  value={cxrView}
                  onChange={(event) => setCxrView(event.target.value)}
                  disabled={pending}
                >
                  <option value="">Projection not specified</option>
                  <option value="PA">PA — standing</option>
                  <option value="AP">AP — bedside</option>
                </Select>
              )}
              <FileDrop
                label="ECG signal (.dat)"
                accept=".dat"
                file={ecgDat}
                onFile={setEcgDat}
                disabled={pending}
                compact
              />
              <FileDrop
                label="ECG header (.hea)"
                accept=".hea"
                file={ecgHea}
                onFile={setEcgHea}
                disabled={pending}
                compact
              />
              <FileDrop
                label="Echocardiogram"
                accept=".avi,.mp4,.mov,.mkv,.webm,.npy"
                file={echoFile}
                onFile={setEchoFile}
                disabled={pending}
                compact
              />
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="ED triage record" description="JSON, optional." />
            <CardBody>
              <Textarea
                rows={5}
                placeholder='{"age": 61, "chief_complaint": "chest pain"}'
                value={triageJson}
                onChange={(event) => setTriageJson(event.target.value)}
                disabled={pending}
                className="font-mono text-xs"
              />
              {jsonError && <p className="mt-1 text-xs text-verdict-withheld">{jsonError}</p>}
            </CardBody>
          </Card>

          <Button className="w-full" loading={pending} disabled={!hasAny} onClick={submit}>
            {pending ? "Running…" : "Run assessment"}
          </Button>
        </div>

        <div className="space-y-4">
          <ErrorNotice error={error} />
          {pending && <ResultSkeleton />}

          {result && !pending && (
            <div className="animate-fade-up space-y-4">
              <VerdictBanner
                reliability={{
                  actionability: result.summary.actionability,
                  level: "aggregate",
                  reasons: result.summary.reasons,
                  guarantees: result.summary.guarantees,
                  guarantees_void: false,
                  coverage: null,
                }}
                headline={result.summary.headline}
              />

              <Card>
                <CardHeader title="Per component" />
                <CardBody className="space-y-2">
                  {(Object.keys(result.components) as ComponentId[]).map((id) => {
                    const envelope = result.components[id]!;
                    const style = VERDICT[envelope.reliability.actionability];
                    return (
                      <div
                        key={id}
                        className={cn("rounded-lg border px-3 py-2.5", style.bg, style.border)}
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-2xs font-semibold text-ink-faint">
                            {COMPONENTS[id].number}
                          </span>
                          <span className="text-sm font-medium text-ink">
                            {COMPONENTS[id].short}
                          </span>
                          <Badge
                            className={cn(style.text, "ml-auto border-line bg-surface/70")}
                            dot={style.dot}
                          >
                            {style.label}
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs leading-relaxed text-ink-muted">
                          {envelope.headline}
                        </p>
                      </div>
                    );
                  })}
                  {Object.entries(result.skipped).map(([id, reason]) => (
                    <div key={id} className="rounded-lg border border-line px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-ink-muted">
                          {COMPONENTS[id as ComponentId]?.short ?? id}
                        </span>
                        <span className="ml-auto text-2xs uppercase tracking-wide text-ink-faint">
                          not run
                        </span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-ink-faint">{reason}</p>
                    </div>
                  ))}
                </CardBody>
              </Card>

              {result.observations.length > 0 && (
                <Card>
                  <CardHeader
                    title="Across modalities"
                    description="Rule-based observations. Each names the values it rests on, so it can be checked against the component payloads."
                  />
                  <CardBody className="space-y-2.5">
                    {result.observations.map((observation, index) => (
                      <div
                        key={index}
                        className={cn("rounded-lg border px-3 py-2.5", OBSERVATION_STYLE[observation.kind])}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-2xs font-semibold uppercase tracking-wide text-ink-muted">
                            {observation.kind}
                          </span>
                          <span className="text-2xs text-ink-faint">
                            {observation.components.join(" + ")}
                          </span>
                        </div>
                        <p className="mt-1 text-xs leading-relaxed text-ink">
                          {observation.statement}
                        </p>
                        <p className="mt-1 text-2xs leading-relaxed text-ink-faint">
                          {observation.basis}
                        </p>
                      </div>
                    ))}
                  </CardBody>
                </Card>
              )}

              <Callout tone="neutral" title="Aggregation, not fusion">
                {result.method_note}
              </Callout>

              <p className="text-2xs text-ink-faint">
                {duration(result.elapsed_ms)} · request {result.request_id}
              </p>
            </div>
          )}

          {!result && !pending && !error && (
            <Card>
              <CardBody className="py-16 text-center">
                <p className="text-sm text-ink-muted">Supply at least one modality.</p>
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
