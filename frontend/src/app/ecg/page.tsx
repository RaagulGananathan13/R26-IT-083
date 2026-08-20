"use client";

import { useCallback, useState } from "react";

import {
  EcgStripViewer,
  ErrorNotice,
  FindingsGrid,
  ModelCardPanel,
  NarrativePanel,
  RawPayload,
  StudyGrid,
  StudyLayout,
  VerdictBanner,
} from "@/components/clinical";
import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Checkbox,
  FileDrop,
  ResultSkeleton,
} from "@/components/ui";
import { useAnalysis } from "@/hooks/useAnalysis";
import { analyzeEcg } from "@/lib/api";
import { duration } from "@/lib/format";
import type { Envelope } from "@/lib/types";

interface EcgInput {
  dat: File;
  hea: File;
  withXai: boolean;
}

export default function EcgPage() {
  const [dat, setDat] = useState<File | null>(null);
  const [hea, setHea] = useState<File | null>(null);
  const [withXai, setWithXai] = useState(true);

  const run = useCallback(
    (input: EcgInput) => analyzeEcg(input.dat, input.hea, input.withXai),
    [],
  );
  const { result, error, pending, execute, reset } = useAnalysis<Envelope, EcgInput>(run);

  const stemMismatch =
    dat && hea && stem(dat.name) !== stem(hea.name)
      ? `The two files must share a base name — got "${stem(dat.name)}" and "${stem(hea.name)}".`
      : null;

  return (
    <StudyLayout
      id="ecg"
      intro="Classifies a 12-lead ECG into five diagnostic superclasses with conformal rule-in / rule-out triage, then writes a verified report. The quality gate runs before the classifier, so a signal that fails it never produces a probability at all."
      pills={[
        "1-D residual CNN + SE",
        "PAC conformal triage",
        "Grad-CAM + integrated gradients",
        "macro-AUROC 0.9343",
      ]}
    >
      <StudyGrid>
        <div className="space-y-4">
          <Card>
            <CardHeader
              title="WFDB record"
              description="Both files are required: the header names the signal file, and the reader resolves it from the header's contents rather than the filename."
            />
            <CardBody className="space-y-3">
              <FileDrop
                label="Signal file (.dat)"
                accept=".dat"
                file={dat}
                onFile={(next) => {
                  setDat(next);
                  reset();
                }}
                disabled={pending}
                compact
              />
              <FileDrop
                label="Header file (.hea)"
                accept=".hea"
                file={hea}
                onFile={(next) => {
                  setHea(next);
                  reset();
                }}
                disabled={pending}
                compact
              />

              {stemMismatch && (
                <p className="text-xs text-verdict-caution">{stemMismatch}</p>
              )}

              <Checkbox
                label="Compute explanations"
                checked={withXai}
                onChange={(event) => setWithXai(event.target.checked)}
                disabled={pending}
              />
              <p className="-mt-1 text-2xs leading-relaxed text-ink-faint">
                Grad-CAM and integrated gradients roughly double latency; the classifier
                itself runs in about 20 ms.
              </p>

              <Button
                className="w-full"
                loading={pending}
                disabled={!dat || !hea || Boolean(stemMismatch)}
                onClick={() => dat && hea && execute({ dat, hea, withXai })}
              >
                {pending ? "Analysing…" : "Analyse ECG"}
              </Button>
            </CardBody>
          </Card>

          <Callout tone="warning" title="Five classes, not all of cardiology">
            Atrial fibrillation and other arrhythmias are outside the label space. Their
            absence from a report is not evidence of their absence — 14.3 % of the source
            dataset carries a finding these five classes cannot express.
          </Callout>
        </div>

        <div className="space-y-4">
          <ErrorNotice error={error} />
          {pending && <ResultSkeleton />}

          {result && !pending && (
            <div className="space-y-4">
              <VerdictBanner
                reliability={result.reliability}
                headline={result.headline}
                confidence={topClass(result)}
              />

              {result.raw?.triage && (
                <Card className="rise-in-1">
                  <CardBody className="flex flex-wrap items-center justify-between gap-4 py-3">
                    <span className="eyebrow">Triage category</span>
                    <span className="display text-lg text-ink">{result.raw.triage}</span>
                  </CardBody>
                </Card>
              )}

              <div className="rise-in-1 space-y-4">
                <FindingsGrid
                  findings={result.findings}
                  actionability={result.reliability.actionability}
                  title="Superclasses"
                  description="Each class carries its conformal zone. Rule-out and rule-in are the two ends; refer means the evidence supports neither."
                />

                <EcgStripViewer
                  imageBase64={(result.explanation.ecg_png_base64 as string | undefined) ?? null}
                  leads={(result.explanation.lead_attribution as never[]) ?? []}
                  explanation={result.explanation}
                />
              </div>

              {result.narrative && (
                <NarrativePanel
                  text={result.narrative}
                  title="Clinical report"
                  description="Template-grounded and passed through an automated verification gate before release."
                />
              )}

              <div className="rise-in-2 space-y-4">
                <ModelCardPanel model={result.model} />
                <RawPayload payload={result.raw} />
                <p className="text-2xs text-ink-faint">
                  {duration(result.elapsed_ms)} · request {result.request_id}
                </p>
              </div>
            </div>
          )}

          {!result && !pending && !error && (
            <Card>
              <CardBody className="py-16 text-center">
                <p className="text-sm text-ink-muted">
                  Upload a matching .dat and .hea pair to begin.
                </p>
                <p className="mt-1 text-xs text-ink-faint">
                  PTB-XL records are bundled under Component 02’s data directory.
                </p>
              </CardBody>
            </Card>
          )}
        </div>
      </StudyGrid>
    </StudyLayout>
  );
}

/** The strongest calibrated class, for the banner's meter. */
function topClass(result: Envelope) {
  const scored = result.findings.filter((f) => f.probability !== null);
  if (scored.length === 0) return null;
  const best = scored.reduce((a, b) => ((b.probability ?? 0) > (a.probability ?? 0) ? b : a));
  return {
    label: best.name,
    value: best.probability ?? 0,
    adverse: best.zone === "rule_in" && !best.name.toLowerCase().includes("normal"),
  };
}

function stem(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}
