"use client";

import { useCallback, useState } from "react";

import {
  ClassDistribution,
  ErrorNotice,
  ExtractionReview,
  FindingsGrid,
  ModelCardPanel,
  RawPayload,
  StudyGrid,
  StudyLayout,
  TextAttribution,
  VerdictBanner,
} from "@/components/clinical";
import { Button, Callout, Card, CardBody, CardHeader, FileDrop, ResultSkeleton } from "@/components/ui";
import { TriageForm, type TriageFormValue } from "@/components/clinical/TriageForm";
import { useAnalysis } from "@/hooks/useAnalysis";
import { analyzeTriage, analyzeTriagePdf } from "@/lib/api";
import { cn, duration } from "@/lib/format";
import type { Envelope, TriagePdfResponse } from "@/lib/types";

type Mode = "pdf" | "form";

const SAMPLES = [
  { file: "sample_01_stemi.pdf", label: "Anterior STEMI", note: "ECG diagnostic, troponin rising" },
  { file: "sample_02_nstemi.pdf", label: "NSTEMI", note: "ST depression, modest troponin rise" },
  {
    file: "sample_03_unstable_angina.pdf",
    label: "Unstable angina",
    note: "The hardest class — expect a referral, not a call",
  },
  { file: "sample_04_non_cardiac.pdf", label: "Non-cardiac", note: "No cardiac workup ordered" },
  { file: "sample_05_sparse.pdf", label: "Sparse triage note", note: "Most fields absent by design" },
];

export default function TriagePage() {
  const [mode, setMode] = useState<Mode>("pdf");
  // A record carried over from a PDF extraction so it can be corrected before
  // being used. This is the answer to the parser's real limitation: it cannot
  // be made infallible on templates it has never seen, but it does not have to
  // be -- it produces a first draft, and a human confirms it.
  const [handoff, setHandoff] = useState<Record<string, any> | null>(null);

  function correctExtraction(record: Record<string, any>) {
    setHandoff(record);
    setMode("form");
  }

  return (
    <StudyLayout
      id="triage"
      intro="Detects acute coronary syndrome from an emergency-department record and, where present, assigns a subtype. Every feature carries a declared availability time, so the question “could the model actually have known this?” has an answer rather than an assumption."
      pills={[
        "LightGBM + XGBoost",
        "Temporal contract H = 0 / 6 / 24 h",
        "Constrained decision layer",
        "NPV 99.41 %",
      ]}
    >
      <div className="mb-6 inline-flex rounded-lg border border-line bg-surface p-0.5">
        {(["pdf", "form"] as Mode[]).map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => setMode(value)}
            className={cn(
              "rounded-md px-4 py-1.5 text-sm font-medium transition-colors",
              mode === value ? "bg-surface-2 text-ink" : "text-ink-muted hover:text-ink",
            )}
          >
            {value === "pdf" ? "Upload PDF record" : "Enter record manually"}
          </button>
        ))}
      </div>

      {mode === "pdf" ? (
        <PdfMode onCorrect={correctExtraction} />
      ) : (
        <FormMode initial={handoff} />
      )}
    </StudyLayout>
  );
}

/* ---------------------------------------------------------------- PDF ---- */

function PdfMode({ onCorrect }: { onCorrect: (record: Record<string, any>) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);

  const run = useCallback((document: File) => analyzeTriagePdf(document), []);
  const { result, error, pending, execute, reset } = useAnalysis<TriagePdfResponse, File>(run);

  async function loadSample(name: string) {
    reset();
    setSampleError(null);
    try {
      // The sample arrives as base64 in JSON, not as a PDF response. Endpoint
      // security software on some machines intercepts any same-origin fetch
      // that would return document bytes and answers 204, which silently
      // yields a 0-byte File and a confusing "file is empty" from the backend.
      // The stem only: a ".pdf" anywhere in the path is what gets intercepted.
      const stem = name.replace(/\.pdf$/i, "");
      const response = await fetch(`/api/samples/${stem}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.message ?? `server returned ${response.status}`);

      const binary = atob(payload.base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      if (bytes.byteLength !== payload.bytes) {
        throw new Error(`expected ${payload.bytes} bytes, reassembled ${bytes.byteLength}`);
      }
      setFile(new File([bytes], name, { type: payload.contentType }));
    } catch (cause) {
      setSampleError(
        `Could not load ${name}: ${cause instanceof Error ? cause.message : cause}. ` +
          "Open backend/samples/triage/ and choose the file manually instead.",
      );
      setFile(null);
    }
  }

  return (
    <StudyGrid>
      <div className="space-y-4">
        <Card>
          <CardHeader
            title="ED record"
            description="A PDF with a text layer. Scans and photographs are not supported — optical character recognition is out of scope, and the parser says so rather than guessing."
          />
          <CardBody className="space-y-4">
            <FileDrop
              label="PDF document"
              hint="Structured ED summary"
              accept=".pdf"
              file={file}
              onFile={(next) => {
                setFile(next);
                reset();
              }}
              disabled={pending}
            />
            <Button
              className="w-full"
              loading={pending}
              disabled={!file}
              onClick={() => file && execute(file)}
            >
              {pending ? "Reading and predicting…" : "Extract and predict"}
            </Button>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Sample records"
            description="Synthetic, fictional patients. No real record or identifier appears in any of them."
          />
          <CardBody className="space-y-1.5">
            {SAMPLES.map((sample) => (
              <button
                key={sample.file}
                type="button"
                disabled={pending}
                onClick={() => loadSample(sample.file)}
                className="block w-full rounded-lg border border-line px-3 py-2 text-left transition-colors hover:bg-surface-2 disabled:opacity-50"
              >
                <span className="block text-sm font-medium text-ink">{sample.label}</span>
                <span className="block text-2xs text-ink-faint">{sample.note}</span>
              </button>
            ))}
            {sampleError && (
              <p className="pt-1 text-xs leading-relaxed text-verdict-withheld">
                {sampleError}
              </p>
            )}
          </CardBody>
        </Card>

        <Callout tone="warning" title="Extraction is not a document AI">
          A regex-and-lexicon parser over the text layer. It handles structured ED
          summaries; a different hospital’s template will extract little, and will report
          what it could not read. Always check the extracted record against the source.
        </Callout>
      </div>

      <div className="space-y-4">
        <ErrorNotice error={error} />
        {pending && <ResultSkeleton />}

        {result && !pending && (
          <div className="animate-fade-up space-y-4">
            <VerdictBanner
              reliability={result.result.reliability}
              headline={result.result.headline}
              confidence={acsConfidence(result.result)}
            />
            <ExtractionReview
              extraction={result.extraction}
              submitted={result.request}
              onCorrect={() => onCorrect(result.request)}
            />
            <TriageResult envelope={result.result} />
          </div>
        )}

        {!result && !pending && !error && (
          <Card>
            <CardBody className="py-16 text-center">
              <p className="text-sm text-ink-muted">Upload an ED record, or load a sample.</p>
              <p className="mt-1 text-xs text-ink-faint">
                What the parser misses is submitted as “not ordered”, so the gaps are shown
                alongside the values.
              </p>
            </CardBody>
          </Card>
        )}
      </div>
    </StudyGrid>
  );
}

/* --------------------------------------------------------------- form ---- */

function FormMode({ initial }: { initial: Record<string, any> | null }) {
  const run = useCallback((value: TriageFormValue) => analyzeTriage(value), []);
  const { result, error, pending, execute } = useAnalysis<Envelope, TriageFormValue>(run);

  return (
    <StudyGrid>
      <TriageForm initial={initial} pending={pending} onSubmit={(value) => execute(value)} />

      <div className="space-y-4">
        <ErrorNotice error={error} />
        {pending && <ResultSkeleton />}
        {result && !pending && (
          <div className="animate-fade-up space-y-4">
            <VerdictBanner
              reliability={result.reliability}
              headline={result.headline}
              confidence={acsConfidence(result)}
            />
            <TriageResult envelope={result} />
          </div>
        )}
        {!result && !pending && !error && (
          <Card>
            <CardBody className="py-16 text-center">
              <p className="text-sm text-ink-muted">
                Every field is optional. Leave what was not measured empty.
              </p>
              <p className="mt-1 text-xs text-ink-faint">
                An untested biomarker is the clinical fact that nobody ordered the test, not
                a number to impute.
              </p>
            </CardBody>
          </Card>
        )}
      </div>
    </StudyGrid>
  );
}

/* ------------------------------------------------------------- shared ---- */

/** P(ACS) drives the banner meter: it is the rule-out screen's own number. */
function acsConfidence(envelope: Envelope) {
  const value = envelope.raw?.p_acs;
  if (typeof value !== "number") return null;
  return {
    label: "P(ACS)",
    value,
    adverse: String(envelope.raw?.prediction ?? "No_ACS") !== "No_ACS",
  };
}

function TriageResult({ envelope }: { envelope: Envelope }) {
  const raw = envelope.raw ?? {};
  return (
    <>
      {raw.risk_level && (
        <Card>
          <CardBody className="flex flex-wrap items-center justify-between gap-4 py-3">
            <div>
              <span className="eyebrow">Risk band</span>
              <p className="mt-0.5 text-sm font-semibold text-ink">{raw.risk_level}</p>
            </div>
            <p className="max-w-md text-xs leading-relaxed text-ink-muted">
              {raw.recommended_action}
            </p>
          </CardBody>
        </Card>
      )}

      {raw.probabilities && (
        <ClassDistribution
          probabilities={raw.probabilities}
          predicted={String(raw.prediction ?? "")}
          actionability={envelope.reliability.actionability}
        />
      )}

      <TextAttribution
        complaint={String(raw.chief_complaint ?? raw.complaint ?? "")}
        tokens={(envelope.explanation.text_attribution as never[]) ?? []}
        modalityNote={envelope.explanation.modality_attribution_note as string | undefined}
      />

      <FindingsGrid
        findings={envelope.findings.filter((f) => f.name !== "Acute coronary syndrome")}
        actionability={envelope.reliability.actionability}
        title="Four-class call"
      />

      <ModelCardPanel model={envelope.model} />
      <RawPayload payload={raw} />

      <p className="text-2xs text-ink-faint">
        {duration(envelope.elapsed_ms)} · request {envelope.request_id}
      </p>
    </>
  );
}
