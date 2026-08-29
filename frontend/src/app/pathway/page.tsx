"use client";

/**
 * The clinical pathway console, walked one stage at a time.
 *
 * WHY A STEPPER AND NOT A FORM
 * ----------------------------
 * The pathway is an ordering, and the studies arrive in that order: the ECG at
 * ten minutes, the film within the hour, the echo hours later, each from a
 * different person. A page that collects all four studies and then runs
 * everything hides the thing the pathway exists to show — that a result can
 * make the next test irrelevant.
 *
 * So each stage is presented alone: whose component it is, what question it
 * answers, the one study it needs, and a button that runs only that stage. The
 * routing decision comes back from the server and says where the pathway goes
 * next, or that it ends here.
 *
 * Every stage can load a study from the curated demo set, grouped by the class
 * it was chosen to demonstrate. That matters for review: an unlabelled upload
 * shows the component produces an answer, a labelled one shows it produces the
 * right answer.
 *
 * The client never decides what comes next. `context` is produced by the engine
 * and handed straight back; reading it here would be a second copy of the
 * routing rules.
 */
import { useCallback, useEffect, useState } from "react";

import {
  AnalysisSplit,
  DispositionBanner,
  ErrorNotice,
  ExtractionReview,
  FindingsTable,
  PathwayTimeline,
  ReportViewer,
  SamplePicker,
  StageEvidence,
  hasStageEvidence,
} from "@/components/clinical";
import { TriageForm, type TriageFormValue } from "@/components/clinical/TriageForm";
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Field,
  FileDrop,
  Hero,
  Select,
} from "@/components/ui";
import { analyzeTriagePdf, runPathwayStage } from "@/lib/api";
import { getDemoCatalog, type DemoCatalog, type DemoSample } from "@/lib/demo";
import { cn, COMPONENTS, STAGE_STATUS, URGENCY, VERDICT } from "@/lib/format";
import type {
  ComponentId,
  Disposition,
  ExtractionReport,
  PathwayContext,
  PathwayStage,
  StageRunResponse,
} from "@/lib/types";

/* ------------------------------------------------------------------ */

const STAGES = [
  { id: "triage_h0", n: 1, clock: "T + 0 min", title: "Triage assessment", component: "triage" as ComponentId, horizon: "H = 0", needs: "record" },
  { id: "ecg", n: 2, clock: "T + 10 min", title: "12-lead ECG", component: "ecg" as ComponentId, horizon: null, needs: "ecg" },
  { id: "cxr", n: 3, clock: "T + 15–60 min", title: "Chest radiograph", component: "cxr" as ComponentId, horizon: null, needs: "cxr" },
  { id: "triage_h6", n: 4, clock: "T + 1–6 h", title: "Troponin 0 h / 1 h", component: "triage" as ComponentId, horizon: "H = 6", needs: "record" },
  { id: "echo", n: 5, clock: "T + 6–24 h", title: "Echocardiogram", component: "echo" as ComponentId, horizon: null, needs: "echo" },
  { id: "triage_h24", n: 6, clock: "T + 24 h", title: "Workup complete", component: "triage" as ComponentId, horizon: "H = 24", needs: "record" },
] as const;

type StageMeta = (typeof STAGES)[number];
const byId = (id: string) => STAGES.find((s) => s.id === id) ?? STAGES[0];

/** Manual fallback when no demo PDF is loaded. */
const BLANK_RECORD = {
  age: 61,
  sex: "M",
  chief_complaint: "central chest pain radiating to left arm, onset 2 hours ago",
  heartrate: 98,
  sbp: 148,
  dbp: 88,
  resprate: 20,
  o2sat: 96,
  pain: 8,
  acuity: 2,
};

/* ------------------------------------------------------------------ */

export default function PathwayPage() {
  const [catalog, setCatalog] = useState<DemoCatalog | null>(null);
  useEffect(() => {
    getDemoCatalog().then(setCatalog);
  }, []);

  /* setup */
  const [seed, setSeed] = useState<Record<string, unknown>>(BLANK_RECORD);
  const [extraction, setExtraction] = useState<ExtractionReport | null>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [pdfPending, setPdfPending] = useState(false);
  const [pdfError, setPdfError] = useState<unknown>(null);
  const [record, setRecord] = useState<Record<string, unknown> | null>(null);

  /* traversal */
  const [context, setContext] = useState<PathwayContext | null>(null);
  const [currentId, setCurrentId] = useState<string>("triage_h0");
  const [done, setDone] = useState<PathwayStage[]>([]);
  const [latest, setLatest] = useState<StageRunResponse | null>(null);
  const [disposition, setDisposition] = useState<Disposition | null>(null);
  const [limits, setLimits] = useState<string[]>([]);

  /* per-stage studies */
  const [cxrFile, setCxrFile] = useState<File | null>(null);
  const [cxrView, setCxrView] = useState("");
  // The Grad-CAM overlay is only checkable against the film underneath it, so
  // the uploaded radiograph is kept as an object URL for the comparison view.
  const [cxrUrl, setCxrUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!cxrFile) {
      setCxrUrl(null);
      return;
    }
    const url = URL.createObjectURL(cxrFile);
    setCxrUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [cxrFile]);
  const [ecgDat, setEcgDat] = useState<File | null>(null);
  const [ecgHea, setEcgHea] = useState<File | null>(null);
  const [echoFile, setEchoFile] = useState<File | null>(null);
  const [expected, setExpected] = useState<Record<string, string>>({});

  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const finished = disposition !== null;
  const current = byId(currentId);
  const awaitingAdvance = latest !== null && !finished;

  /** Parse a PDF into a record the form can show and a human can correct. */
  const ingestPdf = useCallback(async (file: File, label: string) => {
    setPdfPending(true);
    setPdfError(null);
    try {
      const response = await analyzeTriagePdf(file);
      setSeed(response.request as unknown as Record<string, unknown>);
      setExtraction(response.extraction);
      setSourceLabel(label);
    } catch (cause) {
      setPdfError(cause);
    } finally {
      setPdfPending(false);
    }
  }, []);

  function reset() {
    setRecord(null);
    setSeed(BLANK_RECORD);
    setExtraction(null);
    setSourceLabel(null);
    setContext(null);
    setCurrentId("triage_h0");
    setDone([]);
    setLatest(null);
    setDisposition(null);
    setLimits([]);
    setCxrFile(null);
    setEcgDat(null);
    setEcgHea(null);
    setEchoFile(null);
    setExpected({});
    setError(null);
    setPdfError(null);
  }

  const runStage = useCallback(
    async (stageId: string) => {
      if (!record) return;
      setPending(true);
      setError(null);
      try {
        const response = await runPathwayStage({
          stageId, context, triage: record,
          cxrFile, cxrView, ecgDat, ecgHea, echoFile,
        });
        setContext(response.context);
        setLatest(response);
        setLimits(response.limits);
        // The stage that ran, then any the routing bypassed on the way to the
        // next one -- in that order, so the rail reads chronologically. Without
        // this a deliberately skipped stage keeps its "not yet" styling and is
        // indistinguishable from one still waiting to run.
        setDone((previous) => [
          ...previous,
          response.stage,
          ...(response.skipped ?? []),
        ]);
        if (response.finished) setDisposition(response.disposition);
      } catch (cause) {
        setError(cause);
      } finally {
        setPending(false);
      }
    },
    [record, context, cxrFile, cxrView, ecgDat, ecgHea, echoFile],
  );

  function advance() {
    if (!latest?.next_stage) return;
    setCurrentId(latest.next_stage);
    setLatest(null);
  }

  /* ---------------- setup ---------------- */
  if (!record) {
    return (
      <div className="mx-auto w-full max-w-[68rem] space-y-7">
        <PageHeader />

        <Step
          n="1"
          title="Load the emergency-department record"
          note="This record drives stages 1, 4 and 6 — the same record, re-scored at three disclosure horizons."
        >
          <Card>
            <CardHeader
              title="Sample records"
              description="Each is a real ED summary PDF, parsed by the same extractor an uploaded document goes through. The class is what the record was chosen to demonstrate."
            />
            <CardBody className="space-y-4">
              <SamplePicker
                samples={catalog?.samples.triage ?? []}
                disabled={pdfPending}
                onPick={async (files, sample) => {
                  const document = files[0];
                  if (document) await ingestPdf(document, sample.label);
                }}
              />

              <div className="border-t border-line pt-4">
                <FileDrop
                  label="Or upload your own ED record (PDF)"
                  hint="A text-layer PDF. Scanned images are not read — there is no OCR."
                  accept=".pdf"
                  file={null}
                  onFile={(file) => file && ingestPdf(file, file.name)}
                  disabled={pdfPending}
                  compact
                />
              </div>

              {pdfPending && (
                <p className="font-mono text-2xs text-ink-faint">Parsing the document…</p>
              )}
              {pdfError ? <ErrorNotice error={pdfError} /> : null}
              {sourceLabel && !pdfPending && (
                <p className="text-xs text-verdict-actionable">
                  Loaded from <span className="font-semibold">{sourceLabel}</span>. Check the
                  extracted record below before starting.
                </p>
              )}
            </CardBody>
          </Card>
        </Step>

        {extraction && (
          <Step
            n="2"
            title="Check what the parser found"
            note="A parser that silently missed a troponin would produce a confident, wrong answer with no error anywhere."
          >
            <ExtractionReview extraction={extraction} submitted={seed} />
          </Step>
        )}

        <Step
          n={extraction ? "3" : "2"}
          title="Confirm the record and begin"
          note="Every field is optional: an untested biomarker is the fact that nobody ordered the test, not a number to impute."
        >
          <TriageForm
            initial={seed}
            pending={false}
            onSubmit={(value: TriageFormValue) => setRecord(value as Record<string, unknown>)}
            submitLabel="Begin the pathway at stage 1"
          />
        </Step>
      </div>
    );
  }

  /* ---------------- traversal ---------------- */
  return (
    <div className="mx-auto w-full max-w-[68rem] space-y-6">
      <PageHeader onReset={reset} />

      <ProgressRail
        done={done}
        currentId={finished ? null : currentId}
        terminatedAt={context?.terminated_at ?? null}
      />

      {error ? <ErrorNotice error={error} /> : null}

      {latest && (
        <StageResult
          response={latest}
          expected={expected[latest.stage.id]}
          onAdvance={advance}
          cxrUrl={cxrUrl}
        />
      )}

      {!finished && !awaitingAdvance && (
        <CurrentStage
          meta={current}
          catalog={catalog}
          pending={pending}
          onRun={() => runStage(current.id)}
          onSample={(files, sample) => {
            setExpected((previous) => ({ ...previous, [current.id]: sample.klass }));
            if (current.needs === "cxr") {
              setCxrFile(files[0] ?? null);
              if (sample.hint) setCxrView(sample.hint);
            } else if (current.needs === "echo") {
              setEchoFile(files[0] ?? null);
            } else if (current.needs === "ecg") {
              setEcgDat(files.find((f) => f.name.endsWith(".dat")) ?? null);
              setEcgHea(files.find((f) => f.name.endsWith(".hea")) ?? null);
            }
          }}
          cxrFile={cxrFile} setCxrFile={setCxrFile}
          cxrView={cxrView} setCxrView={setCxrView}
          ecgDat={ecgDat} setEcgDat={setEcgDat}
          ecgHea={ecgHea} setEcgHea={setEcgHea}
          echoFile={echoFile} setEchoFile={setEchoFile}
        />
      )}

      {finished && disposition && (
        <div className="space-y-5">
          <DispositionBanner
            disposition={disposition}
            actionability={latest?.actionability ?? "unavailable"}
            terminationReason={context?.termination_reason ?? null}
            stagesCompleted={done.filter((s) => s.status === "completed").length}
            stagesTotal={6}
          />

          <Card>
            <CardHeader
              title="The full traversal"
              description="Every stage that ran, in order, with the routing decision each one produced."
            />
            <CardBody>
              <PathwayTimeline stages={done} terminatedAt={context?.terminated_at ?? null} />
            </CardBody>
          </Card>

          {/* Collapsed, not removed.
              As a block of amber text this dominated the result and read as a
              warning about the answer above it, which it is not — it is scope.
              Deleting it would be worse than either: an unstated limitation
              reads as one that was missed, and "the four cohorts share no
              patients" is the first thing a reviewer asks. Closed by default,
              one click from a complete answer. */}
          {limits.length > 0 && (
            <details className="group rounded-xl border border-line bg-surface">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-6 py-4">
                <span className="text-[0.9375rem] font-bold text-ink">Scope and limitations</span>
                <span className="flex items-center gap-2">
                  <span className="font-mono text-[0.6875rem] text-ink-faint">
                    {limits.length} noted
                  </span>
                  <span className="text-[0.8125rem] font-semibold text-accent group-open:hidden">
                    Show
                  </span>
                  <span className="hidden text-[0.8125rem] font-semibold text-accent group-open:inline">
                    Hide
                  </span>
                </span>
              </summary>
              <ul className="space-y-2.5 border-t border-line px-6 py-4">
                {limits.map((limit) => (
                  <li key={limit} className="flex gap-2.5 text-sm leading-relaxed text-ink-muted">
                    <span
                      className="mt-2 h-1 w-1 flex-none rounded-full bg-ink-faint"
                      aria-hidden
                    />
                    <span>{limit}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          <Button variant="secondary" onClick={reset}>Start another patient</Button>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function PageHeader({ onReset }: { onReset?: () => void }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <Hero
        eyebrow="Clinical pathway"
        title="Emergency-department chest pain, stage by stage"
        subtitle="Six stages in the order a patient meets them. Each result decides whether the next stage happens at all."
        pills={["6 stages", "Component 04 runs 3×", "Gated, not parallel"]}
        className="min-w-0 flex-1 border-0 pb-0"
      />
      {onReset && (
        <Button variant="ghost" size="sm" onClick={onReset}>Start over</Button>
      )}
    </div>
  );
}

function ProgressRail({
  done, currentId, terminatedAt,
}: {
  done: PathwayStage[];
  currentId: string | null;
  terminatedAt: string | null;
}) {
  const stopIndex = terminatedAt ? STAGES.findIndex((s) => s.id === terminatedAt) : -1;
  return (
    <ol className="flex flex-wrap gap-1.5" aria-label="Pathway progress">
      {STAGES.map((stage, index) => {
        const ran = done.find((d) => d.id === stage.id);
        const isCurrent = stage.id === currentId;
        const unreached = stopIndex >= 0 && index > stopIndex;
        const style = ran ? STAGE_STATUS[ran.status] : null;
        return (
          <li
            key={stage.id}
            aria-current={isCurrent ? "step" : undefined}
            className={cn(
              "flex min-w-0 flex-1 items-center gap-2 rounded border px-2.5 py-2",
              isCurrent && "border-accent bg-accent/[0.06]",
              !isCurrent && ran && cn(style?.border, style?.bg),
              !isCurrent && !ran && "border-line bg-surface",
              unreached && "opacity-45",
            )}
          >
            <span className={cn("font-mono text-2xs font-semibold",
              isCurrent ? "text-accent" : ran ? style?.text : "text-ink-faint")}>
              {stage.n}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-2xs font-medium leading-tight text-ink">
                {stage.title}
              </span>
              <span className="block truncate font-mono text-[0.625rem] text-ink-faint">
                {ran ? style?.label : isCurrent ? "Now" : stage.clock}
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function CurrentStage({
  meta, catalog, pending, onRun, onSample,
  cxrFile, setCxrFile, cxrView, setCxrView,
  ecgDat, setEcgDat, ecgHea, setEcgHea, echoFile, setEchoFile,
}: {
  meta: StageMeta;
  catalog: DemoCatalog | null;
  pending: boolean;
  onRun: () => void;
  onSample: (files: File[], sample: DemoSample) => void;
  cxrFile: File | null; setCxrFile: (f: File | null) => void;
  cxrView: string; setCxrView: (v: string) => void;
  ecgDat: File | null; setEcgDat: (f: File | null) => void;
  ecgHea: File | null; setEcgHea: (f: File | null) => void;
  echoFile: File | null; setEchoFile: (f: File | null) => void;
}) {
  const component = COMPONENTS[meta.component];
  const samples =
    meta.needs === "cxr" ? catalog?.samples.cxr
    : meta.needs === "ecg" ? catalog?.samples.ecg
    : meta.needs === "echo" ? catalog?.samples.echo
    : null;

  const ready =
    meta.needs === "record" ||
    (meta.needs === "cxr" && !!cxrFile) ||
    (meta.needs === "echo" && !!echoFile) ||
    (meta.needs === "ecg" && !!ecgDat && !!ecgHea);

  return (
    <Card className="border-accent/40">
      <CardHeader
        title={
          <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <span className="font-mono text-2xs text-accent">
              STAGE {meta.n} OF 6 · {meta.clock}
            </span>
            <span className="display text-base text-ink">{meta.title}</span>
          </span>
        }
        description={
          <span>
            Component {component.number} — {component.title}
            {meta.horizon ? ` · ${meta.horizon}` : ""} · {component.owner}
          </span>
        }
      />
      <CardBody className="space-y-4">
        {meta.needs === "record" ? (
          <p className="text-sm text-ink-muted">
            No upload needed. The record you confirmed is re-scored at this stage&apos;s
            disclosure horizon{meta.horizon ? ` (${meta.horizon})` : ""}, using only what
            would be knowable by then.
          </p>
        ) : (
          <>
            {samples && (
              <div>
                <p className="eyebrow mb-2">Load a sample — grouped by expected class</p>
                <SamplePicker samples={samples} disabled={pending} onPick={onSample} />
              </div>
            )}

            <div className="border-t border-line pt-4">
              {meta.needs === "cxr" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <FileDrop
                    label="Chest radiograph" hint="Frontal film, AP or PA"
                    accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp"
                    file={cxrFile} onFile={setCxrFile} disabled={pending}
                  />
                  <Field label="Projection" hint="Leave unknown rather than guessing — PA applies the stricter threshold.">
                    <Select value={cxrView} onChange={(event) => setCxrView(event.target.value)}>
                      <option value="">Unknown</option>
                      <option value="AP">AP</option>
                      <option value="PA">PA</option>
                    </Select>
                  </Field>
                </div>
              )}

              {meta.needs === "ecg" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <FileDrop label="Signal file (.dat)" hint="WFDB record" accept=".dat"
                    file={ecgDat} onFile={setEcgDat} disabled={pending} />
                  <FileDrop label="Header file (.hea)" hint="Must share the .dat base name" accept=".hea"
                    file={ecgHea} onFile={setEcgHea} disabled={pending} />
                </div>
              )}

              {meta.needs === "echo" && (
                <FileDrop
                  label="Echocardiogram" hint="Apical four-chamber video, or a cached .npy clip"
                  accept=".avi,.mp4,.mov,.mkv,.webm,.npy"
                  file={echoFile} onFile={setEchoFile} disabled={pending}
                />
              )}
            </div>
          </>
        )}

        <div className="flex flex-wrap items-center gap-3 border-t border-line pt-4">
          <Button onClick={onRun} loading={pending} disabled={pending}>
            {pending ? `Running stage ${meta.n}…` : `Run stage ${meta.n}`}
          </Button>
          {!ready && (
            <span className="text-xs text-ink-muted">
              No study attached. Running anyway records this stage as having produced no
              evidence — which is not the same as a normal result.
            </span>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

/** Section headings that mark generated text as a report rather than a line. */
const REPORT_HEADING = /\b(?:FINDINGS|IMPRESSION|TECHNICAL|CONCLUSION)\s*:/;

function StageResult({
  response, expected, onAdvance, cxrUrl,
}: {
  response: StageRunResponse;
  expected?: string;
  onAdvance: () => void;
  cxrUrl?: string | null;
}) {
  const { stage, next_stage: next, finished } = response;
  const routing = stage.routing;
  const envelope = stage.result;
  const status = STAGE_STATUS[stage.status];
  const verdict = envelope ? VERDICT[envelope.reliability.actionability] : null;
  const urgency = routing ? URGENCY[routing.urgency] : null;

  // Tabs rather than a longer page. Both views are one click apart and the
  // panel height does not change, which is what keeps a six-stage walkthrough
  // readable on a laptop. The clinical answer is never behind a tab -- it is
  // the headline above them.
  const evidence = hasStageEvidence(stage.component, envelope);
  const raw = (envelope?.raw ?? {}) as Record<string, unknown>;
  const narrative =
    (envelope?.narrative as string | undefined) ??
    (raw.report_text as string | undefined);
  const groundTruthReport = raw.ground_truth_report as string | undefined;

  // A report has sections; a disposition is one sentence. Component 04 returns
  // "ACS unlikely on current evidence; pursue other causes." as its narrative
  // -- true, and already said in the headline, but putting it behind a tab
  // labelled Report promises a document that does not exist. The radiograph
  // and the ECG both emit section-headed text and genuinely have one.
  const isReport = Boolean(narrative && REPORT_HEADING.test(narrative));
  const reportText = isReport ? narrative : undefined;
  const hasReport = isReport || Boolean(groundTruthReport);
  const hasFindings = (envelope?.findings.length ?? 0) > 0;

  // Explainability left, written analysis right -- and the findings table
  // always below at full width. It was briefly in the right-hand pane whenever
  // a stage had no report, but it renders its own card, so it sat inside the
  // pane as a second frame while the left pane had none. A table also reads
  // badly at half width once it carries a probability, a threshold and a zone.

  return (
    <Card className={cn(routing?.terminates && "border-verdict-withheld/40")}>
      <CardHeader
        title={
          <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <span className="font-mono text-2xs text-ink-faint">STAGE {stage.ordinal} RESULT</span>
            <span className="display text-base text-ink">{stage.title}</span>
          </span>
        }
        actions={
          <span className={cn(
            "rounded-sm border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider",
            status.bg, status.border, status.text)}>
            {status.label}
          </span>
        }
      />
      <CardBody className="space-y-4">
        {/* Ground truth beside the prediction, not as a footnote under it.
            Deliberately NOT auto-scored: the labels are per-component and free
            text ("cardiomegaly" against "Cardiomegaly present (p=0.925)", "MI"
            against "myocardial infarction"), so a match indicator here would be
            a string heuristic dressed up as an evaluation. The two are shown
            plainly and the reader compares them. */}
        {expected && envelope && (
          <div className="grid gap-3 rounded border border-line bg-surface-2 px-3.5 py-3 sm:grid-cols-2">
            <div>
              <p className="font-mono text-2xs uppercase tracking-widest text-ink-faint">
                Ground truth
              </p>
              <p className="mt-1 text-base font-semibold capitalize text-ink">{expected}</p>
            </div>
            <div className="sm:border-l sm:border-line sm:pl-4">
              <p className="font-mono text-2xs uppercase tracking-widest text-ink-faint">
                Model said
              </p>
              <p className="mt-1 text-base font-semibold text-ink">{envelope.headline}</p>
            </div>
          </div>
        )}
        {expected && !envelope && (
          <p className="rounded border border-line bg-surface-2 px-3 py-2 font-mono text-2xs text-ink-muted">
            Sample loaded to demonstrate: <span className="text-ink">{expected}</span>.
          </p>
        )}

        {envelope && (
          <div className="space-y-3">
            {/* The stage's answer, at the size of an answer. It was set in
                body text, so the one line a reader is looking for weighed the
                same as the caption above it. */}
            <p className="flex items-start gap-2.5">
              {verdict && (
                <span
                  className={cn("mt-2.5 h-2 w-2 flex-none rounded-full", verdict.dot)}
                  aria-hidden
                />
              )}
              <span className="display text-[1.375rem] font-semibold leading-snug text-ink">
                {envelope.headline}
              </span>
            </p>

            <AnalysisSplit
              leftLabel="Explainable AI"
              rightLabel="Automated analysis"
              left={
                evidence ? (
                  <StageEvidence
                    component={stage.component}
                    envelope={envelope}
                    originalUrl={cxrUrl}
                  />
                ) : undefined
              }
              right={
                hasReport ? (
                  <ReportViewer
                    bare
                    cleaned={reportText}
                    raw={raw.report_text_raw as string | undefined}
                    groundTruth={groundTruthReport}
                    prompt={raw.classifier_prompt as string | undefined}
                  />
                ) : undefined
              }
            />

            {hasFindings && (
              <FindingsTable
                findings={envelope.findings}
                actionability={envelope.reliability.actionability}
              />
            )}
          </div>
        )}

        {stage.detail && !envelope && <p className="text-sm text-ink-muted">{stage.detail}</p>}

        {routing && (
          <div className={cn("rounded border p-3.5",
            routing.terminates
              ? "border-verdict-withheld/30 bg-verdict-withheld/[0.06]"
              : "border-line bg-surface-2")}>
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <span className="font-mono text-2xs uppercase tracking-widest text-ink-faint">Decision</span>
              {urgency && routing.urgency !== "routine" && (
                <span className={cn(
                  "rounded-sm border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider",
                  urgency.bg, urgency.border, urgency.text)}>
                  {urgency.label}
                </span>
              )}
            </div>
            <p className="text-sm text-ink">{routing.statement}</p>
            <p className="mt-2 font-mono text-2xs leading-relaxed text-ink-faint">{routing.basis}</p>
            {routing.guideline && (
              <p className="mt-2 border-t border-line pt-2 text-xs text-ink-muted">
                <span className="font-semibold">Guideline. </span>{routing.guideline}
              </p>
            )}
          </div>
        )}

        {!finished && next && (
          <div className="flex flex-wrap items-center gap-3 border-t border-line pt-4">
            <Button onClick={onAdvance}>
              Continue to stage {byId(next).n} — {byId(next).title}
            </Button>
            <span className="font-mono text-2xs text-ink-faint">{byId(next).clock}</span>
          </div>
        )}

        {finished && (
          <p className="border-t border-line pt-4 text-sm font-medium text-verdict-withheld">
            The pathway ends here. No further stage is run.
          </p>
        )}
      </CardBody>
    </Card>
  );
}

function Step({
  n, title, note, children,
}: {
  n: string;
  title: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="mb-3 flex items-baseline gap-3">
        <span aria-hidden className="grid h-6 w-6 flex-none place-items-center rounded-sm border border-line bg-surface-2 font-mono text-2xs font-semibold text-ink">
          {n}
        </span>
        <div className="min-w-0">
          <h2 className="display text-[0.95rem] leading-none text-ink">{title}</h2>
          <p className="mt-1 text-xs leading-relaxed text-ink-muted">{note}</p>
        </div>
      </div>
      {children}
    </section>
  );
}
