"use client";

import { useCallback, useEffect, useState } from "react";

import {
  ErrorNotice,
  FindingsGrid,
  GradCamViewer,
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
  Field,
  FileDrop,
  ResultSkeleton,
  Segmented,
} from "@/components/ui";
import { useAnalysis } from "@/hooks/useAnalysis";
import { analyzeCxr } from "@/lib/api";
import { duration } from "@/lib/format";
import type { Envelope } from "@/lib/types";

const VIEWS = [
  { value: "", label: "Not specified" },
  { value: "PA", label: "PA · standing" },
  { value: "AP", label: "AP · bedside" },
];

export default function CxrPage() {
  const [file, setFile] = useState<File | null>(null);
  const [view, setView] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Kept client-side so Grad-CAM can be toggled against the original film.
  // Revoked on replace so a long session does not leak object URLs.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const run = useCallback(
    ({ file: study, view: projection }: { file: File; view: string }) =>
      analyzeCxr(study, projection || null),
    [],
  );
  const { result, error, pending, execute, reset } = useAnalysis<
    Envelope,
    { file: File; view: string }
  >(run);

  const cardiomegaly = result?.findings.find((f) => f.name === "Cardiomegaly");
  const coFindings = result?.findings.filter((f) => f.name !== "Cardiomegaly") ?? [];

  return (
    <StudyLayout
      id="cxr"
      intro="Predicts cardiomegaly and seven co-occurring pathologies from a single frontal chest radiograph, shows where the model looked, and drafts a report. The projection matters: measured AUROC is 0.8224 on AP films against 0.8864 on PA, a gap three separate interventions failed to close."
      pills={[
        "ConvNeXt-Base 384²",
        "Grad-CAM",
        "BioBART report",
        "Cardiomegaly AUROC 0.9189",
      ]}
    >
      <StudyGrid>
        <div className="space-y-4">
          <Card>
            <CardHeader title="Study" />
            <CardBody className="space-y-4">
              <FileDrop
                label="Chest radiograph"
                hint="PNG, JPEG, BMP or TIFF"
                accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp"
                file={file}
                onFile={(next) => {
                  setFile(next);
                  reset();
                }}
                disabled={pending}
              />

              {previewUrl && (
                <div className="image-frame !aspect-[4/3]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrl} alt="Selected radiograph" />
                </div>
              )}

              <Field
                group
                label="Projection"
                hint="Leave unspecified if unknown. The system then uses the global operating point rather than guessing — assuming PA on a bedside film would apply the stricter threshold to the patients least able to tolerate a missed diagnosis."
              >
                <Segmented
                  options={VIEWS}
                  value={view}
                  onChange={setView}
                  disabled={pending}
                  className="w-full"
                  ariaLabel="Radiograph projection"
                />
              </Field>

              <Button
                className="w-full"
                loading={pending}
                disabled={!file}
                onClick={() => file && execute({ file, view })}
              >
                {pending ? "Analysing…" : "Analyse radiograph"}
              </Button>
            </CardBody>
          </Card>

          <Callout tone="neutral" title="Why accuracy is not the headline">
            Five of the eight pathologies score below an always-negative baseline on
            accuracy — an artefact of F1-optimal thresholds on rare disease. Read AUROC
            and sensitivity instead.
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
                confidence={
                  cardiomegaly?.probability != null
                    ? {
                        label: "Cardiomegaly probability",
                        value: cardiomegaly.probability,
                        adverse: cardiomegaly.present === true,
                      }
                    : null
                }
              />

              <div className="rise-in-1 space-y-4">
                <GradCamViewer
                  originalUrl={previewUrl}
                  heatmapBase64={
                    (result.explanation.gradcam_png_base64 as string | undefined) ?? null
                  }
                  target="Cardiomegaly"
                  caveat={result.explanation.gradcam_caveat as string | undefined}
                />

                <FindingsGrid
                  findings={coFindings}
                  actionability={result.reliability.actionability}
                  title="Co-pathologies"
                  description="Reported in the same forward pass. Cardiomegaly is the diagnostic target; these are co-findings."
                />
              </div>

              <div className="rise-in-2 space-y-4">
                {result.narrative && (
                  <NarrativePanel
                    text={result.narrative}
                    title="Draft report"
                    description="Generated by BioBART from the same visual features the classifier used. Prior-study language was removed from the training targets, so fabricated references to non-existent prior scans measure 0.0000 across the test set."
                  />
                )}

                {result.raw?.ground_truth_report && (
                  <NarrativePanel
                    text={result.raw.ground_truth_report}
                    title="Radiologist report (original)"
                    description="The text as dictated, prior-study references and all. Shown only when the uploaded file matches an indexed test study."
                  />
                )}

                <ModelCardPanel model={result.model} />
                <RawPayload payload={result.raw} />

                <p className="text-2xs text-ink-faint">
                  {duration(result.elapsed_ms)} · request {result.request_id}
                </p>
              </div>
            </div>
          )}

          {!result && !pending && !error && <EmptyState />}
        </div>
      </StudyGrid>
    </StudyLayout>
  );
}

function EmptyState() {
  return (
    <Card>
      <CardBody className="py-20 text-center">
        <p className="text-sm text-ink-muted">
          Upload a frontal chest radiograph to begin.
        </p>
        <p className="mt-1.5 text-xs text-ink-faint">
          The projection field changes the operating point that is applied.
        </p>
      </CardBody>
    </Card>
  );
}
