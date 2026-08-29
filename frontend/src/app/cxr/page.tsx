"use client";

import { useCallback, useEffect, useState } from "react";

import {
  ErrorNotice,
  FindingsGrid,
  GradCamViewer,
  AnalysisSplit,
  LoadedStudy,
  RawPayload,
  ReportViewer,
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
      <StudyGrid wide={Boolean(result && !pending)}>
        <div className="space-y-4">
          {result && !pending ? (
            <LoadedStudy
              name={file?.name ?? "Chest radiograph"}
              detail={result.headline}
              onReset={() => {
                setFile(null);
                reset();
              }}
            />
          ) : (
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
          )}

          {!(result && !pending) && (
          <Callout tone="neutral" title="Why accuracy is not the headline">
            Five of the eight pathologies score below an always-negative baseline on
            accuracy — an artefact of F1-optimal thresholds on rare disease. Read AUROC
            and sensitivity instead.
          </Callout>
          )}
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

              {/* Where it looked, beside what it said. One panel holds three
                  report views -- the cleaned text, the decoder's exact output,
                  and the radiologist's original -- because separate cards made
                  the raw output look like a second report rather than the same
                  one, unedited. */}
              <AnalysisSplit
                className="rise-in-1"
                left={
                  <GradCamViewer
                    originalUrl={previewUrl}
                    heatmapBase64={
                      (result.explanation.gradcam_png_base64 as string | undefined) ?? null
                    }
                    target="Cardiomegaly"
                    caveat={result.explanation.gradcam_caveat as string | undefined}
                  />
                }
                right={
                  <ReportViewer
                    cleaned={result.narrative ?? (result.raw?.report_text as string | undefined)}
                    raw={result.raw?.report_text_raw as string | undefined}
                    groundTruth={result.raw?.ground_truth_report as string | undefined}
                    prompt={result.raw?.classifier_prompt as string | undefined}
                  />
                }
              />

              <div className="rise-in-2 space-y-4">
                <FindingsGrid
                  findings={coFindings}
                  actionability={result.reliability.actionability}
                  title="Co-pathologies"
                  description="Reported in the same forward pass. Cardiomegaly is the diagnostic target; these are co-findings."
                />

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
