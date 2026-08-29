"use client";

import { useCallback, useEffect, useState } from "react";

import type { EchoCam } from "@/components/clinical";
import {
  AnalysisSplit,
  EchoGradCam,
  EchoViewer,
  ErrorNotice,
  ExplanationPanel,
  FindingsGrid,
  LoadedStudy,
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
  FileDrop,
  ResultSkeleton,
} from "@/components/ui";
import { useAnalysis } from "@/hooks/useAnalysis";
import { analyzeEcho } from "@/lib/api";
import { duration } from "@/lib/format";
import type { Envelope } from "@/lib/types";

/** Formats a browser can play back. `.avi` and `.npy` cannot be previewed. */
const PLAYABLE = [".mp4", ".webm", ".mov"];

export default function EchoPage() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    const name = file?.name.toLowerCase() ?? "";
    if (!file || !PLAYABLE.some((extension) => name.endsWith(extension))) {
      setVideoUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const run = useCallback((study: File) => analyzeEcho(study), []);
  const { result, error, pending, execute, reset } = useAnalysis<Envelope, File>(run);

  const ordinal =
    result?.findings.filter(
      (finding) =>
        finding.name !== "Severity grade" &&
        finding.name !== "Left-ventricular ejection fraction",
    ) ?? [];

  return (
    <StudyLayout
      id="echo"
      intro="Estimates left-ventricular ejection fraction from an apical four-chamber recording and assigns a four-class severity grade. Clip sampling is label-free: the annotated end-diastole and end-systole frames are never consulted, because they would not exist for a new clinical study."
      pills={[
        "R(2+1)D-18 · four heads",
        "3-seed ensemble · 10 clips",
        "Split-conformal interval",
        "MAE 3.98 EF points",
      ]}
    >
      <StudyGrid wide={Boolean(result && !pending)}>
        <div className="space-y-4">
          {result && !pending ? (
            <LoadedStudy
              name={file?.name ?? "Echocardiogram"}
              detail={`${result.headline}`}
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
                label="Echocardiogram"
                hint="AVI, MP4, MOV, MKV, WEBM — or a cached .npy clip array"
                accept=".avi,.mp4,.mov,.mkv,.webm,.npy"
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
                {pending ? "Analysing…" : "Analyse study"}
              </Button>
              {file && !videoUrl && (
                <p className="text-2xs leading-relaxed text-ink-faint">
                  This format cannot be played back in a browser, so no preview is shown.
                  Analysis is unaffected — the backend decodes it server-side.
                </p>
              )}
            </CardBody>
          </Card>
          )}

          {!(result && !pending) && (
            <>
          <Callout tone="neutral" title="At the level of human disagreement">
            Mean absolute error is 3.98 EF points against reported inter-observer
            variability of 4–5 points. A difference of that size between this estimate
            and a reader is not evidence of model error.
          </Callout>

          <Callout tone="warning" title="Abstention does not fix the floor">
            Deferring the least confident studies raises overall accuracy from 0.730 to
            0.929 while worst-class recall falls, because the minority classes occupy the
            boundary region abstention removes.
          </Callout>
            </>
          )}
        </div>

        <div className="space-y-4">
          <ErrorNotice error={error} />
          {pending && <ResultSkeleton />}

          {result && !pending && (
            <div className="space-y-4">
              <VerdictBanner reliability={result.reliability} headline={result.headline} />

              {/* The recording and its measurements beside the frames the
                  estimate leant on, so a weak-looking pump can be checked
                  against where the model was actually looking. */}
              <AnalysisSplit
                className="rise-in-1"
                left={<EchoViewer result={result} videoUrl={videoUrl} />}
                right={
                  <EchoGradCam
                    cam={(result.explanation.gradcam as EchoCam | undefined) ?? null}
                  />
                }
              />

              <div className="rise-in-2 space-y-4">
                <FindingsGrid
                  findings={ordinal}
                  actionability={result.reliability.actionability}
                  title="Ordinal head distribution"
                  description="From the ordered-cutpoint head, which guarantees rank consistency by construction rather than repairing it afterwards."
                />
                <ExplanationPanel explanation={result.explanation} title="Method" />
                <RawPayload payload={result.raw} />
                <p className="text-2xs text-ink-faint">
                  {duration(result.elapsed_ms)} · request {result.request_id}
                </p>
              </div>
            </div>
          )}

          {!result && !pending && !error && (
            <Card>
              <CardBody className="py-20 text-center">
                <p className="text-sm text-ink-muted">Upload an echocardiogram to begin.</p>
                <p className="mt-1.5 text-xs text-ink-faint">
                  Cached clips live under Component 03’s preprocessing cache.
                </p>
              </CardBody>
            </Card>
          )}
        </div>
      </StudyGrid>
    </StudyLayout>
  );
}
