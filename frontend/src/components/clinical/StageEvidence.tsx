"use client";

import { EchoGradCam, type EchoCam } from "./EchoGradCam";
import { EcgStripViewer } from "./EcgStripViewer";
import { GradCamViewer } from "./GradCamViewer";
import { ShapAttribution } from "./ShapAttribution";
import { TemporalAttention } from "./TemporalAttention";
import { TextAttribution } from "./TextAttribution";
import type { ComponentId, Envelope } from "@/lib/types";

/**
 * The right explanation for whichever component ran this stage.
 *
 * The pathway used to show a headline, a findings table and a routing
 * decision, and nothing about *why* — which is an odd thing to withhold in a
 * system whose whole claim is explainability. Each console already had the
 * right view built; the pathway simply was not reaching for it.
 *
 * Everything renders `bare`, because it is nested inside the stage panel and a
 * card within a card reads as a mistake.
 */
export function StageEvidence({
  component,
  envelope,
  originalUrl,
}: {
  component: ComponentId | null;
  envelope: Envelope;
  /** Object URL of the uploaded radiograph, for the Grad-CAM comparison. */
  originalUrl?: string | null;
}) {
  const explanation = (envelope.explanation ?? {}) as Record<string, unknown>;
  const raw = (envelope.raw ?? {}) as Record<string, unknown>;

  if (component === "cxr") {
    const heatmap = explanation.gradcam_png_base64 as string | undefined;
    if (!heatmap) return <Empty />;
    return (
      <GradCamViewer
        bare
        originalUrl={originalUrl ?? null}
        heatmapBase64={heatmap}
        target={(explanation.gradcam_target as string | undefined) ?? "Cardiomegaly"}
        caveat={explanation.gradcam_caveat as string | undefined}
      />
    );
  }

  if (component === "ecg") {
    const strip = explanation.ecg_png_base64 as string | undefined;
    const curve = (explanation.cam as number[] | undefined) ?? [];
    if (!strip && curve.length === 0) return <Empty />;
    return (
      <div className="space-y-6">
        <EcgStripViewer
          bare
          imageBase64={strip ?? null}
          leads={(explanation.lead_attribution as never[]) ?? []}
          explanation={explanation}
        />
        <TemporalAttention
          bare
          curve={curve}
          durationSeconds={(explanation.camSeconds as number | undefined) ?? 0}
          peaks={explanation.peaksSeconds as number[] | undefined}
          note={explanation.camNote as string | undefined}
        />
      </div>
    );
  }

  if (component === "echo") {
    const cam = explanation.gradcam as EchoCam | undefined;
    if (!cam) return <Empty />;
    return <EchoGradCam bare cam={cam} />;
  }

  if (component === "triage") {
    const features = (explanation.shap_top_features as never[]) ?? [];
    const modality =
      (explanation.shap_modality_contribution as Record<string, number>) ?? {};
    const tokens = (explanation.text_attribution as never[]) ?? [];
    if (features.length === 0 && Object.keys(modality).length === 0 && tokens.length === 0) {
      return <Empty />;
    }
    return (
      <div className="space-y-6">
        <ShapAttribution
          bare
          features={features}
          modality={modality}
          note={explanation.shap_note as string | undefined}
          horizon={raw.horizon_h as number | undefined}
        />
        <TextAttribution
          bare
          complaint={String(raw.chief_complaint ?? raw.complaint ?? "")}
          tokens={tokens}
          modalityNote={explanation.modality_attribution_note as string | undefined}
        />
      </div>
    );
  }

  return <Empty />;
}

/** Whether this stage has anything to put in an evidence tab. */
export function hasStageEvidence(
  component: ComponentId | null,
  envelope: Envelope | null | undefined,
): boolean {
  if (!component || !envelope) return false;
  const explanation = (envelope.explanation ?? {}) as Record<string, unknown>;
  if (component === "cxr") return Boolean(explanation.gradcam_png_base64);
  if (component === "ecg") {
    return Boolean(explanation.ecg_png_base64) ||
      ((explanation.cam as number[] | undefined)?.length ?? 0) > 0;
  }
  if (component === "echo") return Boolean(explanation.gradcam);
  if (component === "triage") {
    return ((explanation.shap_top_features as unknown[] | undefined)?.length ?? 0) > 0 ||
      ((explanation.text_attribution as unknown[] | undefined)?.length ?? 0) > 0;
  }
  return false;
}

function Empty() {
  return (
    <p className="py-6 text-center text-xs text-ink-faint">
      This stage produced no visual attribution.
    </p>
  );
}
