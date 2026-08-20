"use client";

import { useState } from "react";

import { Card, CardBody, CardHeader, Segmented } from "@/components/ui";
import { cn } from "@/lib/format";

type Mode = "original" | "overlay" | "heatmap";

const MODES = [
  { value: "original" as const, label: "Original" },
  { value: "overlay" as const, label: "Overlay" },
  { value: "heatmap" as const, label: "Heatmap" },
];

/**
 * Grad-CAM with the three-way toggle from Component 01's console.
 *
 * The toggle matters more than it looks. A permanently blended overlay makes it
 * impossible to tell whether a bright region is anatomy or attribution — the
 * reader cannot separate the map from the film underneath it. Being able to
 * flick between the radiograph alone, the map alone, and the blend is what
 * turns the heatmap into something checkable.
 *
 * The caveat travels with the image rather than sitting in a footnote: Grad-CAM
 * repeatability on chest radiographs was measured at SSIM 0.12 (Arun et al.,
 * Radiology: AI 2021), so this is a sanity check on where the model looked, not
 * localisation evidence.
 */
export function GradCamViewer({
  originalUrl,
  heatmapBase64,
  target = "Cardiomegaly",
  caveat,
}: {
  /** Object URL of the uploaded study, kept client-side. */
  originalUrl: string | null;
  heatmapBase64: string | null;
  target?: string;
  caveat?: string;
}) {
  const [mode, setMode] = useState<Mode>("overlay");
  const [opacity, setOpacity] = useState(0.7);

  if (!heatmapBase64) return null;
  const heatmapUrl = `data:image/png;base64,${heatmapBase64}`;

  return (
    <Card>
      <CardHeader
        title="Grad-CAM focus map"
        description={`Gradient of the ${target.toLowerCase()} logit, back-propagated to the final convolutional stage.`}
        actions={
          <Segmented
            options={MODES}
            value={mode}
            onChange={setMode}
            ariaLabel="Grad-CAM view mode"
          />
        }
      />
      <CardBody className="space-y-3">
        <div className="image-frame">
          {originalUrl ? (
            <img
              src={originalUrl}
              alt="Uploaded study"
              style={{
                opacity: mode === "heatmap" ? 0 : 1,
                transition: "opacity 240ms ease",
              }}
            />
          ) : (
            <div className="absolute inset-0 grid place-items-center">
              <p className="px-6 text-center text-xs text-white/60">
                The original study is not available for side-by-side comparison.
              </p>
            </div>
          )}
          <img
            src={heatmapUrl}
            alt={`Grad-CAM attribution for ${target}`}
            style={{
              opacity: mode === "original" ? 0 : mode === "heatmap" ? 1 : opacity,
              transition: "opacity 240ms ease",
              // `screen` keeps the underlying anatomy legible through the map;
              // a plain alpha blend washes the film out at useful opacities.
              mixBlendMode: mode === "overlay" ? "screen" : "normal",
            }}
          />
        </div>

        <div
          className={cn(
            "flex items-center gap-3 transition-opacity",
            mode === "overlay" ? "opacity-100" : "pointer-events-none opacity-35",
          )}
        >
          <label htmlFor="cam-opacity" className="eyebrow shrink-0">
            Blend
          </label>
          <input
            id="cam-opacity"
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={opacity}
            disabled={mode !== "overlay"}
            onChange={(event) => setOpacity(Number(event.target.value))}
            className="h-1 w-full cursor-pointer appearance-none rounded-full bg-surface-2 accent-accent"
          />
          <span className="tabular w-10 shrink-0 text-right text-2xs text-ink-faint">
            {Math.round(opacity * 100)} %
          </span>
        </div>

        <p className="rounded-lg border border-verdict-caution/30 bg-verdict-caution/10 px-3 py-2 text-xs leading-relaxed text-ink-muted">
          {caveat ??
            "Grad-CAM shows where the model looked, not whether it was right. Repeatability on chest radiographs was measured at SSIM 0.12, so read this as a sanity check — did it attend to the cardiac silhouette? — never as localisation evidence."}
        </p>
      </CardBody>
    </Card>
  );
}
