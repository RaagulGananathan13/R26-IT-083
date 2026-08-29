"use client";

import { useState } from "react";

import { Segmented, frame } from "@/components/ui";
import { cn, decimal } from "@/lib/format";

interface CamFrame {
  rank: number;
  clip_frame: number;
  source_frame: number;
  importance: number;
  png_base64: string;
}

interface NativeResolution {
  temporal_bins?: number;
  spatial?: string;
  upsampled_to?: string;
  temporal_bins_with_signal?: number;
  caveat?: string;
}

export interface EchoCam {
  frame_importance?: number[];
  frame_importance_peak?: number;
  frame_importance_spread?: number;
  source_frame_indices?: number[];
  clip_index?: number;
  clip_count?: number;
  member_run?: string;
  target?: string;
  note?: string;
  frames?: CamFrame[];
  native_resolution?: NativeResolution;
  concentration?: { "above_0.5"?: number; "above_0.25"?: number; note?: string };
  flat_attribution?: boolean;
  flat_attribution_note?: string;
  partial_temporal_support?: string;
  degenerate?: boolean;
}

/**
 * Grad-CAM for the echocardiogram.
 *
 * The radiograph gets one image and one map. A video does not: the reported
 * ejection fraction is a mean over several clips and two ensemble members, and
 * a map averaged the same way would blur across different phases of the cardiac
 * cycle into something smooth and meaningless. So the map belongs to ONE clip,
 * and the panel says which one rather than letting it be mistaken for an
 * explanation of the reported number.
 *
 * The resolution caveat is not decoration. The map is computed at 4 x 7 x 7 and
 * interpolated up to 32 x 112 x 112 for display — a sixteen-fold spatial
 * stretch. The overlay therefore has smooth, confident-looking edges that are
 * interpolation rather than evidence, and a reader who is not told that will
 * over-read them.
 */
export function EchoGradCam({
  cam,
  bare,
}: {
  cam: EchoCam | null | undefined;
  bare?: boolean;
}) {
  const { Frame, FrameHeader, FrameBody } = frame(bare);
  const frames = cam?.frames ?? [];
  const [selected, setSelected] = useState("0");

  if (!cam) return null;

  if (cam.degenerate) {
    return (
      <Frame>
        <FrameHeader title="Grad-CAM focus map" />
        <FrameBody>
          <p className="rounded-lg border border-verdict-caution/30 bg-verdict-caution/10 px-3 py-2 text-xs leading-relaxed text-ink-muted">
            {cam.note}
          </p>
        </FrameBody>
      </Frame>
    );
  }

  if (frames.length === 0) return null;

  const index = Math.min(Math.max(0, Number(selected) || 0), frames.length - 1);
  const active = frames[index];
  if (!active) return null;
  const curve = cam.frame_importance ?? [];
  const native = cam.native_resolution;
  const options = frames.map((frame, position) => ({
    value: String(position),
    label: `Frame ${frame.source_frame}`,
  }));

  const warnings = [
    cam.partial_temporal_support,
    cam.flat_attribution ? cam.flat_attribution_note : undefined,
  ].filter((line): line is string => Boolean(line));

  const small = [native?.caveat, cam.note].filter(
    (line): line is string => Boolean(line),
  );

  return (
    <Frame>
      <FrameHeader
        title="Grad-CAM focus map"
        description={
          cam.clip_count
            ? `Clip ${(cam.clip_index ?? 0) + 1} of ${cam.clip_count}, from member ${cam.member_run ?? "—"}. Gradient of the continuous ejection fraction.`
            : "Gradient of the continuous ejection fraction, back-propagated to the last spatiotemporal convolution."
        }
        actions={
          options.length > 1 ? (
            <Segmented
              options={options}
              value={String(index)}
              onChange={setSelected}
              ariaLabel="Which frame to show"
            />
          ) : undefined
        }
      />
      <FrameBody className="space-y-4">
        <div className="image-frame">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${active.png_base64}`}
            alt={`Grad-CAM attribution on source frame ${active.source_frame}`}
          />
        </div>

        <dl className="grid gap-3 rounded-xl border border-line bg-surface-2 px-4 py-3 sm:grid-cols-3">
          <div>
            <dt className="eyebrow">Source frame</dt>
            <dd className="tabular mt-0.5 text-sm font-medium text-ink">
              {active.source_frame}
            </dd>
          </div>
          <div>
            <dt className="eyebrow">Frame importance</dt>
            <dd className="tabular mt-0.5 text-sm font-medium text-ink">
              {decimal(active.importance, 3)}
            </dd>
          </div>
          <div>
            <dt className="eyebrow">Attribution spread</dt>
            <dd className="tabular mt-0.5 text-sm font-medium text-ink">
              {cam.frame_importance_spread !== undefined
                ? `${Math.round(cam.frame_importance_spread * 100)} %`
                : "—"}
            </dd>
          </div>
        </dl>

        {curve.length > 1 && (
          <div>
            <p className="eyebrow mb-2">Importance across the clip</p>
            <div className="flex h-14 items-end gap-px" aria-hidden>
              {curve.map((value, position) => (
                <span
                  key={position}
                  className={cn(
                    "flex-1 rounded-sm",
                    position === active.clip_frame ? "bg-accent" : "bg-accent/30",
                  )}
                  style={{ height: `${Math.max(2, value * 100)}%` }}
                />
              ))}
            </div>
            <p className="mt-1.5 text-2xs text-ink-faint">
              Normalised to the peak of this clip. The highlighted bar is the frame shown above.
            </p>
          </div>
        )}

        {warnings.length > 0 && (
          <div className="space-y-1 rounded-lg border border-verdict-caution/30 bg-verdict-caution/10 px-3 py-2">
            {warnings.map((warning) => (
              <p key={warning} className="text-xs leading-relaxed text-ink-muted">
                {warning}
              </p>
            ))}
          </div>
        )}

        {/* One disclosure rather than a stack of grey boxes. The map is the
            single most over-readable thing on this page, so what it cannot
            support stays one click away — never removed, never four blocks
            deep pushing the result off the screen. */}
        {small.length > 0 && (
          <details className="group rounded-lg border border-line bg-surface-2 px-3 py-2">
            <summary className="cursor-pointer list-none text-2xs font-medium text-ink-muted marker:content-none">
              <span className="group-open:hidden">What this map cannot support ▸</span>
              <span className="hidden group-open:inline">What this map cannot support ▾</span>
            </summary>
            <div className="mt-2 space-y-1.5">
              {small.map((line) => (
                <p key={line} className="text-2xs leading-relaxed text-ink-faint">
                  {line}
                </p>
              ))}
            </div>
          </details>
        )}
      </FrameBody>
    </Frame>
  );
}
