"use client";

import { frame } from "@/components/ui";
import { decimal } from "@/lib/format";

/**
 * The temporal Grad-CAM curve over an ECG strip.
 *
 * The rendered strip already carries a shaded attention band, but a band drawn
 * behind a waveform can only say *where* — it cannot show the shape of the
 * attention, and the shape is the part that tells a reader whether to trust the
 * probability. A single sharp spike means the decision rests on one complex. A
 * broad plateau means the model spread its attention across the whole strip,
 * which for a focal diagnosis is a reason for caution regardless of how
 * confident the number looks.
 *
 * So this draws the curve itself, with the independently computed peak times
 * marked on it. Those peaks come from a different code path in the component
 * (`cam_peaks_s`), so their landing on the maxima of this curve is a check that
 * the two agree rather than a redundant restatement.
 */
export function TemporalAttention({
  curve,
  durationSeconds,
  peaks,
  note,
  bare,
}: {
  curve: number[];
  durationSeconds: number;
  peaks?: number[];
  note?: string;
  bare?: boolean;
}) {
  const { Frame, FrameHeader, FrameBody } = frame(bare);
  if (!curve || curve.length < 2 || !durationSeconds) return null;

  const width = 1000;
  const height = 120;
  const peak = Math.max(...curve);
  if (!Number.isFinite(peak) || peak <= 0) return null;

  const x = (index: number) => (index / (curve.length - 1)) * width;
  const y = (value: number) => height - (value / peak) * (height - 8) - 4;

  const line = curve.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const area = `0,${height} ${line} ${width},${height}`;

  // How concentrated the attention is: the share of total mass sitting in the
  // top decile of time. A perfectly focal curve approaches 1; a flat curve
  // approaches 0.1, which is exactly the fraction of the strip it occupies.
  const total = curve.reduce((sum, value) => sum + value, 0);
  const sorted = [...curve].sort((a, b) => b - a);
  const decile = Math.max(1, Math.round(curve.length * 0.1));
  const topMass = sorted.slice(0, decile).reduce((sum, value) => sum + value, 0);
  const concentration = total > 0 ? topMass / total : 0;
  const focal = concentration >= 0.3;

  const ticks = Array.from({ length: Math.floor(durationSeconds) + 1 }, (_, second) => second);

  return (
    <Frame>
      <FrameHeader
        title="Temporal attention"
        description="Grad-CAM over the strip, normalised to its own peak. The shape matters as much as the position."
      />
      <FrameBody className="space-y-3">
        <div className="overflow-x-auto">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="h-28 w-full min-w-[420px]"
            role="img"
            aria-label={`Temporal attention across ${decimal(durationSeconds, 1)} seconds`}
            preserveAspectRatio="none"
          >
            <polygon points={area} className="fill-accent/15" />
            <polyline
              points={line}
              className="stroke-accent"
              fill="none"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
            {peaks?.map((second) => {
              const position = (second / durationSeconds) * width;
              if (!Number.isFinite(position)) return null;
              return (
                <line
                  key={second}
                  x1={position}
                  x2={position}
                  y1={0}
                  y2={height}
                  className="stroke-verdict-caution"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
          </svg>
        </div>

        <div className="flex justify-between px-0.5">
          {ticks.map((second) => (
            <span key={second} className="tabular text-2xs text-ink-faint">
              {second}s
            </span>
          ))}
        </div>

        <dl className="grid gap-3 rounded-xl border border-line bg-surface-2 px-4 py-3 sm:grid-cols-2">
          <div>
            <dt className="eyebrow">Shape</dt>
            <dd className="mt-0.5 text-sm font-medium text-ink">
              {focal ? "Focal" : "Diffuse"}
            </dd>
          </div>
          <div>
            <dt className="eyebrow">Mass in busiest 10 % of the strip</dt>
            <dd className="tabular mt-0.5 text-sm font-medium text-ink">
              {Math.round(concentration * 100)} %
            </dd>
          </div>
        </dl>

        <p className="text-2xs leading-relaxed text-ink-faint">
          {note ??
            "Dashed lines mark the peak times reported by the component. A broad, flat curve means the decision does not rest on any single complex."}
        </p>
      </FrameBody>
    </Frame>
  );
}
