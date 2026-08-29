"use client";

import { frame } from "@/components/ui";
import { cn, decimal } from "@/lib/format";

interface LeadAttribution {
  name: string;
  signed: number;
  magnitude?: number;
}

/**
 * The rendered 12-lead strip, plus signed per-lead attribution.
 *
 * Component 02 bakes the Grad-CAM band into the strip server-side, because the
 * attribution is over time and only means anything drawn against the waveform
 * it belongs to. So unlike the radiograph there is no overlay toggle — there is
 * one image, and the useful second view is the per-lead breakdown beside it.
 *
 * Attribution is *signed* deliberately. Leads pushing towards the class and
 * leads pushing away are different clinical statements, and a magnitude-only
 * bar chart erases that distinction.
 */
export function EcgStripViewer({
  imageBase64,
  leads,
  explanation,
  bare,
}: {
  imageBase64: string | null;
  leads: LeadAttribution[];
  explanation: Record<string, unknown>;
  bare?: boolean;
}) {
  const { Frame, FrameHeader, FrameBody } = frame(bare);
  const target = explanation.target ?? explanation.gradcam_target;
  const territory = explanation.territory as string | undefined;
  const artery = explanation.artery as string | undefined;
  // The heuristic localises whichever class the Grad-CAM targeted, and that is
  // always MI. When MI was not ruled in, the territory still has a value and a
  // named artery -- and printed plainly it reads as a localised occlusion in a
  // patient who has no infarct on this trace.
  const territoryApplies = explanation.territory_applies !== false;
  const topLeads = (explanation.topLeads ?? explanation.top_leads) as string[] | undefined;
  const peaks = (explanation.peaksSeconds ?? explanation.peaks_seconds) as number[] | undefined;

  const maxMagnitude = Math.max(1, ...leads.map((lead) => Math.abs(lead.signed)));

  return (
    <Frame>
      <FrameHeader
        title="Signal and attribution"
        description={
          target
            ? `Attribution computed for ${String(target)}. The highlighted band on the strip is where the model attended.`
            : "Twelve-lead strip as the model received it, after band-pass filtering and resampling."
        }
      />
      <FrameBody className="space-y-4">
        {/* A 12-lead strip is 1924x1146. At full width in the pathway panel
            that is 774 px tall; capping the height keeps all twelve leads
            legible without swallowing the screen. */}
        {imageBase64 && (
          <div className="overflow-hidden rounded-xl border border-line bg-white">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${imageBase64}`}
              alt="Twelve-lead ECG with attribution band"
              className="mx-auto block max-h-[22rem] w-auto max-w-full"
            />
          </div>
        )}

        {(territory || topLeads?.length || peaks?.length) && (
          <dl className="grid gap-3 rounded-xl border border-line bg-surface-2 px-4 py-3 sm:grid-cols-3">
            {territory && (
              <div>
                <dt className="eyebrow">Territory</dt>
                <dd
                  className={cn(
                    "mt-0.5 text-sm font-medium capitalize",
                    territoryApplies ? "text-ink" : "text-ink-faint",
                  )}
                >
                  {territory}
                  {artery && (
                    <span className="ml-1 font-normal text-ink-muted">({artery})</span>
                  )}
                </dd>
                {!territoryApplies && (
                  <p className="mt-1 text-2xs leading-relaxed text-verdict-caution">
                    Infarction was not ruled in on this trace. This is where the
                    MI-class attribution fell, not a localised occlusion.
                  </p>
                )}
              </div>
            )}
            {topLeads && topLeads.length > 0 && (
              <div>
                <dt className="eyebrow">Strongest leads</dt>
                <dd className="mt-0.5 text-sm font-medium text-ink">{topLeads.join(", ")}</dd>
              </div>
            )}
            {peaks && peaks.length > 0 && (
              <div>
                <dt className="eyebrow">Attention peaks</dt>
                <dd className="tabular mt-0.5 text-sm font-medium text-ink">
                  {peaks.map((peak) => `${decimal(peak, 1)} s`).join(", ")}
                </dd>
              </div>
            )}
          </dl>
        )}

        {leads.length > 0 && (
          <div>
            <p className="eyebrow mb-2">Signed lead attribution</p>
            <ul className="space-y-1.5">
              {leads.map((lead) => {
                const width = (Math.abs(lead.signed) / maxMagnitude) * 50;
                const positive = lead.signed >= 0;
                return (
                  <li key={lead.name} className="flex items-center gap-2">
                    <span className="w-10 shrink-0 font-mono text-2xs text-ink-muted">
                      {lead.name}
                    </span>
                    {/* Centre line: towards the class on the right, away on the left. */}
                    <span className="relative h-3 flex-1 rounded bg-surface-2">
                      <span className="absolute inset-y-0 left-1/2 w-px bg-line" aria-hidden />
                      <span
                        className={cn(
                          "absolute inset-y-0 rounded",
                          positive ? "bg-accent" : "bg-verdict-caution",
                        )}
                        style={
                          positive
                            ? { left: "50%", width: `${width}%` }
                            : { right: "50%", width: `${width}%` }
                        }
                      />
                    </span>
                    <span className="tabular w-14 shrink-0 text-right text-2xs text-ink-muted">
                      {lead.signed > 0 ? "+" : ""}
                      {decimal(lead.signed, 1)}
                    </span>
                  </li>
                );
              })}
            </ul>
            <p className="mt-2 text-2xs leading-relaxed text-ink-faint">
              Right of centre pushes towards the reported class; left pushes away.
              Territory localisation is a lead-group heuristic and has not been
              clinically validated.
            </p>
          </div>
        )}
      </FrameBody>
    </Frame>
  );
}
