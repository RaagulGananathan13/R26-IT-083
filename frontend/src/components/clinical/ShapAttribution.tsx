"use client";

import { frame } from "@/components/ui";
import { cn, decimal } from "@/lib/format";

interface ShapFeature {
  feature: string;
  contribution: number;
  value: number | null;
  direction: string;
}

const MODALITY_LABELS: Record<string, string> = {
  vitals: "Vitals",
  demographics: "Demographics",
  text: "Chief complaint",
  medications: "Medications",
  history: "History",
  ecg: "ECG",
  labs: "Laboratory",
  interaction: "Interactions",
  other: "Other",
};

/** `trop_max` reads as noise on a slide; `Troponin (max)` does not. */
function humanise(name: string): string {
  const known: Record<string, string> = {
    trop_max: "Troponin (max)",
    trop_t_first_h: "Hours to first troponin",
    ix_age_x_chestpain: "Age x chest pain",
    ix_heart_score: "HEART score",
    cc_acs_lexicon_score: "ACS lexicon score",
    ecg_n_studies: "ECG studies available",
    ecg_t_axis: "ECG T axis",
    pain_score: "Pain score",
  };
  if (known[name]) return known[name];
  return name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/**
 * Per-case SHAP for the triage model.
 *
 * The published modality percentages describe the whole test set. They are the
 * evidence for the information-horizon claim, but they cannot say why THIS
 * patient was scored the way they were — and a panel asking "why did it say
 * that about him?" is not answered by a cohort average.
 *
 * The modality bars are worth watching at stage 1 specifically. The laboratory
 * channel reads 0 % there, per patient, because no troponin exists ten seconds
 * after someone walks through the door. That is the horizon being enforced
 * rather than asserted, and it is visible live.
 */
export function ShapAttribution({
  features,
  modality,
  note,
  horizon,
  bare,
}: {
  features: ShapFeature[];
  modality: Record<string, number>;
  note?: string;
  horizon?: number;
  bare?: boolean;
}) {
  const { Frame, FrameHeader, FrameBody } = frame(bare);
  const entries = Object.entries(modality ?? {});
  if ((!features || features.length === 0) && entries.length === 0) return null;

  const largest = Math.max(1e-9, ...features.map((f) => Math.abs(f.contribution)));
  const labs = modality?.labs;

  return (
    <Frame>
      <FrameHeader
        title="Why this patient"
        description={
          horizon !== undefined
            ? `Per-case attribution at the H = ${horizon} h disclosure horizon.`
            : "Per-case attribution for this record."
        }
      />
      <FrameBody className="space-y-5">
        {entries.length > 0 && (
          <div>
            <p className="eyebrow mb-2">Evidence by channel</p>
            <ul className="space-y-1.5">
              {entries.map(([name, share]) => (
                <li key={name} className="flex items-center gap-2">
                  <span className="w-28 shrink-0 text-2xs text-ink-muted">
                    {MODALITY_LABELS[name] ?? name}
                  </span>
                  <span className="relative h-3 flex-1 rounded bg-surface-2">
                    <span
                      className="absolute inset-y-0 left-0 rounded bg-accent"
                      style={{ width: `${Math.max(share * 100, share > 0 ? 1 : 0)}%` }}
                    />
                  </span>
                  <span className="tabular w-12 shrink-0 text-right text-2xs text-ink-muted">
                    {(share * 100).toFixed(1)} %
                  </span>
                </li>
              ))}
            </ul>
            {labs !== undefined && labs === 0 && (
              <p className="mt-2 rounded-lg border border-verdict-actionable/30 bg-verdict-actionable/10 px-3 py-2 text-2xs leading-relaxed text-ink-muted">
                The laboratory channel carries exactly zero attribution for this
                patient. No troponin has been drawn yet, and the model is not
                reaching for one — a pipeline with a temporal leak could not
                produce this.
              </p>
            )}
          </div>
        )}

        {features.length > 0 && (
          <div>
            <p className="eyebrow mb-2">Strongest features</p>
            <ul className="space-y-1.5">
              {features.map((feature) => {
                const width = (Math.abs(feature.contribution) / largest) * 50;
                const positive = feature.contribution >= 0;
                return (
                  <li key={feature.feature} className="flex items-center gap-2">
                    <span
                      className="w-36 shrink-0 truncate text-2xs text-ink-muted"
                      title={feature.feature}
                    >
                      {humanise(feature.feature)}
                    </span>
                    <span className="relative h-3 flex-1 rounded bg-surface-2">
                      <span className="absolute inset-y-0 left-1/2 w-px bg-line" aria-hidden />
                      <span
                        className={cn(
                          "absolute inset-y-0 rounded",
                          positive ? "bg-verdict-caution" : "bg-accent",
                        )}
                        style={
                          positive
                            ? { left: "50%", width: `${width}%` }
                            : { right: "50%", width: `${width}%` }
                        }
                      />
                    </span>
                    <span className="tabular w-16 shrink-0 text-right text-2xs text-ink-muted">
                      {feature.value === null ? "absent" : decimal(feature.value, 1)}
                    </span>
                  </li>
                );
              })}
            </ul>
            <p className="mt-2 text-2xs leading-relaxed text-ink-faint">
              Right of centre pushed this patient towards acute coronary syndrome;
              left pushed away. The right-hand column is the feature&rsquo;s value for
              this patient — <span className="italic">absent</span> where it has not been
              measured, which is itself informative to the model.
            </p>
          </div>
        )}

        {note && <p className="text-2xs leading-relaxed text-ink-faint">{note}</p>}
      </FrameBody>
    </Frame>
  );
}
