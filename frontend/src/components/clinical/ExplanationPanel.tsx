import { Card, CardBody, CardHeader } from "@/components/ui";
import { humanise } from "@/lib/format";

const IMAGE_KEYS = ["gradcam_png_base64", "ecg_png_base64"];
const CAVEAT_KEYS = ["gradcam_caveat", "territory_caveat"];

/**
 * Keys that a dedicated component already renders properly.
 *
 * This list is a courtesy, not the safety net — the safety net is `isProse`
 * below, which refuses to print anything structured. Without that rule every
 * new structured field a component starts returning lands here as raw JSON,
 * which is exactly what happened when the echocardiogram gained a Grad-CAM
 * payload: a wall of `{"frame_importance":[0.83,0.83,…` in the middle of the
 * method panel.
 */
const RENDERED_ELSEWHERE = new Set([
  // 01 · radiograph — GradCamViewer
  "gradcam_target",
  "classifier_prompt",
  // 02 · ECG — EcgStripViewer and TemporalAttention
  "cam",
  "camSeconds",
  "camNote",
  "peaksSeconds",
  "peaks_seconds",
  "topLeads",
  "top_leads",
  "lead_attribution",
  "territory",
  "territory_score",
  "artery",
  "target",
  // 03 · echo — EchoGradCam
  "gradcam",
  // 04 · triage — TextAttribution and ShapAttribution
  "text_attribution",
  "shap_top_features",
  "shap_modality_contribution",
  "shap_note",
  "modality_attribution_note",
]);

/** Longer than this and an array is chart data, not a sentence. */
const MAX_LIST = 12;

/**
 * True only for values that read as prose in a definition list.
 *
 * Objects and arrays of objects are always someone else's job. Printing them
 * with `JSON.stringify` produces something that is technically the truth and
 * practically unreadable, and it pushes everything below it off the screen.
 */
function isProse(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value)) {
    if (value.length === 0 || value.length > MAX_LIST) return false;
    return value.every((item) => typeof item !== "object" || item === null);
  }
  return typeof value !== "object";
}

function render(value: unknown): string {
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "number" ? String(Number(item.toFixed(4))) : String(item)))
      .join(", ");
  }
  if (typeof value === "number") return String(Number(value.toFixed(4)));
  return String(value);
}

/**
 * Explanations, with their caveats attached rather than in a footnote.
 *
 * Saliency maps are the part of a system like this most likely to be
 * over-read, so where the component ships a caveat it is rendered next to the
 * image, not below the fold.
 */
export function ExplanationPanel({
  explanation,
  title = "Explanation",
}: {
  explanation: Record<string, unknown>;
  title?: string;
}) {
  const entries = Object.entries(explanation).filter(
    ([, value]) => value !== null && value !== undefined && value !== "",
  );

  const images = entries.filter(
    ([key, value]) => IMAGE_KEYS.includes(key) && typeof value === "string",
  );
  const caveats = entries.filter(([key]) => CAVEAT_KEYS.includes(key));
  const rest = entries.filter(
    ([key, value]) =>
      !IMAGE_KEYS.includes(key) &&
      !CAVEAT_KEYS.includes(key) &&
      !RENDERED_ELSEWHERE.has(key) &&
      isProse(value),
  );

  if (images.length === 0 && caveats.length === 0 && rest.length === 0) return null;

  return (
    <Card>
      <CardHeader title={title} />
      <CardBody className="space-y-4">
        {images.map(([key, value]) => (
          <figure key={key}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${String(value)}`}
              alt={humanise(key)}
              className="w-full rounded-lg border border-line bg-black"
            />
          </figure>
        ))}

        {caveats.length > 0 && (
          <div className="rounded-lg border border-verdict-caution/30 bg-verdict-caution/10 px-3 py-2">
            {caveats.map(([key, value]) => (
              <p key={key} className="text-xs leading-relaxed text-ink-muted">
                {String(value)}
              </p>
            ))}
          </div>
        )}

        {rest.length > 0 && (
          <dl className="grid gap-x-6 gap-y-3 text-xs sm:grid-cols-2">
            {rest.map(([key, value]) => {
              const text = render(value);
              return (
                <div key={key} className={text.length > 60 ? "sm:col-span-2" : ""}>
                  <dt className="eyebrow">{humanise(key)}</dt>
                  <dd className="mt-0.5 leading-relaxed text-ink-muted">{text}</dd>
                </div>
              );
            })}
          </dl>
        )}
      </CardBody>
    </Card>
  );
}
