import { Card, CardBody, CardHeader } from "@/components/ui";
import { humanise } from "@/lib/format";

const IMAGE_KEYS = ["gradcam_png_base64", "ecg_png_base64"];
const CAVEAT_KEYS = ["gradcam_caveat", "territory_caveat"];

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
  if (entries.length === 0) return null;

  const images = entries.filter(([key]) => IMAGE_KEYS.includes(key));
  const caveats = entries.filter(([key]) => CAVEAT_KEYS.includes(key));
  const rest = entries.filter(
    ([key]) => !IMAGE_KEYS.includes(key) && !CAVEAT_KEYS.includes(key),
  );

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
          <dl className="grid gap-x-6 gap-y-2 text-xs sm:grid-cols-2">
            {rest.map(([key, value]) => (
              <div key={key} className={typeof value === "string" && value.length > 60 ? "sm:col-span-2" : ""}>
                <dt className="eyebrow">{humanise(key)}</dt>
                <dd className="mt-0.5 leading-relaxed text-ink-muted">{render(value)}</dd>
              </div>
            ))}
          </dl>
        )}
      </CardBody>
    </Card>
  );
}

function render(value: unknown): string {
  if (Array.isArray(value)) {
    if (value.length === 0) return "—";
    if (typeof value[0] === "object") return JSON.stringify(value);
    return value.join(", ");
  }
  if (typeof value === "object" && value !== null) return JSON.stringify(value);
  return String(value);
}
