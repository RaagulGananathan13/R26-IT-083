import { Meter } from "@/components/ui";
import { cn, percent, VERDICT } from "@/lib/format";
import type { Reliability } from "@/lib/types";

/**
 * The reliability verdict, as the first thing on the page.
 *
 * Component 01's console carried a reliability notice with a coloured left
 * border; this is that idea generalised across four components whose honesty
 * mechanisms differ — projection deferral, conformal refusal, boundary
 * ambiguity, clinician referral — and reduced to one verdict a reader can act
 * on without knowing which mechanism fired.
 *
 * It is placed above the findings and carries the only saturated colour on the
 * page, because a mechanism that declines to commit is worthless if the
 * interface buries it under a large confident percentage.
 */
export function VerdictBanner({
  reliability,
  headline,
  confidence,
  className,
}: {
  reliability: Reliability;
  headline?: string;
  /** Primary probability, where the component has one. 0..1 */
  confidence?: { label: string; value: number; adverse?: boolean } | null;
  className?: string;
}) {
  const style = VERDICT[reliability.actionability];

  return (
    <section
      role="status"
      className={cn(
        "card rise-in overflow-hidden border-l-[3px] p-0",
        style.border,
        className,
      )}
      style={{ borderLeftColor: `rgb(var(--verdict-${reliability.actionability}))` }}
    >
      <div className={cn("px-5 py-4", style.bg)}>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className={cn("h-2 w-2 shrink-0 rounded-full", style.dot)} aria-hidden />
          <span className={cn("eyebrow", style.text)}>{style.label}</span>
          <span className="text-xs text-ink-muted">{style.meaning}</span>
          {reliability.coverage !== null && (
            <span className="ml-auto text-2xs text-ink-faint">
              {/* A selective metric without its coverage is meaningless. */}
              this operating point answers {percent(reliability.coverage, 0)} of studies
            </span>
          )}
        </div>

        <div className="mt-3 grid gap-5 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)] sm:items-center">
          {headline && (
            <p className="display text-[1.35rem] leading-snug text-ink">{headline}</p>
          )}
          {confidence && (
            <div>
              <div className="flex items-baseline justify-between">
                <span className="eyebrow">{confidence.label}</span>
                <span className="tabular text-lg font-semibold text-ink">
                  {percent(confidence.value, 1)}
                </span>
              </div>
              <Meter
                className="mt-2"
                value={confidence.value}
                tone={confidence.adverse ? "caution" : "accent"}
              />
            </div>
          )}
        </div>
      </div>

      {(reliability.guarantees_void ||
        reliability.reasons.length > 0 ||
        reliability.guarantees.length > 0) && (
        <div className="space-y-3 border-t border-line px-5 py-4">
          {reliability.guarantees_void && (
            <p className={cn("text-xs font-semibold", style.text)}>
              The statistical guarantees this component normally offers do NOT apply to
              this record.
            </p>
          )}

          {reliability.reasons.length > 0 && (
            <ul className="space-y-1.5">
              {reliability.reasons.map((reason, index) => (
                <li
                  key={index}
                  className="flex gap-2 text-xs leading-relaxed text-ink-muted"
                >
                  <span
                    aria-hidden
                    className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-faint"
                  />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          )}

          {reliability.guarantees.length > 0 && (
            <div className="border-t border-line pt-3">
              <p className="eyebrow mb-1.5">Guarantees that hold here</p>
              <ul className="space-y-1">
                {reliability.guarantees.map((guarantee, index) => (
                  <li key={index} className="text-xs leading-relaxed text-ink-muted">
                    {guarantee}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
