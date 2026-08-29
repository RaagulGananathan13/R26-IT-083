"use client";

/**
 * Where the pathway ended, and on what evidence.
 *
 * This sits at the top of the result because it is the one thing a reader needs
 * first. Two rules shape it:
 *
 *   Urgency drives the visual weight, not the destination. A catheterisation
 *   laboratory referral and a discharge are both "a destination"; only one of
 *   them has a 90-minute clock attached, and that difference has to be legible
 *   before the text is read.
 *
 *   The overall reliability verdict is shown beside the destination, never
 *   below it. A disposition derived from a deferred or withheld stage is not a
 *   recommendation, and separating the two invites the destination to be read
 *   on its own.
 */
import { cn, DESTINATION, URGENCY, VERDICT } from "@/lib/format";
import type { Actionability, Disposition } from "@/lib/types";

export function DispositionBanner({
  disposition,
  actionability,
  terminationReason,
  stagesCompleted,
  stagesTotal,
}: {
  disposition: Disposition;
  actionability: Actionability;
  terminationReason: string | null;
  stagesCompleted: number;
  stagesTotal: number;
}) {
  const urgency = URGENCY[disposition.urgency];
  const verdict = VERDICT[actionability];
  const destination = DESTINATION[disposition.destination];
  const immediate = disposition.urgency === "immediate";

  return (
    <section
      className={cn(
        "rounded-2xl border p-5 sm:p-6",
        immediate ? "border-verdict-withheld/40 bg-verdict-withheld/5" : urgency.border,
        !immediate && urgency.bg,
      )}
      aria-label="Disposition"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="font-mono text-2xs uppercase tracking-widest text-ink-faint">
            Disposition
          </p>
          <h2 className="mt-1 font-display text-xl font-semibold text-ink sm:text-2xl">
            {disposition.label}
          </h2>
          <p className="mt-1 text-sm text-ink-muted">{destination.note}</p>
        </div>

        <div className="flex flex-none flex-col items-start gap-2 sm:items-end">
          <span
            className={cn(
              "rounded-full border px-2.5 py-1 font-mono text-2xs uppercase tracking-wide",
              immediate
                ? "border-verdict-withheld/40 bg-verdict-withheld/10 text-verdict-withheld"
                : cn(urgency.border, urgency.bg, urgency.text),
            )}
          >
            {urgency.label}
          </span>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-2xs uppercase tracking-wide",
              verdict.border,
              verdict.bg,
              verdict.text,
            )}
          >
            <span className={cn("h-1.5 w-1.5 rounded-full", verdict.dot)} aria-hidden />
            {verdict.label}
          </span>
        </div>
      </div>

      {disposition.time_target && (
        <p className="mt-4 rounded-xl border border-verdict-withheld/30 bg-verdict-withheld/10 px-3.5 py-2.5 text-sm font-semibold text-ink">
          {disposition.time_target}
        </p>
      )}

      {disposition.heart_failure_pathway && (
        <p className="mt-3 rounded-xl border border-verdict-caution/30 bg-verdict-caution/10 px-3.5 py-2.5 text-sm text-ink">
          <span className="font-semibold">Parallel heart-failure pathway. </span>
          The echocardiogram measured an ejection fraction inside the HFrEF range. This
          stands independently of the acute coronary question.
        </p>
      )}

      {terminationReason && (
        <p className="mt-3 text-sm text-ink-muted">
          <span className="font-semibold text-ink">Why it ended here. </span>
          {terminationReason}
        </p>
      )}

      {disposition.rationale.length > 0 && (
        <ul className="mt-4 space-y-1.5 border-t border-line/60 pt-4">
          {disposition.rationale.map((line) => (
            <li key={line} className="text-xs leading-relaxed text-ink-muted">
              {line}
            </li>
          ))}
        </ul>
      )}

      <p className="mt-4 font-mono text-2xs uppercase tracking-wide text-ink-faint">
        {stagesCompleted} of {stagesTotal} stages produced a result
      </p>
    </section>
  );
}
