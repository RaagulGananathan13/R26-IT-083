import { Card, CardBody, CardHeader } from "@/components/ui";
import { cn, percent } from "@/lib/format";
import type { Actionability, Finding } from "@/lib/types";
import { isActionable } from "@/lib/types";

/**
 * Co-findings as chips, in Component 01's idiom: present ones first, absent
 * ones after, each with its probability.
 *
 * Chips rather than a table because these are secondary findings scanned at a
 * glance. The primary finding gets the verdict banner and its own meter; this
 * is the "what else did the multi-label head say" row.
 */
export function FindingsGrid({
  findings,
  actionability,
  title = "Secondary findings",
  description,
}: {
  findings: Finding[];
  actionability: Actionability;
  title?: string;
  description?: string;
}) {
  if (actionability === "withheld") {
    return (
      <Card>
        <CardHeader title={title} />
        <CardBody>
          <p className="text-sm leading-relaxed text-ink-muted">
            No findings are shown. The component suppressed its output, and displaying a
            probability beside that notice would invite it to be read as a result anyway.
          </p>
        </CardBody>
      </Card>
    );
  }
  if (findings.length === 0) return null;

  const present = findings.filter((f) => f.present === true);
  const absent = findings.filter((f) => f.present === false);
  const graded = findings.filter((f) => f.present === null);
  const muted = !isActionable(actionability);

  return (
    <Card>
      <CardHeader
        title={title}
        description={
          description ??
          (muted
            ? "Computed, but the component declined to commit. Working notes, not an answer."
            : undefined)
        }
      />
      <CardBody className={cn(muted && "not-actionable")}>
        <div className="flex flex-wrap gap-2">
          {[...present, ...graded, ...absent].map((finding, index) => (
            <Chip key={`${finding.name}-${index}`} finding={finding} />
          ))}
        </div>
        {present.length === 0 && graded.length === 0 && (
          <p className="mt-3 text-xs text-ink-muted">
            No additional pathologies were called by the classifier.
          </p>
        )}
      </CardBody>
    </Card>
  );
}

function Chip({ finding }: { finding: Finding }) {
  const present = finding.present === true;
  const zone = finding.zone;
  const tone = zone
    ? zone === "rule_in"
      ? "caution"
      : zone === "refer"
        ? "deferred"
        : "muted"
    : present
      ? "caution"
      : "muted";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs",
        tone === "caution" && "border-verdict-caution/40 bg-verdict-caution/10 text-ink",
        tone === "deferred" && "border-verdict-deferred/40 bg-verdict-deferred/10 text-ink",
        tone === "muted" && "border-line bg-surface-2 text-ink-muted",
      )}
      title={finding.evidence ?? undefined}
    >
      <span
        aria-hidden
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          tone === "caution" && "bg-verdict-caution",
          tone === "deferred" && "bg-verdict-deferred",
          tone === "muted" && "bg-ink-faint",
        )}
      />
      <span className="font-medium">{finding.name}</span>
      {zone && <span className="text-ink-muted">{zone.replace("_", " ")}</span>}
      {finding.probability !== null && (
        <span className="tabular text-ink-muted">{percent(finding.probability, 0)}</span>
      )}
      {finding.threshold !== null && (
        <span className="tabular text-2xs text-ink-faint">
          thr {finding.threshold.toFixed(2)}
        </span>
      )}
    </span>
  );
}
