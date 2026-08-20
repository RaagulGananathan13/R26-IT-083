import { Card, CardBody, CardHeader } from "@/components/ui";
import { cn, decimal, percent } from "@/lib/format";
import type { Actionability, Finding } from "@/lib/types";
import { isActionable } from "@/lib/types";

const ZONE_LABEL: Record<string, string> = {
  rule_in: "rule in",
  rule_out: "rule out",
  refer: "refer",
};

/**
 * Findings, gated by the verdict.
 *
 * The gating is the point. When a component withholds its output the findings
 * are not rendered at all — showing a suppressed probability next to a warning
 * invites the reader to use it anyway. When a component defers, the numbers
 * are shown but pushed back visually, because they were computed and are worth
 * seeing without being an answer.
 */
export function FindingsTable({
  findings,
  actionability,
  title = "Findings",
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
            The reason is in the banner above.
          </p>
        </CardBody>
      </Card>
    );
  }

  if (findings.length === 0) {
    return null;
  }

  const muted = !isActionable(actionability);

  return (
    <Card>
      <CardHeader
        title={title}
        description={
          description ??
          (muted
            ? "Computed, but the component declined to commit. Read these as working notes, not as an answer."
            : undefined)
        }
      />
      <div className={cn("overflow-x-auto scrollbar-thin", muted && "not-actionable")}>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th className="eyebrow px-5 py-2 font-semibold">Finding</th>
              <th className="eyebrow px-3 py-2 text-right font-semibold">Value</th>
              <th className="eyebrow px-3 py-2 text-right font-semibold">Operating point</th>
              <th className="eyebrow px-5 py-2 text-right font-semibold">Call</th>
            </tr>
          </thead>
          <tbody>
            {findings.map((finding, index) => (
              <FindingRow key={`${finding.name}-${index}`} finding={finding} />
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function FindingRow({ finding }: { finding: Finding }) {
  const value =
    finding.value !== null
      ? `${decimal(finding.value, 2)}${finding.unit ? ` ${finding.unit}` : ""}`
      : finding.probability !== null
        ? percent(finding.probability, 1)
        : "—";

  return (
    <tr className="border-b border-line/60 last:border-0">
      <td className="px-5 py-2.5">
        <span className="font-medium text-ink">{finding.name}</span>
        {finding.interval && finding.interval.length === 2 && (
          <span className="ml-2 text-xs text-ink-faint">
            95 % interval {decimal(finding.interval[0], 1)}–{decimal(finding.interval[1], 1)}
          </span>
        )}
        {finding.evidence && (
          <p className="mt-0.5 text-xs leading-relaxed text-ink-faint">{finding.evidence}</p>
        )}
      </td>
      <td className="px-3 py-2.5 text-right tabular font-medium text-ink">{value}</td>
      <td className="px-3 py-2.5 text-right tabular text-xs text-ink-faint">
        {finding.threshold !== null ? decimal(finding.threshold, 3) : "—"}
      </td>
      <td className="px-5 py-2.5 text-right">
        <Call finding={finding} />
      </td>
    </tr>
  );
}

function Call({ finding }: { finding: Finding }) {
  if (finding.zone) {
    const zone = ZONE_LABEL[finding.zone] ?? finding.zone;
    return (
      <span
        className={cn(
          "text-xs font-semibold",
          finding.zone === "rule_in" && "text-verdict-caution",
          finding.zone === "rule_out" && "text-ink-muted",
          finding.zone === "refer" && "text-verdict-deferred",
        )}
      >
        {zone}
      </span>
    );
  }
  if (finding.label && finding.present === null) {
    return <span className="text-xs font-semibold text-ink">{finding.label}</span>;
  }
  if (finding.present === null) return <span className="text-xs text-ink-faint">—</span>;
  return (
    <span
      className={cn(
        "text-xs font-semibold",
        finding.present ? "text-verdict-caution" : "text-ink-muted",
      )}
    >
      {finding.present ? "present" : "absent"}
    </span>
  );
}
