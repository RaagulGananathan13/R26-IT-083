import { Card, CardBody, CardHeader } from "@/components/ui";
import { cn, percent } from "@/lib/format";
import type { Actionability } from "@/lib/types";
import { isActionable } from "@/lib/types";

const ORDER = ["No_ACS", "UA", "NSTEMI", "STEMI"];
const FULL: Record<string, string> = {
  No_ACS: "No acute coronary syndrome",
  UA: "Unstable angina",
  NSTEMI: "Non-ST-elevation MI",
  STEMI: "ST-elevation MI",
};

/**
 * The four-class distribution as a bar chart.
 *
 * Deliberately shows the reweighted distribution the decision layer actually
 * argmaxes over, not the raw model output, because the reweighting is what
 * produced the predicted class. Where they disagree the reader should see the
 * one that drove the call.
 */
export function ClassDistribution({
  probabilities,
  predicted,
  actionability,
}: {
  probabilities: Record<string, number>;
  predicted: string;
  actionability: Actionability;
}) {
  const muted = !isActionable(actionability);

  return (
    <Card>
      <CardHeader
        title="Four-class distribution"
        description="After the constrained decision layer's class weights, which is the distribution the prediction is taken from."
      />
      <CardBody className={cn("space-y-3", muted && "not-actionable")}>
        {ORDER.filter((name) => name in probabilities).map((name) => {
          const value = probabilities[name] ?? 0;
          const isPredicted = name === predicted;
          return (
            <div key={name}>
              <div className="flex items-baseline justify-between gap-3">
                <span
                  className={cn(
                    "text-sm",
                    isPredicted ? "font-semibold text-ink" : "text-ink-muted",
                  )}
                >
                  {FULL[name] ?? name}
                </span>
                <span className="tabular text-xs text-ink-muted">{percent(value, 2)}</span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
                <div
                  className={cn(
                    "h-full rounded-full",
                    isPredicted ? "bg-ink" : "bg-ink-faint/50",
                  )}
                  style={{ width: `${Math.max(0.5, value * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </CardBody>
    </Card>
  );
}
