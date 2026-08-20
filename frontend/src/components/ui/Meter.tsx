import { cn } from "@/lib/format";

/**
 * A proportion on a 0-100 track, with its scale shown.
 *
 * The scale labels are not decoration: a bare bar invites reading length as
 * certainty, and these say what the axis actually is.
 */
export function Meter({
  value,
  tone = "accent",
  showScale = true,
  className,
}: {
  /** 0..1 */
  value: number;
  tone?: "accent" | "caution" | "withheld" | "muted";
  showScale?: boolean;
  className?: string;
}) {
  const percent = Math.max(0, Math.min(100, value * 100));
  const fill = {
    accent: "bg-accent",
    caution: "bg-verdict-caution",
    withheld: "bg-verdict-withheld",
    muted: "bg-ink-faint",
  }[tone];

  return (
    <div className={className}>
      <div
        className="meter"
        role="meter"
        aria-valuenow={Math.round(percent)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span className={cn("meter-fill", fill)} style={{ width: `${percent}%` }} />
      </div>
      {showScale && (
        <div className="mt-1.5 flex justify-between text-2xs text-ink-faint">
          <span>0</span>
          <span>50</span>
          <span>100</span>
        </div>
      )}
    </div>
  );
}
