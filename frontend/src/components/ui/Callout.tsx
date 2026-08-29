import { cn } from "@/lib/format";

type Tone = "neutral" | "warning" | "danger" | "info";

/**
 * An aside that must be read but is not the page's subject.
 *
 * A tinted surface with a left rule: enough presence to stop the eye, not
 * enough to compete with a verdict banner, which is the only thing on a page
 * that should read as a saturated state.
 */
const TONES: Record<Tone, string> = {
  neutral: "border-l-ink-faint bg-surface-2 text-ink-muted",
  info: "border-l-verdict-deferred bg-verdict-deferred/[0.07] text-ink",
  warning: "border-l-verdict-caution bg-verdict-caution/[0.08] text-ink",
  danger: "border-l-verdict-withheld bg-verdict-withheld/[0.08] text-ink",
};

const TITLE_TONES: Record<Tone, string> = {
  neutral: "text-ink",
  info: "text-verdict-deferred",
  warning: "text-verdict-caution",
  danger: "text-verdict-withheld",
};

export function Callout({
  tone = "neutral",
  title,
  children,
  className,
}: {
  tone?: Tone;
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-line border-l-[3px] px-4 py-3.5 text-sm leading-relaxed",
        TONES[tone],
        className,
      )}
    >
      {title && (
        <p className={cn("mb-1 text-sm font-bold", TITLE_TONES[tone])}>{title}</p>
      )}
      {children}
    </div>
  );
}
