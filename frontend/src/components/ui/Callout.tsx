import { cn } from "@/lib/format";

type Tone = "neutral" | "warning" | "danger" | "info";

const TONES: Record<Tone, string> = {
  neutral: "border-line bg-surface-2 text-ink-muted",
  info: "border-verdict-deferred/30 bg-verdict-deferred/10 text-ink",
  warning: "border-verdict-caution/30 bg-verdict-caution/10 text-ink",
  danger: "border-verdict-withheld/30 bg-verdict-withheld/10 text-ink",
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
    <div className={cn("rounded-lg border px-4 py-3 text-xs leading-relaxed", TONES[tone], className)}>
      {title && <p className="mb-1 font-semibold text-ink">{title}</p>}
      {children}
    </div>
  );
}
