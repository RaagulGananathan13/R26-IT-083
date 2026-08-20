import { cn } from "@/lib/format";

/** Page opener: eyebrow, display-serif title, subtitle, and fact pills. */
export function Hero({
  eyebrow,
  title,
  subtitle,
  pills,
  className,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
  pills?: string[];
  className?: string;
}) {
  return (
    <header className={cn("rise-in grid gap-4 pb-2 pt-1", className)}>
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="display mt-1.5 text-[clamp(1.75rem,2.6vw,2.5rem)] leading-[1.12] text-ink">
          {title}
        </h1>
        <p className="mt-2.5 max-w-3xl text-[0.95rem] leading-relaxed text-ink-muted">
          {subtitle}
        </p>
      </div>
      {pills && pills.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {pills.map((pill) => (
            <span key={pill} className="pill">
              {pill}
            </span>
          ))}
        </div>
      )}
    </header>
  );
}
