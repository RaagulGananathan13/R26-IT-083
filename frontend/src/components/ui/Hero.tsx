import { cn } from "@/lib/format";

/**
 * Page header.
 *
 * Set at a real display size with a genuine weight jump, so the top of a page
 * establishes hierarchy on its own rather than relying on a rule beneath it.
 * The pills are the page's standing facts — cohort size, operating point, what
 * the page will and will not do — not decoration.
 */
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
    <header className={cn("rise-in", className)}>
      <p className="eyebrow text-accent">{eyebrow}</p>
      <h1 className="display mt-2.5 text-[1.75rem] leading-[1.15] text-ink sm:text-[2.125rem]">
        {title}
      </h1>
      <p className="mt-3 max-w-[62ch] text-[1.0625rem] leading-relaxed text-ink-muted">
        {subtitle}
      </p>
      {pills && pills.length > 0 && (
        <div className="mt-5 flex flex-wrap gap-2">
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
