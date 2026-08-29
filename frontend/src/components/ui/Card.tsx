import { cn } from "@/lib/format";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <section className={cn("card", className)} {...props} />;
}

export function CardHeader({
  title,
  description,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex items-start justify-between gap-5 border-b border-line px-6 py-4",
        className,
      )}
    >
      <div className="min-w-0">
        <h2 className="text-[0.9375rem] font-bold text-ink">{title}</h2>
        {description && (
          <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">{description}</p>
        )}
      </div>
      {actions && <div className="shrink-0">{actions}</div>}
    </header>
  );
}

export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-6 py-5", className)} {...props} />;
}

/* ------------------------------------------------------------------ *
 * Flat variants.
 *
 * A card inside a card reads as a mistake: two borders, two backgrounds
 * and doubled padding around content that belongs to the outer panel. The
 * explainer components are used both standalone on their own console and
 * nested inside a pathway stage, so they pick their frame with `frame()`
 * rather than being duplicated or wrapped in something that strips their
 * styling back off again.
 * ------------------------------------------------------------------ */

function FlatFrame({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("min-w-0", className)} {...props} />;
}

function FlatHeader({
  title,
  description,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header className={cn("mb-3 flex items-start justify-between gap-4", className)}>
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-ink">{title}</h3>
        {description && (
          <p className="mt-1 text-xs leading-relaxed text-ink-muted">{description}</p>
        )}
      </div>
      {actions && <div className="shrink-0">{actions}</div>}
    </header>
  );
}

function FlatBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn(className)} {...props} />;
}

/** Card chrome when standalone, no chrome when nested. */
export function frame(bare?: boolean) {
  return bare
    ? { Frame: FlatFrame, FrameHeader: FlatHeader, FrameBody: FlatBody }
    : { Frame: Card, FrameHeader: CardHeader, FrameBody: CardBody };
}
