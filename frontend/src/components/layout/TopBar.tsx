"use client";

/**
 * The application header: identity, module navigation, service state.
 *
 * WHY A TOP RAIL AND NOT A SIDEBAR
 * --------------------------------
 * A fixed sidebar spends a fifth of the viewport permanently on seven links
 * that are read once. The content here is dense and horizontal — findings
 * tables, six-stage timelines, confusion matrices — so the width goes to the
 * data and navigation gets one row.
 *
 * The active tab is a filled pill rather than an underline. On a page that
 * already uses rules to divide sections, another rule for "you are here" is
 * one signal too many; a filled shape reads instantly and survives being
 * glanced at from the back of a room.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";

import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { cn, COMPONENT_ORDER, COMPONENTS } from "@/lib/format";

const ACROSS = [
  { href: "/pathway", label: "Clinical pathway", note: "Six gated stages, in order" },
];

export function TopBar() {
  const pathname = usePathname();

  return (
    <header className="no-print sticky top-0 z-30 border-b border-line bg-surface/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[88rem] items-center gap-6 px-5 sm:px-8">
        <Link href="/" className="flex shrink-0 items-center gap-3">
          <span
            aria-hidden
            className="grid h-9 w-9 place-items-center rounded-lg bg-accent text-[0.8125rem] font-extrabold text-accent-contrast shadow-sm"
          >
            CV
          </span>
          <span className="hidden sm:block">
            <span className="display block text-[0.9375rem] leading-tight text-ink">
              Cardiovascular XAI
            </span>
            <span className="block font-mono text-[0.6875rem] leading-tight text-ink-faint">
              R26-IT-083
            </span>
          </span>
        </Link>

        <nav
          aria-label="Modules"
          className="no-scrollbar -mx-2 flex flex-1 items-center gap-1 overflow-x-auto px-2"
        >
          {COMPONENT_ORDER.map((id) => {
            const meta = COMPONENTS[id];
            return (
              <Tab
                key={id}
                href={meta.href}
                active={pathname === meta.href}
                label={meta.short}
                index={meta.number}
                note={meta.modality}
              />
            );
          })}

          <span className="mx-2 h-6 w-px flex-none bg-line" aria-hidden />

          {ACROSS.map((item) => (
            <Tab
              key={item.href}
              href={item.href}
              active={pathname === item.href}
              label={item.label}
              note={item.note}
            />
          ))}
        </nav>

        <div className="flex shrink-0 items-center gap-3">
          <ServiceStatus />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

function Tab({
  href,
  active,
  label,
  index,
  note,
}: {
  href: string;
  active: boolean;
  label: string;
  index?: string;
  note: string;
}) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      title={note}
      className={cn(
        "flex flex-none items-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition-colors",
        active
          ? "bg-accent/[0.10] text-accent"
          : "text-ink-muted hover:bg-surface-2 hover:text-ink",
      )}
    >
      {index && (
        <span
          className={cn(
            "font-mono text-[0.6875rem] font-bold",
            active ? "text-accent" : "text-ink-faint",
          )}
        >
          {index}
        </span>
      )}
      <span className="whitespace-nowrap">{label}</span>
    </Link>
  );
}
