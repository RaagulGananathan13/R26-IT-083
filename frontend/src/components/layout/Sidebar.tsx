"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { cn, COMPONENT_ORDER, COMPONENTS } from "@/lib/format";

const EXTRA = [
  { href: "/assessment", label: "Multi-modal", note: "All supplied modalities" },
  { href: "/about", label: "Method & cohorts", note: "Why this aggregates" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="no-print flex h-full w-64 shrink-0 flex-col border-r border-line bg-surface/80 backdrop-blur">
      <div className="border-b border-line px-5 py-5">
        <Link href="/" className="block">
          <span className="flex items-center gap-2.5">
            <span
              aria-hidden
              className="grid h-8 w-8 place-items-center rounded-lg bg-accent text-sm font-semibold text-white"
            >
              CV
            </span>
            <span>
              <span className="display block text-[0.95rem] leading-tight text-ink">
                Cardiovascular XAI
              </span>
              <span className="block font-mono text-2xs tracking-wider text-ink-faint">
                R26-IT-083
              </span>
            </span>
          </span>
        </Link>
      </div>

      <nav className="flex-1 overflow-y-auto scrollbar-thin px-3 py-4">
        <p className="eyebrow px-2 pb-2">Components</p>
        <ul className="space-y-0.5">
          {COMPONENT_ORDER.map((id) => {
            const meta = COMPONENTS[id];
            const active = pathname === meta.href;
            return (
              <li key={id}>
                <Link
                  href={meta.href}
                  className={cn(
                    "flex items-start gap-2.5 rounded-lg px-2 py-2 transition-colors",
                    active ? "bg-surface-2" : "hover:bg-surface-2/70",
                  )}
                >
                  <span
                    className={cn(
                      "mt-px font-mono text-2xs font-semibold",
                      active ? "text-ink" : "text-ink-faint",
                    )}
                  >
                    {meta.number}
                  </span>
                  <span className="min-w-0">
                    <span
                      className={cn(
                        "block text-sm font-medium leading-tight",
                        active ? "text-ink" : "text-ink-muted",
                      )}
                    >
                      {meta.short}
                    </span>
                    <span className="block truncate text-2xs text-ink-faint">
                      {meta.modality}
                    </span>
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>

        <p className="eyebrow px-2 pb-2 pt-5">Across components</p>
        <ul className="space-y-0.5">
          {EXTRA.map((item) => {
            const active = pathname === item.href;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "block rounded-lg px-2 py-2 transition-colors",
                    active ? "bg-surface-2" : "hover:bg-surface-2/70",
                  )}
                >
                  <span
                    className={cn(
                      "block text-sm font-medium leading-tight",
                      active ? "text-ink" : "text-ink-muted",
                    )}
                  >
                    {item.label}
                  </span>
                  <span className="block text-2xs text-ink-faint">{item.note}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="space-y-3 border-t border-line px-3 py-3">
        <ServiceStatus />
        <div className="flex items-center justify-between px-1">
          <span className="text-2xs text-ink-faint">Research prototype</span>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
