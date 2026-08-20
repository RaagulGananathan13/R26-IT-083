"use client";

import { useHealth } from "@/hooks/useHealth";
import { cn } from "@/lib/format";

/** Live backend status in the sidebar, so a cold service is obvious at a glance. */
export function ServiceStatus() {
  const { health, error, loading } = useHealth();

  const tone = error
    ? "bg-verdict-withheld"
    : loading
      ? "bg-ink-faint"
      : health?.status === "ok"
        ? "bg-verdict-actionable"
        : "bg-verdict-caution";

  const label = error
    ? "Backend unreachable"
    : loading
      ? "Checking…"
      : `${health?.components.filter((c) => c.status !== "unavailable" && c.status !== "failed").length ?? 0} of 4 serviceable`;

  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span className={cn("h-2 w-2 shrink-0 rounded-full", tone)} aria-hidden />
        <span className="text-xs font-medium text-ink">{label}</span>
      </div>
      {health && (
        <p className="mt-1 truncate text-2xs text-ink-faint" title={health.device}>
          {health.device}
        </p>
      )}
      {error ? (
        <p className="mt-1 font-mono text-2xs text-ink-faint">python run.py --warm</p>
      ) : null}
    </div>
  );
}
