"use client";

import { useHealth } from "@/hooks/useHealth";
import { cn } from "@/lib/format";

/**
 * Live service state in the application header.
 *
 * WHAT THIS DELIBERATELY DOES NOT SHOW
 * ------------------------------------
 * The compute device. `cuda:NVIDIA GeForce RTX 4060 Laptop GPU` is operator
 * telemetry: it tells a clinician nothing about whether a result can be
 * trusted, and putting hardware detail in the chrome of a clinical tool makes
 * it read as a development build. It remains available on /health and on the
 * Method page for anyone who needs it.
 *
 * What a clinician needs from the header is one thing: whether the service can
 * answer right now. That is a state, so it reads as one — a dot and a short
 * phrase, sized to be legible at a glance rather than studied.
 */
export function ServiceStatus() {
  const { health, error, loading } = useHealth();

  const ready =
    health?.components.filter(
      (component) => component.status !== "unavailable" && component.status !== "failed",
    ).length ?? 0;
  const total = health?.components.length ?? 4;
  const allReady = ready === total && total > 0;

  const tone = error
    ? { dot: "bg-verdict-withheld", text: "text-verdict-withheld", ring: "bg-verdict-withheld/15" }
    : loading
      ? { dot: "bg-ink-faint", text: "text-ink-muted", ring: "bg-ink-faint/15" }
      : allReady
        ? { dot: "bg-verdict-actionable", text: "text-verdict-actionable", ring: "bg-verdict-actionable/15" }
        : { dot: "bg-verdict-caution", text: "text-verdict-caution", ring: "bg-verdict-caution/15" };

  const label = error
    ? "Service offline"
    : loading
      ? "Connecting"
      : allReady
        ? "All systems ready"
        : `${ready} of ${total} ready`;

  return (
    <div
      className="flex items-center gap-2.5 rounded-lg border border-line bg-surface px-3 py-2"
      title={
        error
          ? "The backend is not responding. Start it with: python run.py --warm"
          : health
            ? `${ready} of ${total} diagnostic components available`
            : undefined
      }
    >
      {/* A haloed dot: readable from across a room, which a 6px dot is not. */}
      <span className={cn("relative grid h-2.5 w-2.5 place-items-center")} aria-hidden>
        <span className={cn("absolute inset-[-4px] rounded-full", tone.ring)} />
        <span className={cn("h-2 w-2 rounded-full", tone.dot)} />
      </span>
      <span className={cn("whitespace-nowrap text-[0.8125rem] font-semibold", tone.text)}>
        {label}
      </span>
    </div>
  );
}
